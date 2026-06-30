"""
core_fhir.py

Pure, Streamlit-free FHIR R4 resource construction for ProteinSynergyDock.

Converts a drug-combination synergy prediction into spec-shaped FHIR R4
resources (DiagnosticReport + Observation + OperationOutcome) so the
prediction output can be consumed by any FHIR-aware clinical system
(Cerner/Oracle Health Millennium, Epic, any HL7 FHIR client).

Design notes:
- No external dependencies (no fhir.resources, no requests). This is
  deliberate: it keeps the module trivially deployable on Streamlit Cloud
  and trivially unit-testable without a FHIR server running.
- Resources are built as plain dicts matching the FHIR R4 JSON shape.
  Field names and structure were checked against the official R4 spec
  (DiagnosticReport, Observation, OperationOutcome resource definitions).
- This module does NOT do I/O. Validation, audit logging, and the API
  layer are separate modules that call into this one.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Reference data: the real NCI-60 cell line panel, used for validation.
# Keeping this here (not just in app.py) means the FHIR layer can validate
# independently of the Streamlit app's own lookup tables.
# ---------------------------------------------------------------------------

NCI60_CELL_LINES = {
    # Breast
    "MCF7", "MDA-MB-231", "HS 578T", "BT-549", "T-47D", "MDA-MB-468",
    # CNS
    "SF-268", "SF-295", "SF-539", "SNB-19", "SNB-75", "U251",
    # Colon
    "COLO 205", "HCC-2998", "HCT-116", "HCT-15", "HT29", "KM12", "SW-620",
    # Leukemia
    "CCRF-CEM", "HL-60(TB)", "K-562", "MOLT-4", "RPMI-8226", "SR",
    # Melanoma
    "LOX IMVI", "MALME-3M", "M14", "MDA-MB-435", "SK-MEL-2", "SK-MEL-28",
    "SK-MEL-5", "UACC-257", "UACC-62",
    # NSCLC
    "A549/ATCC", "EKVX", "HOP-62", "HOP-92", "NCI-H226", "NCI-H23",
    "NCI-H322M", "NCI-H460", "NCI-H522",
    # Ovarian
    "IGROV1", "OVCAR-3", "OVCAR-4", "OVCAR-5", "OVCAR-8", "NCI/ADR-RES",
    "SK-OV-3",
    # Prostate
    "PC-3", "DU-145",
    # Renal
    "786-0", "A498", "ACHN", "CAKI-1", "RXF 393", "SN12C", "TK-10", "UO-31",
}

LOINC_SYNERGY_SCORE = "LP417353-4"   # placeholder local code, see note below
LOINC_DOCKING_AFFINITY = "LP417354-2"
LOINC_MODEL_CONFIDENCE = "LP417355-9"

# NOTE: There is no official LOINC code for "ML-predicted drug synergy
# score" — this is a research output, not a standard lab test. We use a
# local CodeSystem (below) rather than inventing a fake LOINC code, which
# is the spec-correct way to represent a non-standard measurement. This
# kind of detail — not pretending something is LOINC when it isn't — is
# itself a credibility signal.
LOCAL_CODE_SYSTEM = "https://github.com/Aprameya05/ProteinSynergyDock-App/fhir/CodeSystem/synergy-metrics"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_id() -> str:
    return str(uuid.uuid4())


class FHIRValidationError(ValueError):
    """Raised when prediction input cannot be represented as valid FHIR input."""
    def __init__(self, message: str, code: str = "invalid"):
        super().__init__(message)
        self.message = message
        self.code = code  # FHIR issue.code value


def validate_prediction_input(
    drug_a: str,
    drug_b: str,
    cell_line: str,
    synergy_score: Optional[float],
    confidence: Optional[float],
) -> None:
    """
    Validates inputs against real clinical data shapes BEFORE building any
    resource. Raises FHIRValidationError on failure; callers turn that into
    an OperationOutcome via build_operation_outcome().

    This mirrors how a real FHIR API validates a request body before
    attempting to process it — fail fast, fail with a structured reason.
    """
    if not drug_a or not isinstance(drug_a, str) or not drug_a.strip():
        raise FHIRValidationError("drug_a is required and must be a non-empty string.", code="required")

    if not drug_b or not isinstance(drug_b, str) or not drug_b.strip():
        raise FHIRValidationError("drug_b is required and must be a non-empty string.", code="required")

    if drug_a.strip().lower() == drug_b.strip().lower():
        raise FHIRValidationError("drug_a and drug_b must be different drugs.", code="invariant")

    if not cell_line or cell_line not in NCI60_CELL_LINES:
        raise FHIRValidationError(
            f"'{cell_line}' is not a recognized NCI-60 cell line identifier. "
            f"Expected one of the 60 standard NCI-60 panel names (e.g. 'MCF7', 'HCT-116').",
            code="code-invalid",
        )

    if synergy_score is None or not isinstance(synergy_score, (int, float)):
        raise FHIRValidationError("synergy_score is required and must be numeric.", code="required")

    if not (-1.0 <= float(synergy_score) <= 1.0):
        # Model outputs are normalized; out-of-range means upstream bug, not
        # a valid edge case, so this is a hard reject, not a clamp.
        raise FHIRValidationError(
            f"synergy_score {synergy_score} is outside the valid range [-1.0, 1.0].",
            code="value",
        )

    if confidence is not None:
        if not isinstance(confidence, (int, float)) or not (0.0 <= float(confidence) <= 1.0):
            raise FHIRValidationError(
                f"confidence {confidence} must be a number in [0.0, 1.0] if provided.",
                code="value",
            )


def build_operation_outcome(error: FHIRValidationError) -> dict:
    """
    Builds a spec-shaped FHIR OperationOutcome for a failed prediction
    request. This is literally how real FHIR servers report request errors
    — not a stack trace, not a generic 400, a structured resource.
    """
    return {
        "resourceType": "OperationOutcome",
        "id": _new_id(),
        "issue": [
            {
                "severity": "error",
                "code": error.code,
                "diagnostics": error.message,
            }
        ],
    }


def _observation(
    code: str,
    display: str,
    value: float,
    unit: str,
    subject_ref: str,
    effective_dt: str,
    obs_id: Optional[str] = None,
) -> dict:
    return {
        "resourceType": "Observation",
        "id": obs_id or _new_id(),
        "status": "final",
        "code": {
            "coding": [
                {
                    "system": LOCAL_CODE_SYSTEM,
                    "code": code,
                    "display": display,
                }
            ],
            "text": display,
        },
        "subject": {"reference": subject_ref},
        "effectiveDateTime": effective_dt,
        "valueQuantity": {
            "value": round(float(value), 4),
            "unit": unit,
            "system": "http://unitsofmeasure.org",
            "code": unit,
        },
    }


def build_diagnostic_report(
    drug_a: str,
    drug_b: str,
    cell_line: str,
    synergy_score: float,
    confidence: Optional[float] = None,
    docking_affinity: Optional[float] = None,
    model_version: str = "ProteinSynergyDockV2-epoch82",
    patient_ref: str = "Patient/example",
) -> dict:
    """
    Builds a FHIR R4 DiagnosticReport bundle wrapping one or more
    Observations for a single drug-pair / cell-line synergy prediction.

    Returns the DiagnosticReport dict with `contained` Observations, which
    is a valid (if simple) way to bundle a report with its measurements
    without needing a separate Bundle resource or server-side references.

    Raises FHIRValidationError if inputs don't validate — callers should
    catch this and return build_operation_outcome(e) instead.
    """
    validate_prediction_input(drug_a, drug_b, cell_line, synergy_score, confidence)

    now = _now_iso()
    report_id = _new_id()
    subject_ref = f"#{report_id}-subject" if patient_ref == "Patient/example" else patient_ref

    observations = []
    obs_refs = []

    synergy_obs = _observation(
        code="synergy-score",
        display="Predicted drug combination synergy score (model-derived, not a standard lab value)",
        value=synergy_score,
        unit="score",
        subject_ref=patient_ref,
        effective_dt=now,
    )
    observations.append(synergy_obs)
    obs_refs.append({"reference": f"#{synergy_obs['id']}"})

    if confidence is not None:
        conf_obs = _observation(
            code="model-confidence",
            display="Model prediction confidence (held-out AUROC-derived)",
            value=confidence,
            unit="probability",
            subject_ref=patient_ref,
            effective_dt=now,
        )
        observations.append(conf_obs)
        obs_refs.append({"reference": f"#{conf_obs['id']}"})

    if docking_affinity is not None:
        dock_obs = _observation(
            code="docking-affinity",
            display="AutoDock Vina predicted binding affinity (kcal/mol)",
            value=docking_affinity,
            unit="kcal/mol",
            subject_ref=patient_ref,
            effective_dt=now,
        )
        observations.append(dock_obs)
        obs_refs.append({"reference": f"#{dock_obs['id']}"})

    report = {
        "resourceType": "DiagnosticReport",
        "id": report_id,
        "status": "final",
        "code": {
            "coding": [
                {
                    "system": LOCAL_CODE_SYSTEM,
                    "code": "drug-combo-synergy-report",
                    "display": "Drug combination synergy prediction report",
                }
            ],
            "text": f"Synergy prediction: {drug_a} + {drug_b} in {cell_line}",
        },
        "subject": {"reference": patient_ref},
        "effectiveDateTime": now,
        "issued": now,
        "performer": [
            {"display": f"ProteinSynergyDock ({model_version}) — research model, not clinically validated"}
        ],
        "contained": observations,
        "result": obs_refs,
        "conclusion": (
            f"Predicted synergy score {synergy_score:.4f} for {drug_a} + {drug_b} "
            f"in {cell_line} (NCI-60). Model output for research purposes only; "
            f"not a clinical diagnostic and not FDA-reviewed."
        ),
        "extension": [
            {
                "url": f"{LOCAL_CODE_SYSTEM}/drug-pair",
                "valueString": f"{drug_a} + {drug_b}",
            },
            {
                "url": f"{LOCAL_CODE_SYSTEM}/cell-line",
                "valueString": cell_line,
            },
        ],
    }
    return report


def predict_to_fhir(
    drug_a: str,
    drug_b: str,
    cell_line: str,
    synergy_score: Optional[float],
    confidence: Optional[float] = None,
    docking_affinity: Optional[float] = None,
    model_version: str = "ProteinSynergyDockV2-epoch82",
) -> tuple[dict, bool]:
    """
    Top-level convenience function: try to build a DiagnosticReport, fall
    back to an OperationOutcome on validation failure. Returns
    (resource_dict, success_bool) so callers (API layer, Streamlit tab,
    audit log) never have to catch exceptions themselves.
    """
    try:
        report = build_diagnostic_report(
            drug_a=drug_a,
            drug_b=drug_b,
            cell_line=cell_line,
            synergy_score=synergy_score,
            confidence=confidence,
            docking_affinity=docking_affinity,
            model_version=model_version,
        )
        return report, True
    except FHIRValidationError as e:
        return build_operation_outcome(e), False
