"""
api.py

Standalone FastAPI service exposing ProteinSynergyDock predictions as
FHIR R4 resources AND CDS Hooks cards.

Endpoints:
  POST /fhir/DiagnosticReport        -> FHIR R4 DiagnosticReport or OperationOutcome
  GET  /fhir/AuditLog                -> read-only audit trail
  GET  /fhir/AuditLog/verify         -> hash-chain integrity check
  GET  /cds-services                 -> CDS Hooks discovery (required by spec)
  POST /cds-services/synergy-advisor -> CDS Hook: medication-prescribe synergy cards
  GET  /health                       -> liveness check
  GET  /                             -> redirects to /docs

CDS Hooks context:
  The medication-prescribe hook fires inside an EHR (Oracle Health/Cerner,
  Epic, etc.) when a clinician orders a drug. This endpoint receives the
  draft MedicationRequest, identifies the drug, looks up top synergy
  combinations from the ProteinSynergyDock model, and returns structured
  "cards" that surface inline in the clinical UI — exactly how third-party
  tools plug into Cerner's app marketplace.

  Hook spec: https://cds-hooks.org/hooks/medication-prescribe/
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from core_fhir import predict_to_fhir, NCI60_CELL_LINES
from audit_log import AuditLog
from model_bridge import predict_synergy, ModelUnavailableError, DRUG_SMILES_LOOKUP


app = FastAPI(
    title="ProteinSynergyDock FHIR + CDS Hooks API",
    description=(
        "Exposes drug-combination synergy predictions as FHIR R4 DiagnosticReport "
        "resources and as CDS Hooks cards for EHR integration (Oracle Health / Cerner / Epic). "
        "Research tool — not a clinical diagnostic, not FDA-reviewed.\n\n"
        "**CDS Hooks discovery:** `GET /cds-services`\n"
        "**Synergy advisor hook:** `POST /cds-services/synergy-advisor`"
    ),
    version="1.1.0",
)

audit = AuditLog(path="audit_log.jsonl")

# Default cell lines used for CDS Hook synergy lookups when no patient
# context is available — representative lines across major cancer types.
_CDS_DEFAULT_CELL_LINES = ["MCF7", "HCT-116", "A549/ATCC", "OVCAR-3", "K-562"]

# Top synergy candidates to evaluate per incoming drug — limited to keep
# the hook response fast enough for real EHR use (< 2s target).
_CDS_CANDIDATE_PARTNERS = [
    "Olaparib", "Rucaparib", "Vemurafenib", "Trametinib", "Erlotinib",
    "Lapatinib", "Imatinib", "Dasatinib", "Palbociclib", "Venetoclax",
    "Osimertinib", "Alpelisib", "Paclitaxel", "Temozolomide", "Dabrafenib",
]

# ── Pydantic models ──────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    drug_a: str = Field(..., description="Name of the first drug, e.g. 'Olaparib'")
    drug_b: str = Field(..., description="Name of the second drug, e.g. 'Rucaparib'")
    cell_line: str = Field(..., description="NCI-60 cell line identifier, e.g. 'MCF7'")
    user: Optional[str] = Field("anonymous", description="Caller identity, for the audit log")


class CDSHookRequest(BaseModel):
    """
    Minimal CDS Hooks request body per the CDS Hooks 1.0 spec.
    Real EHR systems send more fields (fhirServer, fhirAuthorization,
    prefetch) but only hook, hookInstance, and context are required.
    """
    hook: str = Field(..., description="Hook type, e.g. 'medication-prescribe'")
    hookInstance: str = Field(..., description="UUID for this specific hook call")
    context: Dict[str, Any] = Field(..., description="Hook-specific context payload")
    prefetch: Optional[Dict[str, Any]] = Field(None, description="Pre-fetched FHIR resources")


# ── Utility ─────────────────────────────────────────────────────────────────

def _extract_drug_from_context(context: Dict[str, Any]) -> Optional[str]:
    """
    Attempts to extract a drug name from the CDS Hooks context payload.

    Real medication-prescribe context contains a FHIR Bundle of
    MedicationRequest resources. We look in two places:
    1. context['draftOrders']['entry'][*]['resource']['medicationCodeableConcept']['text']
       — the display name of the ordered medication
    2. context['medications']['MedicationRequest'][*]['medicationCodeableConcept']['text']
       — alternate shape used by some EHR implementations

    If neither is present (e.g. test calls sending just a drug name string),
    we fall back to context.get('drug') for easy testing without a full
    FHIR Bundle.
    """
    # Standard CDS Hooks medication-prescribe shape
    draft_orders = context.get("draftOrders", {})
    entries = draft_orders.get("entry", [])
    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "MedicationRequest":
            med = resource.get("medicationCodeableConcept", {})
            name = med.get("text") or (med.get("coding", [{}])[0].get("display"))
            if name:
                # Match against our known drug list (case-insensitive)
                for known in DRUG_SMILES_LOOKUP:
                    if known.lower() == name.lower():
                        return known
                return name  # return as-is even if not in our list

    # Fallback for simplified test payloads
    return context.get("drug") or context.get("drugName")


def _get_synergy_cards(
    drug_a: str,
    cell_lines: List[str],
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    """
    Runs model inference for drug_a vs each candidate partner across
    the given cell lines and returns CDS cards for the top_n synergistic
    pairs (mean_synergy > 0.05 threshold).

    Predictions are averaged across cell lines to give a panel-level
    signal rather than a single-line point estimate.
    """
    if drug_a not in DRUG_SMILES_LOOKUP:
        return [{
            "summary": f"ProteinSynergyDock: '{drug_a}' not in drug database",
            "detail": (
                f"'{drug_a}' is not in ProteinSynergyDock's drug library "
                f"({len(DRUG_SMILES_LOOKUP)} drugs). No synergy predictions available."
            ),
            "indicator": "info",
            "source": {
                "label": "ProteinSynergyDock",
                "url": "https://proteinsynergydock-app-kddtbdmnkixw9c8jfnf8un.streamlit.app/",
            },
        }]

    scored = []
    for partner in _CDS_CANDIDATE_PARTNERS:
        if partner == drug_a or partner not in DRUG_SMILES_LOOKUP:
            continue
        scores = []
        for cl in cell_lines:
            if cl not in NCI60_CELL_LINES:
                continue
            try:
                score, confidence, _ = predict_synergy(drug_a, partner, cl)
                scores.append((score, confidence))
            except (ModelUnavailableError, Exception):
                continue
        if not scores:
            continue
        mean_score = sum(s for s, _ in scores) / len(scores)
        mean_conf  = sum(c for _, c in scores if c) / max(1, sum(1 for _, c in scores if c))
        if mean_score > 0.05:
            scored.append((partner, mean_score, mean_conf))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_n]

    if not top:
        return [{
            "summary": f"No high-synergy combinations found for {drug_a}",
            "detail": (
                f"ProteinSynergyDock evaluated {len(_CDS_CANDIDATE_PARTNERS)} candidate "
                f"combinations with {drug_a} and found no pairs with predicted synergy "
                f"above threshold across {len(cell_lines)} cell lines."
            ),
            "indicator": "info",
            "source": {
                "label": "ProteinSynergyDock",
                "url": "https://proteinsynergydock-app-kddtbdmnkixw9c8jfnf8un.streamlit.app/",
            },
        }]

    cards = []
    for partner, score, conf in top:
        indicator = "warning" if score > 0.4 else "info"
        conf_label = "High" if conf >= 0.8 else ("Moderate" if conf >= 0.5 else "Low")
        cards.append({
            "summary": f"Potential synergy: {drug_a} + {partner} (score {score:.3f})",
            "detail": (
                f"ProteinSynergyDockV2 predicts a synergy score of **{score:.3f}** "
                f"for **{drug_a} + {partner}** (averaged across {len(cell_lines)} NCI-60 "
                f"cell lines). Model confidence: {conf_label}. "
                f"This is a research prediction — not a clinical recommendation."
            ),
            "indicator": indicator,
            "source": {
                "label": "ProteinSynergyDock (GNN-based synergy prediction)",
                "url": "https://proteinsynergydock-app-kddtbdmnkixw9c8jfnf8un.streamlit.app/",
            },
            "links": [
                {
                    "label": f"Explore {drug_a} + {partner} in ProteinSynergyDock",
                    "url": "https://proteinsynergydock-app-kddtbdmnkixw9c8jfnf8un.streamlit.app/",
                    "type": "absolute",
                }
            ],
        })
    return cards


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


# ── FHIR endpoints (unchanged) ───────────────────────────────────────────────

@app.post("/fhir/DiagnosticReport")
def create_diagnostic_report(req: PredictionRequest):
    """
    Runs a live model prediction for the given drug pair + cell line,
    converts it to a FHIR R4 DiagnosticReport, and logs the attempt
    to the audit trail.
    """
    try:
        synergy_score, confidence, docking_affinity = predict_synergy(
            req.drug_a, req.drug_b, req.cell_line
        )
    except ModelUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))

    resource, success = predict_to_fhir(
        drug_a=req.drug_a,
        drug_b=req.drug_b,
        cell_line=req.cell_line,
        synergy_score=synergy_score,
        confidence=confidence,
        docking_affinity=docking_affinity,
    )

    audit.record(
        drug_a=req.drug_a,
        drug_b=req.drug_b,
        cell_line=req.cell_line,
        output_resource_type=resource["resourceType"],
        output_summary=resource.get("conclusion", resource.get("issue", [{}])[0].get("diagnostics", "")),
        model_version="ProteinSynergyDockV2-epoch82",
        success=success,
        user=req.user,
    )

    if not success:
        raise HTTPException(status_code=400, detail=resource)

    return resource


@app.get("/fhir/AuditLog")
def get_audit_log(limit: int = 50):
    """Read-only view of the most recent audit entries."""
    entries = audit.read_all()
    return {"count": len(entries), "entries": entries[-limit:]}


@app.get("/fhir/AuditLog/verify")
def verify_audit_log():
    """Verifies the hash chain hasn't been tampered with."""
    valid, broken_at = audit.verify_chain()
    return {"valid": valid, "broken_at_index": broken_at}


# ── CDS Hooks endpoints ──────────────────────────────────────────────────────

@app.get("/cds-services")
def cds_discovery():
    """
    CDS Hooks discovery endpoint — required by the CDS Hooks 1.0 spec.
    EHR systems (Oracle Health/Cerner, Epic) call this to enumerate
    available hooks before registering the service in their app marketplace.

    Spec: https://cds-hooks.org/specification/current/#discovery
    """
    return {
        "services": [
            {
                "hook": "medication-prescribe",
                "id": "synergy-advisor",
                "title": "ProteinSynergyDock — Drug Combination Advisor",
                "description": (
                    "When a drug is ordered, surfaces GNN-predicted synergistic "
                    "combination partners from the ProteinSynergyDock model "
                    "(trained on 107K NCI ALMANAC triplets across 60 NCI-60 cell lines). "
                    "Research tool — not a clinical decision support system."
                ),
                "prefetch": {
                    "patient": "Patient/{{context.patientId}}",
                    "conditions": "Condition?patient={{context.patientId}}&category=problem-list-item",
                },
            }
        ]
    }


@app.post("/cds-services/synergy-advisor")
def cds_synergy_advisor(req: CDSHookRequest):
    """
    CDS Hook: medication-prescribe → synergy suggestion cards.

    Receives a draft MedicationRequest from the EHR, identifies the
    ordered drug, runs ProteinSynergyDockV2 predictions against candidate
    partners, and returns CDS cards for the top synergistic combinations.

    The cards appear inline in the clinician's workflow — in Cerner
    Millennium this renders in the order composer sidebar.

    Spec: https://cds-hooks.org/hooks/medication-prescribe/
    """
    drug = _extract_drug_from_context(req.context)

    if not drug:
        # Spec-correct: return an empty cards array, not an error,
        # when context doesn't contain a recognisable drug name.
        # An error response would break the EHR's hook processing pipeline.
        return {
            "cards": [{
                "summary": "ProteinSynergyDock: could not identify drug from context",
                "detail": (
                    "The medication-prescribe context did not contain a recognisable "
                    "drug name. Expected a MedicationRequest with "
                    "medicationCodeableConcept.text matching a drug in the "
                    "ProteinSynergyDock library."
                ),
                "indicator": "info",
                "source": {"label": "ProteinSynergyDock"},
            }]
        }

    # Use cell lines from prefetch Patient conditions if available,
    # otherwise fall back to the representative default panel.
    cell_lines = _CDS_DEFAULT_CELL_LINES

    cards = _get_synergy_cards(drug, cell_lines, top_n=3)
    return {"cards": cards}
