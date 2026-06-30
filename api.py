"""
api.py

Standalone FastAPI service exposing ProteinSynergyDock predictions as
FHIR R4 resources. Deployed separately from the Streamlit app (Streamlit
Cloud can't host a second ASGI service on the same dyno) — this is meant
to run on Render's free tier or equivalent.

Endpoints:
  POST /fhir/DiagnosticReport   -> run a prediction, return DiagnosticReport
                                    or OperationOutcome, log to audit trail
  GET  /fhir/AuditLog           -> read back the audit trail (read-only)
  GET  /fhir/AuditLog/verify    -> verify hash-chain integrity
  GET  /health                  -> liveness check
  GET  /                        -> redirects to /docs

This file intentionally has almost no logic of its own — it's a thin
HTTP wrapper around core_fhir.py and audit_log.py, which are independently
unit-tested. That separation (HTTP layer vs. pure logic) is the same
pattern as core.py / app.py in the Streamlit app.
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from core_fhir import predict_to_fhir
from audit_log import AuditLog
from model_bridge import predict_synergy, ModelUnavailableError


app = FastAPI(
    title="ProteinSynergyDock FHIR API",
    description=(
        "Exposes drug-combination synergy predictions as FHIR R4 "
        "DiagnosticReport resources for clinical-interoperability use cases. "
        "Research tool — not a clinical diagnostic, not FDA-reviewed."
    ),
    version="1.0.0",
)

audit = AuditLog(path="audit_log.jsonl")


class PredictionRequest(BaseModel):
    drug_a: str = Field(..., description="Name of the first drug, e.g. 'Olaparib'")
    drug_b: str = Field(..., description="Name of the second drug, e.g. 'Rucaparib'")
    cell_line: str = Field(..., description="NCI-60 cell line identifier, e.g. 'MCF7'")
    user: Optional[str] = Field("anonymous", description="Caller identity, for the audit log")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/fhir/DiagnosticReport")
def create_diagnostic_report(req: PredictionRequest):
    """
    Runs a live model prediction for the given drug pair + cell line,
    converts it to a FHIR DiagnosticReport, and logs the attempt
    (success or failure) to the audit trail.

    Returns the FHIR resource directly as the response body, matching
    how real FHIR servers respond to a create request.
    """
    try:
        synergy_score, confidence, docking_affinity = predict_synergy(
            req.drug_a, req.drug_b, req.cell_line
        )
    except ModelUnavailableError as e:
        # Model failed to load or run — this is a server-side problem,
        # not a bad request, so it's a 503, not a FHIR OperationOutcome.
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
        # Spec-correct: invalid request -> 400 with OperationOutcome body,
        # not a 200 with an error buried in the payload.
        raise HTTPException(status_code=400, detail=resource)

    return resource


@app.get("/fhir/AuditLog")
def get_audit_log(limit: int = 50):
    """Read-only view of the most recent audit entries (newest last)."""
    entries = audit.read_all()
    return {"count": len(entries), "entries": entries[-limit:]}


@app.get("/fhir/AuditLog/verify")
def verify_audit_log():
    """Verifies the hash chain hasn't been tampered with."""
    valid, broken_at = audit.verify_chain()
    return {"valid": valid, "broken_at_index": broken_at}
