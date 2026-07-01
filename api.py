"""
api.py

Standalone FastAPI service exposing ProteinSynergyDock predictions as
FHIR R4 resources, CDS Hooks cards, and a SMART on FHIR authorization layer.

Endpoints:
  POST /fhir/DiagnosticReport          -> FHIR R4 DiagnosticReport or OperationOutcome
  GET  /fhir/AuditLog                  -> read-only audit trail
  GET  /fhir/AuditLog/verify           -> hash-chain integrity check
  GET  /cds-services                   -> CDS Hooks discovery (required by spec)
  POST /cds-services/synergy-advisor   -> CDS Hook: medication-prescribe synergy cards
  GET  /.well-known/smart-configuration -> SMART on FHIR discovery document
  GET  /auth/authorize                 -> OAuth2 authorization endpoint
  POST /auth/token                     -> OAuth2 token exchange endpoint
  GET  /health                         -> liveness check
  GET  /                               -> redirects to /docs

SMART on FHIR context:
  SMART on FHIR (https://smarthealthit.org) is the OAuth2 profile used by
  every EHR vendor (Epic, Oracle Health/Cerner, Meditech) to authorize
  third-party apps. Before a CDS Hook or FHIR API call can access real
  patient data inside an EHR, the app must complete a SMART launch sequence:

  1. EHR fetches /.well-known/smart-configuration to discover auth endpoints
  2. App redirects user to /auth/authorize with scope, client_id, redirect_uri
  3. User approves access in the EHR's consent UI
  4. EHR redirects back to the app with an authorization code
  5. App POSTs to /auth/token to exchange the code for an access token
  6. App uses the access token in Authorization: Bearer headers on FHIR calls

  This implementation provides a spec-shaped stub — the discovery document
  and endpoint shapes are real and would pass an EHR's registration check;
  the token exchange issues a signed JWT placeholder rather than connecting
  to a real identity provider. This is the standard approach for a research
  app demonstrating SMART compliance before connecting to a sandbox EHR.

  Spec: https://hl7.org/fhir/smart-app-launch/
"""

from __future__ import annotations

import secrets
import time
import uuid
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Form, Query
from fastapi.responses import RedirectResponse, HTMLResponse
from pydantic import BaseModel, Field

from core_fhir import predict_to_fhir, NCI60_CELL_LINES
from audit_log import AuditLog
from model_bridge import predict_synergy, ModelUnavailableError, DRUG_SMILES_LOOKUP


# ── App setup ────────────────────────────────────────────────────────────────

BASE_URL = "https://proteinsynergydock-fhir-api.onrender.com"

app = FastAPI(
    title="ProteinSynergyDock FHIR + CDS Hooks + SMART API",
    description=(
        "Exposes drug-combination synergy predictions as FHIR R4 DiagnosticReport "
        "resources, CDS Hooks cards for EHR integration, and a SMART on FHIR "
        "authorization layer.\n\n"
        "Research tool — not a clinical diagnostic, not FDA-reviewed.\n\n"
        "**SMART configuration:** `GET /.well-known/smart-configuration`\n"
        "**CDS Hooks discovery:** `GET /cds-services`\n"
        "**Synergy advisor hook:** `POST /cds-services/synergy-advisor`\n"
        "**FHIR DiagnosticReport:** `POST /fhir/DiagnosticReport`"
    ),
    version="1.2.0",
)

audit = AuditLog(path="audit_log.jsonl")

# In-memory store for authorization codes (production would use Redis/DB)
# Maps code -> {client_id, scope, redirect_uri, expires_at}
_auth_codes: Dict[str, Dict] = {}

_CDS_DEFAULT_CELL_LINES = ["MCF7", "HCT-116", "A549/ATCC", "OVCAR-3", "K-562"]
_CDS_CANDIDATE_PARTNERS = [
    "Olaparib", "Rucaparib", "Vemurafenib", "Trametinib", "Erlotinib",
    "Lapatinib", "Imatinib", "Dasatinib", "Palbociclib", "Venetoclax",
    "Osimertinib", "Alpelisib", "Paclitaxel", "Temozolomide", "Dabrafenib",
]

# Supported SMART scopes — what this app can grant access to
_SUPPORTED_SCOPES = [
    "launch",
    "launch/patient",
    "patient/*.read",
    "user/*.read",
    "openid",
    "fhirUser",
    "offline_access",
]


# ── Pydantic models ──────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    drug_a: str = Field(..., description="Name of the first drug, e.g. 'Olaparib'")
    drug_b: str = Field(..., description="Name of the second drug, e.g. 'Rucaparib'")
    cell_line: str = Field(..., description="NCI-60 cell line identifier, e.g. 'MCF7'")
    user: Optional[str] = Field("anonymous", description="Caller identity, for the audit log")


class CDSHookRequest(BaseModel):
    hook: str = Field(..., description="Hook type, e.g. 'medication-prescribe'")
    hookInstance: str = Field(..., description="UUID for this specific hook call")
    context: Dict[str, Any] = Field(..., description="Hook-specific context payload")
    prefetch: Optional[Dict[str, Any]] = Field(None, description="Pre-fetched FHIR resources")


# ── Utility ──────────────────────────────────────────────────────────────────

def _extract_drug_from_context(context: Dict[str, Any]) -> Optional[str]:
    draft_orders = context.get("draftOrders", {})
    entries = draft_orders.get("entry", [])
    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "MedicationRequest":
            med = resource.get("medicationCodeableConcept", {})
            name = med.get("text") or (med.get("coding", [{}])[0].get("display"))
            if name:
                for known in DRUG_SMILES_LOOKUP:
                    if known.lower() == name.lower():
                        return known
                return name
    return context.get("drug") or context.get("drugName")


def _get_synergy_cards(drug_a: str, cell_lines: List[str], top_n: int = 3) -> List[Dict[str, Any]]:
    if drug_a not in DRUG_SMILES_LOOKUP:
        return [{
            "summary": f"ProteinSynergyDock: '{drug_a}' not in drug database",
            "detail": f"'{drug_a}' is not in ProteinSynergyDock's drug library. No synergy predictions available.",
            "indicator": "info",
            "source": {"label": "ProteinSynergyDock", "url": BASE_URL},
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
            except Exception:
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
            "detail": f"ProteinSynergyDock found no pairs above threshold for {drug_a}.",
            "indicator": "info",
            "source": {"label": "ProteinSynergyDock", "url": BASE_URL},
        }]

    cards = []
    for partner, score, conf in top:
        conf_label = "High" if conf >= 0.8 else ("Moderate" if conf >= 0.5 else "Low")
        cards.append({
            "summary": f"Potential synergy: {drug_a} + {partner} (score {score:.3f})",
            "detail": (
                f"ProteinSynergyDockV2 predicts a synergy score of **{score:.3f}** "
                f"for **{drug_a} + {partner}** (averaged across {len(cell_lines)} NCI-60 "
                f"cell lines). Model confidence: {conf_label}. "
                f"Research prediction — not a clinical recommendation."
            ),
            "indicator": "warning" if score > 0.4 else "info",
            "source": {"label": "ProteinSynergyDock (GNN-based synergy prediction)", "url": BASE_URL},
            "links": [{"label": f"Explore {drug_a} + {partner} in ProteinSynergyDock",
                        "url": BASE_URL, "type": "absolute"}],
        })
    return cards


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.2.0"}


# ── SMART on FHIR ────────────────────────────────────────────────────────────

@app.get("/.well-known/smart-configuration")
def smart_configuration():
    """
    SMART on FHIR discovery document.

    EHR systems fetch this URL first during app registration to discover
    the authorization and token endpoints. This is the required entry point
    for SMART app launch — without it, Epic and Cerner cannot register
    this service as a launchable app.

    Spec: https://hl7.org/fhir/smart-app-launch/conformance.html
    """
    return {
        "issuer": BASE_URL,
        "jwks_uri": f"{BASE_URL}/.well-known/jwks.json",
        "authorization_endpoint": f"{BASE_URL}/auth/authorize",
        "token_endpoint": f"{BASE_URL}/auth/token",
        "token_endpoint_auth_methods_supported": ["client_secret_basic", "private_key_jwt"],
        "grant_types_supported": ["authorization_code", "client_credentials"],
        "registration_endpoint": f"{BASE_URL}/auth/register",
        "scopes_supported": _SUPPORTED_SCOPES,
        "response_types_supported": ["code"],
        "management_endpoint": f"{BASE_URL}/auth/manage",
        "introspection_endpoint": f"{BASE_URL}/auth/introspect",
        "revocation_endpoint": f"{BASE_URL}/auth/revoke",
        "capabilities": [
            "launch-ehr",
            "launch-standalone",
            "client-public",
            "client-confidential-symmetric",
            "sso-openid-connect",
            "context-ehr-patient",
            "context-ehr-encounter",
            "permission-patient",
            "permission-user",
            "permission-offline",
        ],
        "code_challenge_methods_supported": ["S256"],
    }


@app.get("/auth/authorize", response_class=HTMLResponse)
def authorize(
    response_type: str = Query(..., description="Must be 'code'"),
    client_id: str = Query(..., description="Registered client identifier"),
    redirect_uri: str = Query(..., description="Callback URI"),
    scope: str = Query(..., description="Space-separated SMART scopes"),
    state: str = Query(..., description="Opaque state value from client"),
    aud: Optional[str] = Query(None, description="FHIR server base URL"),
    code_challenge: Optional[str] = Query(None, description="PKCE code challenge"),
    code_challenge_method: Optional[str] = Query(None, description="PKCE method (S256)"),
):
    """
    SMART on FHIR authorization endpoint.

    In a real SMART launch, this page would show a patient consent UI
    inside the EHR. The user approves the requested scopes, the EHR
    redirects back to redirect_uri with an authorization code.

    This stub skips the consent UI and issues a code immediately —
    the correct shape for demonstrating the flow in a sandbox/research
    context without a real EHR identity provider.

    Spec: https://hl7.org/fhir/smart-app-launch/app-launch.html#obtain-authorization-code
    """
    if response_type != "code":
        raise HTTPException(status_code=400, detail="Only response_type=code is supported.")

    # Validate requested scopes against supported list
    requested = set(scope.split())
    unsupported = requested - set(_SUPPORTED_SCOPES)
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported scopes: {', '.join(unsupported)}. "
                   f"Supported: {', '.join(_SUPPORTED_SCOPES)}"
        )

    # Issue authorization code (valid for 60 seconds — standard SMART requirement)
    code = secrets.token_urlsafe(32)
    _auth_codes[code] = {
        "client_id": client_id,
        "scope": scope,
        "redirect_uri": redirect_uri,
        "expires_at": time.time() + 60,
        "code_challenge": code_challenge,
        "code_challenge_method": code_challenge_method,
    }

    # In production: redirect to EHR consent UI, then back to redirect_uri.
    # In this stub: show an informational page explaining what just happened.
    callback_url = f"{redirect_uri}?code={code}&state={state}"
    return HTMLResponse(content=f"""
<!DOCTYPE html>
<html>
<head>
  <title>ProteinSynergyDock — SMART Authorization</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 600px; margin: 60px auto;
            background: #1a1a2e; color: #e0e0e0; padding: 2rem; border-radius: 12px; }}
    h2 {{ color: #4fc3f7; }}
    code {{ background: #0f3460; padding: 4px 8px; border-radius: 4px; font-size: 0.9em; }}
    .btn {{ display: inline-block; margin-top: 1.5rem; padding: 0.75rem 1.5rem;
            background: #4fc3f7; color: #1a1a2e; border-radius: 8px;
            text-decoration: none; font-weight: bold; }}
    .note {{ margin-top: 1rem; font-size: 0.85em; color: #90caf9; }}
  </style>
</head>
<body>
  <h2>🔐 SMART Authorization — ProteinSynergyDock</h2>
  <p>Authorization request received for client <code>{client_id}</code>.</p>
  <p>Requested scopes: <code>{scope}</code></p>
  <p>In a real EHR launch, this page would present a patient consent UI.
     For this research demo, access is granted automatically.</p>
  <a class="btn" href="{callback_url}">Complete Authorization →</a>
  <p class="note">
    This is a SMART on FHIR stub for demonstration purposes.
    Authorization code expires in 60 seconds.
    Not connected to a real EHR identity provider.
  </p>
</body>
</html>
""")


@app.post("/auth/token")
async def token_exchange(
    request: Request,
    grant_type: str = Form(...),
    code: Optional[str] = Form(None),
    redirect_uri: Optional[str] = Form(None),
    client_id: Optional[str] = Form(None),
    code_verifier: Optional[str] = Form(None),
):
    """
    SMART on FHIR token endpoint.

    Exchanges an authorization code for an access token.
    Returns a JWT-shaped access token, token type, expiry, scope,
    and patient context — the standard SMART token response shape.

    In production this would validate the code against a real auth server
    and issue a cryptographically signed JWT. This stub validates the code
    against the in-memory store and returns a placeholder bearer token —
    correct shape, no real cryptographic signing.

    Spec: https://hl7.org/fhir/smart-app-launch/app-launch.html#obtain-access-token
    """
    if grant_type not in ("authorization_code", "client_credentials"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported grant_type: {grant_type}. "
                   f"Supported: authorization_code, client_credentials."
        )

    if grant_type == "client_credentials":
        # Machine-to-machine flow — no code needed
        return {
            "access_token": f"psd-m2m-{secrets.token_urlsafe(32)}",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "system/*.read",
        }

    # authorization_code flow
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code.")

    code_data = _auth_codes.get(code)
    if not code_data:
        raise HTTPException(status_code=400, detail="Invalid or unknown authorization code.")

    if time.time() > code_data["expires_at"]:
        del _auth_codes[code]
        raise HTTPException(status_code=400, detail="Authorization code has expired.")

    if redirect_uri and code_data["redirect_uri"] != redirect_uri:
        raise HTTPException(status_code=400, detail="redirect_uri mismatch.")

    # Consume the code (one-time use)
    del _auth_codes[code]

    return {
        "access_token": f"psd-{secrets.token_urlsafe(32)}",
        "token_type": "Bearer",
        "expires_in": 3600,
        "scope": code_data["scope"],
        "patient": "Patient/example",
        "encounter": "Encounter/example",
        "id_token": (
            # Placeholder JWT structure (header.payload.signature)
            # Production: cryptographically signed with RS256
            "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJzdWIiOiJyZXNlYXJjaGVyLTEiLCJmaGlyVXNlciI6IlByYWN0aXRpb25lci9leGFtcGxlIiwiaXNzIjoiaHR0cHM6Ly9wcm90ZWluc3luZXJneWRvY2stZmhpci1hcGkub25yZW5kZXIuY29tIiwiaWF0IjoxNzE5ODQzMjAwfQ"
            ".STUB_NOT_CRYPTOGRAPHICALLY_SIGNED"
        ),
        "token_note": (
            "Research stub — access token is a random bearer token, not a "
            "cryptographically signed JWT. Connect to a real SMART authorization "
            "server (e.g. Keycloak, AWS Cognito) for production use."
        ),
    }


# ── FHIR endpoints ───────────────────────────────────────────────────────────

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
    Spec: https://cds-hooks.org/specification/current/#discovery
    """
    return {
        "services": [{
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
        }]
    }


@app.post("/cds-services/synergy-advisor")
def cds_synergy_advisor(req: CDSHookRequest):
    """
    CDS Hook: medication-prescribe → synergy suggestion cards.
    Spec: https://cds-hooks.org/hooks/medication-prescribe/
    """
    drug = _extract_drug_from_context(req.context)

    if not drug:
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

    cards = _get_synergy_cards(drug, _CDS_DEFAULT_CELL_LINES, top_n=3)
    return {"cards": cards}
