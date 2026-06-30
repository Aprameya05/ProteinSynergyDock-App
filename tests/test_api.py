"""
test_api.py

Tests the FastAPI HTTP layer using FastAPI's TestClient and a stubbed
predict_synergy (via monkeypatch), so these tests never depend on the
real trained model being loadable in CI.
"""

import os
import pytest
from fastapi.testclient import TestClient

import api as api_module
from audit_log import AuditLog


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Point the module-level audit log at a temp file for test isolation.
    test_log_path = str(tmp_path / "test_audit.jsonl")
    api_module.audit = AuditLog(path=test_log_path)

    def fake_predict(drug_a, drug_b, cell_line):
        return 0.42, 0.79, -8.3

    monkeypatch.setattr(api_module, "predict_synergy", fake_predict)
    return TestClient(api_module.app)


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestDiagnosticReportEndpoint:
    def test_valid_request_returns_diagnostic_report(self, client):
        resp = client.post(
            "/fhir/DiagnosticReport",
            json={"drug_a": "Olaparib", "drug_b": "Rucaparib", "cell_line": "MCF7"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["resourceType"] == "DiagnosticReport"

    def test_invalid_cell_line_returns_400_with_operation_outcome(self, client):
        resp = client.post(
            "/fhir/DiagnosticReport",
            json={"drug_a": "Olaparib", "drug_b": "Rucaparib", "cell_line": "NOT-REAL"},
        )
        assert resp.status_code == 400
        assert resp.json()["detail"]["resourceType"] == "OperationOutcome"

    def test_missing_field_returns_422(self, client):
        resp = client.post("/fhir/DiagnosticReport", json={"drug_a": "Olaparib"})
        assert resp.status_code == 422

    def test_successful_request_is_audited(self, client):
        client.post(
            "/fhir/DiagnosticReport",
            json={"drug_a": "Olaparib", "drug_b": "Rucaparib", "cell_line": "MCF7"},
        )
        log_resp = client.get("/fhir/AuditLog")
        assert log_resp.json()["count"] == 1
        assert log_resp.json()["entries"][0]["success"] is True

    def test_failed_request_is_audited(self, client):
        client.post(
            "/fhir/DiagnosticReport",
            json={"drug_a": "Olaparib", "drug_b": "Rucaparib", "cell_line": "NOT-REAL"},
        )
        log_resp = client.get("/fhir/AuditLog")
        assert log_resp.json()["count"] == 1
        assert log_resp.json()["entries"][0]["success"] is False


class TestAuditLogEndpoints:
    def test_verify_chain_valid_after_requests(self, client):
        for cl in ["MCF7", "HCT-116", "PC-3"]:
            client.post(
                "/fhir/DiagnosticReport",
                json={"drug_a": "Olaparib", "drug_b": "Rucaparib", "cell_line": cl},
            )
        resp = client.get("/fhir/AuditLog/verify")
        assert resp.json()["valid"] is True

    def test_audit_log_respects_limit(self, client):
        for cl in ["MCF7", "HCT-116", "PC-3"]:
            client.post(
                "/fhir/DiagnosticReport",
                json={"drug_a": "Olaparib", "drug_b": "Rucaparib", "cell_line": cl},
            )
        resp = client.get("/fhir/AuditLog?limit=2")
        assert len(resp.json()["entries"]) == 2
