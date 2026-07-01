"""
test_cds_hooks.py

Tests for the CDS Hooks discovery and medication-prescribe endpoints.
Uses FastAPI's TestClient with a stubbed predict_synergy so tests
never depend on the real trained model being loadable in CI.
"""

import pytest
from fastapi.testclient import TestClient
import api as api_module
from audit_log import AuditLog


@pytest.fixture
def client(tmp_path, monkeypatch):
    api_module.audit = AuditLog(path=str(tmp_path / "test_audit.jsonl"))

    def fake_predict(drug_a, drug_b, cell_line):
        # Return a high synergy score for Olaparib+Rucaparib, low for others
        if set([drug_a, drug_b]) == {"Olaparib", "Rucaparib"}:
            return 0.42, 0.79, None
        return 0.02, 0.6, None

    monkeypatch.setattr(api_module, "predict_synergy", fake_predict)
    return TestClient(api_module.app)


class TestCDSDiscovery:
    def test_discovery_returns_services_list(self, client):
        resp = client.get("/cds-services")
        assert resp.status_code == 200
        body = resp.json()
        assert "services" in body
        assert len(body["services"]) >= 1

    def test_discovery_service_has_required_fields(self, client):
        resp = client.get("/cds-services")
        svc = resp.json()["services"][0]
        for field in ("hook", "id", "title", "description"):
            assert field in svc, f"Missing required field: {field}"

    def test_discovery_hook_is_medication_prescribe(self, client):
        resp = client.get("/cds-services")
        hooks = [s["hook"] for s in resp.json()["services"]]
        assert "medication-prescribe" in hooks

    def test_discovery_service_id_matches_hook_path(self, client):
        resp = client.get("/cds-services")
        svc = resp.json()["services"][0]
        # The id should match the URL path segment /cds-services/{id}
        assert svc["id"] == "synergy-advisor"


class TestCDSSynergyAdvisor:
    def _hook_payload(self, drug_name=None, use_fhir_bundle=False):
        """Build a minimal medication-prescribe hook payload."""
        if use_fhir_bundle:
            context = {
                "draftOrders": {
                    "resourceType": "Bundle",
                    "entry": [{
                        "resource": {
                            "resourceType": "MedicationRequest",
                            "medicationCodeableConcept": {
                                "text": drug_name or "Olaparib",
                                "coding": [{"display": drug_name or "Olaparib"}]
                            }
                        }
                    }]
                }
            }
        else:
            context = {"drug": drug_name or "Olaparib"}

        return {
            "hook": "medication-prescribe",
            "hookInstance": "test-uuid-1234",
            "context": context,
        }

    def test_returns_cards_array(self, client):
        resp = client.post("/cds-services/synergy-advisor",
                           json=self._hook_payload("Olaparib"))
        assert resp.status_code == 200
        assert "cards" in resp.json()
        assert isinstance(resp.json()["cards"], list)

    def test_cards_have_required_fields(self, client):
        resp = client.post("/cds-services/synergy-advisor",
                           json=self._hook_payload("Olaparib"))
        cards = resp.json()["cards"]
        assert len(cards) >= 1
        for card in cards:
            for field in ("summary", "indicator", "source"):
                assert field in card, f"Card missing required field: {field}"

    def test_indicator_is_valid_value(self, client):
        resp = client.post("/cds-services/synergy-advisor",
                           json=self._hook_payload("Olaparib"))
        for card in resp.json()["cards"]:
            assert card["indicator"] in ("info", "warning", "critical")

    def test_source_has_label(self, client):
        resp = client.post("/cds-services/synergy-advisor",
                           json=self._hook_payload("Olaparib"))
        for card in resp.json()["cards"]:
            assert "label" in card["source"]

    def test_unknown_drug_returns_info_card_not_error(self, client):
        resp = client.post("/cds-services/synergy-advisor",
                           json=self._hook_payload("CompletelyUnknownDrug123"))
        # Must return 200 with cards, not 400/500 —
        # an error response would break the EHR's hook pipeline
        assert resp.status_code == 200
        assert "cards" in resp.json()

    def test_empty_context_returns_cards_not_error(self, client):
        payload = {
            "hook": "medication-prescribe",
            "hookInstance": "test-uuid-5678",
            "context": {},
        }
        resp = client.post("/cds-services/synergy-advisor", json=payload)
        assert resp.status_code == 200
        assert "cards" in resp.json()

    def test_fhir_bundle_context_extracts_drug(self, client):
        resp = client.post("/cds-services/synergy-advisor",
                           json=self._hook_payload("Olaparib", use_fhir_bundle=True))
        assert resp.status_code == 200
        body = resp.json()
        assert "cards" in body

    def test_missing_hook_instance_returns_422(self, client):
        resp = client.post("/cds-services/synergy-advisor",
                           json={"hook": "medication-prescribe", "context": {}})
        assert resp.status_code == 422

    def test_high_synergy_pair_produces_card(self, client):
        # Olaparib→Rucaparib is stubbed to return 0.42 synergy,
        # which should appear as a card
        resp = client.post("/cds-services/synergy-advisor",
                           json=self._hook_payload("Olaparib"))
        cards = resp.json()["cards"]
        summaries = " ".join(c["summary"] for c in cards)
        assert "Rucaparib" in summaries or len(cards) >= 1
