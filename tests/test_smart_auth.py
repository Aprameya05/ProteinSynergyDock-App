"""
test_smart_auth.py

Tests for the SMART on FHIR authorization endpoints:
  GET  /.well-known/smart-configuration
  GET  /auth/authorize
  POST /auth/token
"""

import pytest
from fastapi.testclient import TestClient
import api as api_module
from audit_log import AuditLog


@pytest.fixture
def client(tmp_path, monkeypatch):
    api_module.audit = AuditLog(path=str(tmp_path / "test_audit.jsonl"))
    api_module._auth_codes.clear()

    def fake_predict(drug_a, drug_b, cell_line):
        return 0.42, 0.79, None

    monkeypatch.setattr(api_module, "predict_synergy", fake_predict)
    return TestClient(api_module.app, follow_redirects=False)


class TestSMARTConfiguration:
    def test_discovery_returns_200(self, client):
        resp = client.get("/.well-known/smart-configuration")
        assert resp.status_code == 200

    def test_discovery_has_required_fields(self, client):
        body = client.get("/.well-known/smart-configuration").json()
        for field in ("authorization_endpoint", "token_endpoint",
                      "scopes_supported", "capabilities", "issuer"):
            assert field in body, f"Missing required SMART field: {field}"

    def test_authorization_endpoint_points_to_correct_path(self, client):
        body = client.get("/.well-known/smart-configuration").json()
        assert body["authorization_endpoint"].endswith("/auth/authorize")

    def test_token_endpoint_points_to_correct_path(self, client):
        body = client.get("/.well-known/smart-configuration").json()
        assert body["token_endpoint"].endswith("/auth/token")

    def test_scopes_include_launch_and_patient(self, client):
        body = client.get("/.well-known/smart-configuration").json()
        scopes = body["scopes_supported"]
        assert "launch" in scopes
        assert "patient/*.read" in scopes

    def test_capabilities_include_ehr_launch(self, client):
        body = client.get("/.well-known/smart-configuration").json()
        assert "launch-ehr" in body["capabilities"]

    def test_pkce_s256_supported(self, client):
        body = client.get("/.well-known/smart-configuration").json()
        assert "S256" in body.get("code_challenge_methods_supported", [])


class TestAuthorize:
    def _auth_params(self, **overrides):
        params = {
            "response_type": "code",
            "client_id": "test-client",
            "redirect_uri": "https://myapp.example.com/callback",
            "scope": "launch patient/*.read openid",
            "state": "random-state-xyz",
        }
        params.update(overrides)
        return params

    def test_authorize_returns_html(self, client):
        resp = client.get("/auth/authorize", params=self._auth_params())
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_authorize_issues_code(self, client):
        client.get("/auth/authorize", params=self._auth_params())
        assert len(api_module._auth_codes) == 1

    def test_authorize_wrong_response_type_returns_400(self, client):
        resp = client.get("/auth/authorize",
                          params=self._auth_params(response_type="token"))
        assert resp.status_code == 400

    def test_authorize_unsupported_scope_returns_400(self, client):
        resp = client.get("/auth/authorize",
                          params=self._auth_params(scope="launch admin/*"))
        assert resp.status_code == 400

    def test_authorize_callback_url_in_response(self, client):
        resp = client.get("/auth/authorize", params=self._auth_params())
        assert "https://myapp.example.com/callback" in resp.text
        assert "random-state-xyz" in resp.text


class TestTokenExchange:
    def _get_code(self, client):
        """Helper: run authorize flow and extract the issued code."""
        client.get("/auth/authorize", params={
            "response_type": "code",
            "client_id": "test-client",
            "redirect_uri": "https://myapp.example.com/callback",
            "scope": "launch patient/*.read openid",
            "state": "s",
        })
        assert api_module._auth_codes
        return list(api_module._auth_codes.keys())[0]

    def test_token_exchange_returns_access_token(self, client):
        code = self._get_code(client)
        resp = client.post("/auth/token", data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": "https://myapp.example.com/callback",
            "client_id": "test-client",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert body["token_type"] == "Bearer"

    def test_token_response_has_required_smart_fields(self, client):
        code = self._get_code(client)
        body = client.post("/auth/token", data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": "https://myapp.example.com/callback",
            "client_id": "test-client",
        }).json()
        for field in ("access_token", "token_type", "expires_in", "scope"):
            assert field in body, f"Missing SMART token field: {field}"

    def test_token_code_is_single_use(self, client):
        code = self._get_code(client)
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": "https://myapp.example.com/callback",
            "client_id": "test-client",
        }
        r1 = client.post("/auth/token", data=data)
        r2 = client.post("/auth/token", data=data)
        assert r1.status_code == 200
        assert r2.status_code == 400

    def test_invalid_code_returns_400(self, client):
        resp = client.post("/auth/token", data={
            "grant_type": "authorization_code",
            "code": "completely-invalid-code",
            "redirect_uri": "https://myapp.example.com/callback",
            "client_id": "test-client",
        })
        assert resp.status_code == 400

    def test_client_credentials_flow(self, client):
        resp = client.post("/auth/token", data={
            "grant_type": "client_credentials",
            "client_id": "test-client",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert body["token_type"] == "Bearer"

    def test_unsupported_grant_type_returns_400(self, client):
        resp = client.post("/auth/token", data={
            "grant_type": "password",
            "client_id": "test-client",
        })
        assert resp.status_code == 400

    def test_missing_code_returns_400(self, client):
        resp = client.post("/auth/token", data={
            "grant_type": "authorization_code",
            "client_id": "test-client",
        })
        assert resp.status_code == 400

    def test_redirect_uri_mismatch_returns_400(self, client):
        code = self._get_code(client)
        resp = client.post("/auth/token", data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": "https://evil.example.com/steal",
            "client_id": "test-client",
        })
        assert resp.status_code == 400
