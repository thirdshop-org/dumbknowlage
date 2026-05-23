from __future__ import annotations


class TestHealth:
    def test_server_is_alive(self, http_client):
        resp = http_client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["whisper_model"] != ""

    def test_upstream_services(self, http_client):
        resp = http_client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["arango"] is True, "ArangoDB should be reachable"
        assert data["whisper_model"] != "", "Whisper model should be configured"
