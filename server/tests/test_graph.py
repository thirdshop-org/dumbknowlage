from __future__ import annotations


class TestGraph:
    def test_aql_count_collections(self, http_client):
        resp = http_client.post(
            "/api/graph/aql",
            params={"query": "RETURN LENGTH(Word)"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "count" in data
        assert isinstance(data["count"], int)

    def test_aql_list_databases(self, http_client):
        resp = http_client.post(
            "/api/graph/aql",
            params={"query": "FOR i IN 1..10 RETURN i"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 10

    def test_aql_invalid_syntax(self, http_client):
        resp = http_client.post(
            "/api/graph/aql",
            params={"query": "INVALID SYNTAX!!!"},
        )
        assert resp.status_code == 400

    def test_revalidate_dry_run(self, http_client):
        resp = http_client.post(
            "/api/graph/revalidate",
            params={"dry_run": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "scanned" in data

    def test_cleanup_dry_run(self, http_client):
        resp = http_client.post(
            "/api/graph/cleanup",
            params={"dry_run": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "scanned" in data

    def test_list_rules(self, http_client):
        resp = http_client.get("/api/rules")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_list_corrections(self, http_client):
        resp = http_client.get("/api/corrections", params={"limit": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert "stats" in data
        assert "corrections" in data
