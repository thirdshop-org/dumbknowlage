from __future__ import annotations


class TestEntities:
    def test_list_entities(self, http_client):
        resp = http_client.get("/api/entities", params={"limit": 10})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_search_entities(self, http_client):
        resp = http_client.get("/api/entities", params={"q": "test", "limit": 10})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_list_entities_by_type(self, http_client):
        for typ in ("Person", "Organization", "Location", "Event"):
            resp = http_client.get(
                "/api/entities", params={"type": typ, "limit": 5}
            )
            assert resp.status_code == 200, f"Failed for type={typ}"
            data = resp.json()
            assert isinstance(data, list)

    def test_get_entity_detail_not_found(self, http_client):
        resp = http_client.get("/api/entities/Person/nonexistent_key_xyz")
        assert resp.status_code == 404
