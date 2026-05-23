from __future__ import annotations

import re

import pytest

SESSION_ID_PATTERN = re.compile(r"^[0-9a-f]{8}$")


class TestSessions:
    def test_list_sessions(self, http_client):
        resp = http_client.get("/api/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_create_session_sync(self, http_client_long):
        text = (
            "Les réseaux de neurones convolutifs sont largement utilisés "
            "en vision par ordinateur. Ils excellent dans la classification "
            "d'images et la détection d'objets."
        )
        resp = http_client_long.post(
            "/api/sessions/ingest",
            params={"text": text, "filename": "test_cnn.txt", "defer": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert SESSION_ID_PATTERN.match(data["session_id"]), (
            f"Invalid session_id format: {data['session_id']}"
        )
        assert data["chunks_count"] > 0

    def test_get_session(self, http_client, sample_session):
        resp = http_client.get(f"/api/sessions/{sample_session}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session"]["id"] == sample_session
        assert len(data["chunks"]) > 0

    def test_get_session_not_found(self, http_client):
        resp = http_client.get("/api/sessions/00000000")
        assert resp.status_code == 404

    def test_create_session_defer_true(self, http_client):
        """defer=True: crée la session + chunks, le NLP pipeline tourne en arrière-plan."""
        text = (
            "Les transformers ont révolutionné le traitement automatique "
            "des langues. BERT et GPT sont les architectures les plus "
            "emblématiques de cette famille de modèles."
        )
        resp = http_client.post(
            "/api/sessions/ingest",
            params={
                "text": text,
                "filename": "test_defer_true.txt",
                "defer": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert SESSION_ID_PATTERN.match(data["session_id"])
        assert data["chunks_count"] > 0
