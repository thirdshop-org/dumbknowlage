from __future__ import annotations

import pytest
import httpx


def pytest_addoption(parser):
    parser.addoption(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the deployed server (e.g. https://dumbknowlage.thirdshop.fr)",
    )


@pytest.fixture(scope="session")
def base_url(request):
    return request.config.getoption("--base-url")


@pytest.fixture(scope="session")
def http_client(base_url):
    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        yield client


@pytest.fixture
async def async_client(base_url):
    async with httpx.AsyncClient(base_url=base_url, timeout=60.0) as client:
        yield client


@pytest.fixture
def http_client_long(base_url):
    with httpx.Client(base_url=base_url, timeout=180.0) as client:
        yield client


@pytest.fixture
def sample_session(http_client):
    text = (
        "L'intelligence artificielle transforme profondément "
        "le secteur de la santé. Les algorithmes de deep learning "
        "permettent de détecter des anomalies dans les examens "
        "médicaux avec une précision inégalée."
    )
    resp = http_client.post(
        "/api/sessions/ingest",
        params={"text": text, "filename": "test_ia_sante.txt", "defer": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    yield data["session_id"]
