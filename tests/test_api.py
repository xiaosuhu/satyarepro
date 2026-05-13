import pytest
from httpx import ASGITransport, AsyncClient

from satyarepro.api.app import create_app
from satyarepro.api.schemas import AuditStatus


@pytest.fixture
def app():
    return create_app()


async def test_health(app) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_submit_audit_returns_202(app) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/audit", json={"query": "PMID 12345678"})
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == AuditStatus.pending


async def test_get_audit_not_found(app) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/audit/does-not-exist")
    assert resp.status_code == 404


async def test_submit_and_retrieve_job(app) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        submit = await ac.post("/audit", json={"query": "reproducibility in cancer biology"})
        assert submit.status_code == 202
        job_id = submit.json()["job_id"]

        get = await ac.get(f"/audit/{job_id}")
    assert get.status_code == 200
    data = get.json()
    assert data["job_id"] == job_id
    assert data["query"] == "reproducibility in cancer biology"
    assert data["status"] in {AuditStatus.pending, AuditStatus.running, AuditStatus.completed, AuditStatus.failed}


async def test_submit_invalid_query_too_short(app) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/audit", json={"query": "ab"})
    assert resp.status_code == 422
