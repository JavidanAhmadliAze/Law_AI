"""End-to-end API flow against real Postgres (requires `make up` + migrations)."""

import uuid

import pytest
from fastapi.testclient import TestClient

from law_ai.main import create_app

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client():  # type: ignore[no-untyped-def]
    with TestClient(create_app()) as test_client:  # context manager runs the lifespan
        yield test_client


@pytest.fixture(scope="module")
def credentials() -> dict[str, str]:
    return {"username": f"it-{uuid.uuid4().hex[:10]}", "password": "integration-pass-1"}


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["database"] == "up"


def test_register_login_chat_lifecycle(client: TestClient, credentials: dict[str, str]) -> None:
    assert client.post("/auth/register", json=credentials).status_code == 201
    assert client.post("/auth/register", json=credentials).status_code == 409  # distinct usernames

    token = client.post("/auth/login", json=credentials).json()["access_token"]
    headers = {"authorization": f"Bearer {token}"}

    chat = client.post("/chats", json={"title": "IT chat"}, headers=headers).json()
    listed = client.get("/chats", headers=headers).json()
    assert [c["id"] for c in listed] == [chat["id"]]

    assert client.delete(f"/chats/{chat['id']}", headers=headers).status_code == 204
    assert client.get("/chats", headers=headers).json() == []


def test_ownership_is_enforced(client: TestClient, credentials: dict[str, str]) -> None:
    other = {"username": f"it-{uuid.uuid4().hex[:10]}", "password": "integration-pass-2"}
    client.post("/auth/register", json=other)
    other_token = client.post("/auth/login", json=other).json()["access_token"]

    token = client.post("/auth/login", json=credentials).json()["access_token"]
    chat = client.post("/chats", json={}, headers={"authorization": f"Bearer {token}"}).json()

    # another user can neither see nor delete it (404, not 403 — ids not probeable)
    response = client.delete(
        f"/chats/{chat['id']}", headers={"authorization": f"Bearer {other_token}"}
    )
    assert response.status_code == 404
