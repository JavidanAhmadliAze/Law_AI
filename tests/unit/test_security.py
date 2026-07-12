import uuid

import pytest

from law_ai.config import AppSettings
from law_ai.exceptions import AuthenticationError
from law_ai.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)

SETTINGS = AppSettings(secret_key="test-secret", access_token_expire_minutes=5)


def test_password_roundtrip() -> None:
    hashed = hash_password("s3cret-pass")
    assert hashed != "s3cret-pass"
    assert verify_password("s3cret-pass", hashed)
    assert not verify_password("wrong", hashed)


def test_jwt_roundtrip() -> None:
    user_id = uuid.uuid4()
    token = create_access_token(user_id, SETTINGS)
    assert decode_access_token(token, SETTINGS) == user_id


def test_jwt_rejects_tampered_token() -> None:
    token = create_access_token(uuid.uuid4(), SETTINGS)
    with pytest.raises(AuthenticationError):
        decode_access_token(token + "x", SETTINGS)


def test_jwt_rejects_wrong_secret() -> None:
    token = create_access_token(uuid.uuid4(), SETTINGS)
    other = AppSettings(secret_key="other-secret")
    with pytest.raises(AuthenticationError):
        decode_access_token(token, other)
