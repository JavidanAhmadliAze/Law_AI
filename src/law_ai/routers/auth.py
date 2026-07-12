"""Registration and login. Simple username + password, distinct usernames."""

from fastapi import APIRouter, status

from law_ai.dependencies import SettingsDep, UserRepoDep
from law_ai.exceptions import AuthenticationError, ConflictError
from law_ai.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserOut
from law_ai.security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest, users: UserRepoDep) -> UserOut:
    if await users.get_by_username(payload.username) is not None:
        raise ConflictError("Username already taken")
    user = await users.create(
        {"username": payload.username, "password_hash": hash_password(payload.password)}
    )
    return UserOut.model_validate(user)


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, users: UserRepoDep, settings: SettingsDep) -> TokenResponse:
    user = await users.get_by_username(payload.username)
    if user is None or not verify_password(payload.password, user.password_hash):
        raise AuthenticationError("Invalid username or password")
    return TokenResponse(access_token=create_access_token(user.id, settings.app))
