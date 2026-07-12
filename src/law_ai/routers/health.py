"""Liveness / readiness probe."""

from fastapi import APIRouter, Response, status

from law_ai.dependencies import DatabaseDep

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(db: DatabaseDep, response: Response) -> dict[str, str]:
    db_ok = await db.health_check()
    if not db_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ok" if db_ok else "degraded",
        "database": "up" if db_ok else "down",
    }
