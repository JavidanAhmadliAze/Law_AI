from law_ai.config import Settings
from law_ai.services.langfuse.client import LangfuseService


def create_langfuse(settings: Settings) -> LangfuseService:
    return LangfuseService(settings.langfuse)
