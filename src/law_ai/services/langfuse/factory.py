from law_ai.config import Settings
from law_ai.services.langfuse.client import LangfuseService


class LangfuseFactory:
    @staticmethod
    def create(settings: Settings) -> LangfuseService:
        return LangfuseService(settings.langfuse)
