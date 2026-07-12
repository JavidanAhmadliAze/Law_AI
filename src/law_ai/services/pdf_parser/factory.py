from law_ai.config import Settings
from law_ai.services.pdf_parser.base import BasePDFParser
from law_ai.services.pdf_parser.client import PyMuPDFParser


class PDFParserFactory:
    @staticmethod
    def create(settings: Settings) -> BasePDFParser:  # noqa: ARG004 — uniform factory signature
        return PyMuPDFParser()
