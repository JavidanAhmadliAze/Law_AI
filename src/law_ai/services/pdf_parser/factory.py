from law_ai.config import Settings
from law_ai.services.pdf_parser.base import BasePDFParser
from law_ai.services.pdf_parser.client import PyMuPDFParser


def create_pdf_parser(settings: Settings) -> BasePDFParser:  # noqa: ARG001 — uniform factory signature
    return PyMuPDFParser()
