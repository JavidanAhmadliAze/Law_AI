"""PDF parser contract."""

from abc import ABC, abstractmethod

from pydantic import BaseModel


class ParsedDocument(BaseModel):
    text: str
    page_count: int
    source_url: str = ""


class BasePDFParser(ABC):
    @abstractmethod
    def parse(self, pdf_bytes: bytes, source_url: str = "") -> ParsedDocument:
        """Extract plain text (reading order) from a PDF."""
