import pymupdf

from law_ai.logging import get_logger
from law_ai.services.pdf_parser.base import BasePDFParser, ParsedDocument

logger = get_logger(__name__)


class PyMuPDFParser(BasePDFParser):
    def parse(self, pdf_bytes: bytes, source_url: str = "") -> ParsedDocument:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            pages = [page.get_text("text") for page in doc]
        text = "\n".join(pages)
        logger.info("pdf_parser.parsed", pages=len(pages), chars=len(text))
        return ParsedDocument(text=text, page_count=len(pages), source_url=source_url)
