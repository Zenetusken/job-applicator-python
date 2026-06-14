"""OCR service for extracting text from scanned PDFs and images."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from job_applicator.exceptions import DocumentError
from job_applicator.utils.logging import get_logger

logger = get_logger("documents.ocr")


class OCRService:
    """Local OCR using PaddleOCR."""

    def __init__(self) -> None:
        self._ocr: Any | None = None

    def _get_ocr(self) -> Any:
        """Lazy-load PaddleOCR instance."""
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise DocumentError(
                    "paddleocr is not installed. Run: pip install paddleocr"
                ) from exc
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                show_log=False,
                use_gpu=False,
            )
        return self._ocr

    def extract_text_from_image(self, path: Path) -> str:
        """Run OCR on a single image file."""
        logger.info("Running OCR on image: %s", path)
        ocr = self._get_ocr()
        try:
            result = ocr.ocr(str(path), cls=True)
        except Exception as exc:
            raise DocumentError(f"OCR failed for image {path}: {exc}") from exc
        return self._parse_result(result).strip()

    def extract_text_from_pdf(self, path: str | Path) -> str:
        """Extract text from a PDF using OCR."""
        ocr = self._get_ocr()
        path = Path(path)
        try:
            import fitz
        except ImportError as exc:
            raise DocumentError(
                "PyMuPDF (fitz) is required for OCR on PDFs. Run: pip install pymupdf"
            ) from exc
        with fitz.open(str(path)) as doc:
            page_texts = []
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(pix.tobytes())
                    tmp.flush()
                    try:
                        result = ocr.ocr(tmp.name, cls=True)
                        page_text = self._parse_result(result).strip()
                        page_texts.append(page_text)
                    finally:
                        Path(tmp.name).unlink(missing_ok=True)
            return "\n\n".join(page_texts).strip()

    def _parse_result(self, result: Any) -> str:
        """Flatten PaddleOCR result into plain text."""
        if not result or result[0] is None:
            return ""
        lines: list[str] = []
        for line in result[0]:
            if line is None:
                continue
            # Each line is ([bbox], (text, confidence))
            text = line[1][0]
            lines.append(text)
        return "\n".join(lines)
