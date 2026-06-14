"""OCR service for extracting text from scanned PDFs and images."""

from __future__ import annotations

from pathlib import Path
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

    def extract_text_from_pdf(self, path: Path) -> str:
        """Run OCR on every page of a PDF."""
        logger.info("Running OCR on PDF: %s", path)
        try:
            import fitz
        except ImportError as exc:
            raise DocumentError(
                "PyMuPDF is required for OCR on PDFs. Install: pip install pymupdf"
            ) from exc

        try:
            page_texts: list[str] = []
            with fitz.open(str(path)) as doc:
                for _page_num, page in enumerate(doc, start=1):
                    pix = page.get_pixmap(dpi=200)
                    import tempfile

                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        tmp_path = Path(tmp.name)
                        pix.save(str(tmp_path))
                    try:
                        text = self.extract_text_from_image(tmp_path)
                        page_texts.append(text)
                    finally:
                        tmp_path.unlink(missing_ok=True)
        except Exception as exc:
            raise DocumentError(f"OCR failed for PDF {path}: {exc}") from exc

        full_text = "\n\n".join(page_texts).strip()
        logger.info("OCR extracted %d characters from PDF", len(full_text))
        return full_text.strip()

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
