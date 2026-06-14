# OCR Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use compose:subagent (recommended) or compose:execute to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add PaddleOCR-based fallback text extraction for scanned PDFs and image resumes, controllable via `--ocr-mode` and `--force-ocr` CLI flags.

**Architecture:** A new `OCRService` in `documents/ocr.py` wraps PaddleOCR and exposes PDF and image extraction. `ResumeLoader` accepts an `ocr_mode` parameter and decides when to call the service. CLI commands receive two new flags and pass the resolved mode to `ResumeLoader.load`.

**Tech Stack:** Python 3.12, PaddleOCR, PyMuPDF (existing optional fallback), pytest.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/job_applicator/documents/ocr.py` | New `OCRService` class; lazy PaddleOCR initialization; PDF/image extraction. |
| `src/job_applicator/documents/resume.py` | Modified `ResumeLoader` to accept `ocr_mode`, trigger OCR fallback, and support image formats. |
| `src/job_applicator/cli.py` | Add `--ocr-mode` and `--force-ocr` flags to `apply`, `generate-cover-letter`, `match`, `batch`, `tailor`; thread mode through `ResumeLoader.load`. |
| `pyproject.toml` | Add `paddleocr` to `[project.dependencies]`. |
| `tests/unit/test_documents.py` | Unit tests for OCR fallback, force flag, off mode, failure handling, image support. |

---

### Task 1: Add PaddleOCR dependency

**Covers:** [S10]

**Files:**
- Modify: `pyproject.toml:10-24`

- [ ] **Step 1: Add `paddleocr` to `[project.dependencies]`**

Insert `"paddleocr>=2.10",` after `python-docx` in the document processing group.

```toml
[project]
dependencies = [
    # Core
    "playwright>=1.53",
    "pydantic>=2.12",
    "pydantic-settings>=2.10",
    "typer>=0.16",
    "rich>=14.2",
    "jinja2>=3.1",
    "numpy>=1.26",
    # LLM - Modern stack
    "litellm>=1.88",
    "instructor>=1.15",
    # Document processing
    "python-docx>=1.1",
    "paddleocr>=2.10",
]
```

- [ ] **Step 2: Install dependency in the local venv**

Run: `.venv/bin/pip install paddleocr>=2.10`

Expected: Package installs successfully (may take a few minutes due to model downloads).

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add paddleocr for scanned PDF OCR fallback"
```

---

### Task 2: Implement OCRService

**Covers:** [S2], [S5], [S11], [S12]

**Files:**
- Create: `src/job_applicator/documents/ocr.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_documents.py`:

```python
from unittest.mock import MagicMock, patch


def test_ocr_service_extracts_text_from_image(tmp_path: Path) -> None:
    from job_applicator.documents.ocr import OCRService

    service = OCRService()
    # PaddleOCR is lazy-loaded; mock it to avoid heavy model init in unit tests.
    service._ocr = MagicMock()
    service._ocr.ocr.return_value = [[([[0, 0], [10, 0], [10, 10], [0, 10]], ("Hello", 0.99))]]

    img_path = tmp_path / "resume.png"
    # Create a tiny blank PNG using PIL
    from PIL import Image
    Image.new("RGB", (50, 50), color="white").save(img_path)

    text = service.extract_text_from_image(img_path)
    assert "Hello" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_documents.py::test_ocr_service_extracts_text_from_image -v`

Expected: `ModuleNotFoundError: No module named 'job_applicator.documents.ocr'`

- [ ] **Step 3: Create OCRService module**

Create `src/job_applicator/documents/ocr.py`:

```python
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
                from paddleocr import PaddleOCR  # type: ignore[import-untyped]
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
        return self._parse_result(result)

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
                for page_num, page in enumerate(doc, start=1):
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

        full_text = "\n\n".join(page_texts)
        logger.info("OCR extracted %d characters from PDF", len(full_text))
        return full_text

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_documents.py::test_ocr_service_extracts_text_from_image -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/job_applicator/documents/ocr.py tests/unit/test_documents.py
git commit -m "feat(ocr): add OCRService wrapper around PaddleOCR"
```

---

### Task 3: Integrate OCR into ResumeLoader

**Covers:** [S3], [S6], [S8], [S11]

**Files:**
- Modify: `src/job_applicator/documents/resume.py`

- [ ] **Step 1: Write the failing tests**

Ensure `tests/unit/test_documents.py` imports `DocumentError`:

```python
from job_applicator.exceptions import DocumentError
```

Append the OCR tests:

```python
from unittest.mock import MagicMock, patch


def test_ocr_fallback_triggers_on_short_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    loader = ResumeLoader()
    with patch.object(loader, "_run_pdftotext", return_value=" "), patch.object(
        loader, "_run_pymupdf", return_value="  "
    ), patch.object(loader, "_ocr_service", MagicMock()) as mock_ocr:
        mock_ocr.extract_text_from_pdf.return_value = "John Doe\nSkills: Python"
        result = loader._load_pdf(pdf_path, ocr_mode="auto")

    mock_ocr.extract_text_from_pdf.assert_called_once_with(pdf_path)
    assert "John Doe" in result.raw_text


def test_force_ocr_skips_text_extraction(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    loader = ResumeLoader()
    with patch.object(loader, "_run_pdftotext") as mock_pdftotext, patch.object(
        loader, "_run_pymupdf"
    ) as mock_pymupdf, patch.object(loader, "_ocr_service", MagicMock()) as mock_ocr:
        mock_ocr.extract_text_from_pdf.return_value = "OCR text"
        result = loader._load_pdf(pdf_path, ocr_mode="on")

    mock_pdftotext.assert_not_called()
    mock_pymupdf.assert_not_called()
    mock_ocr.extract_text_from_pdf.assert_called_once_with(pdf_path)
    assert result.raw_text == "OCR text"


def test_ocr_mode_off_disables_ocr(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    loader = ResumeLoader()
    with patch.object(loader, "_run_pdftotext", return_value="X"), patch.object(
        loader, "_run_pymupdf", return_value="X"
    ), patch.object(loader, "_ocr_service", MagicMock()) as mock_ocr:
        result = loader._load_pdf(pdf_path, ocr_mode="off")

    mock_ocr.extract_text_from_pdf.assert_not_called()
    assert result.raw_text == "X"


def test_ocr_failure_falls_back_to_extracted_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    loader = ResumeLoader()
    with patch.object(loader, "_run_pdftotext", return_value="short"), patch.object(
        loader, "_run_pymupdf", return_value="Some extracted text"
    ), patch.object(loader, "_ocr_service", MagicMock()) as mock_ocr:
        mock_ocr.extract_text_from_pdf.side_effect = DocumentError("OCR failed")
        result = loader._load_pdf(pdf_path, ocr_mode="auto")

    assert "Some extracted text" in result.raw_text


def test_ocr_failure_with_no_text_raises(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    loader = ResumeLoader()
    with patch.object(loader, "_run_pdftotext", return_value=""), patch.object(
        loader, "_run_pymupdf", return_value=""
    ), patch.object(loader, "_ocr_service", MagicMock()) as mock_ocr:
        mock_ocr.extract_text_from_pdf.side_effect = DocumentError("OCR failed")
        with pytest.raises(DocumentError):
            loader._load_pdf(pdf_path, ocr_mode="auto")


def test_image_resume_uses_ocr(tmp_path: Path) -> None:
    img_path = tmp_path / "resume.png"
    from PIL import Image
    Image.new("RGB", (50, 50), color="white").save(img_path)

    loader = ResumeLoader()
    with patch.object(loader, "_ocr_service", MagicMock()) as mock_ocr:
        mock_ocr.extract_text_from_image.return_value = "OCR text"
        result = loader.load(img_path, ocr_mode="on")

    mock_ocr.extract_text_from_image.assert_called_once_with(img_path)
    assert result.raw_text == "OCR text"


def test_force_ocr_failure_raises(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    loader = ResumeLoader()
    with patch.object(loader, "_ocr_service", MagicMock()) as mock_ocr:
        mock_ocr.extract_text_from_pdf.side_effect = DocumentError("OCR failed")
        with pytest.raises(DocumentError):
            loader._load_pdf(pdf_path, ocr_mode="on")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_documents.py -k "ocr" -v`

Expected: Multiple failures because `_load_pdf` does not accept `ocr_mode` and `_ocr_service` does not exist.

- [ ] **Step 3: Refactor ResumeLoader**

Replace `src/job_applicator/documents/resume.py` with a version that supports OCR. Key changes:

```python
from job_applicator.documents.ocr import OCRService

# Add constant
OCR_THRESHOLD = 100

class ResumeLoader:
    def __init__(self) -> None:
        self._ocr_service: OCRService | None = None

    @property
    def ocr_service(self) -> OCRService:
        if self._ocr_service is None:
            self._ocr_service = OCRService()
        return self._ocr_service

    def load(self, path: str | Path, ocr_mode: str = "auto") -> ResumeData:
        file_path = Path(path)
        if not file_path.exists():
            raise ResumeNotFoundError(f"Resume not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf(file_path, ocr_mode=ocr_mode)
        elif suffix == ".docx":
            return self._load_docx(file_path)
        elif suffix in (".txt", ".md"):
            return self._load_text(file_path)
        elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"):
            return self._load_image(file_path, ocr_mode=ocr_mode)
        else:
            raise DocumentError(f"Unsupported resume format: {suffix}")

    def _load_pdf(self, path: Path, ocr_mode: str = "auto") -> ResumeData:
        if ocr_mode == "on":
            text = self.ocr_service.extract_text_from_pdf(path)
            return self._parse_text(text)

        if ocr_mode == "off":
            text = self._run_pdftotext(path)
            if len(text.strip()) >= OCR_THRESHOLD:
                return self._parse_text(text)
            text = self._run_pymupdf(path)
            if len(text.strip()) >= OCR_THRESHOLD:
                return self._parse_text(text)
            return self._parse_text(text)

        # ocr_mode == "auto"
        text = self._run_pdftotext(path)
        if len(text.strip()) >= OCR_THRESHOLD:
            return self._parse_text(text)

        text = self._run_pymupdf(path)
        if len(text.strip()) >= OCR_THRESHOLD:
            return self._parse_text(text)

        logger.info("OCR fallback triggered for %s (extracted %d chars)", path, len(text))
        try:
            ocr_text = self.ocr_service.extract_text_from_pdf(path)
            logger.info("OCR fallback completed for %s (%d chars)", path, len(ocr_text))
            return self._parse_text(ocr_text)
        except DocumentError:
            if text:
                logger.warning("OCR failed; using extracted text (%d chars)", len(text))
                return self._parse_text(text)
            raise

    def _run_pdftotext(self, path: Path) -> str:
        """Extract text using pdftotext; return empty string on failure."""
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            try:
                result = subprocess.run(  # noqa: S603
                    ["pdftotext", "-layout", str(path), tmp.name],  # noqa: S607
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    return Path(tmp.name).read_text(encoding="utf-8")
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                pass
            finally:
                Path(tmp.name).unlink(missing_ok=True)
        return ""

    def _run_pymupdf(self, path: Path) -> str:
        """Extract text using PyMuPDF; return empty string on failure."""
        try:
            import fitz
        except ImportError:
            return ""
        try:
            doc = fitz.open(str(path))
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception:
            return ""

    def _load_image(self, path: Path, ocr_mode: str = "auto") -> ResumeData:
        if ocr_mode == "off":
            raise DocumentError(
                f"Image resume {path} requires OCR, but ocr_mode is 'off'"
            )
        text = self.ocr_service.extract_text_from_image(path)
        return self._parse_text(text)
```

Keep `_load_docx`, `_load_text`, `_parse_text` unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_documents.py -k "ocr" -v`

Expected: All 6 OCR tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/job_applicator/documents/resume.py tests/unit/test_documents.py
git commit -m "feat(resume): integrate OCR fallback for scanned PDFs and images"
```

---

### Task 4: Add CLI flags and thread mode through commands

**Covers:** [S4], [S7]

**Files:**
- Modify: `src/job_applicator/cli.py:168-180`, `317-335`, `400-420`, `523-545`, `1049-1070`

- [ ] **Step 1: Add a helper to resolve OCR flags**

At the top of `src/job_applicator/cli.py` (after imports), add:

```python
def _resolve_ocr_mode(ocr_mode: str, force_ocr: bool) -> str:
    """Return effective OCR mode from CLI flags."""
    if force_ocr:
        return "on"
    return ocr_mode
```

- [ ] **Step 2: Add flags to each resume-loading command**

For each of the five commands (`apply`, `generate_cover_letter`, `match`, `batch`, `tailor`), add these two parameters immediately after `headed: bool = ...` (or create a matching parameter block):

```python
ocr_mode: str = typer.Option(
    "auto",
    "--ocr-mode",
    help="OCR mode: auto (fallback), on (always), off (never).",
    show_choices=True,
    choices=["auto", "on", "off"],
    case_sensitive=False,
),
force_ocr: bool = typer.Option(
    False,
    "--force-ocr",
    help="Force OCR; equivalent to --ocr-mode on.",
),
```

Then, at the start of each command body:

```python
effective_ocr_mode = _resolve_ocr_mode(ocr_mode, force_ocr)
```

Replace every `loader.load(settings.resume_path)` with `loader.load(settings.resume_path, ocr_mode=effective_ocr_mode)`.

In `batch`, the loop `resume = loader.load(p)` becomes `resume = loader.load(p, ocr_mode=effective_ocr_mode)`.

- [ ] **Step 3: Run CLI help to verify flags appear**

Run: `.venv/bin/job-applicator match --help`

Expected: Output includes `--ocr-mode` and `--force-ocr` options.

- [ ] **Step 4: Commit**

```bash
git add src/job_applicator/cli.py
git commit -m "feat(cli): add --ocr-mode and --force-ocr flags"
```

---

### Task 5: Update existing tests if needed

**Covers:** [S9]

**Files:**
- Modify: `tests/unit/test_documents.py` (existing `test_resume_loader_unsupported_format` and similar)

- [ ] **Step 1: Run the full document test suite**

Run: `.venv/bin/python -m pytest tests/unit/test_documents.py -v`

Expected: All tests pass, including existing PDF/DOCX/text tests.

- [ ] **Step 2: Fix any regressions**

If `test_resume_loader_unsupported_format` fails because `.png` is now supported, update it to use a different unsupported extension such as `.xyz`.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_documents.py
git commit -m "test(documents): adjust unsupported-format test for image support"
```

---

### Task 6: Run full verification

**Covers:** [S9]

**Files:**
- None

- [ ] **Step 1: Run lint and format checks**

Run:
```bash
.venv/bin/ruff check src/ tests/
.venv/bin/ruff format --check src/ tests/
```

Expected: All checks pass. Auto-fix if needed with `.venv/bin/ruff check --fix src/ tests/` and `.venv/bin/ruff format src/ tests/`.

- [ ] **Step 2: Run typecheck**

Run: `.venv/bin/mypy src/job_applicator/ --ignore-missing-imports`

Expected: Success: no issues found.

- [ ] **Step 3: Run unit tests**

Run: `.venv/bin/python -m pytest tests/unit/ -q`

Expected: 280+ tests pass.

- [ ] **Step 4: Run live OCR smoke test (optional but recommended)**

Create a simple scanned PDF or image with text and a jobs JSON file, then run:

```bash
# Create a jobs file if you don't have one
.venv/bin/python -c "import json; json.dump([{'title': 'Dev', 'company': 'Co', 'description': 'Python', 'requirements': ['Python'], 'url': 'https://example.com/job', 'location': 'Remote', 'board': 'linkedin'}], open('/tmp/jobs.json','w'))"

.venv/bin/job-applicator match --resume scanned_resume.pdf --ocr-mode on --jobs-file /tmp/jobs.json
```

Expected: Resume loads without `DocumentError` and match score is computed.

- [ ] **Step 5: Commit final verification state**

```bash
git commit --allow-empty -m "chore: verification passed for OCR fallback"
```

---

## Self-Review Checklist

- [ ] Spec coverage: Every `[Sn]` section from `2026-06-13-ocr-fallback-design.md` is covered by at least one task.
- [ ] Placeholder scan: No "TBD", "TODO", or vague steps remain.
- [ ] Type consistency: `ocr_mode` is always a `str`; `_resolve_ocr_mode` returns `str`.
- [ ] CLI flag interaction: `--force-ocr` overrides `--ocr-mode` as specified in [S7].
- [ ] Backwards compatibility: `ResumeLoader.load()` defaults to `ocr_mode="auto"`, preserving existing behavior.
