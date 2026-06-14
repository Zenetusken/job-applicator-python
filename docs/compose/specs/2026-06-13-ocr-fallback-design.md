# OCR Fallback for Scanned PDFs and Images

## [S1] Problem

The current `ResumeLoader` only extracts text from PDFs via `pdftotext` and PyMuPDF. Scanned PDFs or image-based resumes contain no extractable text layer, so these loaders return empty or near-empty `ResumeData`, breaking downstream matching and tailoring.

## [S2] Goals

1. Automatically detect when a PDF or image resume needs OCR.
2. Extract text from scanned PDFs and image files using PaddleOCR.
3. Allow users to force or disable OCR via CLI flags.
4. Keep the existing text-extraction path fast and unchanged when OCR is not needed.
5. Provide clear error messages when OCR is unavailable or fails.

## [S3] Non-Goals

1. No support for handwritten resumes.
2. No layout reconstruction (tables, columns); only plain text extraction.
3. No cloud OCR APIs in this iteration.
4. No batch OCR progress UI beyond existing status messages.

## [S4] Architecture

A new `OCRService` lives in `src/job_applicator/documents/ocr.py`. `ResumeLoader` delegates to it when needed. CLI commands that load resumes receive two new flags: `--ocr-mode` and `--force-ocr`. The OCR service is lazy-loaded so startup time is unaffected when OCR is not used.

## [S5] OCRService API

```python
class OCRService:
    def __init__(self) -> None: ...
    def extract_text_from_pdf(self, path: Path) -> str: ...
    def extract_text_from_image(self, path: Path) -> str: ...
```

- `__init__` instantiates `PaddleOCR(use_angle_cls=True, lang="en", show_log=False)` on first call.
- `extract_text_from_pdf` converts each page to a PIL image and runs OCR, joining results with newlines.
- `extract_text_from_image` runs OCR directly on the image file.
- Both methods return stripped text; raise `DocumentError` on failure.

## [S6] ResumeLoader Integration

`ResumeLoader.load(self, path, ocr_mode="auto")` accepts a new parameter.

`ResumeLoader._load_pdf(self, path, ocr_mode="auto")` behaves as follows:

1. If `ocr_mode == "on"`, call `OCRService.extract_text_from_pdf(path)` and return.
2. If `ocr_mode == "off"`, run the existing `pdftotext` → PyMuPDF path and return.
3. If `ocr_mode == "auto"`:
   - Try `pdftotext`.
   - If insufficient text (< 100 chars after stripping), try PyMuPDF.
   - If still insufficient text, call `OCRService.extract_text_from_pdf(path)`.
   - If OCR also fails, raise `DocumentError`.

`ResumeLoader.load` also gains support for image formats (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp`) by calling `OCRService.extract_text_from_image`.

## [S7] CLI Flags

Add to every command that loads a resume: `match`, `tailor`, `apply`, `generate-cover-letter`, `batch`.

- `--ocr-mode {auto,on,off}` (default: `auto`)
- `--force-ocr` convenience flag equivalent to `--ocr-mode on`

If both `--force-ocr` and `--ocr-mode` are passed, `--force-ocr` wins and the effective mode is `on`. Flag values are threaded through to `ResumeLoader.load(..., ocr_mode=...)`.

## [S8] Error Handling

- If OCR is requested but `paddleocr` is not installed, raise `DocumentError` with installation instructions.
- If OCR fails but normal extraction produced some text, log a warning and return the extracted text in `auto` mode.
- If no text was extracted and OCR fails, raise `DocumentError`.
- If the user passes `--force-ocr` and OCR fails, raise `DocumentError` (do not silently fall back).

## [S9] Testing

Unit tests in `tests/unit/test_documents.py` with mocked `PaddleOCR`:

1. `test_ocr_fallback_triggers_on_short_text`: simulate `pdftotext`/`fitz` returning < 100 chars and assert OCR is called.
2. `test_force_ocr_skips_text_extraction`: assert neither `pdftotext` nor PyMuPDF is called when `ocr_mode="on"`.
3. `test_ocr_mode_off_disables_ocr`: assert OCR is not called even with short extracted text.
4. `test_ocr_failure_falls_back_to_extracted_text`: in `auto` mode, if some text exists and OCR fails, return the text.
5. `test_ocr_failure_with_no_text_raises`: when no text was extracted and OCR fails, raise `DocumentError`.
6. `test_image_resume_uses_ocr`: loading a `.png` calls `extract_text_from_image`.

## [S10] Dependency

Add `paddleocr` to `pyproject.toml` under `[project.dependencies]`.

## [S11] Logging

Log at `INFO` level when OCR fallback is triggered and when it completes, including the number of characters extracted.

## [S12] Security

No new secrets. OCR runs locally. Temporary image conversions, if any, use standard libraries and are cleaned up.
