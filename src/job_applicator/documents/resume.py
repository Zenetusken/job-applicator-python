"""Resume parser and loader."""

from __future__ import annotations

import re
from pathlib import Path

from job_applicator.documents.ocr import OCRService
from job_applicator.exceptions import DocumentError, ResumeNotFoundError
from job_applicator.models import ResumeData
from job_applicator.utils.logging import get_logger

logger = get_logger("documents.resume")

OCR_THRESHOLD = 100

# Section headers used for confidence scoring and skills-section boundaries.
_KNOWN_SECTIONS = frozenset(
    ("experience", "education", "skills", "certifications", "languages", "projects")
)


class ResumeLoader:
    """Load and parse resume files."""

    def __init__(self) -> None:
        self._ocr_service: OCRService | None = None

    @property
    def ocr_service(self) -> OCRService:
        if self._ocr_service is None:
            self._ocr_service = OCRService()
        return self._ocr_service

    def load(self, path: str | Path, ocr_mode: str = "auto") -> ResumeData:
        """Load a resume from file. Supports PDF, DOCX, plain text, and images."""
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

    def _load_docx(self, path: Path) -> ResumeData:
        """Load a DOCX resume."""
        try:
            from docx import Document
        except ImportError as exc:
            raise DocumentError("python-docx not installed. Run: pip install python-docx") from exc

        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs)
        return self.parse_text(text, method="docx")

    def _load_pdf(self, path: Path, ocr_mode: str = "auto") -> ResumeData:
        if ocr_mode not in {"auto", "on", "off"}:
            raise DocumentError(f"Invalid ocr_mode '{ocr_mode}'. Valid modes: auto, on, off")

        if self._is_password_protected(path):
            raise DocumentError(
                f"PDF {path} is password-protected. "
                "Remove the password or save an unprotected copy."
            )

        if ocr_mode == "on":
            text = self.ocr_service.extract_text_from_pdf(path)
            return self.parse_text(text, method="ocr")

        if ocr_mode == "off":
            result = self._pdf_consensus(path, methods=("pdftotext", "pymupdf"))
            if len(result.raw_text.strip()) < OCR_THRESHOLD:
                raise DocumentError(
                    f"PDF {path} contains insufficient extractable text; "
                    "enable OCR with --ocr-mode auto or --force-ocr"
                )
            return result

        # ocr_mode == "auto": run text extractors, fall back to OCR if they are short.
        # ocr_mode == "auto": run text extractors, fall back to OCR if they are short.
        try:
            auto_result = self._pdf_consensus(path, methods=("pdftotext", "pymupdf"))
            if len(auto_result.raw_text.strip()) >= OCR_THRESHOLD:
                return auto_result
            fallback_confidence = auto_result.parse_confidence
            fallback_result = auto_result
        except DocumentError:
            fallback_confidence = 0.0
            fallback_result = None

        logger.info("OCR fallback triggered for %s", path)
        try:
            ocr_text = self.ocr_service.extract_text_from_pdf(path)
            ocr_result = self.parse_text(ocr_text, method="ocr")
            if fallback_result is None or ocr_result.parse_confidence > fallback_confidence:
                logger.info(
                    "OCR result is better for %s (confidence=%.2f)",
                    path,
                    ocr_result.parse_confidence,
                )
                return ocr_result
            logger.info("OCR did not improve confidence for %s; keeping text extractor", path)
            return fallback_result
        except DocumentError:
            if fallback_result is not None:
                logger.warning("OCR failed for %s; using best text extractor", path)
                return fallback_result
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

    def _pdf_consensus(self, path: Path, methods: tuple[str, ...]) -> ResumeData:
        """Extract PDF text with the given methods and return the best parse.

        Each method produces a candidate; the candidate with the highest
        heuristic confidence wins. This protects against a single parser
        returning garbled or incomplete text.
        """
        candidates: list[tuple[ResumeData, str]] = []
        for method in methods:
            raw = self._extract_pdf_with_method(path, method)
            if raw.strip():
                parsed = self.parse_text(raw, method=method)
                candidates.append((parsed, method))
                logger.debug(
                    "PDF %s via %s: confidence=%.2f", path, method, parsed.parse_confidence
                )

        if not candidates:
            raise DocumentError(
                f"PDF {path} contains insufficient extractable text; "
                "enable OCR with --ocr-mode auto or --force-ocr"
            )

        candidates.sort(key=lambda item: item[0].parse_confidence, reverse=True)
        best, method = candidates[0]
        logger.info(
            "Selected PDF parser %s for %s (confidence=%.2f)", method, path, best.parse_confidence
        )
        return best

    def _extract_pdf_with_method(self, path: Path, method: str) -> str:
        if method == "pdftotext":
            return self._run_pdftotext(path)
        if method == "pymupdf":
            return self._run_pymupdf(path)
        if method == "ocr":
            return self.ocr_service.extract_text_from_pdf(path)
        raise DocumentError(f"Unknown PDF extraction method: {method}")

    def _is_password_protected(self, path: Path) -> bool:
        """Best-effort detection of password-protected PDFs."""
        try:
            import fitz
        except ImportError:
            return False
        try:
            doc = fitz.open(str(path))
            doc.close()
            return False
        except Exception as exc:
            message = str(exc).lower()
            return "password" in message or "encrypted" in message or "crypt" in message

    def _compute_confidence(self, text: str) -> float:
        """Heuristic confidence based on length, contact info, and section coverage."""
        stripped = text.strip()
        if not stripped:
            return 0.0

        score = 0.0
        # Length signal: up to 0.3 for a reasonably long resume.
        score += min(len(stripped) / 2000, 0.3)

        # Contact signals.
        if re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", stripped):
            score += 0.2
        if any(
            sum(c.isdigit() for c in m.group(0)) >= 10
            for m in re.finditer(r"[\+]?[\d\s\-\(\)]{10,}", stripped)
        ):
            score += 0.2

        # Section coverage.
        lower = stripped.lower()
        for section in _KNOWN_SECTIONS:
            if section in lower:
                score += 0.1

        return min(round(score, 2), 1.0)

    def _load_image(self, path: Path, ocr_mode: str = "auto") -> ResumeData:
        if ocr_mode == "off":
            raise DocumentError(f"Image resume {path} requires OCR, but ocr_mode is 'off'")
        text = self.ocr_service.extract_text_from_image(path)
        return self.parse_text(text, method="ocr")

    def _load_text(self, path: Path) -> ResumeData:
        """Load plain text resume."""
        text = path.read_text(encoding="utf-8")
        return self.parse_text(text, method="text")

    def parse_text(self, text: str, method: str = "") -> ResumeData:
        """Parse raw text into structured ResumeData."""
        lines = text.strip().split("\n")
        name = lines[0].strip() if lines else ""

        # Extract email
        email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
        email = email_match.group(0) if email_match else ""

        # Extract phone — find sequences with at least 10 actual digits
        phone_pattern = r"[\+]?[\d\s\-\(\)]{10,}"
        phone = ""
        for match in re.finditer(phone_pattern, text):
            candidate = match.group(0)
            digit_count = sum(c.isdigit() for c in candidate)
            if digit_count >= 10:
                phone = candidate.strip()
                break

        # Extract skills section
        skills: list[str] = []
        summary = ""
        skills_section = self._extract_skills_section(text)

        if skills_section is not None:
            # Parse skills - handle various formats:
            # - Comma separated: "Python, Java, C++"
            # - Bullet list: "•\n\nSkill Name"
            # - Line list: "Skill1\nSkill2\nSkill3"
            # - Two-column bullets: "Skill1 • Skill2"

            # Remove bullet characters and empty lines
            lines = skills_section.split("\n")
            clean_lines = []
            for line in lines:
                stripped = line.strip()
                # Skip empty lines, bullets, and separators
                if stripped and stripped not in ("•", "·", "-", "|", "/"):
                    # Split on middle bullets (two-column format)
                    parts = re.split(r"\s+[•·]\s+", stripped)
                    for part in parts:
                        # Remove leading bullets
                        part = re.sub(r"^[•·\-\|/]\s*", "", part).strip()
                        if part and len(part) > 2:
                            clean_lines.append(part)

            # Determine if skills are comma-separated on one line or one-per-line
            if len(clean_lines) == 1 and "," in clean_lines[0]:
                # All skills on one line, comma-separated
                raw = clean_lines[0].split(",")
                skills = [s.strip() for s in raw if s.strip() and len(s.strip()) > 2]
            elif len(clean_lines) > 1:
                # One skill per line
                skills = [s.strip() for s in clean_lines if len(s.strip()) > 2]
            else:
                # Single skill or empty
                skills = [s.strip() for s in clean_lines if len(s.strip()) > 2]

        # Extract summary/objective
        if "Summary" in text:
            # Prefer **Professional Summary** or **Summary** header
            summary_match = re.search(
                r"\*?\*?(?:Professional\s+)?Summary\*?\*?\s*[:\n](.*?)(?:\n\n|\n\*\*)",
                text,
                re.IGNORECASE | re.DOTALL,
            )
            if summary_match:
                summary = summary_match.group(1).strip()
            else:
                summary = text.split("Summary", 1)[-1].split("\n\n")[0].strip()
                summary = re.sub(r"^[:\s]+", "", summary)
        elif "Objective" in text:
            summary = text.split("Objective", 1)[-1].split("\n\n")[0].strip()
            summary = re.sub(r"^[:\s]+", "", summary)
        elif "objective" in text.lower():
            idx = text.lower().index("objective")
            summary = text[idx:].split("\n\n")[0].strip()
            summary = re.sub(r"^objective[:\s]*", "", summary, flags=re.IGNORECASE)
        else:
            # Fallback: detect first paragraph after contact info
            # Skip name line and contact line (email/phone)
            lines = text.strip().split("\n")
            contact_end = 0
            for i, line in enumerate(lines[:5]):
                has_email = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", line)
                has_phone = re.search(r"[\+]?[\d\s\-\(\)]{10,}", line)
                if has_email or has_phone:
                    contact_end = i + 1
            # Find first non-empty paragraph after contact
            paragraph_lines: list[str] = []
            for line in lines[contact_end:]:
                stripped = line.strip()
                if not stripped:
                    if paragraph_lines:
                        break
                    continue
                # Stop at section headers
                if stripped in ("Skills", "Experience", "Education", "Certifications", "Languages"):
                    break
                paragraph_lines.append(stripped)
            if paragraph_lines:
                summary = " ".join(paragraph_lines)
                # Only keep it if it looks like a summary, not a stray header.
                if len(summary) <= 50 or summary.isupper():
                    summary = ""

        confidence = self._compute_confidence(text)
        logger.info(
            "Parsed resume: name=%s, skills=%d, confidence=%.2f", name, len(skills), confidence
        )
        return ResumeData(
            raw_text=text,
            name=name,
            email=email,
            phone=phone,
            summary=summary,
            skills=skills,
            parse_confidence=confidence,
            parse_method=method,
        )

    def _extract_skills_section(self, text: str) -> str | None:
        """Extract the text between the Skills header and the next section header.

        Returns None if no standalone Skills section header is found.
        Handles markdown bold headers ("**Skills**") and optional colon.
        """
        import re

        # Match a standalone Skills header (optional markdown bold, optional
        # colon, optional leading whitespace). Also allow inline skills after
        # the colon on the same line, e.g. "Skills: Python, Java". Recognizes
        # the common qualified variants the tailor uses too ("Technical
        # Skills", "Core Competencies", "Key Skills", etc.).
        pattern = re.compile(
            r"^\s*\*{0,2}\s*"
            r"(?:(?:Technical|Core|Key|Professional|Relevant|Soft)\s+)?"
            r"(?:Skills|Competencies|Proficiencies)"
            r"\s*\*{0,2}\s*:?\s*(.*)$",
            re.IGNORECASE | re.MULTILINE,
        )
        match = pattern.search(text)
        if not match:
            return None

        inline_skills = match.group(1).strip()
        start = match.end()
        remaining = text[start:]

        # If there were inline skills after the colon, prepend them to the section
        if inline_skills:
            remaining = inline_skills + "\n" + remaining

        # Find the next known section header (allow inline content after colon too)
        next_header = re.compile(
            r"^\s*\*{0,2}\s*(?:Experience|Education|Certifications|Languages|Interests|Projects|Volunteer|References|Awards)\s*\*{0,2}\s*:?",
            re.IGNORECASE | re.MULTILINE,
        )
        next_match = next_header.search(remaining)
        if next_match:
            return remaining[: next_match.start()]
        return remaining
