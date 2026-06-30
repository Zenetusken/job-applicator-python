"""Resume parser and loader."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from job_applicator.documents.ocr import OCRService
from job_applicator.exceptions import DocumentError, JobApplicatorError, ResumeNotFoundError
from job_applicator.models import ResumeData
from job_applicator.utils.logging import get_logger

logger = get_logger("documents.resume")

OCR_THRESHOLD = 100

# Section headers used for confidence scoring. Vocabulary aligned with
# _extract_skills_section so a "Core Competencies"/"Proficiencies" resume earns
# section credit too. Matched as line-anchored HEADERS (see _SECTION_HEADER_RE),
# never as bare substrings.
_KNOWN_SECTIONS = frozenset(
    (
        "experience",
        "education",
        "skills",
        "competencies",
        "proficiencies",
        "certifications",
        "languages",
        "projects",
    )
)
_SECTION_HEADER_RE = re.compile(
    r"(?im)^\s*\*{0,2}\s*"
    r"(?:(?:technical|core|key|professional|relevant|soft)\s+)?"  # match "Core Competencies"
    r"(" + "|".join(sorted(_KNOWN_SECTIONS)) + r")\b"
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
        """Load a resume from file. Supports PDF, DOCX, plain text, and images.

        Guarantees a TYPED failure: any underlying parser error (python-docx,
        PyMuPDF, OCR, ...) is wrapped as ``DocumentError`` so callers only ever see a
        ``JobApplicatorError`` — never a raw third-party traceback.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise ResumeNotFoundError(f"Resume not found: {file_path}")

        suffix = file_path.suffix.lower()
        try:
            if suffix == ".pdf":
                data = self._load_pdf(file_path, ocr_mode=ocr_mode)
            elif suffix == ".docx":
                data = self._load_docx(file_path)
            elif suffix in (".txt", ".md"):
                data = self._load_text(file_path)
            elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"):
                data = self._load_image(file_path, ocr_mode=ocr_mode)
            else:
                fmt = suffix or "(no file extension)"
                raise DocumentError(f"Unsupported resume format: {fmt} ({file_path.name})")
        except JobApplicatorError:
            raise
        except Exception as exc:
            raise DocumentError(
                f"Could not parse resume {file_path.name}: {type(exc).__name__}: {exc}"
            ) from exc

        if not data.raw_text.strip():
            raise DocumentError(
                f"Resume has no extractable text: {file_path.name} (the file is empty, or its "
                "text could not be extracted — for a scanned PDF try --ocr-mode on)"
            )
        return data

    def _load_docx(self, path: Path) -> ResumeData:
        """Load a DOCX resume.

        Walks the document body in order, extracting BOTH paragraphs and table cells. Real
        résumés routinely put the contact header and the skills section in tables; reading
        only ``doc.paragraphs`` (the prior behaviour) silently dropped that content from the
        text, so the email/phone/skills never reached matching or tailoring — a tailored CV
        could even go out missing the contact block (audit AI-H5). Document order is preserved
        so a leading contact table stays first, where ``parse_text``'s first-line name
        heuristic can find it.
        """
        try:
            from docx import Document
            from docx.oxml.table import CT_Tbl
            from docx.oxml.text.paragraph import CT_P
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError as exc:
            raise DocumentError("python-docx not installed. Run: pip install python-docx") from exc

        doc = Document(str(path))
        parts: list[str] = []
        for child in doc.element.body.iterchildren():
            if isinstance(child, CT_P):
                line = Paragraph(child, doc).text
                if line.strip():
                    parts.append(line)
            elif isinstance(child, CT_Tbl):
                # Append each unique cell's text. python-docx repeats the same cell across a
                # merged grid span, so dedup by the underlying <w:tc> element to avoid doubling
                # merged contact/header cells. Best-effort PER TABLE: a structurally malformed
                # grid (e.g. an orphan vMerge continuation — a real export artifact) raises in
                # python-docx's grid walk, so skip that one table with a logged warning rather
                # than failing the whole résumé. The table is collected atomically (extend only
                # on success) so a mid-walk failure contributes nothing, never a partial table.
                # Paragraphs and well-formed tables are unaffected — strictly more robust than
                # the prior paragraphs-only reader, and the skip is disclosed (not masked).
                try:
                    cell_texts: list[str] = []
                    seen: set[object] = set()
                    for row in Table(child, doc).rows:
                        for cell in row.cells:
                            if cell._tc in seen:
                                continue
                            seen.add(cell._tc)
                            cell_text = cell.text.strip()
                            if cell_text:
                                cell_texts.append(cell_text)
                    parts.extend(cell_texts)
                except Exception as exc:
                    logger.warning("Skipping a malformed table in %s: %s", path.name, exc)
        text = "\n".join(parts)
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
                best = ocr_result
            else:
                logger.info("OCR did not improve confidence for %s; keeping text extractor", path)
                best = fallback_result
        except DocumentError:
            if fallback_result is None:
                raise
            logger.warning("OCR failed for %s; using best text extractor", path)
            best = fallback_result

        # Guard against a silently-empty/short parse: if every text extractor
        # AND OCR yielded less than the OCR_THRESHOLD, fail loudly rather than
        # return a ResumeData with too little text for downstream match/tailor.
        if len(best.raw_text.strip()) < OCR_THRESHOLD:
            raise DocumentError(
                f"PDF {path} contains insufficient extractable text "
                f"({len(best.raw_text.strip())} chars); "
                "enable OCR with --ocr-mode on or --force-ocr"
            )
        return best

    def _run_pdftotext(self, path: Path) -> str:
        """Extract text using pdftotext; return empty string on failure."""
        import subprocess  # nosec B404
        import tempfile

        exe = shutil.which("pdftotext")
        if not exe:
            return ""

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            try:
                result = subprocess.run(  # noqa: S603 # nosec B603
                    [exe, "-layout", str(path), tmp.name],
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
            try:
                return "".join(page.get_text() for page in doc)
            finally:
                doc.close()
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
        """Best-effort detection of password-protected PDFs.

        PyMuPDF OPENS an encrypted PDF without raising and sets ``needs_pass`` —
        so check that flag rather than only whether ``fitz.open`` raised (which it
        usually doesn't). Falls back to sniffing the exception message for the
        rarer open-time failure.
        """
        try:
            import fitz
        except ImportError:
            return False
        try:
            doc = fitz.open(str(path))
            try:
                # needs_pass: encrypted AND not yet authenticated (the real gate).
                return bool(getattr(doc, "needs_pass", False))
            finally:
                doc.close()
        except Exception as exc:
            message = str(exc).lower()
            return "password" in message or "encrypted" in message or "crypt" in message

    @staticmethod
    def _extract_phone(text: str) -> str:
        """First phone-like run (10-15 digits), or "".

        Rejects numeric tables / year lists like "2019 2020 2021 2022 2023"
        (3+ space-separated 4-digit groups), which the old ``{10,}``-char pattern
        false-matched as a phone — polluting the ``phone`` field and inflating
        parse confidence.
        """
        for match in re.finditer(r"[\+]?\d[\d\s\-\(\)]{8,}\d", text):
            candidate = match.group(0).strip()
            digits = sum(c.isdigit() for c in candidate)
            if not 10 <= digits <= 15:
                continue
            groups = candidate.split()
            if len(groups) >= 3 and all(g.isdigit() and len(g) == 4 for g in groups):
                continue
            return candidate
        return ""

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
        if self._extract_phone(stripped):
            score += 0.2

        # Section coverage — count DISTINCT section headers anchored at line
        # starts (a bare "skills" inside "no skills listed" must not inflate it).
        found = {w.lower() for w in _SECTION_HEADER_RE.findall(stripped)}
        score += 0.1 * len(found)

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
        name = re.sub(r"^[*_]+|[*_]+$", "", name).strip()

        # Extract email
        email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
        email = email_match.group(0) if email_match else ""

        # Extract phone (rejects numeric tables / year lists — see _extract_phone).
        phone = self._extract_phone(text)

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
                        # Drop a leading "Category<tab>" label: two-column skills grids put the
                        # bold row label before a tab, which otherwise contaminates the first
                        # skill of each row (e.g. "Networking\tTCP/IP" must parse as "TCP/IP").
                        part = part.rsplit("\t", 1)[-1]
                        # Remove leading bullets
                        part = re.sub(r"^[•·\-\|/]\s*", "", part).strip()
                        # >= 2 so common short skills survive (Go, C#, AI, ML, UX);
                        # single chars are dropped as too noisy/ambiguous.
                        if part and len(part) >= 2:
                            clean_lines.append(part)

            # Skills may be comma-separated (often wrapping across several lines),
            # one-per-line, or a mix. Split any line containing commas into its
            # tokens and keep comma-free lines as single skills, so a wrapped comma
            # list ("Python, asyncio,\npytest, Git") parses to individual skills
            # instead of one blob per line. (The old "comma-split only when a single
            # line" path mis-parsed every wrapped comma list into per-line blobs,
            # which then matched nothing during skill-coverage scoring.)
            for clean_line in clean_lines:
                if "," in clean_line:
                    skills.extend(
                        tok.strip() for tok in clean_line.split(",") if len(tok.strip()) >= 2
                    )
                else:
                    skills.append(clean_line)

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

        # Drop a leading setext/markdown underline that can follow a header.
        summary = re.sub(r"^[\-=~*]+\s*\n?", "", summary).strip()

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

        # Strip markdown/setext underline lines (e.g. "------" or "======") that
        # immediately follow the header, so they are not parsed as a skill.
        remaining = re.sub(r"^[\-=~*]+\s*\n", "", remaining, count=1)

        # Find the next known section header (allow inline content after colon too)
        next_header = re.compile(
            r"^\s*\*{0,2}\s*(?:Experience|Education|Certifications|Languages|Interests|Projects|Volunteer|References|Awards)\s*\*{0,2}\s*:?",
            re.IGNORECASE | re.MULTILINE,
        )
        next_match = next_header.search(remaining)
        if next_match:
            return remaining[: next_match.start()]
        return remaining
