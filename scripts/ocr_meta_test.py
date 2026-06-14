"""End-to-end and edge-case tests for OCR fallback through the CLI."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path("/home/drei/project/job-applicator-python")
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
CLI = PROJECT_ROOT / ".venv" / "bin" / "job-applicator"


def create_text_pdf(path: Path, text: str) -> None:
    """Create a normal text-layer PDF using reportlab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        raise RuntimeError("reportlab required for test file creation") from None

    c = canvas.Canvas(str(path), pagesize=letter)
    y = 700
    for line in text.split("\n"):
        c.drawString(72, y, line)
        y -= 20
    c.save()


def create_image_only_pdf(path: Path, text: str) -> None:
    """Create a scanned PDF: text rendered as an image, no text layer."""
    img = Image.new("RGB", (612, 792), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    draw.text((72, 72), text, fill="black", font=font)

    with TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "page.png"
        img.save(img_path, dpi=(300, 300))

        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
        except ImportError:
            raise RuntimeError("reportlab required for test file creation") from None

        c = canvas.Canvas(str(path), pagesize=letter)
        c.drawImage(str(img_path), 0, 0, width=letter[0], height=letter[1])
        c.save()


def create_text_image(path: Path, text: str) -> None:
    """Create a PNG image containing text."""
    img = Image.new("RGB", (600, 400), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    draw.text((30, 30), text, fill="black", font=font)
    img.save(path)


def run_cli(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    """Run the CLI and return the completed process."""
    cmd = [str(CLI), *args]
    start = time.monotonic()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = time.monotonic() - start
    return result, elapsed


def assert_contains(haystack: str, needle: str, description: str) -> None:
    if needle not in haystack:
        raise AssertionError(f"{description}: expected '{needle}' in:\n{haystack}")


def assert_not_contains(haystack: str, needle: str, description: str) -> None:
    if needle in haystack:
        raise AssertionError(f"{description}: did not expect '{needle}' in:\n{haystack}")


def main() -> int:
    results: list[tuple[str, str, float, str]] = []

    with TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        real_pdf = base / "real_text.pdf"
        scanned_pdf = base / "scanned.pdf"
        resume_png = base / "resume.png"
        jobs_file = base / "jobs.json"
        corrupted = base / "corrupted.png"

        resume_text = (
            "John Doe\n"
            "Senior Python Developer\n"
            "Email: john.doe@example.com\n"
            "Phone: (555) 123-4567\n\n"
            "Summary:\n"
            "Experienced Python developer with expertise in machine learning, "
            "data engineering, and cloud infrastructure.\n\n"
            "Skills: Python, Machine Learning, TensorFlow, PyTorch, AWS, Docker, Kubernetes"
        )
        create_text_pdf(real_pdf, resume_text)
        create_image_only_pdf(scanned_pdf, resume_text)
        create_text_image(resume_png, resume_text)
        corrupted.write_bytes(b"not an image")

        jobs = [
            {
                "title": "Python Developer",
                "company": "TestCo",
                "description": "We need a Python developer with machine learning experience.",
                "requirements": ["Python", "Machine Learning"],
                "url": "https://example.com/job1",
                "location": "Remote",
                "board": "linkedin",
            }
        ]
        jobs_file.write_text(json.dumps(jobs))

        # Scenario 1: Help shows new flags
        try:
            result, elapsed = run_cli(["match", "--help"])
            out = result.stdout + result.stderr
            assert result.returncode == 0, f"match --help failed:\n{out}"
            assert_contains(out, "--ocr-mode", "help shows --ocr-mode")
            assert_contains(out, "--force-ocr", "help shows --force-ocr")
            results.append(("help shows OCR flags", "PASS", elapsed, ""))
        except Exception as exc:
            results.append(("help shows OCR flags", "FAIL", 0.0, str(exc)))

        # Scenario 2: Normal PDF auto mode should NOT trigger OCR
        try:
            result, elapsed = run_cli(
                [
                    "match",
                    "--resume",
                    str(real_pdf),
                    "--jobs-file",
                    str(jobs_file),
                    "--top-k",
                    "1",
                    "--ocr-mode",
                    "auto",
                    "--json",
                ]
            )
            out = result.stdout + result.stderr
            assert result.returncode == 0, f"normal PDF auto failed:\n{out}"
            assert_not_contains(out, "OCR fallback triggered", "normal PDF should not OCR")
            assert_contains(out, "Python Developer", "match output includes job title")
            results.append(("normal PDF auto (no OCR)", "PASS", elapsed, ""))
        except Exception as exc:
            results.append(("normal PDF auto (no OCR)", "FAIL", 0.0, str(exc)))

        # Scenario 3: Scanned PDF auto mode should trigger OCR fallback
        try:
            result, elapsed = run_cli(
                [
                    "match",
                    "--resume",
                    str(scanned_pdf),
                    "--jobs-file",
                    str(jobs_file),
                    "--top-k",
                    "1",
                    "--ocr-mode",
                    "auto",
                    "--json",
                ]
            )
            out = result.stdout + result.stderr
            assert result.returncode == 0, f"scanned PDF auto failed:\n{out}"
            assert_contains(out, "OCR fallback triggered", "scanned PDF should trigger OCR")
            assert_contains(out, "Python Developer", "match output includes job title")
            results.append(("scanned PDF auto (OCR fallback)", "PASS", elapsed, ""))
        except Exception as exc:
            results.append(("scanned PDF auto (OCR fallback)", "FAIL", 0.0, str(exc)))

        # Scenario 4: Scanned PDF off mode should fail
        try:
            result, elapsed = run_cli(
                [
                    "match",
                    "--resume",
                    str(scanned_pdf),
                    "--jobs-file",
                    str(jobs_file),
                    "--top-k",
                    "1",
                    "--ocr-mode",
                    "off",
                    "--json",
                ]
            )
            out = result.stdout + result.stderr
            assert result.returncode != 0, "scanned PDF off should error"
            assert_contains(out, "OCR", "off mode error mentions OCR")
            results.append(("scanned PDF off (error expected)", "PASS", elapsed, ""))
        except Exception as exc:
            results.append(("scanned PDF off (error expected)", "FAIL", 0.0, str(exc)))

        # Scenario 5: --force-ocr on normal PDF should OCR
        try:
            result, elapsed = run_cli(
                [
                    "match",
                    "--resume",
                    str(real_pdf),
                    "--jobs-file",
                    str(jobs_file),
                    "--top-k",
                    "1",
                    "--force-ocr",
                    "--json",
                ]
            )
            out = result.stdout + result.stderr
            assert result.returncode == 0, f"force-ocr failed:\n{out}"
            assert_contains(out, "Running OCR on PDF", "force-ocr should run OCR")
            assert_contains(out, "Python Developer", "match output includes job title")
            results.append(("force-ocr on normal PDF", "PASS", elapsed, ""))
        except Exception as exc:
            results.append(("force-ocr on normal PDF", "FAIL", 0.0, str(exc)))

        # Scenario 6: Image resume with --ocr-mode on should succeed
        try:
            result, elapsed = run_cli(
                [
                    "match",
                    "--resume",
                    str(resume_png),
                    "--jobs-file",
                    str(jobs_file),
                    "--top-k",
                    "1",
                    "--ocr-mode",
                    "on",
                    "--json",
                ]
            )
            out = result.stdout + result.stderr
            assert result.returncode == 0, f"image on failed:\n{out}"
            assert_contains(out, "Running OCR on image", "image resume should run OCR")
            assert_contains(out, "Python Developer", "match output includes job title")
            results.append(("image PNG --ocr-mode on", "PASS", elapsed, ""))
        except Exception as exc:
            results.append(("image PNG --ocr-mode on", "FAIL", 0.0, str(exc)))

        # Scenario 7: Image resume with --ocr-mode off should fail
        try:
            result, elapsed = run_cli(
                [
                    "match",
                    "--resume",
                    str(resume_png),
                    "--jobs-file",
                    str(jobs_file),
                    "--top-k",
                    "1",
                    "--ocr-mode",
                    "off",
                    "--json",
                ]
            )
            out = result.stdout + result.stderr
            assert result.returncode != 0, "image off should error"
            assert_contains(out, "OCR", "image off error mentions OCR")
            results.append(("image PNG --ocr-mode off (error)", "PASS", elapsed, ""))
        except Exception as exc:
            results.append(("image PNG --ocr-mode off (error)", "FAIL", 0.0, str(exc)))

        # Scenario 8: Invalid --ocr-mode value
        try:
            result, elapsed = run_cli(
                [
                    "match",
                    "--resume",
                    str(real_pdf),
                    "--jobs-file",
                    str(jobs_file),
                    "--top-k",
                    "1",
                    "--ocr-mode",
                    "maybe",
                    "--json",
                ]
            )
            out = result.stdout + result.stderr
            assert result.returncode != 0, "invalid mode should error"
            results.append(("invalid --ocr-mode value (error)", "PASS", elapsed, ""))
        except Exception as exc:
            results.append(("invalid --ocr-mode value (error)", "FAIL", 0.0, str(exc)))

        # Scenario 9: Corrupted image with --ocr-mode on should fail gracefully
        try:
            result, elapsed = run_cli(
                [
                    "match",
                    "--resume",
                    str(corrupted),
                    "--jobs-file",
                    str(jobs_file),
                    "--top-k",
                    "1",
                    "--ocr-mode",
                    "on",
                    "--json",
                ]
            )
            out = result.stdout + result.stderr
            assert result.returncode != 0, "corrupted image should error"
            assert_contains(out, "OCR failed", "corrupted image error mentions OCR failure")
            results.append(("corrupted image --ocr-mode on (error)", "PASS", elapsed, ""))
        except Exception as exc:
            results.append(("corrupted image --ocr-mode on (error)", "FAIL", 0.0, str(exc)))

    # Print report
    print("\n=== OCR Fallback Meta-Test Report ===\n")
    pass_count = sum(1 for _, status, _, _ in results if status == "PASS")
    fail_count = len(results) - pass_count
    for name, status, elapsed, detail in results:
        line = f"[{status}] {name} ({elapsed:.2f}s)"
        if detail:
            line += f"\n    -> {detail}"
        print(line)

    print(f"\nTotal: {len(results)} | PASS: {pass_count} | FAIL: {fail_count}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
