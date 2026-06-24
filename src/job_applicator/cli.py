"""CLI entry point — Typer + Rich for terminal UX."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from job_applicator import __version__
from job_applicator.config import AppSettings, LLMConfig
from job_applicator.exceptions import CookieError, JobApplicatorError
from job_applicator.factories import (
    _make_applicator,
    _make_browser,
    _make_runtime,
    _make_scraper,
)
from job_applicator.models import BatchRunSpec, DoctorReport
from job_applicator.utils.console import console, err_console
from job_applicator.utils.cookies import (
    _cookies_from_browser,
    _normalize_cookie,
    _site_specs,
    save_cookies,
)
from job_applicator.utils.llm import SERVE_SCRIPT
from job_applicator.utils.logging import setup_logging
from job_applicator.utils.profile import _detect_tone, _load_user_profile
from job_applicator.utils.verbose import VerboseReporter
from job_applicator.workflows.apply import _apply_to_jobs
from job_applicator.workflows.tailor import _tailor_workflow

if TYPE_CHECKING:
    from job_applicator.batch_state import BatchState
    from job_applicator.jobs_store import JobStore
    from job_applicator.models import (
        ATSCompatibilityResult,
        JobListing,
        ResumeData,
        TailoredResume,
    )

app = typer.Typer(
    name="job-applicator",
    help="Automated job application tool with AI-powered cover letters.",
    add_completion=False,
)

T = TypeVar("T")


@dataclass
class VerboseContext:
    verbose: bool
    log_file: str | None = None


def _get_reporter(
    ctx: typer.Context,
    command: str,
    args: dict[str, Any],
    config: dict[str, Any],
) -> VerboseReporter | None:
    vctx = ctx.obj
    if not isinstance(vctx, VerboseContext) or not vctx.verbose:
        return None
    return VerboseReporter(command=command, args=args, config=config)


async def _llm_with_retry(  # noqa: UP047 — mypy doesn't support PEP 695 yet
    console: Console,
    operation: Callable[[], Awaitable[T]],
    status_message: str = "Processing...",
    on_fail_choices: str = "[R] Retry or [Q] Quit",
) -> T | None:
    """Execute an async LLM operation with retry on failure.

    Returns the result on success, or None if the user chooses to quit.
    """
    while True:
        try:
            with console.status(status_message):
                return await operation()
        except Exception as exc:
            err_console.print(f"[red]LLM error: {escape(str(exc))}[/red]")
            choice = console.input(f"[bold cyan]{on_fail_choices}? [/bold cyan]").strip().upper()
            if choice == "Q":
                return None


class OCRMode(StrEnum):
    """Valid --ocr-mode values; typer rejects anything else at parse time (exit 2)."""

    AUTO = "auto"
    ON = "on"
    OFF = "off"


def _resolve_ocr_mode(ocr_mode: OCRMode, force_ocr: bool) -> str:
    """Return effective OCR mode from CLI flags."""
    if force_ocr:
        return "on"
    return str(ocr_mode)


def _load_jobs_file(jobs_file: str) -> list[JobListing]:
    """Load + validate a JSON jobs file into JobListings.

    Raises a clean typed ``DocumentError`` for the realistic bad-input modes — missing file,
    unreadable/directory path, non-UTF-8 bytes, malformed JSON, not a JSON array, a non-object
    entry, or an entry with invalid/missing fields — so a caller's ``except JobApplicatorError``
    renders a one-line message instead of a raw traceback. (Pathological inputs — deeply-nested
    JSON → RecursionError, multi-GB → MemoryError — are out of scope and still surface.)
    JobListing is annotation-only at module scope, so it's imported at runtime here.
    """
    import json

    from pydantic import ValidationError

    from job_applicator.exceptions import DocumentError
    from job_applicator.models import JobListing

    try:
        with open(jobs_file, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise DocumentError(f"Jobs file not found: {jobs_file}") from exc
    except json.JSONDecodeError as exc:
        raise DocumentError(f"Jobs file is not valid JSON ({jobs_file}): {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        # directory / unreadable (IsADirectoryError, PermissionError) / non-UTF-8 bytes —
        # the rest of the OSError family + decode errors, kept typed so callers stay clean.
        raise DocumentError(
            f"Could not read jobs file {jobs_file}: {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(data, list):
        raise DocumentError(
            f"Jobs file must be a JSON array of job objects ({jobs_file}); "
            f"got {type(data).__name__}"
        )

    jobs: list[JobListing] = []
    for i, item in enumerate(data, 1):
        if not isinstance(item, dict):
            raise DocumentError(f"Jobs file {jobs_file}: entry #{i} is not a job object")
        try:
            jobs.append(JobListing(**item))
        except ValidationError as exc:
            fields = ", ".join(str(e["loc"][0]) for e in exc.errors() if e.get("loc")) or "?"
            raise DocumentError(
                f"Jobs file {jobs_file}: entry #{i} has invalid/missing fields: {fields}"
            ) from exc
    return jobs


def _get_jobs_store() -> JobStore:
    """Construct the funnel store (DI seam — tests patch this to isolate the DB).

    Lazily imports the (light, sqlite-only) store module so it isn't loaded until a
    command touches the funnel; the runtime construct uses this local import, not the
    ``TYPE_CHECKING`` one, so there is no construct-time ``NameError``.
    """
    from job_applicator.jobs_store import JobStore

    return JobStore()


def _run_ats_preflight(resume: ResumeData) -> ATSCompatibilityResult:
    """Run ATS compatibility check and warn if issues found."""
    from job_applicator.documents.ats_checker import ATSChecker

    checker = ATSChecker()
    result = checker.check(resume)

    if result.is_compatible:
        return result

    console.print(f"\n[yellow]⚠ ATS Compatibility: {result.score:.0%} (Not Compatible)[/yellow]")
    for warning in result.warnings[:3]:
        console.print(f"  [yellow]![/yellow] {warning}")
    console.print(
        "  [dim]Tip: Run 'job-applicator ats-check --resume <path>' for full report[/dim]"
    )
    console.print()
    return result


def _run_ats_post_tailor(original_text: str, tailored_text: str) -> ATSCompatibilityResult | None:
    """Compare ATS compatibility before and after tailoring."""
    from job_applicator.documents.ats_checker import ATSChecker
    from job_applicator.documents.resume import ResumeLoader

    checker = ATSChecker()
    loader = ResumeLoader()

    original = loader.parse_text(original_text)
    tailored = loader.parse_text(tailored_text)

    original_result = checker.check(original)
    tailored_result = checker.check(tailored)

    before = original_result.score
    after = tailored_result.score

    if after >= before:
        console.print(
            f"\n[green]ATS Compatibility (before → after): {before:.0%} → {after:.0%} ✓[/green]"
        )
        if after >= 0.6:
            console.print("  [green]✓ All checks passing after tailoring[/green]")
    else:
        console.print(
            f"\n[yellow]⚠ ATS Compatibility (before → after): {before:.0%} → {after:.0%}[/yellow]"
        )
        original_checks = {c["name"]: c["passed"] for c in original_result.checks}
        for check in tailored_result.checks:
            if not check["passed"] and original_checks.get(check["name"], False):
                console.print(f"  [yellow]![/yellow] New issue: {check['details']}")
    console.print()
    return tailored_result


def version_callback(value: bool) -> None:
    if value:
        console.print(f"job-applicator v{__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        help="Emit structured observability report.",
    ),
    log_file: str | None = typer.Option(
        None,
        "--log-file",
        help="Write verbose report to file (requires --verbose).",
    ),
) -> None:
    """Automated job application tool with AI-powered cover letters.

    Run with no command in a terminal to open the full-screen UI (`tui`).
    """
    if log_file and not verbose:
        raise typer.BadParameter("--log-file requires --verbose")
    ctx.obj = VerboseContext(verbose=verbose, log_file=log_file)

    if ctx.invoked_subcommand is not None:
        return
    # Bare invocation: open the TUI in an interactive terminal; otherwise print help
    # (so pipes / CI / `job-applicator | cat` get usable output, never a hung UI).
    if _tui_tty_ok():
        _launch_tui()
    else:
        typer.echo(ctx.get_help())
        raise typer.Exit(0)


def _tui_tty_ok() -> bool:
    """True only when BOTH stdout and stdin are a real terminal. Textual reads keys from
    stdin, so a TTY stdout with a piped/redirected stdin (`producer | job-applicator`,
    `job-applicator < file`) would launch a UI that hangs waiting for input."""
    return sys.stdout.isatty() and sys.stdin.isatty()


def _launch_tui() -> None:
    """Run the TUI, turning a store-construction failure into a clean message — the
    stores are built before the event loop, outside the app's own error handling."""
    from job_applicator.tui import run_tui

    try:
        run_tui(_get_settings())
    except JobApplicatorError as exc:
        err_console.print(f"[yellow]⚠ {escape(str(exc))}[/yellow]")
        raise typer.Exit(1) from exc


def _verbose_option() -> bool:
    """Reusable verbose flag for subcommands."""
    return typer.Option(False, "--verbose", "-V", help="Emit structured observability report.")  # type: ignore[no-any-return]


def _log_file_option() -> str | None:
    """Reusable log-file flag for subcommands."""
    return typer.Option(  # type: ignore[no-any-return]
        None,
        "--log-file",
        help="Write verbose report to file (requires --verbose).",
    )


def _merge_verbose_ctx(ctx: typer.Context, verbose: bool, log_file: str | None) -> None:
    """Merge subcommand --verbose/--log-file into global VerboseContext."""
    existing = ctx.obj
    global_verbose = isinstance(existing, VerboseContext) and existing.verbose
    if log_file and not verbose and not global_verbose:
        raise typer.BadParameter("--log-file requires --verbose")
    if isinstance(existing, VerboseContext):
        if verbose or existing.verbose:
            ctx.obj = VerboseContext(verbose=True, log_file=log_file or existing.log_file)
    else:
        ctx.obj = VerboseContext(verbose=verbose, log_file=log_file)


@app.command()
def search(
    ctx: typer.Context,
    site: str = typer.Option("linkedin", "--site", "-s", help="Job board to search."),
    query: str = typer.Option(..., "--query", "-q", help="Search query."),
    location: str = typer.Option("", "--location", "-l", help="Location filter."),
    remote: bool = typer.Option(False, "--remote", "-r", help="Remote jobs only."),
    max_results: int = typer.Option(25, "--max", "-n", help="Max results."),
    headed: bool = typer.Option(False, "--headed", help="Run browser in headed mode."),
    as_json: bool = typer.Option(False, "--json", help="Output results as JSON."),
    verbose: bool = _verbose_option(),
    log_file: str | None = _log_file_option(),
) -> None:
    """Search for jobs on a job board."""
    _merge_verbose_ctx(ctx, verbose, log_file)
    settings = _get_settings(headed)
    setup_logging(settings.log_level)

    reporter = _get_reporter(
        ctx=ctx,
        command="search",
        args={"query": query, "board": site, "location": location, "limit": max_results},
        config=_sanitize_config(settings),
    )

    async def _run() -> None:
        from job_applicator.models import JobBoard
        from job_applicator.scrapers.base import SearchParams

        board_map = {"linkedin": JobBoard.LINKEDIN, "indeed": JobBoard.INDEED}
        if site not in board_map:
            err_console.print(f"[red]Unsupported site: {site}[/red]")
            raise typer.Exit(1)

        params = SearchParams(
            query=query,
            location=location,
            remote_only=remote,
            max_results=max_results,
            board=board_map[site],
        )

        async with _make_browser(site, settings) as browser:
            scraper = _make_scraper(site, browser, settings)

            with console.status(f"Searching {site} for '{query}'..."):
                jobs = await scraper.scrape(params)

        if not jobs:
            if as_json:
                console.print("[]")
            else:
                console.print("[yellow]No jobs found.[/yellow]")
            return

        # Persist discovered jobs so they flow into match/tailor/apply and `status`.
        # Best-effort: a store hiccup must not sink the freshly-scraped results.
        try:
            store = _get_jobs_store()
            for job in jobs:
                store.upsert_job(job, source_query=query)
        except JobApplicatorError as exc:
            err_console.print(
                f"[yellow]⚠ Could not save jobs to the store: {escape(str(exc))}[/yellow]"
            )

        if as_json:
            import json

            output = [
                {
                    "title": j.title,
                    "company": j.company,
                    "location": j.location,
                    "url": str(j.url),
                    "description": j.description,
                    "requirements": j.requirements,
                    "board": j.board.value,
                }
                for j in jobs
            ]
            sys.stdout.write(json.dumps(output, indent=2) + "\n")
            return

        table = Table(title=f"Found {len(jobs)} jobs")
        table.add_column("Title", style="cyan")
        table.add_column("Company", style="green")
        table.add_column("Location")
        table.add_column("URL", style="blue")

        for job in jobs:
            table.add_row(job.title, job.company, job.location, str(job.url))

        console.print(table)

    try:
        asyncio.run(_run())
    except JobApplicatorError as exc:
        # Typed, expected failures (no session, anti-bot block, missing resume)
        # — show the message cleanly instead of a raw Python traceback.
        if reporter:
            reporter.record_error(str(exc))
        err_console.print(f"[yellow]⚠ {escape(str(exc))}[/yellow]")
        raise typer.Exit(1) from exc
    except Exception as exc:
        if reporter:
            reporter.record_error(str(exc))
        raise
    finally:
        if reporter:
            log_file = None
            vctx = ctx.obj
            if isinstance(vctx, VerboseContext):
                log_file = vctx.log_file
            reporter.render(err_console, log_file=log_file)


@app.command()
def login(
    ctx: typer.Context,
    site: str = typer.Option("linkedin", "--site", "-s", help="Job board to sign in to."),
    timeout: int = typer.Option(
        300, "--timeout", help="Seconds to wait for you to complete the manual sign-in."
    ),
    verbose: bool = _verbose_option(),
    log_file: str | None = _log_file_option(),
) -> None:
    """Sign in once in a real browser window; the session is saved and reused.

    LinkedIn blocks automated password logins with a CAPTCHA, so this opens a
    headed browser, pre-fills your configured credentials, and waits while you
    click Sign in and solve any CAPTCHA/2FA. The authenticated session is stored
    in the persistent browser profile and reused by `search`/`apply` headlessly,
    so you only do this once (until the session expires). You submit the form
    yourself — nothing is automated — which is far safer than a programmatic
    login, though no automated LinkedIn use is entirely risk-free.
    """
    _merge_verbose_ctx(ctx, verbose, log_file)
    settings = _get_settings(headed=True)  # manual sign-in needs a visible window
    setup_logging(settings.log_level)

    if site != "linkedin":
        err_console.print(f"[yellow]{site} login not yet implemented[/yellow]")
        raise typer.Exit(1)

    async def _run() -> bool:
        from job_applicator.browser.manager import BrowserManager
        from job_applicator.scrapers.linkedin import LinkedInScraper

        async with BrowserManager(settings.browser) as browser:
            scraper = LinkedInScraper(browser, settings)
            return await scraper.interactive_login(timeout_s=timeout)

    console.print(
        Panel(
            "A browser window will open. Click [bold]Sign in[/bold] and solve any "
            "CAPTCHA / 2FA. Your credentials are pre-filled from config; nothing is "
            "submitted automatically.",
            title="LinkedIn sign-in",
            style="cyan",
        )
    )
    if asyncio.run(_run()):
        console.print(
            "[green]✓ Signed in. Session saved — `search`/`apply` will reuse it headlessly.[/green]"
        )
    else:
        err_console.print("[red]✗ Sign-in not detected. Re-run `job-applicator login`.[/red]")
        raise typer.Exit(1)


@app.command(name="import-cookies")
def import_cookies(
    ctx: typer.Context,
    site: str = typer.Option("linkedin", "--site", "-s", help="Job board: linkedin or indeed."),
    li_at: str = typer.Option(
        "",
        "--li-at",
        help="The `li_at` cookie value. NOTE: a value here is visible in shell history — "
        "prefer --from-browser, or pass '-' to read the token from stdin.",
    ),
    jsessionid: str = typer.Option(
        "", "--jsessionid", help="Optional JSESSIONID value (needed only for some write actions)."
    ),
    file: str = typer.Option(
        "", "--file", help="Path to a cookie JSON export (alternative to --li-at)."
    ),
    from_browser: str = typer.Option(
        "",
        "--from-browser",
        help="Read the session straight from a local browser's cookie store "
        "(chrome/chromium/brave/edge/firefox). Needs the 'browser' extra; reads/decrypts "
        "your browser cookie store, so it only runs when you pass this flag.",
    ),
    verify: bool = typer.Option(
        True, "--verify/--no-verify", help="Confirm the session by loading the feed once."
    ),
    verbose: bool = _verbose_option(),
    log_file: str | None = _log_file_option(),
) -> None:
    """Import a LinkedIn session cookie exported from your normal browser.

    This keeps the actual sign-in in your everyday browser (a normal, human
    event) — the tool only *reuses* the resulting session, so the login never
    happens inside an automation-controlled browser.

    Get the cookie: log into LinkedIn normally, open DevTools -> Application ->
    Cookies -> https://www.linkedin.com -> copy the `li_at` value, then run:

        job-applicator import-cookies --li-at "<value>"

    Or pass a JSON export from a cookie-manager extension with --file.

    To avoid copying anything at all, read the session straight from a local
    browser (this decrypts that browser's cookie store, so it only happens when
    you ask for it):

        job-applicator import-cookies --from-browser chrome

    Use --site indeed to import an Indeed session the same way.
    """
    import json

    _merge_verbose_ctx(ctx, verbose, log_file)
    settings = _get_settings()
    setup_logging(settings.log_level)

    specs = _site_specs()
    if site not in specs:
        err_console.print(f"[red]Unsupported site '{site}'. Choose: {', '.join(specs)}.[/red]")
        raise typer.Exit(1)
    spec = specs[site]
    if (li_at or jsessionid) and not spec.session_flags:
        err_console.print(
            "[red]--li-at/--jsessionid are LinkedIn-only; "
            f"use --from-browser/--file for {site}.[/red]"
        )
        raise typer.Exit(1)

    cookies: list[dict[str, Any]] = []
    if from_browser:
        try:
            cookies = _cookies_from_browser(from_browser, spec.base_domain)
        except CookieError as exc:
            err_console.print(f"[red]{escape(str(exc))}[/red]")
            raise typer.Exit(1) from exc
        console.print(f"[green]Read {len(cookies)} {site} cookie(s) from {from_browser}.[/green]")
    elif file:
        try:
            raw = json.loads(Path(file).read_text())
        except (OSError, ValueError) as exc:
            err_console.print(f"[red]Could not read cookie file: {escape(str(exc))}[/red]")
            raise typer.Exit(1) from exc
        entries = raw.get("cookies", raw) if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            err_console.print('[red]Cookie file must be a JSON list or {"cookies": [...]}.[/red]')
            raise typer.Exit(1)
        cookies = [c for c in (_normalize_cookie(e) for e in entries) if c]
    elif li_at:
        if li_at == "-":  # read from stdin to keep the token out of shell history
            li_at = sys.stdin.readline().strip()
        if not li_at:
            err_console.print("[red]No token provided on stdin.[/red]")
            raise typer.Exit(1)
        seed = _normalize_cookie(
            {
                "name": "li_at",
                "value": li_at,
                "domain": ".linkedin.com",
                "path": "/",
                "httpOnly": True,
                "secure": True,
                "sameSite": "None",
            }
        )
        if seed:
            cookies.append(seed)
        if jsessionid:
            js = _normalize_cookie(
                {
                    "name": "JSESSIONID",
                    "value": jsessionid,
                    "domain": ".www.linkedin.com",
                    "path": "/",
                    "secure": True,
                    "sameSite": "None",
                }
            )
            if js:
                cookies.append(js)
    else:
        err_console.print(
            "[red]Provide --from-browser <name>, --li-at <value>, or --file <path>.[/red]"
        )
        raise typer.Exit(1)

    if not cookies:
        err_console.print("[red]No usable cookies found in the input.[/red]")
        raise typer.Exit(1)
    names = {c.get("name") for c in cookies}
    if spec.required_cookie and spec.required_cookie not in names:
        err_console.print(
            f"[red]No `{spec.required_cookie}` cookie in the import — it would not authenticate. "
            f"Are you logged into {site} in that browser / export?[/red]"
        )
        raise typer.Exit(1)
    if spec.preferred_cookie and spec.preferred_cookie not in names:
        # Not fatal: the search is public, but without this cookie a fresh
        # automation session is more likely to be challenged.
        console.print(
            f"[yellow]No `{spec.preferred_cookie}` cookie in the import — {site} may still "
            f"challenge the scrape. Visiting {site} once in your browser first can "
            f"seed it.[/yellow]"
        )

    save_cookies(spec.cookie_path, cookies)
    console.print(f"[green]Wrote {len(cookies)} cookie(s) to {spec.cookie_path}[/green]")

    if not verify:
        return
    if not spec.feed_verify:
        # A logged-in feed check confirms a LinkedIn session; for a public,
        # Cloudflare-fronted board like Indeed there's no equivalent cheap probe,
        # so leave validation to `search --site <site>`.
        console.print(
            f"[green]Cookies saved. Run `job-applicator search --site {site}` to test.[/green]"
        )
        return

    async def _verify() -> bool:
        from job_applicator.browser.manager import BrowserManager
        from job_applicator.scrapers.linkedin import LinkedInScraper

        async with BrowserManager(settings.browser) as browser:
            scraper = LinkedInScraper(browser, settings)
            return await scraper.has_active_session()

    with console.status("Verifying session by loading your LinkedIn feed once..."):
        ok = asyncio.run(_verify())
    if ok:
        console.print("[green]✓ Session valid — `search` will reuse it headlessly.[/green]")
    else:
        err_console.print(
            "[yellow]Imported, but the feed did not load as logged-in. The li_at value may be "
            "stale — re-copy it from a freshly logged-in browser and try again.[/yellow]"
        )
        raise typer.Exit(1)


@app.command()
def apply(
    ctx: typer.Context,
    site: str = typer.Option("linkedin", "--site", "-s", help="Job board."),
    query: str = typer.Option(
        "", "--query", "-q", help="Search query (empty = apply to saved jobs from the store)."
    ),
    from_ref: str = typer.Option(
        "", "--from", help="Apply to a stored job by id or URL (from `status`); skips searching."
    ),
    limit: int = typer.Option(5, "--limit", "-n", help="Max applications."),
    cover_letter: bool = typer.Option(True, "--cover-letter/--no-cover-letter", help="AI cover."),
    headed: bool = typer.Option(False, "--headed", help="Run browser in headed mode."),
    resume_path: str = typer.Option("", "--resume", help="Path to resume file."),
    style_guide: str = typer.Option("", "--style-guide", help="Style example file(s) to mimic."),
    as_json: bool = typer.Option(False, "--json", help="Output results as JSON."),
    ocr_mode: OCRMode = typer.Option(
        OCRMode.AUTO,
        "--ocr-mode",
        help="OCR mode: auto (fallback), on (always), off (never).",
    ),
    force_ocr: bool = typer.Option(
        False,
        "--force-ocr",
        help="Force OCR; equivalent to --ocr-mode on.",
    ),
    submit: bool = typer.Option(
        False,
        "--submit/--no-submit",
        help="Actually submit applications (default: dry run — fills forms, never submits).",
    ),
    validate: bool = typer.Option(
        False,
        "--validate",
        help="Exit non-zero if a dry run does not reach the Submit button.",
    ),
    verbose: bool = _verbose_option(),
    log_file: str | None = _log_file_option(),
) -> None:
    """Auto-apply to jobs with optional AI cover letters.

    By default this is a DRY RUN: each job's Easy Apply form is opened and
    filled, but never submitted. Pass --submit to send real applications.
    Pass --validate to treat an unreached Submit step as a failure.
    """
    _merge_verbose_ctx(ctx, verbose, log_file)
    if submit:
        err_console.print(
            "[bold red]--submit set: real applications WILL be sent on your account.[/bold red]"
        )
    else:
        err_console.print(
            "[dim]Dry run: forms are filled but NOT submitted. Pass --submit to apply.[/dim]"
        )
    settings = _get_settings(headed)
    if resume_path:
        settings.resume_path = resume_path
    if style_guide:
        settings.style_guide_path = style_guide
    setup_logging(settings.log_level)
    effective_ocr_mode = _resolve_ocr_mode(ocr_mode, force_ocr)

    reporter = _get_reporter(
        ctx=ctx,
        command="apply",
        args={"resume": settings.resume_path, "jobs_file": "", "limit": limit},
        config=_sanitize_config(settings),
    )

    async def _run() -> None:
        from job_applicator.exceptions import DocumentError
        from job_applicator.models import JobBoard
        from job_applicator.scrapers.base import SearchParams

        # Resolve target jobs from the store BEFORE launching a browser when they need
        # no scraping (--from a stored job, or the saved-jobs list); only --query scrapes.
        # The board to apply on comes from the stored job(s), not the --site default, so a
        # stored Indeed job isn't pushed through the LinkedIn applicator.
        store_jobs: list[JobListing] = []
        effective_site = site
        if from_ref:
            stored = _get_jobs_store().get(from_ref)
            if stored is None:
                raise DocumentError(
                    f"No stored job matches --from {from_ref!r}. "
                    "Run `job-applicator status` to list saved jobs."
                )
            store_jobs = [stored.job]
            effective_site = stored.job.board.value
        elif not query:
            # The applicator is board-specific, so a saved-list run targets one board:
            # the requested --site (default linkedin). Filter in SQL (before LIMIT) so we
            # don't miss older jobs of this board hidden behind newer ones of another.
            store_jobs = [s.job for s in _get_jobs_store().list_jobs(board=site, limit=limit)]
            if not store_jobs:
                err_console.print(
                    f"[yellow]No saved {site} jobs to apply to. Run "
                    f"`job-applicator search -q ...` first, pass --from <id>, "
                    f"or use --query to search.[/yellow]"
                )
                raise typer.Exit(1)

        async with _make_browser(effective_site, settings) as browser:
            # Scrape only for a --query without --from; otherwise apply to the store jobs.
            if query and not from_ref:
                scraper = _make_scraper(effective_site, browser, settings)
                params = SearchParams(
                    query=query,
                    max_results=limit,
                    board=JobBoard(effective_site),
                )
                with err_console.status(f"Searching {effective_site}..."):
                    jobs = await scraper.scrape(params)
            else:
                jobs = store_jobs

            if not jobs:
                console.print("[yellow]No jobs found to apply to.[/yellow]")
                return

            # Generate cover letters whenever they are requested and a résumé is
            # available. Dry runs use them as a preview; real submissions use them
            # in the form. LLM calls are local, so the preview is worth the cost.
            cover_letters: dict[str, str] = {}
            if cover_letter and settings.resume_path:
                from job_applicator.documents.cover_letter import CoverLetterGenerator
                from job_applicator.documents.resume import ResumeLoader

                loader = ResumeLoader()
                resume_data = loader.load(settings.resume_path, ocr_mode=effective_ocr_mode)
                ats_result = _run_ats_preflight(resume_data)
                if reporter:
                    reporter.record_ats(
                        score=ats_result.score,
                        checks=ats_result.checks,
                        warnings=ats_result.warnings,
                        suggestions=ats_result.suggestions,
                    )

                user_profile = _load_user_profile(settings)

                generator = CoverLetterGenerator(settings.llm, runtime=_make_runtime(settings))
                style = None
                if settings.style_guide_path:
                    with err_console.status("Analyzing writing style..."):
                        style = await generator.load_style_guide(
                            settings.style_guide_path, ocr_mode=effective_ocr_mode
                        )
                    err_console.print(f"[green]Style loaded: {style.tone}[/green]")
                sem = asyncio.Semaphore(3)

                async def _gen_one(
                    job: JobListing,
                ) -> tuple[str, str] | None:
                    async with sem:
                        try:
                            letter = await generator.generate(
                                job, user_profile, resume_data, style_guide=style
                            )
                            return str(job.url), letter
                        except Exception as exc:
                            msg = f"Cover letter failed for {job.title}: {exc}"
                            err_console.print(f"[yellow]{msg}[/yellow]")
                            return None

                with err_console.status("Generating cover letters (parallel)..."):
                    results_cl = await asyncio.gather(*(_gen_one(j) for j in jobs[:limit]))
                    for entry in results_cl:
                        if entry is not None:
                            url, letter = entry
                            cover_letters[url] = letter

            # Apply to jobs

            applicator = _make_applicator(effective_site, browser, settings)

            await _apply_to_jobs(
                jobs,
                applicator,
                cover_letters,
                settings,
                effective_site,
                limit,
                submit=submit,
                validate=validate,
                as_json=as_json,
                console=err_console if as_json else console,
                reporter=reporter,
            )

    try:
        asyncio.run(_run())
    except JobApplicatorError as exc:
        # Typed, expected failures (no session, anti-bot block, missing resume)
        # — show the message cleanly instead of a raw Python traceback.
        if reporter:
            reporter.record_error(str(exc))
        err_console.print(f"[yellow]⚠ {escape(str(exc))}[/yellow]")
        raise typer.Exit(1) from exc
    except Exception as exc:
        if reporter:
            reporter.record_error(str(exc))
        raise
    finally:
        if reporter:
            log_file = None
            vctx = ctx.obj
            if isinstance(vctx, VerboseContext):
                log_file = vctx.log_file
            reporter.render(err_console, log_file=log_file)


@app.command()
def generate_cover_letter(
    ctx: typer.Context,
    job_title: str = typer.Option(..., "--job-title", "-t", help="Job title."),
    company: str = typer.Option(..., "--company", "-c", help="Company name."),
    job_description: str = typer.Option("", "--description", "-d", help="Job description."),
    as_json: bool = typer.Option(False, "--json", help="Output the cover letter as JSON."),
    resume_path: str = typer.Option("", "--resume", help="Path to resume file."),
    style_guide: str = typer.Option("", "--style-guide", help="Style example file(s) to mimic."),
    headed: bool = typer.Option(False, "--headed", help="Run browser in headed mode."),
    ocr_mode: OCRMode = typer.Option(
        OCRMode.AUTO,
        "--ocr-mode",
        help="OCR mode: auto (fallback), on (always), off (never).",
    ),
    force_ocr: bool = typer.Option(
        False,
        "--force-ocr",
        help="Force OCR; equivalent to --ocr-mode on.",
    ),
    verbose: bool = _verbose_option(),
    log_file: str | None = _log_file_option(),
) -> None:
    """Generate an AI cover letter for a specific job."""
    _merge_verbose_ctx(ctx, verbose, log_file)
    settings = _get_settings(headed)
    if resume_path:
        settings.resume_path = resume_path
    if style_guide:
        settings.style_guide_path = style_guide
    setup_logging(settings.log_level)
    effective_ocr_mode = _resolve_ocr_mode(ocr_mode, force_ocr)

    reporter = _get_reporter(
        ctx=ctx,
        command="generate-cover-letter",
        args={
            "resume": settings.resume_path,
            "job_title": job_title,
            "company": company,
            "style_guide": settings.style_guide_path,
        },
        config=_sanitize_config(settings),
    )

    async def _run() -> None:
        from pydantic import HttpUrl

        from job_applicator.documents.cover_letter import CoverLetterGenerator
        from job_applicator.documents.resume import ResumeLoader
        from job_applicator.models import JobBoard, JobListing

        if not settings.resume_path:
            err_console.print("[red]Resume path required. Use --resume or set RESUME_PATH.[/red]")
            raise typer.Exit(1)

        loader = ResumeLoader()
        resume_data = loader.load(settings.resume_path, ocr_mode=effective_ocr_mode)
        if reporter:
            reporter.record_resume(
                source=settings.resume_path,
                ocr_mode=effective_ocr_mode,
                text_length=len(resume_data.raw_text),
                parsed_name=resume_data.name or "",
                parsed_email=resume_data.email or "",
                parsed_phone=resume_data.phone or "",
                parsed_skills=resume_data.skills,
                parsed_summary_preview=resume_data.summary[:200] if resume_data.summary else "",
            )
        user_profile = _load_user_profile(settings)

        job = JobListing(
            title=job_title,
            company=company,
            description=job_description,
            url=HttpUrl("https://example.com/placeholder"),
            board=JobBoard.LINKEDIN,
        )

        tone_profile = _detect_tone(job)
        err_console.print(  # progress/info → stderr (keeps --json stdout clean)
            f"[dim]Detected tone: {tone_profile.primary} "
            f"(confidence: {tone_profile.confidence:.0%})[/dim]"
        )

        generator = CoverLetterGenerator(settings.llm, runtime=_make_runtime(settings))

        # Load style guide if provided (supports comma-separated paths)
        style = None
        if settings.style_guide_path:
            with err_console.status("Analyzing writing style..."):
                style = await generator.load_style_guide(
                    settings.style_guide_path, ocr_mode=effective_ocr_mode
                )
            err_console.print(f"[green]Style loaded: {style.tone}[/green]")

        if reporter:
            reporter.record_llm_call(
                model=settings.llm.model,
                endpoint=settings.llm.api_base,
                temperature=settings.llm.temperature,
                details={"style_guide": settings.style_guide_path or "default"},
            )

        from job_applicator.documents.tone_detector import ToneDetector

        tone_section = ToneDetector().format_for_prompt(tone_profile)

        with err_console.status("Generating cover letter..."):  # progress → stderr
            letter = await generator.generate(
                job, user_profile, resume_data, style, tone_section=tone_section
            )

        if as_json:
            import json

            sys.stdout.write(
                json.dumps(
                    {"cover_letter": letter, "job_title": job_title, "company": company}, indent=2
                )
                + "\n"
            )
        else:
            console.print("\n[bold]Generated Cover Letter:[/bold]\n")
            console.print(letter)

    try:
        asyncio.run(_run())
    except JobApplicatorError as exc:
        # Typed, expected failures (no session, anti-bot block, missing resume)
        # — show the message cleanly instead of a raw Python traceback.
        if reporter:
            reporter.record_error(str(exc))
        err_console.print(f"[yellow]⚠ {escape(str(exc))}[/yellow]")
        raise typer.Exit(1) from exc
    except Exception as exc:
        if reporter:
            reporter.record_error(str(exc))
        raise
    finally:
        if reporter:
            log_file = ctx.obj.log_file if isinstance(ctx.obj, VerboseContext) else None
            reporter.render(err_console, log_file)


@app.command()
def match(
    ctx: typer.Context,
    resume_path: str = typer.Option("", "--resume", help="Path to resume file."),
    jobs_file: str = typer.Option("", "--jobs-file", help="JSON file with job listings."),
    top_k: int = typer.Option(5, "--top-k", "-k", min=1, help="Number of top matches."),
    min_score: float = typer.Option(
        0.0, "--min-score", min=0.0, max=1.0, help="Minimum match score (0.0-1.0)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Output results as JSON."),
    ocr_mode: OCRMode = typer.Option(
        OCRMode.AUTO,
        "--ocr-mode",
        help="OCR mode: auto (fallback), on (always), off (never).",
    ),
    force_ocr: bool = typer.Option(
        False,
        "--force-ocr",
        help="Force OCR; equivalent to --ocr-mode on.",
    ),
    verbose: bool = _verbose_option(),
    log_file: str | None = _log_file_option(),
) -> None:
    """Match resume to job listings using semantic embeddings."""
    _merge_verbose_ctx(ctx, verbose, log_file)
    settings = _get_settings()
    if resume_path:
        settings.resume_path = resume_path
    setup_logging(settings.log_level)
    effective_ocr_mode = _resolve_ocr_mode(ocr_mode, force_ocr)

    reporter = _get_reporter(
        ctx=ctx,
        command="match",
        args={"resume": settings.resume_path, "jobs_file": jobs_file, "top_k": top_k},
        config=_sanitize_config(settings),
    )

    async def _run() -> None:
        from job_applicator.documents.resume import ResumeLoader
        from job_applicator.embeddings.matching import JobMatcher

        if not settings.resume_path:
            err_console.print("[red]Resume path required. Use --resume.[/red]")
            raise typer.Exit(1)

        # Load resume
        loader = ResumeLoader()
        resume_data = loader.load(settings.resume_path, ocr_mode=effective_ocr_mode)
        if reporter:
            reporter.record_resume(
                source=settings.resume_path,
                ocr_mode=effective_ocr_mode,
                text_length=len(resume_data.raw_text),
                parsed_name=resume_data.name or "",
                parsed_email=resume_data.email or "",
                parsed_phone=resume_data.phone or "",
                parsed_skills=resume_data.skills,
                parsed_summary_preview=resume_data.summary[:200] if resume_data.summary else "",
            )
        if not as_json:
            console.print(f"[green]Loaded resume: {resume_data.name}[/green]")
            ats_result = _run_ats_preflight(resume_data)
        else:
            from job_applicator.documents.ats_checker import ATSChecker

            ats_result = ATSChecker().check(resume_data)
        if reporter:
            reporter.record_ats(
                score=ats_result.score,
                checks=ats_result.checks,
                warnings=ats_result.warnings,
                suggestions=ats_result.suggestions,
            )

        # Load jobs
        jobs: list[JobListing] = []
        if jobs_file:
            jobs = _load_jobs_file(jobs_file)
        else:
            err_console.print(
                "[red]No jobs to match. Provide --jobs-file <path> "
                "(a JSON array of job listings).[/red]"
            )
            raise typer.Exit(1)

        if not as_json:
            console.print(f"[green]Loaded {len(jobs)} jobs[/green]")

        # Match
        with console.status("Computing embeddings and matching..."):
            matcher = JobMatcher(settings.embedding)
            matches = matcher.rank_jobs(resume_data, jobs, top_k=top_k)

        # Filter by min score
        if min_score > 0:
            matches = [m for m in matches if m.score >= min_score]

        # Persist scored jobs so they flow into tailor/apply and `status`.
        # Best-effort: a store hiccup must not sink the computed match results.
        try:
            store = _get_jobs_store()
            for m in matches:
                store.upsert_match(m)
        except JobApplicatorError as exc:
            err_console.print(
                f"[yellow]⚠ Could not save matches to the store: {escape(str(exc))}[/yellow]"
            )

        if reporter and matches:
            reporter.record_match(
                embedding_model=settings.embedding.model_name,
                device=settings.embedding.device,
                load_time_ms=0,
                results=[
                    {
                        "rank": i + 1,
                        "title": m.job.title,
                        "company": m.job.company,
                        "score": round(m.score, 4),
                        "semantic_score": round(m.semantic_score, 4),
                        "skill_score": round(m.skill_score, 4),
                        "matched_skills": m.matched_skills,
                        "missing_skills": m.missing_skills,
                    }
                    for i, m in enumerate(matches)
                ],
            )

        # JSON output
        if as_json:
            import json

            output = [
                {
                    "rank": i + 1,
                    "score": round(m.score, 4),
                    "title": m.job.title,
                    "company": m.job.company,
                    "url": str(m.job.url),
                    "matched_skills": m.matched_skills,
                    "missing_skills": m.missing_skills,
                    "summary": m.summary,
                }
                for i, m in enumerate(matches)
            ]
            sys.stdout.write(json.dumps(output, indent=2) + "\n")
            return

        # Display results
        table = Table(title=f"Top {len(matches)} Job Matches")
        table.add_column("Rank", style="dim")
        table.add_column("Score", style="cyan")
        table.add_column("Job", style="green")
        table.add_column("Company")
        table.add_column("Matched Skills")
        table.add_column("Missing Skills")

        for i, match in enumerate(matches, 1):
            if match.score >= 0.7:
                score_style = "green"
            elif match.score >= 0.5:
                score_style = "yellow"
            else:
                score_style = "red"
            table.add_row(
                str(i),
                f"[{score_style}]{match.score:.0%}[/{score_style}]",
                match.job.title,
                match.job.company,
                ", ".join(match.matched_skills[:3]) or "-",
                ", ".join(match.missing_skills[:3]) or "-",
            )

        console.print(table)

    try:
        asyncio.run(_run())
    except JobApplicatorError as exc:
        # Typed, expected failures (no session, anti-bot block, missing resume)
        # — show the message cleanly instead of a raw Python traceback.
        if reporter:
            reporter.record_error(str(exc))
        err_console.print(f"[yellow]⚠ {escape(str(exc))}[/yellow]")
        raise typer.Exit(1) from exc
    except Exception as exc:
        if reporter:
            reporter.record_error(str(exc))
        raise
    finally:
        if reporter:
            log_file = ctx.obj.log_file if isinstance(ctx.obj, VerboseContext) else None
            reporter.render(err_console, log_file)


@app.command()
def status(
    ctx: typer.Context,
    as_json: bool = typer.Option(False, "--json", help="Output the funnel as JSON."),
    limit: int = typer.Option(20, "--limit", "-n", min=1, help="Max recent jobs to show."),
    verbose: bool = _verbose_option(),
    log_file: str | None = _log_file_option(),
) -> None:
    """Show your job funnel: counts by stage and the most recent jobs.

    Composes the funnel head (found/matched/tailored, from the job store) with the
    applied tail (submitted applications, from the application-state store), keyed by
    URL so each job shows its *furthest* stage (no double counting). Offline and
    account-safe — reads local state only, never the network or your account.
    """
    _merge_verbose_ctx(ctx, verbose, log_file)
    from datetime import UTC

    from job_applicator.models import ApplicationStatus, FunnelStatus
    from job_applicator.state import ApplicationState

    # Read both stores up front so a DB hiccup (e.g. a concurrent apply/batch holding
    # the write lock past the 5s timeout) surfaces as a clean message, not a raw
    # traceback — matching the other commands' typed-error handling.
    try:
        store = _get_jobs_store()
        app_state = ApplicationState()
        stored_jobs = store.list_jobs(limit=10_000)
        recent_apps = app_state.list_recent(limit=10_000)
    except JobApplicatorError as exc:
        err_console.print(f"[yellow]⚠ {escape(str(exc))}[/yellow]")
        raise typer.Exit(1) from exc

    # URL-keyed view: stage from the store, overridden to APPLIED when the
    # application-state store has a SUBMITTED record (its authority for "applied").
    view: dict[str, dict[str, Any]] = {}
    for s in stored_jobs:
        url = str(s.job.url)
        view[url] = {
            "title": s.job.title,
            "company": s.job.company,
            "board": s.job.board.value,
            "stage": s.funnel_status.value,
            "score": s.match_score,
            "url": url,
            "when": s.updated_at,
        }
    for appres in recent_apps:
        if appres.status != ApplicationStatus.SUBMITTED:
            continue
        url = str(appres.job.url)
        row = view.get(
            url,
            {
                "title": appres.job.title,
                "company": appres.job.company,
                "board": appres.job.board.value,
                "score": None,
                "url": url,
            },
        )
        row["stage"] = FunnelStatus.APPLIED.value
        row["when"] = appres.timestamp
        view[url] = row

    def _when_key(row: dict[str, Any]) -> Any:
        # Normalize naive→aware (assume UTC) so the two stores' timestamps compare.
        when = row["when"]
        return when.replace(tzinfo=UTC) if when.tzinfo is None else when

    rows = sorted(view.values(), key=_when_key, reverse=True)

    order = [s.value for s in FunnelStatus]
    counts = dict.fromkeys(order, 0)
    for r in rows:
        counts[r["stage"]] = counts.get(r["stage"], 0) + 1

    if as_json:
        import json

        payload = {
            "counts": counts,
            "total": len(rows),
            "recent": [
                {
                    "title": r["title"],
                    "company": r["company"],
                    "board": r["board"],
                    "stage": r["stage"],
                    "score": r["score"],
                    "url": r["url"],
                    "when": r["when"].isoformat(),
                }
                for r in rows[:limit]
            ],
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return

    summary = " · ".join(f"{counts[st]} {st.replace('_', ' ')}" for st in order)
    console.print(Panel(summary, title="Funnel", border_style="cyan"))

    if not rows:
        console.print(
            "[dim]No jobs yet. Run `job-applicator search -q ...` to discover jobs.[/dim]"
        )
        return

    table = Table(title=f"Recent jobs (showing {min(limit, len(rows))} of {len(rows)})")
    table.add_column("Stage", style="cyan")
    table.add_column("Score")
    table.add_column("Title", style="green")
    table.add_column("Company")
    table.add_column("Board", style="dim")
    for r in rows[:limit]:
        score = f"{r['score']:.0%}" if r["score"] is not None else "—"
        table.add_row(r["stage"].replace("_", " "), score, r["title"], r["company"], r["board"])
    console.print(table)


@app.command()
def tui() -> None:
    """Open the full-screen terminal UI — a navigable home over your job funnel.

    Browse the search → tailor → cover-letter → apply pipeline and act on the selected job
    from inside the app. Launching, navigating, and filtering touch only local state; the
    search and apply actions touch your real account only behind an explicit confirm. Same
    as running `job-applicator` with no command in a terminal.
    """
    if not _tui_tty_ok():
        err_console.print("[yellow]The TUI needs an interactive terminal (a TTY).[/yellow]")
        raise typer.Exit(1)
    _launch_tui()


def _resume_tailored_resume(
    batch_state: BatchState, run_id: str, job_url: str
) -> tuple[TailoredResume, str, str] | None:
    """Mid-job resume: if a job is persisted TAILORED with a readable meta.json,
    reconstruct its TailoredResume so the batch can skip re-tailoring and go straight
    to the cover letter. Returns (tailored, resume_path, meta_path), or None when the
    job isn't reusably tailored or the artifact is missing/corrupt (→ caller re-tailors).
    """
    from job_applicator.batch_state import BatchJobStatus
    from job_applicator.models import TailoredResume

    persisted = batch_state.get_job(run_id, job_url)
    if not persisted or persisted[0] != BatchJobStatus.TAILORED or not persisted[1]:
        return None
    resume_path = persisted[1]
    meta_path = str(Path(resume_path).with_suffix(".meta.json"))
    if not Path(meta_path).exists():
        return None
    try:
        tailored = TailoredResume.model_validate_json(Path(meta_path).read_text())
    except Exception:
        return None
    return tailored, resume_path, meta_path


@app.command()
def batch(
    ctx: typer.Context,
    resume_path: str = typer.Option("", "--resume", help="Path to resume file."),
    jobs_file: str = typer.Option("", "--jobs-file", help="JSON file with job listings."),
    query: str = typer.Option(
        "", "--query", "-q", help="Search query (alternative to --jobs-file)."
    ),
    site: str = typer.Option("linkedin", "--site", "-s", help="Job board for --query."),
    top_k: int = typer.Option(5, "--top-k", "-k", min=1, help="Max jobs to tailor."),
    min_score: float = typer.Option(
        0.0, "--min-score", min=0.0, max=1.0, help="Skip jobs below this score."
    ),
    cover_letter: bool = typer.Option(
        True, "--cover-letter/--no-cover-letter", help="Generate cover letters."
    ),
    style_guide: str = typer.Option("", "--style-guide", help="Style example file(s) to mimic."),
    headed: bool = typer.Option(False, "--headed", help="Run browser in headed mode."),
    as_json: bool = typer.Option(False, "--json", help="Output results as JSON."),
    ocr_mode: OCRMode = typer.Option(
        OCRMode.AUTO,
        "--ocr-mode",
        help="OCR mode: auto (fallback), on (always), off (never).",
    ),
    force_ocr: bool = typer.Option(
        False,
        "--force-ocr",
        help="Force OCR; equivalent to --ocr-mode on.",
    ),
    run_id: str = typer.Option("", "--run-id", help="Unique ID for this batch run."),
    resume_run: bool = typer.Option(
        False,
        "--resume-run",
        help="Resume an existing incomplete batch run with matching parameters.",
    ),
    verbose: bool = _verbose_option(),
    log_file: str | None = _log_file_option(),
) -> None:
    """Batch tailor resumes (and optionally cover letters) for multiple jobs."""
    _merge_verbose_ctx(ctx, verbose, log_file)
    settings = _get_settings(headed)
    if resume_path:
        settings.resume_path = resume_path
    if style_guide:
        settings.style_guide_path = style_guide
    setup_logging(settings.log_level)
    effective_ocr_mode = _resolve_ocr_mode(ocr_mode, force_ocr)

    reporter = _get_reporter(
        ctx=ctx,
        command="batch",
        args={
            "resume_path": settings.resume_path,
            "jobs_file": jobs_file,
            "query": query,
            "top_k": top_k,
            "cover_letter": cover_letter,
            "run_id": run_id or "auto",
            "resume_run": resume_run,
        },
        config=_sanitize_config(settings),
    )
    written_paths: list[str] = []

    async def _run() -> None:
        import json
        from datetime import datetime as dt
        from pathlib import Path

        from job_applicator.batch_state import BatchJobStatus, BatchState
        from job_applicator.documents.resume import ResumeLoader
        from job_applicator.documents.resume_tailor import ResumeTailor
        from job_applicator.embeddings.matching import JobMatcher, MatchResult
        from job_applicator.models import JobBoard, TailoringReport

        if not settings.resume_path:
            err_console.print("[red]Resume path required. Use --resume.[/red]")
            raise typer.Exit(1)

        loader = ResumeLoader()
        resume_data = loader.load(settings.resume_path, ocr_mode=effective_ocr_mode)
        if reporter:
            reporter.record_resume(
                source=settings.resume_path,
                ocr_mode=effective_ocr_mode,
                text_length=len(resume_data.raw_text),
                parsed_name=resume_data.name or "",
                parsed_email=resume_data.email or "",
                parsed_phone=resume_data.phone or "",
                parsed_skills=resume_data.skills,
                parsed_summary_preview=resume_data.summary[:200] if resume_data.summary else "",
            )
        if not as_json:
            console.print(f"[green]Loaded resume: {resume_data.name}[/green]")
        ats_result = _run_ats_preflight(resume_data)
        if reporter:
            reporter.record_ats(
                score=ats_result.score,
                checks=ats_result.checks,
                warnings=ats_result.warnings,
                suggestions=ats_result.suggestions,
            )

        jobs: list[JobListing] = []
        if jobs_file:
            jobs = _load_jobs_file(jobs_file)
        elif query:
            from job_applicator.scrapers.base import SearchParams

            async with _make_browser(site, settings) as browser:
                scraper = _make_scraper(site, browser, settings)
                params = SearchParams(query=query, max_results=top_k * 2, board=JobBoard(site))
                with console.status(f"Searching {site}..."):
                    jobs = await scraper.scrape(params)
        else:
            err_console.print("[red]Provide --jobs-file or --query.[/red]")
            raise typer.Exit(1)

        if not jobs:
            console.print("[yellow]No jobs found.[/yellow]")
            return

        if not as_json:
            console.print(f"[green]Loaded {len(jobs)} jobs[/green]")

        matcher = JobMatcher(settings.embedding)
        with console.status("Computing match scores..."):
            matches = matcher.rank_jobs(resume_data, jobs, top_k=top_k)

        if reporter and matches:
            reporter.record_match(
                embedding_model=settings.embedding.model_name,
                device=settings.embedding.device,
                load_time_ms=0,
                results=[
                    {
                        "rank": i + 1,
                        "title": m.job.title,
                        "company": m.job.company,
                        "score": round(m.score, 4),
                        "semantic_score": round(m.semantic_score, 4),
                        "skill_score": round(m.skill_score, 4),
                        "matched_skills": m.matched_skills,
                        "missing_skills": m.missing_skills,
                    }
                    for i, m in enumerate(matches)
                ],
            )

        if min_score > 0:
            before = len(matches)
            matches = [m for m in matches if m.score >= min_score]
            skipped = before - len(matches)
            if skipped and not as_json:
                console.print(
                    f"[yellow]Skipped {skipped} jobs below {min_score:.0%} threshold[/yellow]"
                )

        if not matches:
            console.print("[yellow]No jobs above minimum score threshold.[/yellow]")
            return

        batch_state = BatchState()
        run_spec = BatchRunSpec(
            site=site,
            query=query or None,
            jobs_file=jobs_file or None,
            resume_path=settings.resume_path,
            top_k=top_k,
            min_score=min_score,
            cover_letter=cover_letter,
        )
        # One spec is the single source for the run id AND the resume-match key.
        effective_run_id = run_id or run_spec.run_id()

        resuming = False
        if resume_run:
            existing = batch_state.find_existing_run(run_spec)
            if existing:
                effective_run_id = existing
                resuming = True
                if not as_json:
                    console.print(f"[cyan]Resuming batch run {effective_run_id}...[/cyan]")
            else:
                if not as_json:
                    console.print(
                        "[yellow]No incomplete batch run found; starting a new run.[/yellow]"
                    )

        if not resuming:
            batch_state.start_run(run_spec, run_id=effective_run_id)
        completed_urls = set(batch_state.list_completed_jobs(effective_run_id))

        pending_matches = [m for m in matches if str(m.job.url) not in completed_urls]
        skipped_already = len(matches) - len(pending_matches)
        if skipped_already and not as_json:
            console.print(
                f"[cyan]Skipping {skipped_already} already-completed jobs from previous run.[/cyan]"
            )

        if not pending_matches:
            batch_state.complete_run(effective_run_id)
            if not as_json:
                console.print("[green]All jobs already completed.[/green]")
            return

        matches = pending_matches
        if not as_json:
            n = len(matches)
            console.print(f"[cyan]Tailoring {n} job{'s' if n != 1 else ''}...[/cyan]")

        # One breaker shared across cover-letter generation + résumé tailoring for
        # this whole batch run (every job goes through the same circuit breaker).
        runtime = _make_runtime(settings)
        style = None
        cl_generator = None
        if settings.style_guide_path or cover_letter:
            from job_applicator.documents.cover_letter import CoverLetterGenerator

            cl_generator = CoverLetterGenerator(settings.llm, runtime=runtime)
            if settings.style_guide_path:
                with err_console.status("Loading style guide..."):
                    style = await cl_generator.load_style_guide(
                        settings.style_guide_path, ocr_mode=effective_ocr_mode
                    )
                err_console.print(f"[green]Style loaded: {style.tone}[/green]")

        tailor_engine = ResumeTailor(settings.llm, runtime=runtime)
        user_profile = _load_user_profile(settings)
        sem = asyncio.Semaphore(3)
        timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        output_dir = settings.output_dir
        await asyncio.to_thread(settings.ensure_output_dir)
        tailoring_scores: list[tuple[float, float]] = []
        batch_reports: list[TailoringReport] = []

        async def _process_one(match_result: MatchResult) -> dict[str, object]:
            job = match_result.job
            safe_company = "".join(c if c.isalnum() or c in "-_" else "_" for c in job.company)[:30]
            safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in job.title)[:30]
            user_instructions = ""

            async with sem:
                result: dict[str, object] = {
                    "title": job.title,
                    "company": job.company,
                    "url": str(job.url),
                }

                # Mid-job resume: reuse a persisted TAILORED résumé instead of
                # re-tailoring (a TAILORED job is re-processed on resume so its cover
                # letter gets generated). Missing/corrupt artifact → re-tailor below.
                reused = await asyncio.to_thread(
                    _resume_tailored_resume, batch_state, effective_run_id, str(job.url)
                )
                if reused is not None:
                    tailored, resume_path_out, meta_path = reused
                    result["match_score"] = round(tailored.match_score, 4)
                    result["semantic_score"] = round(tailored.semantic_score, 4)
                    result["skill_score"] = round(tailored.skill_score, 4)
                    result["resume_path"] = resume_path_out
                    result["tailored"] = True
                    result["resumed"] = True
                else:
                    try:
                        tone_profile = _detect_tone(job)
                        if reporter:
                            reporter.record_llm_call(
                                model=settings.llm.model,
                                endpoint=settings.llm.api_base,
                                temperature=settings.llm.temperature,
                                details={
                                    "job_title": job.title,
                                    "company": job.company,
                                    "type": "tailor",
                                },
                            )
                        tailored = await tailor_engine.tailor(
                            resume=resume_data,
                            job=job,
                            user_instructions=user_instructions,
                            style_guide=style,
                            tone_profile=tone_profile,
                            matcher=matcher,
                        )
                        result["match_score"] = round(tailored.match_score, 4)
                        result["semantic_score"] = round(tailored.semantic_score, 4)
                        result["skill_score"] = round(tailored.skill_score, 4)
                        post_ats = _run_ats_post_tailor(
                            resume_data.raw_text, tailored.tailored_text
                        )
                        ats_score = post_ats.score if post_ats else 1.0
                        tailoring_scores.append((tailored.match_score, ats_score))
                        batch_reports.append(
                            TailoringReport(
                                job_title=job.title,
                                company=job.company,
                                tone=tone_profile.primary,
                                tone_confidence=tone_profile.confidence,
                                attempts=1,
                                ats_before=ats_result.score if ats_result else 0.0,
                                ats_after=ats_score,
                                changes_summary=tailored.changes_summary or "",
                            )
                        )

                        resume_filename = f"tailored_{safe_company}_{safe_title}_{timestamp}.txt"
                        resume_path_out = str(Path(output_dir) / resume_filename)
                        await asyncio.to_thread(
                            Path(resume_path_out).write_text, tailored.tailored_text
                        )
                        written_paths.append(resume_path_out)

                        meta_filename = f"{resume_filename.rsplit('.', 1)[0]}.meta.json"
                        meta_path = str(Path(output_dir) / meta_filename)
                        tailored.output_path = resume_path_out
                        await asyncio.to_thread(
                            Path(meta_path).write_text, tailored.model_dump_json(indent=2)
                        )
                        written_paths.append(meta_path)
                        result["resume_path"] = resume_path_out
                        result["tailored"] = True
                        batch_state.record_job(
                            effective_run_id,
                            job,
                            BatchJobStatus.TAILORED,
                            resume_path=resume_path_out,
                        )
                    except Exception as exc:
                        result["tailored"] = False
                        result["error"] = str(exc)
                        batch_state.record_job(
                            effective_run_id,
                            job,
                            BatchJobStatus.FAILED,
                            error_message=str(exc),
                        )
                        return result

                if cl_generator is not None and cover_letter:
                    try:
                        if reporter:
                            reporter.record_llm_call(
                                model=settings.llm.model,
                                endpoint=settings.llm.api_base,
                                temperature=settings.llm.temperature,
                                details={
                                    "job_title": job.title,
                                    "company": job.company,
                                    "type": "cover_letter",
                                },
                            )
                        letter = await cl_generator.generate(
                            job,
                            user_profile,
                            resume_data,
                            style_guide=style,
                            tailored_resume_text=tailored.tailored_text,
                        )
                        cl_filename = f"cover_letter_{safe_company}_{safe_title}_{timestamp}.txt"
                        cl_path = str(Path(output_dir) / cl_filename)
                        await asyncio.to_thread(Path(cl_path).write_text, letter)
                        written_paths.append(cl_path)
                        tailored.cover_letter_path = cl_path
                        result["cover_letter_path"] = cl_path
                        result["cover_letter"] = True
                        # Re-write meta.json with cover_letter_path
                        await asyncio.to_thread(
                            Path(meta_path).write_text, tailored.model_dump_json(indent=2)
                        )
                        batch_state.record_job(
                            effective_run_id,
                            job,
                            BatchJobStatus.COMPLETED,
                            resume_path=resume_path_out,
                            cover_letter_path=cl_path,
                        )
                    except Exception as exc:
                        result["cover_letter"] = False
                        result["cl_error"] = str(exc)
                        batch_state.record_job(
                            effective_run_id,
                            job,
                            BatchJobStatus.FAILED,
                            resume_path=resume_path_out,
                            error_message=str(exc),
                        )
                else:
                    batch_state.record_job(
                        effective_run_id,
                        job,
                        BatchJobStatus.COMPLETED,
                        resume_path=resume_path_out,
                    )

                return result

        with console.status("Processing jobs in parallel..."):
            batch_results = await asyncio.gather(*(_process_one(m) for m in matches))

        batch_state.complete_run(effective_run_id)

        if reporter and batch_reports:
            reporter.record_batch_tailoring(batch_reports)

        summary = {
            "timestamp": timestamp,
            "resume": settings.resume_path,
            "total_jobs": len(jobs),
            "matched": len(matches),
            "results": list(batch_results),
        }
        summary_path = str(Path(output_dir) / f"batch_summary_{timestamp}.json")
        await asyncio.to_thread(Path(summary_path).write_text, json.dumps(summary, indent=2))
        written_paths.append(summary_path)

        if reporter:
            reporter.record_io(
                files_written=written_paths,
                batch_summary_path=summary_path,
            )

        if as_json:
            sys.stdout.write(json.dumps(summary, indent=2) + "\n")
        else:
            table = Table(title="Batch Results")
            table.add_column("Job", style="cyan")
            table.add_column("Company", style="green")
            table.add_column("Score")
            table.add_column("Tailored")
            table.add_column("Cover Letter")
            table.add_column("Notes")

            for r in batch_results:
                score_raw = r.get("match_score", 0)
                score_val = float(score_raw) if score_raw else 0.0  # type: ignore[arg-type]
                score_style = (
                    "green" if score_val >= 0.7 else "yellow" if score_val >= 0.5 else "red"
                )
                score_str = (
                    f"[{score_style}]{score_val:.0%}[/{score_style}]"
                    if r.get("tailored")
                    else "[dim]N/A[/dim]"
                )
                table.add_row(
                    str(r.get("title", "")),
                    str(r.get("company", "")),
                    score_str,
                    "✓" if r.get("tailored") else "✗",
                    "✓" if r.get("cover_letter") else ("✗" if cover_letter else "-"),
                    str(r.get("error", r.get("cl_error", ""))),
                )

            console.print(table)
            tailored_ok = sum(1 for r in batch_results if r.get("tailored"))
            cl_ok = sum(1 for r in batch_results if r.get("cover_letter"))
            console.print(
                f"\n[green]{tailored_ok}[/green] tailored, [green]{cl_ok}[/green] cover letters"
            )
            console.print(f"Summary: {summary_path}")

    try:
        asyncio.run(_run())
    except JobApplicatorError as exc:
        # Typed, expected failures (no session, anti-bot block, missing resume)
        # — show the message cleanly instead of a raw Python traceback.
        if reporter:
            reporter.record_error(str(exc))
        err_console.print(f"[yellow]⚠ {escape(str(exc))}[/yellow]")
        raise typer.Exit(1) from exc
    except Exception as exc:
        if reporter:
            reporter.record_error(str(exc))
        raise
    finally:
        if reporter:
            log_file = ctx.obj.log_file if isinstance(ctx.obj, VerboseContext) else None
            reporter.render(err_console, log_file)


@app.command()
def tailor(
    ctx: typer.Context,
    resume_path: str = typer.Option("", "--resume", help="Path to resume file."),
    from_ref: str = typer.Option(
        "", "--from", help="Tailor a stored job by id or URL (from `status`) instead of -t/-c/-d."
    ),
    job_title: str = typer.Option("", "--job-title", "-t", help="Job title (or use --from)."),
    company: str = typer.Option("", "--company", "-c", help="Company name (or use --from)."),
    job_description: str = typer.Option("", "--description", "-d", help="Job description."),
    job_url: str = typer.Option("", "--url", help="Job posting URL."),
    requirements: str = typer.Option(
        "", "--requirements", "-r", help="Comma-separated requirements."
    ),
    location: str = typer.Option("", "--location", "-l", help="Job location."),
    style_guide: str = typer.Option("", "--style-guide", help="Style example file(s) to mimic."),
    headed: bool = typer.Option(False, "--headed", help="Run browser in headed mode."),
    min_score: float = typer.Option(
        0.0, "--min-score", min=0.0, max=1.0, help="Abort if below this match threshold (0.0-1.0)."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip interactive prompts (auto-answer yes)."
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Emit the tailored résumé as JSON (implies --yes / non-interactive)."
    ),
    ocr_mode: OCRMode = typer.Option(
        OCRMode.AUTO,
        "--ocr-mode",
        help="OCR mode: auto (fallback), on (always), off (never).",
    ),
    force_ocr: bool = typer.Option(
        False,
        "--force-ocr",
        help="Force OCR; equivalent to --ocr-mode on.",
    ),
    verbose: bool = _verbose_option(),
    log_file: str | None = _log_file_option(),
) -> None:
    """Tailor a resume for a specific job with interactive preview."""
    _merge_verbose_ctx(ctx, verbose, log_file)
    yes = yes or as_json  # --json is non-interactive: auto-accept + reserve stdout for the JSON
    real_stdout = sys.stdout  # captured before _run redirects stdout→stderr (the JSON lands here)
    settings = _get_settings(headed)
    if resume_path:
        settings.resume_path = resume_path
    if style_guide:
        settings.style_guide_path = style_guide
    setup_logging(settings.log_level)
    effective_ocr_mode = _resolve_ocr_mode(ocr_mode, force_ocr)

    reporter = _get_reporter(
        ctx=ctx,
        command="tailor",
        args={
            "resume": settings.resume_path,
            "job": job_description,
            "min_score": min_score,
            "interactive": not yes,
        },
        config=_sanitize_config(settings),
    )

    async def _run() -> None:
        from pydantic import HttpUrl

        from job_applicator.documents.resume import ResumeLoader
        from job_applicator.documents.resume_tailor import ResumeTailor
        from job_applicator.exceptions import DocumentError
        from job_applicator.models import JobBoard, JobListing

        if not settings.resume_path:
            err_console.print("[red]Resume path required. Use --resume.[/red]")
            raise typer.Exit(1)

        loader = ResumeLoader()
        resume_data = loader.load(settings.resume_path, ocr_mode=effective_ocr_mode)
        if reporter:
            reporter.record_resume(
                source=settings.resume_path,
                ocr_mode=effective_ocr_mode,
                text_length=len(resume_data.raw_text),
                parsed_name=resume_data.name or "",
                parsed_email=resume_data.email or "",
                parsed_phone=resume_data.phone or "",
                parsed_skills=resume_data.skills,
                parsed_summary_preview=resume_data.summary[:200] if resume_data.summary else "",
            )
        console.print(f"[green]Loaded resume: {resume_data.name}[/green]")
        ats_result = _run_ats_preflight(resume_data)
        if reporter:
            reporter.record_ats(
                score=ats_result.score,
                checks=ats_result.checks,
                warnings=ats_result.warnings,
                suggestions=ats_result.suggestions,
            )

        if from_ref:
            stored = _get_jobs_store().get(from_ref)
            if stored is None:
                raise DocumentError(
                    f"No stored job matches --from {from_ref!r}. "
                    "Run `job-applicator status` to list saved jobs."
                )
            job = stored.job
        else:
            if not job_title or not company:
                err_console.print(
                    "[red]Provide --job-title and --company, or --from <id|url>.[/red]"
                )
                raise typer.Exit(1)
            req_list = (
                [r.strip() for r in requirements.split(",") if r.strip()] if requirements else []
            )
            url = HttpUrl(job_url) if job_url else HttpUrl("https://example.com/placeholder")
            job = JobListing(
                title=job_title,
                company=company,
                description=job_description,
                url=url,
                requirements=req_list,
                location=location,
                board=JobBoard.INDEED,
            )

        tone_profile = _detect_tone(job)
        console.print(
            f"[dim]Detected tone: {tone_profile.primary} "
            f"(confidence: {tone_profile.confidence:.0%})[/dim]"
        )

        # One breaker shared across style-loading + résumé tailoring for this command.
        runtime = _make_runtime(settings)
        style = None
        if settings.style_guide_path:
            from job_applicator.documents.cover_letter import CoverLetterGenerator

            generator = CoverLetterGenerator(settings.llm, runtime=runtime)
            with err_console.status("Analyzing writing style..."):
                style = await generator.load_style_guide(
                    settings.style_guide_path, ocr_mode=effective_ocr_mode
                )
            err_console.print(f"[green]Style loaded: {style.tone}[/green]")

        tailor_engine = ResumeTailor(settings.llm, runtime=runtime)
        user_instructions = ""

        from job_applicator.models import TailorSession

        session = TailorSession(
            original_text=resume_data.raw_text,
            job_title=job.title,
            job_company=job.company,
        )

        # Pre-ingestion date audit
        from job_applicator.documents.resume_tailor import ResumeDateValidator

        validator = ResumeDateValidator()
        audit = validator.audit(resume_data)

        console.print("\n[bold]📋 CV Date Audit[/bold]")
        audit_table = Table(title="Date Analysis", show_lines=True)
        audit_table.add_column("Section", style="dim")
        audit_table.add_column("Entry", style="bold")
        audit_table.add_column("Start")
        audit_table.add_column("End")
        for entry in audit.entries:
            audit_table.add_row(
                entry.section,
                entry.label,
                entry.start,
                entry.end,
            )
        console.print(audit_table)

        console.print(f"\n[dim]Date range: {audit.earliest_date} → {audit.latest_date}[/dim]")

        if audit.warnings:
            console.print("\n[bold yellow]⚠ Warnings:[/bold yellow]")
            for w in audit.warnings:
                console.print(f"  [yellow]• {w}[/yellow]")

        if audit.staleness_issues:
            console.print("\n[bold red]⚠ Staleness Warnings:[/bold red]")
            for s in audit.staleness_issues:
                console.print(f"  [red]• {s}[/red]")

        if audit.ordering_issues:
            console.print("\n[bold red]⚠ Ordering Issues:[/bold red]")
            for o in audit.ordering_issues:
                console.print(f"  [red]• {o}[/red]")

        # Confirm gate at TOP LEVEL — fires on staleness OR ordering issues (not nested under
        # `if ordering_issues`), so a stale-but-correctly-ordered CV is gated too; the else
        # (coherent) now runs for a genuinely clean CV instead of being dead code.
        if audit.is_stale or audit.ordering_issues:
            console.print(
                "\n[bold yellow]This CV may be outdated or have ordering "
                "issues. Please verify your CV is up to date before "
                "proceeding.[/bold yellow]"
            )
            if yes:
                console.print("[dim]--yes flag set, proceeding automatically.[/dim]")
            else:
                confirm = (
                    console.input("\n[bold cyan]Proceed anyway? (y/n): [/bold cyan]")
                    .strip()
                    .lower()
                )
                if confirm != "y":
                    console.print("[yellow]Aborted. Please update your CV.[/yellow]")
                    raise typer.Exit(0)
        else:
            console.print("[green]✓ Dates look coherent and current.[/green]")

        # Pre-tailor match score check
        pre_match_score = None
        if min_score > 0:
            from job_applicator.embeddings.matching import JobMatcher

            with console.status("Computing match score..."):
                matcher = JobMatcher(settings.embedding)
                pre_match = matcher.match_resume_to_job(resume_data, job)
            pre_match_score = pre_match.score
            console.print(
                f"[cyan]Match score: {pre_match.score:.0%} (threshold: {min_score:.0%})[/cyan]"
            )
            if pre_match.score < min_score:
                console.print(
                    f"[red]Match score {pre_match.score:.0%} is below threshold "
                    f"{min_score:.0%}. Aborting.[/red]"
                )
                # Under --json the abort message is on stderr; signal "no result" via a non-zero
                # exit so a `tailor --json | jq` pipeline isn't a silent empty-stdout success.
                raise typer.Exit(1 if as_json else 0)

        try:
            with console.status("Tailoring resume..."):
                result = await tailor_engine.tailor(
                    resume_data, job, user_instructions, style, tone_profile
                )
            session.add_attempt(result)

            if reporter:
                reporter.record_llm_call(
                    model=settings.llm.model,
                    endpoint=settings.llm.api_base,
                    temperature=settings.llm.temperature,
                    details={"job_title": job.title, "interactive": True},
                )
            if reporter and result:
                ats_before = ats_result.score if ats_result else 0.0
                post_ats = _run_ats_post_tailor(resume_data.raw_text, result.tailored_text)
                ats_after = post_ats.score if post_ats else ats_before
                reporter.record_tailoring(
                    job_title=job.title,
                    company=job.company,
                    tone=tone_profile.primary if tone_profile else "",
                    tone_confidence=tone_profile.confidence if tone_profile else 0.0,
                    pre_match_score=pre_match_score,
                    attempts=1,
                    ats_before=ats_before,
                    ats_after=ats_after,
                    hallucination_actions=[],
                    changes_summary=result.changes_summary or "",
                )
        except Exception as exc:
            err_console.print(f"[red]LLM error: {escape(str(exc))}[/red]")
            err_console.print("[yellow]Could not generate tailored resume.[/yellow]")
            if reporter:
                reporter.record_error(str(exc))
            raise typer.Exit(1) from exc

        await _tailor_workflow(
            console,
            settings,
            job,
            resume_data,
            style,
            tone_profile,
            tailor_engine,
            session,
            result,
            reporter,
            yes=yes,
        )

        # Reflect an accepted tailor in the funnel store so `status` shows it — only for
        # a job with a real identity (a stored --from job, or one given --url); a manual
        # job with no URL would collide on the placeholder URL. Best-effort: a store
        # hiccup must not sink an already-saved tailoring (especially under --json).
        if result.output_path and (from_ref or job_url):
            try:
                _get_jobs_store().mark_tailored(
                    job,
                    tailored_resume_path=result.output_path,
                    cover_letter_path=result.cover_letter_path,
                )
            except JobApplicatorError as exc:
                err_console.print(
                    f"[yellow]⚠ Could not update the job store: {escape(str(exc))}[/yellow]"
                )

        if as_json:
            # All human/Rich output above was redirected to stderr (below); the workflow's many
            # console helpers (ATS preflight, date audit, preview…) thus stay off stdout. Write
            # only the JSON result to the real stdout so `tailor --json | jq` is clean.
            real_stdout.write(result.model_dump_json(indent=2) + "\n")

    import contextlib

    try:
        if as_json:
            with contextlib.redirect_stdout(sys.stderr):
                asyncio.run(_run())
        else:
            asyncio.run(_run())
    except JobApplicatorError as exc:
        # Typed, expected failures (no session, anti-bot block, missing resume)
        # — show the message cleanly instead of a raw Python traceback.
        if reporter:
            reporter.record_error(str(exc))
        err_console.print(f"[yellow]⚠ {escape(str(exc))}[/yellow]")
        raise typer.Exit(1) from exc
    except Exception as exc:
        if reporter:
            reporter.record_error(str(exc))
        raise
    finally:
        if reporter:
            report_log_file = ctx.obj.log_file if isinstance(ctx.obj, VerboseContext) else None
            reporter.render(err_console, report_log_file)


def _sanitize_config(settings: AppSettings) -> dict[str, Any]:
    data = settings.model_dump()
    _redact_secrets(data)
    return data


def _redact_secrets(obj: Any) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str) and (
                "password" in key.lower()
                or "secret" in key.lower()
                or "api_key" in key.lower()
                or ("token" in key.lower() and "max_tokens" not in key.lower())
            ):
                obj[key] = "[REDACTED]"
            else:
                _redact_secrets(value)
    elif isinstance(obj, list):
        for item in obj:
            _redact_secrets(item)


@app.command()
def ats_check(
    ctx: typer.Context,
    resume_path: str = typer.Option("", "--resume", help="Path to resume file."),
    as_json: bool = typer.Option(False, "--json", help="Output results as JSON."),
    strict: bool = typer.Option(
        False, "--strict", help="Exit non-zero if the résumé is not ATS-compatible (for CI gating)."
    ),
    ocr_mode: OCRMode = typer.Option(
        OCRMode.AUTO,
        "--ocr-mode",
        help="OCR mode: auto (fallback), on (always), off (never).",
    ),
    force_ocr: bool = typer.Option(
        False,
        "--force-ocr",
        help="Force OCR; equivalent to --ocr-mode on.",
    ),
    verbose: bool = _verbose_option(),
    log_file: str | None = _log_file_option(),
) -> None:
    """Check resume ATS (Applicant Tracking System) compatibility."""
    _merge_verbose_ctx(ctx, verbose, log_file)
    settings = _get_settings()
    if resume_path:
        settings.resume_path = resume_path
    setup_logging(settings.log_level)
    effective_ocr_mode = _resolve_ocr_mode(ocr_mode, force_ocr)

    from job_applicator.documents.ats_checker import ATSChecker
    from job_applicator.documents.resume import ResumeLoader

    if not settings.resume_path:
        reporter = _get_reporter(
            ctx=ctx,
            command="ats-check",
            args={"resume": "", "ocr_mode": effective_ocr_mode},
            config=_sanitize_config(settings),
        )
        if reporter:
            reporter.record_error("Resume path required. Use --resume.")
            reporter.render(err_console, log_file=None)
        err_console.print("[red]Resume path required. Use --resume.[/red]")
        raise typer.Exit(1)

    reporter = _get_reporter(
        ctx=ctx,
        command="ats-check",
        args={"resume": settings.resume_path, "ocr_mode": effective_ocr_mode},
        config=_sanitize_config(settings),
    )

    try:
        loader = ResumeLoader()
        resume_data = loader.load(settings.resume_path, ocr_mode=effective_ocr_mode)

        if reporter:
            reporter.record_resume(
                source=str(settings.resume_path),
                ocr_mode=effective_ocr_mode,
                text_length=len(resume_data.raw_text),
                parsed_name=resume_data.name,
                parsed_email=resume_data.email,
                parsed_phone=resume_data.phone,
                parsed_skills=resume_data.skills,
                parsed_summary_preview=resume_data.summary[:100],
            )

        if not as_json:
            console.print(f"[green]Loaded resume: {resume_data.name}[/green]")

        checker = ATSChecker()
        result = checker.check(resume_data)

        if reporter:
            reporter.record_ats(
                score=result.score,
                checks=result.checks,
                warnings=result.warnings,
                suggestions=result.suggestions,
            )

        if as_json:
            import json

            output = {
                "score": result.score,
                "is_compatible": result.is_compatible,
                "checks": result.checks,
                "warnings": result.warnings,
                "suggestions": result.suggestions,
            }
            sys.stdout.write(json.dumps(output, indent=2) + "\n")
            if strict and not result.is_compatible:
                raise typer.Exit(1)
            return

        # Display results
        color = "green" if result.is_compatible else "red"
        console.print(f"\n[bold {color}]ATS Score: {result.score:.0%}[/bold {color}]")
        status = "Compatible" if result.is_compatible else "Not Compatible"
        console.print(f"[{color}]Status: {status}[/{color}]\n")

        # Check results table
        table = Table(title="ATS Checks")
        table.add_column("Check", style="cyan")
        table.add_column("Status")
        table.add_column("Details")

        for check in result.checks:
            status = "[green]PASS[/green]" if check["passed"] else "[red]FAIL[/red]"
            table.add_row(str(check["name"]), status, str(check["details"]))

        console.print(table)

        # Warnings
        if result.warnings:
            console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in result.warnings:
                console.print(f"  [yellow]![/yellow] {warning}")

        # Suggestions
        if result.suggestions:
            console.print("\n[bold cyan]Suggestions:[/bold cyan]")
            for suggestion in result.suggestions:
                console.print(f"  [cyan]*[/cyan] {suggestion}")

        if strict and not result.is_compatible:
            raise typer.Exit(1)
    except typer.Exit:
        raise  # --strict gate (an explicit Exit) — propagate, don't record/wrap as an error
    except JobApplicatorError as exc:
        # Typed, expected failures (unreadable / unsupported / missing résumé) get a clean
        # message + exit 1 — matching the sibling commands, not a raw traceback.
        if reporter:
            reporter.record_error(str(exc))
        err_console.print(f"[yellow]⚠ {escape(str(exc))}[/yellow]")
        raise typer.Exit(1) from exc
    except Exception as exc:
        if reporter:
            reporter.record_error(str(exc))
        raise
    finally:
        if reporter:
            log_file = None
            vctx = ctx.obj
            if isinstance(vctx, VerboseContext):
                log_file = vctx.log_file
            reporter.render(err_console, log_file=log_file)


@app.command()
def config_init(
    ctx: typer.Context,
    output_path: str = typer.Option("config.toml", "--output", "-o", help="Output file path."),
    verbose: bool = _verbose_option(),
    log_file: str | None = _log_file_option(),
) -> None:
    """Create a sample config.toml file."""
    _merge_verbose_ctx(ctx, verbose, log_file)
    config_content = """# Job Applicator Configuration

# Profile
profile_name = "default"
resume_path = "/path/to/your/resume.pdf"
# Example résumé/cover letter(s) whose writing style the AI should mimic. Supports a single
# file or comma-separated paths (e.g. "example1.txt,example2.pdf").
# style_guide_path = "cover_letter_example.txt"
output_dir = "output"
log_level = "INFO"

# Browser
[browser]
headless = true
slow_mo = 0
timeout_ms = 30000

# LLM (for AI cover letters)
[llm]
api_base = "__LLM_API_BASE__"
api_key = "__LLM_API_KEY__"
model = "__LLM_MODEL__"
max_tokens = __LLM_MAX_TOKENS__
temperature = __LLM_TEMPERATURE__

# Targets
[target]
max_applications_per_day = 20
delay_between_applications_s = 2.0
# linkedin_email = "your-email@example.com"
# linkedin_password = "your-password"
"""
    # Fill the [llm] section from the LLMConfig defaults so config-init never drifts
    # from the code (brace-safe placeholders → .replace, not an f-string over the template).
    for _token, _field in (
        ("__LLM_API_BASE__", "api_base"),
        ("__LLM_API_KEY__", "api_key"),
        ("__LLM_MODEL__", "model"),
        ("__LLM_MAX_TOKENS__", "max_tokens"),
        ("__LLM_TEMPERATURE__", "temperature"),
    ):
        config_content = config_content.replace(_token, str(LLMConfig.model_fields[_field].default))

    config_path = Path(output_path)
    if config_path.is_dir():
        err_console.print(f"[red]Output path is a directory, not a file: {output_path}[/red]")
        raise typer.Exit(1)
    if config_path.exists():
        console.print(f"[yellow]{output_path} already exists. Skipping.[/yellow]")
        return

    reporter = _get_reporter(
        ctx=ctx,
        command="config-init",
        args={"output": output_path},
        config={},
    )

    try:
        config_path.write_text(config_content)
        console.print("[green]Created config.toml[/green]")
        console.print("Edit it with your credentials, or set environment variables.")

        if reporter:
            reporter.record_io(files_written=[str(output_path)])
    except OSError as exc:
        # Unwritable path / permission error → clean message + exit 1, not a raw traceback.
        if reporter:
            reporter.record_error(str(exc))
        # escape the user-supplied path too — a path with `[...]` is Rich markup otherwise.
        msg = f"Could not write config to {escape(output_path)}: {escape(str(exc))}"
        err_console.print(f"[red]{msg}[/red]")
        raise typer.Exit(1) from exc
    except Exception as exc:
        if reporter:
            reporter.record_error(str(exc))
        raise
    finally:
        if reporter:
            log_file = None
            vctx = ctx.obj
            if isinstance(vctx, VerboseContext):
                log_file = vctx.log_file
            reporter.render(err_console, log_file=log_file)


@app.command("check-session")
def check_session(
    ctx: typer.Context,
    site: str = typer.Argument("linkedin", help="Job board to check (linkedin, indeed)."),
    headed: bool = typer.Option(False, "--headed", help="Show the browser window."),
    verbose: bool = _verbose_option(),
    log_file: str | None = _log_file_option(),
) -> None:
    """Verify that an authenticated board session is ready (or not required)."""
    _merge_verbose_ctx(ctx, verbose, log_file)
    settings = _get_settings(headed=headed)
    setup_logging(settings.log_level)

    async def _run() -> None:
        async with _make_browser(site, settings) as browser:
            scraper = _make_scraper(site, browser, settings)
            health = await scraper.check_session()

        if health.healthy:
            console.print(f"[green]✓ {health.board.value}:[/green] {health.details}")
        else:
            console.print(f"[red]✗ {health.board.value}:[/red] {health.details}")
            raise typer.Exit(1)

    asyncio.run(_run())


@app.command()
def doctor(
    ctx: typer.Context,
    as_json: bool = typer.Option(False, "--json", help="Output the health report as JSON."),
    verbose: bool = _verbose_option(),
    log_file: str | None = _log_file_option(),
) -> None:
    """Check the AI backend, browser, system binaries, and config."""
    _merge_verbose_ctx(ctx, verbose, log_file)
    from job_applicator.diagnostics import run_diagnostics

    settings = _get_settings()
    report = asyncio.run(run_diagnostics(settings))
    if as_json:
        import json

        # `ok` is a @property (excluded from model_dump) — include the headline verdict explicitly.
        sys.stdout.write(json.dumps({"ok": report.ok, **report.model_dump()}, indent=2) + "\n")
    else:
        _render_doctor(report)
    if not report.ok:
        raise typer.Exit(1)


def _render_doctor(report: DoctorReport) -> None:
    """Render a DoctorReport as a human-readable health check.

    HTTP-200 reachability is the only blocking signal; a reachable-but-rejected
    endpoint (401/403) and a model/embeddings mismatch are surfaced distinctly. All
    dynamic values are markup-escaped, so an error string or model id containing
    brackets can't corrupt or crash the report.
    """
    good = "[green]✓[/green]"
    bad = "[red]✗[/red]"
    warn = "[yellow]•[/yellow]"
    llm = report.llm
    api_base = escape(llm.api_base)

    console.print("\n[bold]job-applicator doctor[/bold]\n")

    if not llm.reachable:
        console.print(f"  LLM endpoint   {bad} not reachable  {api_base}")
        if llm.error:
            console.print(f"                 [dim]{escape(llm.error)}[/dim]")
        console.print(f"                 → start one: [cyan]{SERVE_SCRIPT}[/cyan]")
        console.print("                 → or point your llm.api_base at a running provider")
    elif llm.http_status == 200:
        console.print(f"  LLM endpoint   {good} reachable  {api_base}")
        if llm.model_available:
            console.print(f"    model        {good} {escape(llm.model_configured)}")
        else:
            console.print(
                f"    model        {warn} '{escape(llm.model_configured)}' not listed by "
                "the endpoint (fine for cloud/Ollama; for a local vLLM, check the id)"
            )
            if llm.models_seen:
                listed = ", ".join(escape(m) for m in llm.models_seen[:5])
                extra = len(llm.models_seen) - 5
                more = f" (+{extra} more)" if extra > 0 else ""
                console.print(f"                 endpoint serves: {listed}{more}")
    elif llm.http_status in (401, 403):
        console.print(
            f"  LLM endpoint   {bad} reachable but rejected (HTTP {llm.http_status})  {api_base}"
        )
        console.print("                 → the server is up; check your llm.api_key / credentials")
    else:
        console.print(f"  LLM endpoint   {bad} reachable but /models failed  {api_base}")
        if llm.error:
            console.print(f"                 [dim]{escape(llm.error)}[/dim]")

    emb = report.embeddings
    if emb.cached:
        console.print(f"  Embeddings     {good} {escape(emb.model_name)} cached")
    else:
        console.print(
            f"  Embeddings     {warn} {escape(emb.model_name)} not cached "
            "(auto-downloads on first match)"
        )

    sh = report.self_host
    vllm_part = f"{good} vllm" if sh.vllm_installed else f"{warn} vllm not installed"
    token_part = f"{good} HF token" if sh.hf_token_present else f"{warn} no HF token"
    console.print(f"  Self-host      {vllm_part} · {token_part}  [dim](only if self-hosting)[/dim]")

    browser = report.browser
    if browser.playwright_installed and browser.chromium_executable:
        console.print(f"  Browser        {good} Playwright + Chromium")
        console.print(f"                 [dim]{escape(str(browser.chromium_executable))}[/dim]")
    elif browser.playwright_installed:
        console.print(f"  Browser        {warn} Playwright installed, Chromium not found")
        if browser.error:
            console.print(f"                 [dim]{escape(browser.error)}[/dim]")
    else:
        console.print(f"  Browser        {warn} Playwright not installed")
        if browser.error:
            console.print(f"                 [dim]{escape(browser.error)}[/dim]")
        console.print("                 → run: [cyan]playwright install chromium[/cyan]")

    sys = report.system
    pdf_part = f"{good} pdftotext" if sys.pdftotext_available else f"{warn} pdftotext"
    xvfb_part = f"{good} Xvfb" if sys.xvfb_available else f"{warn} Xvfb"
    console.print(f"  System bins    {pdf_part} · {xvfb_part}  [dim](optional)[/dim]")

    cfg = report.config
    if cfg.config_file_found and cfg.config_file_parseable:
        console.print(f"  Config         {good} {escape(str(cfg.config_file_path))}")
    elif cfg.config_file_found:
        console.print(f"  Config         {bad} parse error  {escape(str(cfg.config_file_path))}")
        if cfg.error:
            console.print(f"                 [dim]{escape(cfg.error)}[/dim]")
    else:
        console.print(f"  Config         {warn} not found  {escape(str(cfg.config_file_path))}")
        console.print("                 → run: [cyan]job-applicator config-init[/cyan]")
    if cfg.plaintext_credentials:
        console.print(
            f"                 {warn} [yellow]plaintext board credentials in config file — "
            "consider removing them (login is headed)[/yellow]"
        )
    if cfg.resume_path_set and not cfg.resume_path_exists:
        console.print(
            f"                 {warn} [yellow]configured resume_path does not exist[/yellow]"
        )

    console.print()
    if report.ok and llm.model_available:
        console.print("[green]All systems go — AI features ready.[/green]\n")
    elif report.ok:
        console.print("[yellow]Reachable; configured model not listed (advisory).[/yellow]\n")
    elif llm.reachable:
        console.print("[red]Reachable but not usable — AI features will fail until fixed.[/red]\n")
    else:
        console.print(
            "[red]LLM endpoint unreachable — AI features will fail until it is up.[/red]\n"
        )


def _get_settings(headed: bool = False) -> AppSettings:
    """Build AppSettings, overriding headless if --headed."""
    settings = AppSettings()
    if headed:
        settings.browser.headless = False
    return settings
