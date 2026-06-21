"""CLI entry point — Typer + Rich for terminal UX."""

from __future__ import annotations

import asyncio
import sys
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from job_applicator import __version__
from job_applicator.config import AppSettings, LLMConfig
from job_applicator.exceptions import JobApplicatorError
from job_applicator.models import DoctorReport, UserProfile
from job_applicator.state import ApplicationState
from job_applicator.utils.cookies import save_cookies
from job_applicator.utils.diff import render_diff
from job_applicator.utils.llm import SERVE_SCRIPT
from job_applicator.utils.logging import setup_logging
from job_applicator.utils.url import host_matches
from job_applicator.utils.verbose import VerboseReporter

if TYPE_CHECKING:
    from job_applicator.applicators.base import BaseApplicator
    from job_applicator.browser.manager import BrowserManager
    from job_applicator.documents.tone_detector import ToneProfile
    from job_applicator.models import (
        ApplicationResult,
        ATSCompatibilityResult,
        CoverLetterResult,
        CoverLetterSession,
        JobListing,
        ResumeData,
        StyleGuide,
        TailoredResume,
    )
    from job_applicator.scrapers.base import BaseScraper

app = typer.Typer(
    name="job-applicator",
    help="Automated job application tool with AI-powered cover letters.",
    add_completion=False,
)
console = Console()

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
            console.print(f"[red]LLM error: {escape(str(exc))}[/red]")
            choice = console.input(f"[bold cyan]{on_fail_choices}? [/bold cyan]").strip().upper()
            if choice == "Q":
                return None


def _resolve_ocr_mode(ocr_mode: str, force_ocr: bool) -> str:
    """Return effective OCR mode from CLI flags."""
    if force_ocr:
        return "on"
    return ocr_mode


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


def _detect_tone(job: JobListing) -> ToneProfile:
    """Detect job posting tone deterministically via keyword matching."""
    from job_applicator.documents.tone_detector import ToneDetector

    return ToneDetector().detect(
        title=job.title,
        description=job.description,
        requirements=job.requirements,
    )


def version_callback(value: bool) -> None:
    if value:
        console.print(f"job-applicator v{__version__}")
        raise typer.Exit()


@app.callback()
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
    """Automated job application tool with AI-powered cover letters."""
    if log_file and not verbose:
        raise typer.BadParameter("--log-file requires --verbose")
    ctx.obj = VerboseContext(verbose=verbose, log_file=log_file)


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
            console.print(f"[red]Unsupported site: {site}[/red]")
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
        console.print(f"[yellow]⚠ {escape(str(exc))}[/yellow]")
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
            reporter.render(console, log_file=log_file)


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
        console.print(f"[yellow]{site} login not yet implemented[/yellow]")
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
        console.print("[red]✗ Sign-in not detected. Re-run `job-applicator login`.[/red]")
        raise typer.Exit(1)


def _normalize_cookie(entry: Any) -> dict[str, Any] | None:
    """Best-effort conversion of an exported cookie dict to Playwright format.

    Handles common browser-extension exports (e.g. `expirationDate` instead of
    `expires`, `sameSite: "no_restriction"`). Returns None for unusable entries.
    """
    if not isinstance(entry, dict):
        return None
    name = entry.get("name")
    value = entry.get("value")
    if not name or value is None:
        return None
    out: dict[str, Any] = {"name": str(name), "value": str(value)}
    domain = entry.get("domain")
    if domain:
        out["domain"] = str(domain)
        out["path"] = str(entry.get("path", "/"))
    else:
        out["url"] = str(entry.get("url", "https://www.linkedin.com"))
    exp = entry.get("expires", entry.get("expirationDate"))
    if isinstance(exp, int | float) and not isinstance(exp, bool) and exp > 0:
        out["expires"] = float(exp)
    for key in ("httpOnly", "secure"):
        if key in entry:
            out[key] = bool(entry[key])
    same = entry.get("sameSite")
    if isinstance(same, str):
        mapped = {"no_restriction": "None", "none": "None", "lax": "Lax", "strict": "Strict"}.get(
            same.lower()
        )
        if mapped:
            out["sameSite"] = mapped
    # Chromium rejects a SameSite=None cookie that is not Secure, so an export
    # that omits `secure` would otherwise yield a silently-dropped session cookie.
    if out.get("sameSite") == "None":
        out["secure"] = True
    return out


def _cookiejar_to_playwright(cookie: Any) -> dict[str, Any] | None:
    """Convert a stdlib cookiejar cookie (from browser_cookie3) to Playwright form."""
    raw: dict[str, Any] = {
        "name": cookie.name,
        "value": cookie.value,
        "domain": cookie.domain,
        "path": cookie.path or "/",
        "secure": bool(getattr(cookie, "secure", False)),
    }
    expires = getattr(cookie, "expires", None)
    if expires:
        raw["expires"] = expires
    # cookiejar keeps httpOnly as a nonstandard attr (browser_cookie3 sets it);
    # propagate it so the imported cookie matches the real browser cookie.
    rest = getattr(cookie, "_rest", None) or {}
    if any(str(key).lower() == "httponly" for key in rest):
        raw["httpOnly"] = True
    return _normalize_cookie(raw)


def _cookies_from_browser(browser: str, base_domain: str) -> list[dict[str, Any]]:
    """Read a site's cookies directly from a local browser's cookie store.

    Uses browser_cookie3, which decrypts the browser's on-disk cookie database —
    this reaches httpOnly cookies (like LinkedIn `li_at` or Cloudflare
    `cf_clearance`) that page scripts cannot. Only invoked via `--from-browser`.
    """
    try:
        import browser_cookie3
    except ImportError as exc:
        console.print(
            "[red]--from-browser needs the optional dependency: "
            'pip install "job-applicator[browser]"[/red]'
        )
        raise typer.Exit(1) from exc

    loaders = {
        "chrome": browser_cookie3.chrome,
        "chromium": browser_cookie3.chromium,
        "brave": browser_cookie3.brave,
        "edge": browser_cookie3.edge,
        "firefox": browser_cookie3.firefox,
    }
    loader = loaders.get(browser.lower())
    if loader is None:
        console.print(f"[red]Unsupported browser '{browser}'. Choose: {', '.join(loaders)}.[/red]")
        raise typer.Exit(1)
    try:
        jar = loader(domain_name=base_domain)
    except Exception as exc:  # browser_cookie3 raises various OS/keyring/db errors
        console.print(
            f"[red]Could not read {browser} cookies: {exc}. Is {browser} installed and your "
            "login keyring unlocked?[/red]"
        )
        raise typer.Exit(1) from exc
    # browser_cookie3's domain filter is a SUBSTRING match, so it can sweep in
    # look-alike hosts (e.g. notlinkedin.com); keep only genuine site cookies.
    return [
        c
        for c in (_cookiejar_to_playwright(ck) for ck in jar)
        if c and host_matches(str(c.get("domain", "")), base_domain)
    ]


@dataclass(frozen=True)
class _SiteSpec:
    """Per-board rules for ``import-cookies``, so the command body stays board-agnostic.

    ``required_cookie`` is a hard gate (absent => refuse, since the session can't
    work without it). ``preferred_cookie`` is a soft signal (absent => warn but
    save). ``session_flags`` enables the LinkedIn ``--li-at``/``--jsessionid``
    seed inputs. ``feed_verify`` runs the post-import logged-in feed check.
    """

    cookie_path: Path
    base_domain: str
    required_cookie: str | None
    preferred_cookie: str | None
    session_flags: bool
    feed_verify: bool


def _site_specs() -> dict[str, _SiteSpec]:
    from job_applicator.scrapers.indeed import IndeedScraper
    from job_applicator.scrapers.linkedin import LinkedInScraper

    return {
        # li_at is the LinkedIn session token: nothing authenticates without it.
        "linkedin": _SiteSpec(
            cookie_path=LinkedInScraper.COOKIE_PATH,
            base_domain="linkedin.com",
            required_cookie="li_at",
            preferred_cookie=None,
            session_flags=True,
            feed_verify=True,
        ),
        # Indeed search is public — no cookie is strictly required. cf_clearance
        # (Cloudflare) is what actually helps a warm session avoid challenges, so
        # it's preferred-not-required; CTK and friends are mere tracking cookies.
        "indeed": _SiteSpec(
            cookie_path=IndeedScraper.COOKIE_PATH,
            base_domain="indeed.com",
            required_cookie=None,
            preferred_cookie="cf_clearance",
            session_flags=False,
            feed_verify=False,
        ),
    }


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
        "(chrome/chromium/brave/edge/firefox). Needs the [browser] extra; reads/decrypts "
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
        console.print(f"[red]Unsupported site '{site}'. Choose: {', '.join(specs)}.[/red]")
        raise typer.Exit(1)
    spec = specs[site]
    if (li_at or jsessionid) and not spec.session_flags:
        console.print(
            "[red]--li-at/--jsessionid are LinkedIn-only; "
            f"use --from-browser/--file for {site}.[/red]"
        )
        raise typer.Exit(1)

    cookies: list[dict[str, Any]] = []
    if from_browser:
        cookies = _cookies_from_browser(from_browser, spec.base_domain)
        console.print(f"[green]Read {len(cookies)} {site} cookie(s) from {from_browser}.[/green]")
    elif file:
        try:
            raw = json.loads(Path(file).read_text())
        except (OSError, ValueError) as exc:
            console.print(f"[red]Could not read cookie file: {escape(str(exc))}[/red]")
            raise typer.Exit(1) from exc
        entries = raw.get("cookies", raw) if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            console.print('[red]Cookie file must be a JSON list or {"cookies": [...]}.[/red]')
            raise typer.Exit(1)
        cookies = [c for c in (_normalize_cookie(e) for e in entries) if c]
    elif li_at:
        if li_at == "-":  # read from stdin to keep the token out of shell history
            li_at = sys.stdin.readline().strip()
        if not li_at:
            console.print("[red]No token provided on stdin.[/red]")
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
        console.print(
            "[red]Provide --from-browser <name>, --li-at <value>, or --file <path>.[/red]"
        )
        raise typer.Exit(1)

    if not cookies:
        console.print("[red]No usable cookies found in the input.[/red]")
        raise typer.Exit(1)
    names = {c.get("name") for c in cookies}
    if spec.required_cookie and spec.required_cookie not in names:
        console.print(
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
        console.print(
            "[yellow]Imported, but the feed did not load as logged-in. The li_at value may be "
            "stale — re-copy it from a freshly logged-in browser and try again.[/yellow]"
        )
        raise typer.Exit(1)


@app.command()
def apply(
    ctx: typer.Context,
    site: str = typer.Option("linkedin", "--site", "-s", help="Job board."),
    query: str = typer.Option("", "--query", "-q", help="Search query (empty = use saved list)."),
    limit: int = typer.Option(5, "--limit", "-n", help="Max applications."),
    cover_letter: bool = typer.Option(True, "--cover-letter/--no-cover-letter", help="AI cover."),
    headed: bool = typer.Option(False, "--headed", help="Run browser in headed mode."),
    resume_path: str = typer.Option("", "--resume", help="Path to resume file."),
    style_guide: str = typer.Option("", "--style-guide", help="Example to mimic style."),
    as_json: bool = typer.Option(False, "--json", help="Output results as JSON."),
    ocr_mode: str = typer.Option(
        "auto",
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
    verbose: bool = _verbose_option(),
    log_file: str | None = _log_file_option(),
) -> None:
    """Auto-apply to jobs with optional AI cover letters.

    By default this is a DRY RUN: each job's Easy Apply form is opened and
    filled, but never submitted. Pass --submit to send real applications.
    """
    _merge_verbose_ctx(ctx, verbose, log_file)
    if submit:
        console.print(
            "[bold red]--submit set: real applications WILL be sent on your account.[/bold red]"
        )
    else:
        console.print(
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
        from job_applicator.models import JobBoard
        from job_applicator.scrapers.base import SearchParams

        async with _make_browser(site, settings) as browser:
            # Search for jobs
            if query:
                scraper = _make_scraper(site, browser, settings)
                params = SearchParams(
                    query=query,
                    max_results=limit,
                    board=JobBoard(site),
                )
                with console.status(f"Searching {site}..."):
                    jobs = await scraper.scrape(params)
            else:
                console.print("[yellow]No query provided. Use --query to search.[/yellow]")
                raise typer.Exit(1)

            if not jobs:
                console.print("[yellow]No jobs found to apply to.[/yellow]")
                return

            # Generate cover letters only when actually submitting — a dry run
            # never sends them, so generating up front would waste LLM calls.
            cover_letters: dict[str, str] = {}
            if cover_letter and settings.resume_path and submit:
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

                generator = CoverLetterGenerator(settings.llm)
                sem = asyncio.Semaphore(3)

                async def _gen_one(
                    job: JobListing,
                ) -> tuple[str, str] | None:
                    async with sem:
                        try:
                            letter = await generator.generate(job, user_profile, resume_data)
                            return str(job.url), letter
                        except Exception as exc:
                            msg = f"Cover letter failed for {job.title}: {exc}"
                            console.print(f"[yellow]{msg}[/yellow]")
                            return None

                with console.status("Generating cover letters (parallel)..."):
                    results_cl = await asyncio.gather(*(_gen_one(j) for j in jobs[:limit]))
                    for entry in results_cl:
                        if entry is not None:
                            url, letter = entry
                            cover_letters[url] = letter

            # Apply to jobs
            from job_applicator.models import ApplicationStatus

            applicator = _make_applicator(site, browser, settings)
            state = ApplicationState()

            if submit:
                today_count = state.count_today(board=site)
                daily_cap = settings.target.max_applications_per_day
                if today_count >= daily_cap:
                    console.print(
                        f"[yellow]Daily application cap reached ({today_count}/{daily_cap}). "
                        "Skipping apply loop.[/yellow]"
                    )
                    return

            app_results: list[ApplicationResult] = []
            for job in jobs[:limit]:
                job_url = str(job.url)
                if submit and state.has_applied(
                    job_url,
                    statuses={ApplicationStatus.SUBMITTED, ApplicationStatus.ALREADY_APPLIED},
                ):
                    console.print(
                        f"[dim]Skipping {job.title} at {job.company} — already applied.[/dim]"
                    )
                    app_results.append(
                        ApplicationResult(
                            job=job,
                            status=ApplicationStatus.ALREADY_APPLIED,
                            notes=(
                                "Skipped by local state store "
                                "(previous submitted/already-applied record)."
                            ),
                        )
                    )
                    continue

                if submit:
                    today_count = state.count_today(board=site)
                    if today_count >= daily_cap:
                        console.print(
                            f"[yellow]Daily application cap reached ({today_count}/{daily_cap}). "
                            "Stopping.[/yellow]"
                        )
                        break

                with console.status(f"Applying to {job.title} at {job.company}..."):
                    job_letter = cover_letters.get(job_url)
                    ar: ApplicationResult = await applicator.apply(job, job_letter, submit=submit)
                    app_results.append(ar)
                    state.record(ar)

            if reporter and app_results:
                reporter.record_io(files_written=[])

            # Display results
            if as_json:
                import json

                output = [
                    {
                        "job": r.job.title,
                        "company": r.job.company,
                        "status": r.status.value,
                        "error": r.error_message,
                        "notes": r.notes,
                    }
                    for r in app_results
                ]
                sys.stdout.write(json.dumps(output, indent=2) + "\n")
            else:
                table = Table(title="Application Results")
                table.add_column("Job", style="cyan")
                table.add_column("Company", style="green")
                table.add_column("Status")
                table.add_column("Notes")

                for r in app_results:
                    status_style = {
                        "submitted": "green",
                        "failed": "red",
                        "skipped": "yellow",
                        "already_applied": "magenta",
                        "pending": "blue",
                    }.get(r.status.value, "white")
                    table.add_row(
                        r.job.title,
                        r.job.company,
                        f"[{status_style}]{r.status.value}[/{status_style}]",
                        r.error_message or r.notes or "",
                    )

                console.print(table)
                # Count every status (incl. already_applied) so the summary
                # never silently under-reports outcomes.
                counts = Counter(r.status.value for r in app_results)
                summary = ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
                console.print(f"\n{summary}")

    try:
        asyncio.run(_run())
    except JobApplicatorError as exc:
        # Typed, expected failures (no session, anti-bot block, missing resume)
        # — show the message cleanly instead of a raw Python traceback.
        if reporter:
            reporter.record_error(str(exc))
        console.print(f"[yellow]⚠ {escape(str(exc))}[/yellow]")
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
            reporter.render(console, log_file=log_file)


@app.command()
def generate_cover_letter(
    ctx: typer.Context,
    job_title: str = typer.Option(..., "--job-title", "-t", help="Job title."),
    company: str = typer.Option(..., "--company", "-c", help="Company name."),
    job_description: str = typer.Option("", "--description", "-d", help="Job description."),
    resume_path: str = typer.Option("", "--resume", help="Path to resume file."),
    style_guide: str = typer.Option("", "--style-guide", help="Style examples."),
    headed: bool = typer.Option(False, "--headed", help="Run browser in headed mode."),
    ocr_mode: str = typer.Option(
        "auto",
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
            console.print("[red]Resume path required. Use --resume or set RESUME_PATH.[/red]")
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
        console.print(
            f"[dim]Detected tone: {tone_profile.primary} "
            f"(confidence: {tone_profile.confidence:.0%})[/dim]"
        )

        generator = CoverLetterGenerator(settings.llm)

        # Load style guide if provided (supports comma-separated paths)
        style = None
        if settings.style_guide_path:
            paths = [p.strip() for p in settings.style_guide_path.split(",") if p.strip()]

            if len(paths) == 1:
                with console.status("Analyzing writing style..."):
                    style = await generator.load_style_guide(paths[0])
                console.print(f"[green]Style loaded: {style.tone}[/green]")
            elif len(paths) > 1:
                with console.status(f"Analyzing {len(paths)} style examples..."):
                    from job_applicator.documents.style_analyzer import StyleAnalyzer

                    analyzer = StyleAnalyzer(settings.llm)

                    texts = []
                    for path in paths:
                        from pathlib import Path

                        p = Path(path)
                        if await asyncio.to_thread(p.exists):
                            if p.suffix.lower() == ".pdf":
                                resume = loader.load(p, ocr_mode=effective_ocr_mode)
                                texts.append(resume.raw_text)
                            else:
                                texts.append(await asyncio.to_thread(p.read_text, encoding="utf-8"))

                    if texts:
                        style = await analyzer.analyze_multiple(texts)
                        msg = f"Combined style from {len(texts)} examples"
                        console.print(f"[green]{msg}: {style.tone}[/green]")

        if reporter:
            reporter.record_llm_call(
                model=settings.llm.model,
                endpoint=settings.llm.api_base,
                temperature=settings.llm.temperature,
                details={"style_guide": settings.style_guide_path or "default"},
            )

        from job_applicator.documents.tone_detector import ToneDetector

        tone_section = ToneDetector().format_for_prompt(tone_profile)

        with console.status("Generating cover letter..."):
            letter = await generator.generate(
                job, user_profile, resume_data, style, tone_section=tone_section
            )

        console.print("\n[bold]Generated Cover Letter:[/bold]\n")
        console.print(letter)

    try:
        asyncio.run(_run())
    except JobApplicatorError as exc:
        # Typed, expected failures (no session, anti-bot block, missing resume)
        # — show the message cleanly instead of a raw Python traceback.
        if reporter:
            reporter.record_error(str(exc))
        console.print(f"[yellow]⚠ {escape(str(exc))}[/yellow]")
        raise typer.Exit(1) from exc
    except Exception as exc:
        if reporter:
            reporter.record_error(str(exc))
        raise
    finally:
        if reporter:
            log_file = ctx.obj.log_file if isinstance(ctx.obj, VerboseContext) else None
            reporter.render(console, log_file)


@app.command()
def match(
    ctx: typer.Context,
    resume_path: str = typer.Option("", "--resume", help="Path to resume file."),
    jobs_file: str = typer.Option("", "--jobs-file", help="JSON file with job listings."),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of top matches."),
    min_score: float = typer.Option(0.0, "--min-score", help="Minimum match score (0.0-1.0)."),
    as_json: bool = typer.Option(False, "--json", help="Output results as JSON."),
    ocr_mode: str = typer.Option(
        "auto",
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
        from job_applicator.models import JobBoard, JobListing

        if not settings.resume_path:
            console.print("[red]Resume path required. Use --resume.[/red]")
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
            import json

            with open(jobs_file) as f:  # noqa: ASYNC230
                data = json.load(f)
                for item in data:
                    jobs.append(JobListing(**item))
        else:
            # Example jobs for demo
            from pydantic import HttpUrl

            jobs = [
                JobListing(
                    title="Python Developer",
                    company="TechCorp",
                    url=HttpUrl("https://example.com/1"),
                    description="Looking for Python developer with FastAPI experience",
                    requirements=["Python", "FastAPI", "PostgreSQL"],
                    board=JobBoard.LINKEDIN,
                ),
                JobListing(
                    title="Backend Engineer",
                    company="StartupXYZ",
                    url=HttpUrl("https://example.com/2"),
                    description="Backend engineer for microservices",
                    requirements=["Python", "Docker", "AWS"],
                    board=JobBoard.LINKEDIN,
                ),
            ]

        if not as_json:
            console.print(f"[green]Loaded {len(jobs)} jobs[/green]")

        # Match
        with console.status("Computing embeddings and matching..."):
            matcher = JobMatcher(settings.embedding)
            matches = matcher.rank_jobs(resume_data, jobs, top_k=top_k)

        # Filter by min score
        if min_score > 0:
            matches = [m for m in matches if m.score >= min_score]

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
        console.print(f"[yellow]⚠ {escape(str(exc))}[/yellow]")
        raise typer.Exit(1) from exc
    except Exception as exc:
        if reporter:
            reporter.record_error(str(exc))
        raise
    finally:
        if reporter:
            log_file = ctx.obj.log_file if isinstance(ctx.obj, VerboseContext) else None
            reporter.render(console, log_file)


@app.command()
def batch(
    ctx: typer.Context,
    resume_path: str = typer.Option("", "--resume", help="Path to resume file."),
    jobs_file: str = typer.Option("", "--jobs-file", help="JSON file with job listings."),
    query: str = typer.Option(
        "", "--query", "-q", help="Search query (alternative to --jobs-file)."
    ),
    site: str = typer.Option("linkedin", "--site", "-s", help="Job board for --query."),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Max jobs to tailor."),
    min_score: float = typer.Option(0.0, "--min-score", help="Skip jobs below this score."),
    cover_letter: bool = typer.Option(
        True, "--cover-letter/--no-cover-letter", help="Generate cover letters."
    ),
    style_guide: str = typer.Option("", "--style-guide", help="Style example file."),
    headed: bool = typer.Option(False, "--headed", help="Run browser in headed mode."),
    as_json: bool = typer.Option(False, "--json", help="Output results as JSON."),
    ocr_mode: str = typer.Option(
        "auto",
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
            "resume": settings.resume_path,
            "jobs_file": jobs_file,
            "query": query,
            "top_k": top_k,
            "cover_letter": cover_letter,
        },
        config=_sanitize_config(settings),
    )
    written_paths: list[str] = []

    async def _run() -> None:
        import json
        from datetime import datetime as dt
        from pathlib import Path

        from job_applicator.documents.resume import ResumeLoader
        from job_applicator.documents.resume_tailor import ResumeTailor
        from job_applicator.embeddings.matching import JobMatcher, MatchResult
        from job_applicator.models import JobBoard, JobListing, TailoringReport

        if not settings.resume_path:
            console.print("[red]Resume path required. Use --resume.[/red]")
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
            try:
                with open(jobs_file) as f:  # noqa: ASYNC230
                    data = json.load(f)
                    for item in data:
                        jobs.append(JobListing(**item))
            except FileNotFoundError:
                console.print(f"[red]Jobs file not found: {jobs_file}[/red]")
                raise typer.Exit(1) from None
            except Exception as exc:
                console.print(f"[red]Error reading jobs file: {escape(str(exc))}[/red]")
                raise typer.Exit(1) from exc
        elif query:
            from job_applicator.scrapers.base import SearchParams

            async with _make_browser(site, settings) as browser:
                scraper = _make_scraper(site, browser, settings)
                params = SearchParams(query=query, max_results=top_k * 2, board=JobBoard(site))
                with console.status(f"Searching {site}..."):
                    jobs = await scraper.scrape(params)
        else:
            console.print("[red]Provide --jobs-file or --query.[/red]")
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

        if not as_json:
            console.print(f"[cyan]Tailoring {len(matches)} jobs...[/cyan]")

        style = None
        cl_generator = None
        if settings.style_guide_path:
            from job_applicator.documents.cover_letter import CoverLetterGenerator

            cl_generator = CoverLetterGenerator(settings.llm)
            with console.status("Loading style guide..."):
                style = await cl_generator.load_style_guide(settings.style_guide_path)
        elif cover_letter:
            from job_applicator.documents.cover_letter import CoverLetterGenerator

            cl_generator = CoverLetterGenerator(settings.llm)

        tailor_engine = ResumeTailor(settings.llm)
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
                    post_ats = _run_ats_post_tailor(resume_data.raw_text, tailored.tailored_text)
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
                except Exception as exc:
                    result["tailored"] = False
                    result["error"] = str(exc)
                    return result

                if cl_generator is not None:
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
                    except Exception as exc:
                        result["cover_letter"] = False
                        result["cl_error"] = str(exc)

                return result

        with console.status("Processing jobs in parallel..."):
            batch_results = await asyncio.gather(*(_process_one(m) for m in matches))

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
        console.print(f"[yellow]⚠ {escape(str(exc))}[/yellow]")
        raise typer.Exit(1) from exc
    except Exception as exc:
        if reporter:
            reporter.record_error(str(exc))
        raise
    finally:
        if reporter:
            log_file = ctx.obj.log_file if isinstance(ctx.obj, VerboseContext) else None
            reporter.render(console, log_file)


async def _generate_cover_letter(
    console: Console,
    settings: AppSettings,
    job: JobListing,
    resume_data: ResumeData,
    style: StyleGuide | None,
    tone_section: str,
    tailored_resume_text: str,
    session: CoverLetterSession,
    attempt: int = 1,
) -> CoverLetterResult | None:
    """Generate a cover letter via LLM. Returns None on failure."""
    from job_applicator.documents.cover_letter import CoverLetterGenerator

    generator = CoverLetterGenerator(settings.llm)
    try:
        with console.status("Generating cover letter..."):
            letter = await generator.generate(
                job,
                _load_user_profile(settings),
                resume_data,
                style_guide=style,
                tone_section=tone_section,
                tailored_resume_text=tailored_resume_text,
            )
        result = CoverLetterResult(
            job_title=job.title,
            job_company=job.company,
            job_url=str(job.url),
            cover_letter_text=letter,
            attempt=attempt,
            prompt_version="1.0",
        )
        session.add_attempt(result)
        return result
    except Exception as exc:
        console.print(f"[red]LLM error: {escape(str(exc))}[/red]")
        return None


async def _save_cover_letter(
    console: Console,
    settings: AppSettings,
    job: JobListing,
    result: CoverLetterResult,
) -> Path:
    """Save cover letter to disk and return the path."""
    from datetime import datetime as dt

    output_dir = await asyncio.to_thread(settings.ensure_output_dir)
    safe_company = job.company.replace(" ", "_").replace("/", "_")
    safe_title = job.title.replace(" ", "_").replace("/", "_")
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    cl_filename = f"cover_letter_{safe_company}_{safe_title}_{timestamp}.txt"
    cl_path = output_dir / cl_filename
    await asyncio.to_thread(cl_path.write_text, result.cover_letter_text, encoding="utf-8")
    result.output_path = str(cl_path)
    cl_meta_path = cl_path.with_suffix(".meta.json")
    await asyncio.to_thread(
        cl_meta_path.write_text, result.model_dump_json(indent=2), encoding="utf-8"
    )
    console.print(f"\n[green]Cover letter saved: {cl_path}[/green]")
    return cl_path


async def _refine_cover_letter(
    console: Console,
    settings: AppSettings,
    job: JobListing,
    result: CoverLetterResult,
    user_instructions: str,
    session: CoverLetterSession,
    attempt: int,
    resume_data: ResumeData | None = None,
    style: StyleGuide | None = None,
    tone_section: str = "",
) -> bool:
    """Refine a cover letter with user instructions via LLM."""
    from job_applicator.documents.cover_letter import CoverLetterGenerator
    from job_applicator.models import CoverLetterResult as CLResult
    from job_applicator.models import ResumeData

    try:
        generator = CoverLetterGenerator(settings.llm)
        with console.status("Refining cover letter..."):
            refined = await generator.refine(
                job=job,
                resume=resume_data or ResumeData(raw_text=""),
                current_text=result.cover_letter_text,
                user_feedback=user_instructions,
                style_guide=style,
                tone_section=tone_section,
            )
        new_result = CLResult(
            job_title=job.title,
            job_company=job.company,
            job_url=str(job.url),
            cover_letter_text=refined,
            user_modifications=user_instructions,
            attempt=attempt + 1,
        )
        session.add_attempt(new_result)
        return True
    except Exception as exc:
        console.print(f"[red]LLM error: {escape(str(exc))}[/red]")
        return False


async def _cover_letter_workflow(
    console: Console,
    settings: AppSettings,
    job: JobListing,
    resume_data: ResumeData,
    style: StyleGuide | None,
    tone_profile: ToneProfile | None,
    tailored_resume_text: str,
) -> Path | None:
    """Generate and save a cover letter with accept/retry workflow.

    Returns the Path to the saved cover letter, or None if skipped.
    """
    from job_applicator.models import CoverLetterSession

    tone_section = ""
    if tone_profile is None:
        tone_profile = _detect_tone(job)
    from job_applicator.documents.tone_detector import ToneDetector

    tone_section = ToneDetector().format_for_prompt(tone_profile)

    session = CoverLetterSession(job_title=job.title, job_company=job.company)
    attempt = 0

    result = await _generate_cover_letter(
        console, settings, job, resume_data, style, tone_section, tailored_resume_text, session
    )
    if result is None:
        retry = console.input("[bold cyan][R] Retry or [Q] Skip? [/bold cyan]").strip().upper()
        if retry == "R":
            result = await _generate_cover_letter(
                console,
                settings,
                job,
                resume_data,
                style,
                tone_section,
                tailored_resume_text,
                session,
            )
            if result is None:
                console.print("[red]Cover letter generation failed. Skipping.[/red]")
                return None
        else:
            return None

    while True:
        attempt += 1
        if attempt > 10:
            console.print("[red]Maximum retry limit (10) reached.[/red]")
            break
        if attempt >= 8:
            console.print("[yellow]Warning: approaching retry limit (10 max).[/yellow]")

        result = session.current
        console.print(f"\n[bold blue]--- Cover Letter Attempt #{attempt} ---[/bold blue]")

        console.print("\n[bold]Cover Letter Preview:[/bold]\n")
        console.print(Panel(result.cover_letter_text, title="Cover Letter", border_style="green"))

        if len(session.attempts) > 1:
            render_diff(
                console,
                session.attempts[0].cover_letter_text,
                result.cover_letter_text,
                max_lines=30,
            )

        console.print("\n[bold]What would you like to do?[/bold]")
        action_table = Table(show_header=False, box=None)
        action_table.add_column("Option", style="cyan bold")
        action_table.add_column("Description")
        action_table.add_row("[A] Accept", "Save this cover letter")
        action_table.add_row("[R] Retry", "Regenerate")
        action_table.add_row("[I] Input", "Give custom instructions")
        action_table.add_row("[D] Diff", "Show full diff from first attempt")
        action_table.add_row("[V] History", "Browse previous attempts")
        action_table.add_row("[Q] Skip", "Discard (resume already saved)")
        console.print(action_table)

        choice = (
            console.input("\n[bold cyan]Your choice (A/R/I/D/V/Q): [/bold cyan]").strip().upper()
        )

        if choice == "A":
            return await _save_cover_letter(console, settings, job, result)

        elif choice == "R":
            console.print("[yellow]Regenerating...[/yellow]")
            new_result = await _generate_cover_letter(
                console,
                settings,
                job,
                resume_data,
                style,
                tone_section,
                tailored_resume_text,
                session,
                attempt=attempt + 1,
            )
            if new_result is None:
                console.print("[red]Generation failed. Please try again.[/red]")
            continue

        elif choice == "I":
            user_instructions = console.input(
                "\n[bold]Instructions (e.g., 'emphasize customer service'): [/bold]"
            ).strip()
            if not user_instructions:
                console.print("[yellow]No instructions provided.[/yellow]")
                continue
            refined_ok = await _refine_cover_letter(
                console,
                settings,
                job,
                result,
                user_instructions,
                session,
                attempt,
                resume_data,
                style,
                tone_section,
            )
            if not refined_ok:
                console.print("[red]Refinement failed. Please try again.[/red]")
                continue
            result = session.current
            continue

        elif choice == "D":
            if len(session.attempts) > 1:
                render_diff(
                    console,
                    session.attempts[0].cover_letter_text,
                    result.cover_letter_text,
                    max_lines=0,
                )
            else:
                console.print("[yellow]Only one attempt so far.[/yellow]")
            continue

        elif choice == "V":
            if len(session.attempts) < 2:
                console.print("[yellow]No previous attempts yet.[/yellow]")
                continue
            hist_table = Table(title="Cover Letter History")
            hist_table.add_column("#", style="dim")
            hist_table.add_column("Attempt")
            hist_table.add_column("Preview", style="dim")
            for i, att in enumerate(session.attempts):
                preview = att.cover_letter_text[:60].replace("\n", " ")
                marker = "\u2192" if i == session.current_index else " "
                hist_table.add_row(marker, str(att.attempt), preview + "...")
            console.print(hist_table)
            sel = console.input(
                "\n[bold cyan]Select attempt # (or Enter back): [/bold cyan]"
            ).strip()
            if sel.isdigit():
                idx = int(sel) - 1
                if 0 <= idx < len(session.attempts):
                    session.select(idx)
                    console.print(f"[green]Switched to attempt #{session.current.attempt}[/green]")
                else:
                    console.print("[red]Invalid attempt number.[/red]")
            continue

        elif choice == "Q":
            console.print("[yellow]Cover letter skipped. Resume already saved.[/yellow]")
            return None

        else:
            console.print("[red]Invalid choice. Please enter A, R, I, D, V, or Q.[/red]")

    return None


@app.command()
def tailor(
    ctx: typer.Context,
    resume_path: str = typer.Option("", "--resume", help="Path to resume file."),
    job_title: str = typer.Option(..., "--job-title", "-t", help="Job title."),
    company: str = typer.Option(..., "--company", "-c", help="Company name."),
    job_description: str = typer.Option("", "--description", "-d", help="Job description."),
    job_url: str = typer.Option("", "--url", help="Job posting URL."),
    requirements: str = typer.Option(
        "", "--requirements", "-r", help="Comma-separated requirements."
    ),
    location: str = typer.Option("", "--location", "-l", help="Job location."),
    style_guide: str = typer.Option("", "--style-guide", help="Style examples."),
    headed: bool = typer.Option(False, "--headed", help="Run browser in headed mode."),
    min_score: float = typer.Option(
        0.0, "--min-score", help="Abort if match score is below this threshold (0.0-1.0)."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip interactive prompts (auto-answer yes)."
    ),
    ocr_mode: str = typer.Option(
        "auto",
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
        from job_applicator.models import JobBoard, JobListing

        if not settings.resume_path:
            console.print("[red]Resume path required. Use --resume.[/red]")
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

        req_list = [r.strip() for r in requirements.split(",") if r.strip()] if requirements else []
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

        style = None
        if settings.style_guide_path:
            from job_applicator.documents.cover_letter import CoverLetterGenerator

            generator = CoverLetterGenerator(settings.llm)
            with console.status("Analyzing writing style..."):
                style = await generator.load_style_guide(settings.style_guide_path)
            console.print(f"[green]Style loaded: {style.tone}[/green]")

        tailor_engine = ResumeTailor(settings.llm)
        attempt = 0
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
                raise typer.Exit(0)

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
            console.print(f"[red]LLM error: {escape(str(exc))}[/red]")
            console.print("[yellow]Could not generate tailored resume.[/yellow]")
            if reporter:
                reporter.record_error(str(exc))
            raise typer.Exit(1) from exc

        while True:
            attempt += 1
            if attempt > 10:
                console.print("[red]Maximum retry limit (10) reached.[/red]")
                break
            if attempt >= 8:
                console.print("[yellow]Warning: approaching retry limit (10 max).[/yellow]")

            console.print(f"\n[bold blue]--- Attempt #{attempt} ---[/bold blue]")

            console.print("\n[bold]Tailored Resume Preview:[/bold]\n")
            console.print(
                Panel(
                    result.tailored_text,
                    title="Tailored Resume",
                    border_style="cyan",
                )
            )
            render_diff(console, session.original_text, result.tailored_text, max_lines=30)

            console.print("\n[bold]Metadata:[/bold]")
            meta_table = Table(show_header=False, box=None)
            meta_table.add_column("Key", style="dim")
            meta_table.add_column("Value")
            meta_table.add_row("Job", f"{job.title} at {job.company}")
            meta_table.add_row("Match Score", f"{result.match_score:.0%}")
            meta_table.add_row(
                "Matched Skills",
                ", ".join(result.matched_skills[:5]) or "—",
            )
            meta_table.add_row(
                "Missing Skills",
                ", ".join(result.missing_skills[:5]) or "—",
            )
            meta_table.add_row("Attempt", str(attempt))
            if result.user_modifications:
                meta_table.add_row("User Input", result.user_modifications)
            console.print(meta_table)

            console.print("\n[bold]Changes Made:[/bold]")
            console.print(result.changes_summary)

            console.print("\n[bold]What would you like to do?[/bold]")
            action_table = Table(show_header=False, box=None)
            action_table.add_column("Option", style="cyan bold")
            action_table.add_column("Description")
            action_table.add_row("[A] Accept", "Save this version as final")
            action_table.add_row("[R] Retry", "Regenerate with same instructions")
            action_table.add_row("[I] Input", "Give custom instructions to refine")
            action_table.add_row("[D] Diff", "Show changes from original resume")
            action_table.add_row("[V] History", "Browse previous attempts")
            action_table.add_row("[S] Section", "Edit a specific section")
            action_table.add_row("[Q] Quit", "Discard and exit")
            console.print(action_table)

            choice = (
                console.input("\n[bold cyan]Your choice (A/R/I/D/V/S/Q): [/bold cyan]")
                .strip()
                .upper()
            )

            if choice == "A":
                from datetime import datetime as dt

                output_dir = await asyncio.to_thread(settings.ensure_output_dir)

                safe_company = job.company.replace(" ", "_").replace("/", "_")
                safe_title = job.title.replace(" ", "_").replace("/", "_")
                timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
                filename = f"tailored_{safe_company}_{safe_title}_{timestamp}.txt"
                output_path = output_dir / filename

                await asyncio.to_thread(
                    output_path.write_text, result.tailored_text, encoding="utf-8"
                )
                result.output_path = str(output_path)

                if reporter:
                    reporter.record_io(files_written=[str(output_path)])

                console.print(f"\n[green]Tailored resume saved: {output_path}[/green]")
                console.print(f"[dim]Attempt #{attempt} | Score: {result.match_score:.0%}[/dim]")

                # Offer cover letter generation
                cover_letter_path = None
                cl_choice = (
                    console.input(
                        f"\n[bold cyan]Generate a matching cover letter "
                        f"for {job.title} at {job.company}? (Y/N): [/bold cyan]"
                    )
                    .strip()
                    .upper()
                )

                if cl_choice == "Y":
                    cover_letter_path = await _cover_letter_workflow(
                        console,
                        settings,
                        job,
                        resume_data,
                        style,
                        tone_profile,
                        result.tailored_text,
                    )

                # Write resume meta.json (with or without cover_letter_path)
                if cover_letter_path:
                    result.cover_letter_path = str(cover_letter_path)
                meta_path = output_path.with_suffix(".meta.json")
                await asyncio.to_thread(
                    meta_path.write_text, result.model_dump_json(indent=2), encoding="utf-8"
                )
                console.print(f"[green]Metadata saved: {meta_path}[/green]")

                break

            elif choice == "R":
                console.print("[yellow]Regenerating...[/yellow]")
                user_instructions = ""
                refined: TailoredResume | None = await _llm_with_retry(
                    console,
                    partial(
                        tailor_engine.refine,
                        resume_data,
                        result,
                        "",
                        job,
                        tone_profile=tone_profile,
                    ),
                    "Tailoring resume...",
                )
                if refined is None:
                    break
                result = refined
                result.attempt = attempt
                session.add_attempt(result)
                continue

            elif choice == "I":
                user_instructions = console.input(
                    "\n[bold]Enter your instructions (e.g., 'emphasize "
                    "customer service', 'add troubleshooting detail'): "
                    "[/bold]"
                ).strip()
                if not user_instructions:
                    console.print("[yellow]No instructions provided, retrying.[/yellow]")
                refined = await _llm_with_retry(
                    console,
                    partial(
                        tailor_engine.refine,
                        resume_data,
                        result,
                        user_instructions,
                        job,
                        tone_profile=tone_profile,
                    ),
                    "Tailoring resume...",
                )
                if refined is None:
                    break
                result = refined
                result.attempt = attempt
                session.add_attempt(result)
                continue

            elif choice == "D":
                render_diff(console, session.original_text, result.tailored_text, max_lines=0)
                continue

            elif choice == "V":
                if len(session.attempts) < 2:
                    console.print("[yellow]No previous attempts yet.[/yellow]")
                    continue
                hist_table = Table(title="Version History")
                hist_table.add_column("#", style="dim")
                hist_table.add_column("Attempt")
                hist_table.add_column("Score", style="cyan")
                hist_table.add_column("Instructions")
                hist_table.add_column("Preview", style="dim")
                for i, att in enumerate(session.attempts):
                    preview = att.tailored_text[:60].replace("\n", " ")
                    marker = "\u2192" if i == session.current_index else " "
                    hist_table.add_row(
                        marker,
                        str(att.attempt),
                        f"{att.match_score:.0%}",
                        att.user_modifications or "\u2014",
                        preview + "...",
                    )
                console.print(hist_table)
                sel = console.input(
                    "\n[bold cyan]Select attempt # to view (or Enter to go back): [/bold cyan]"
                ).strip()
                if sel.isdigit():
                    idx = int(sel) - 1
                    if 0 <= idx < len(session.attempts):
                        session.select(idx)
                        result = session.current
                        console.print(f"[green]Switched to attempt #{result.attempt}[/green]")
                    else:
                        console.print("[red]Invalid attempt number.[/red]")
                continue

            elif choice == "S":
                from job_applicator.documents.resume_tailor import parse_sections

                sections = parse_sections(result.tailored_text)
                if len(sections) <= 1 and sections[0].name == "Full Document":
                    console.print(
                        "[yellow]Could not detect sections. "
                        "Use [I] for full-resume instructions.[/yellow]"
                    )
                    continue

                console.print("\n[bold]Sections:[/bold]")
                sec_table = Table(show_header=False, box=None)
                sec_table.add_column("#", style="cyan")
                sec_table.add_column("Section", style="bold")
                sec_table.add_column("Lines", style="dim")
                for i, sec in enumerate(sections, 1):
                    line_count = sec.text.count("\n") + 1
                    sec_table.add_row(str(i), sec.name, f"{line_count} lines")
                console.print(sec_table)

                sec_choice = console.input(
                    "\n[bold cyan]Section # to edit (or Enter to go back): [/bold cyan]"
                ).strip()
                if not sec_choice.isdigit():
                    continue
                sec_idx = int(sec_choice) - 1
                if sec_idx < 0 or sec_idx >= len(sections):
                    console.print("[red]Invalid section number.[/red]")
                    continue

                target_section = sections[sec_idx]
                console.print(f"\n[dim]Editing: {target_section.name}[/dim]")
                console.print(f"[dim]{target_section.text[:200]}...[/dim]\n")

                sec_instructions = console.input(
                    "[bold]Instructions for this section: [/bold]"
                ).strip()
                if not sec_instructions:
                    console.print("[yellow]No instructions provided.[/yellow]")
                    continue

                user_instructions = (
                    f"ONLY modify the {target_section.name} section. "
                    f"Keep all other sections unchanged.\n\n"
                    f"Current {target_section.name} content:\n{target_section.text}\n\n"
                    f"User instructions for this section: {sec_instructions}"
                )
                refined = await _llm_with_retry(
                    console,
                    partial(
                        tailor_engine.refine,
                        resume_data,
                        result,
                        user_instructions,
                        job,
                        tone_profile=tone_profile,
                    ),
                    "Refining section...",
                )
                if refined is None:
                    break
                result = refined
                result.attempt = attempt
                session.add_attempt(result)
                continue

            elif choice == "Q":
                console.print("[yellow]Discarded. No changes saved.[/yellow]")
                break

            else:
                console.print("[red]Invalid choice. Please enter A, R, I, D, V, S, or Q.[/red]")

    try:
        asyncio.run(_run())
    except JobApplicatorError as exc:
        # Typed, expected failures (no session, anti-bot block, missing resume)
        # — show the message cleanly instead of a raw Python traceback.
        if reporter:
            reporter.record_error(str(exc))
        console.print(f"[yellow]⚠ {escape(str(exc))}[/yellow]")
        raise typer.Exit(1) from exc
    except Exception as exc:
        if reporter:
            reporter.record_error(str(exc))
        raise
    finally:
        if reporter:
            report_log_file = ctx.obj.log_file if isinstance(ctx.obj, VerboseContext) else None
            reporter.render(console, report_log_file)


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
    ocr_mode: str = typer.Option(
        "auto",
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
            reporter.render(console, log_file=None)
        console.print("[red]Resume path required. Use --resume.[/red]")
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
            reporter.render(console, log_file=log_file)


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
    if config_path.exists():
        console.print("[yellow]config.toml already exists. Skipping.[/yellow]")
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
            reporter.render(console, log_file=log_file)


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
    verbose: bool = _verbose_option(),
    log_file: str | None = _log_file_option(),
) -> None:
    """Check the AI backend, browser, system binaries, and config."""
    _merge_verbose_ctx(ctx, verbose, log_file)
    from job_applicator.diagnostics import run_diagnostics

    settings = _get_settings()
    report = asyncio.run(run_diagnostics(settings))
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
        console.print(
            f"  Browser        {good} Playwright + Chromium  "
            f"[dim]{escape(str(browser.chromium_executable))}[/dim]"
        )
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


def _scraper_class(site: str) -> type[BaseScraper]:
    """Resolve a board's scraper class, or exit if unsupported.

    Site validation lives here so it happens BEFORE any browser is launched.
    """
    if site == "linkedin":
        from job_applicator.scrapers.linkedin import LinkedInScraper

        return LinkedInScraper
    if site == "indeed":
        from job_applicator.scrapers.indeed import IndeedScraper

        return IndeedScraper
    console.print(f"[yellow]{site} scraper not yet implemented[/yellow]")
    raise typer.Exit(1)


def _make_browser(site: str, settings: AppSettings) -> BrowserManager:
    """Build a browser per the board's declared ``BrowserPolicy``.

    The policy lives on the scraper class (not here), so a board's anti-bot needs
    can't drift from the CLI and every caller building a browser gets them right.
    Indeed declares headed + ephemeral-profile + virtual-display (its Cloudflare
    managed challenge fails headless); LinkedIn keeps the default headless shared
    profile. ``--headed`` (config headless=False) shows a real window instead of a
    virtual one. Validates the site before constructing anything (so an unknown
    board never launches a browser).
    """
    from job_applicator.browser.manager import BrowserManager

    policy = _scraper_class(site).browser_policy()
    cfg = settings.browser
    if policy.headed:
        cfg = cfg.model_copy(update={"headless": False})
    # Use a virtual display only when forcing headed AND the user didn't ask to
    # watch (--headed leaves config headless=False → show a real window).
    use_virtual = policy.virtual_display and settings.browser.headless
    return BrowserManager(
        cfg, ephemeral_profile=policy.ephemeral_profile, virtual_display=use_virtual
    )


def _make_scraper(site: str, browser: BrowserManager, settings: AppSettings) -> BaseScraper:
    """Construct the scraper for a job board, or exit if unsupported."""
    return _scraper_class(site)(browser, settings)


def _make_applicator(site: str, browser: BrowserManager, settings: AppSettings) -> BaseApplicator:
    """Construct the applicator for a job board, or exit if unsupported."""
    if site == "linkedin":
        from job_applicator.applicators.linkedin import LinkedInApplicator

        return LinkedInApplicator(browser, settings)
    if site == "indeed":
        from job_applicator.applicators.indeed import IndeedApplicator

        return IndeedApplicator(browser, settings)
    console.print(f"[yellow]{site} applicator not yet implemented[/yellow]")
    raise typer.Exit(1)


def _load_user_profile(settings: AppSettings) -> UserProfile:
    """Load user profile from settings."""
    name_parts = settings.profile_name.split() if settings.profile_name else ["User"]
    return UserProfile(
        first_name=name_parts[0],
        last_name=name_parts[-1] if len(name_parts) > 1 else "",
        email=settings.target.linkedin_email,
        phone="",
        resume_path=settings.resume_path,
    )
