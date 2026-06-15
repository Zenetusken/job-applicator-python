# Cookie-Based Authentication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use compose:subagent (recommended) or compose:execute to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add cookie-based authentication to the LinkedIn scraper so users can save a logged-in session and reuse it across runs, avoiding repeated CAPTCHA challenges.

**Architecture:** Store cookies in a JSON file at `~/.job-applicator/cookies/linkedin.json`. After a successful login, save the browser context's cookies. On subsequent runs, load cookies into the persistent context before navigating to LinkedIn. If cookies are valid, skip the login form entirely. Add a `search --save-cookies` flag to force a fresh login and save new cookies.

**Tech Stack:** Playwright cookie API (`context.cookies()`, `context.add_cookies()`), Python `pathlib`, `json`.

---

## File Structure

| File | Role |
|------|------|
| `src/job_applicator/scrapers/linkedin.py` | Cookie save/load logic in `login()` and `scrape()` |
| `src/job_applicator/browser/manager.py` | Cookie persistence hooks in `persistent_context()` |
| `src/job_applicator/cli.py` | `--save-cookies` flag on `search` command |
| `tests/unit/test_scrapers.py` | Cookie save/load tests |

---

### Task 1: Cookie Storage Path Utility

**Files:**
- Modify: `src/job_applicator/scrapers/linkedin.py`

- [ ] **Step 1: Add cookie path property**

Add a `COOKIE_PATH` class constant and a `_cookie_file` property to `LinkedInScraper`:

```python
import json
from pathlib import Path

# ... existing imports ...

class LinkedInScraper(BaseScraper):
    """Scrapes job listings from LinkedIn."""

    COOKIE_PATH = Path.home() / ".job-applicator" / "cookies" / "linkedin.json"

    def __init__(self, browser: BrowserManager, config: AppSettings) -> None:
        self._browser = browser
        self._config = config
        self._logged_in = False

    @property
    def _cookie_file(self) -> Path:
        return self.COOKIE_PATH
```

- [ ] **Step 2: Commit**

```bash
git add src/job_applicator/scrapers/linkedin.py
git commit -m "feat: add cookie storage path for LinkedIn scraper"
```

---

### Task 2: Cookie Save/Load Methods

**Files:**
- Modify: `src/job_applicator/scrapers/linkedin.py`

- [ ] **Step 1: Add `_load_cookies()` method**

```python
    async def _load_cookies(self, context: BrowserContext) -> bool:
        """Load saved cookies into the browser context.

        Returns True if cookies were loaded and appear valid.
        """
        if not self._cookie_file.exists():
            return False
        try:
            data = json.loads(self._cookie_file.read_text())
            cookies = data.get("cookies", [])
            if not cookies:
                return False
            await context.add_cookies(cookies)
            logger.info("Loaded %d cookies from %s", len(cookies), self._cookie_file)
            return True
        except Exception as exc:
            logger.warning("Failed to load cookies: %s", exc)
            return False
```

- [ ] **Step 2: Add `_save_cookies()` method**

```python
    async def _save_cookies(self, context: BrowserContext) -> None:
        """Save current browser cookies to disk."""
        try:
            cookies = await context.cookies()
            self._cookie_file.parent.mkdir(parents=True, exist_ok=True)
            self._cookie_file.write_text(
                json.dumps({"cookies": cookies}, indent=2)
            )
            logger.info("Saved %d cookies to %s", len(cookies), self._cookie_file)
        except Exception as exc:
            logger.warning("Failed to save cookies: %s", exc)
```

- [ ] **Step 3: Commit**

```bash
git add src/job_applicator/scrapers/linkedin.py
git commit -m "feat: add cookie save/load methods for LinkedIn"
```

---

### Task 3: Integrate Cookies into Login Flow

**Files:**
- Modify: `src/job_applicator/scrapers/linkedin.py`

- [ ] **Step 1: Modify `login()` to save cookies after success**

After the login success check, add:

```python
            if "feed" in current_url or "mynetwork" in current_url or "jobs" in current_url:
                self._logged_in = True
                logger.info("LinkedIn login successful (redirected to %s)", current_url)
                # Save cookies for future runs
                await self._save_cookies(context)
                return True
```

- [ ] **Step 2: Modify `scrape()` to try cookies first**

Before the login check, add cookie loading:

```python
    async def scrape(self, params: SearchParams) -> list[JobListing]:
        """Scrape LinkedIn job listings."""
        jobs: list[JobListing] = []
        context = await self._get_context()

        # Try cookie-based auth first
        if not self._logged_in:
            cookies_loaded = await self._load_cookies(context)
            if cookies_loaded:
                # Verify cookies are still valid by checking a LinkedIn page
                page = await context.new_page()
                try:
                    await page.goto(f"{LINKEDIN_BASE}/feed", wait_until="domcontentloaded", timeout=10_000)
                    await random_delay(1.0, 2.0)
                    if "feed" in page.url or "mynetwork" in page.url:
                        self._logged_in = True
                        logger.info("Cookie-based login successful")
                    else:
                        logger.info("Cookies expired, will try password login")
                except Exception:
                    logger.info("Cookie validation failed, will try password login")
                finally:
                    await page.close()

        if not self._logged_in:
            # ... existing login code ...
```

- [ ] **Step 3: Commit**

```bash
git add src/job_applicator/scrapers/linkedin.py
git commit -m "feat: integrate cookies into LinkedIn login flow"
```

---

### Task 4: Add `--save-cookies` CLI Flag

**Files:**
- Modify: `src/job_applicator/cli.py`

- [ ] **Step 1: Add flag to `search` command**

Find the `search` command signature and add:

```python
save_cookies: bool = typer.Option(
    False,
    "--save-cookies",
    help="Force a fresh login and save new cookies for future runs",
),
```

- [ ] **Step 2: Pass flag to scraper**

In the `search` command, pass `save_cookies` to the scraper. Add a `save_cookies` parameter to `LinkedInScraper.__init__` or handle it in the CLI:

```python
if site == "linkedin":
    from job_applicator.scrapers.linkedin import LinkedInScraper
    scraper = LinkedInScraper(browser, settings)
    if save_cookies:
        # Delete existing cookies to force fresh login
        scraper._cookie_file.unlink(missing_ok=True)
        logger.info("Cookie cache cleared, will perform fresh login")
```

- [ ] **Step 3: Commit**

```bash
git add src/job_applicator/cli.py
git commit -m "feat: add --save-cookies flag to search command"
```

---

### Task 5: Tests

**Files:**
- Modify: `tests/unit/test_scrapers.py`

- [ ] **Step 1: Test cookie save/load**

```python
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from job_applicator.scrapers.linkedin import LinkedInScraper


@pytest.mark.asyncio
async def test_load_cookies_success(mock_browser_manager, mock_settings):
    scraper = LinkedInScraper(mock_browser_manager, mock_settings)
    mock_context = AsyncMock()
    mock_context.add_cookies = AsyncMock()

    # Create a fake cookie file
    cookie_data = {"cookies": [{"name": "li_at", "value": "test", "domain": ".linkedin.com"}]}
    scraper._cookie_file.parent.mkdir(parents=True, exist_ok=True)
    scraper._cookie_file.write_text(json.dumps(cookie_data))

    result = await scraper._load_cookies(mock_context)
    assert result is True
    mock_context.add_cookies.assert_called_once()

    # Cleanup
    scraper._cookie_file.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_save_cookies(mock_browser_manager, mock_settings):
    scraper = LinkedInScraper(mock_browser_manager, mock_settings)
    mock_context = AsyncMock()
    mock_context.cookies = AsyncMock(return_value=[{"name": "li_at", "value": "test"}])

    await scraper._save_cookies(mock_context)

    assert scraper._cookie_file.exists()
    data = json.loads(scraper._cookie_file.read_text())
    assert data["cookies"] == [{"name": "li_at", "value": "test"}]

    # Cleanup
    scraper._cookie_file.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_load_cookies_missing_file(mock_browser_manager, mock_settings):
    scraper = LinkedInScraper(mock_browser_manager, mock_settings)
    mock_context = AsyncMock()

    # Ensure file doesn't exist
    scraper._cookie_file.unlink(missing_ok=True)

    result = await scraper._load_cookies(mock_context)
    assert result is False
```

- [ ] **Step 2: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_scrapers.py -v
```

Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_scrapers.py
git commit -m "test: add cookie save/load tests for LinkedIn scraper"
```

---

### Task 6: Verification

- [ ] **Step 1: Run unit tests**

```bash
.venv/bin/python -m pytest tests/unit/ -q
```

Expected: All tests pass.

- [ ] **Step 2: Run lint/format**

```bash
ruff check src/ tests/
ruff format --check src/ tests/
```

Expected: Clean.

- [ ] **Step 3: Run typecheck**

```bash
mypy src/job_applicator/ --ignore-missing-imports
```

Expected: Clean.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: verify cookie auth feature passes all checks"
```

---

## Self-Review Checklist

- [ ] Spec coverage: All sections covered by tasks.
- [ ] Placeholder scan: No TBDs, TODOs, or vague steps.
- [ ] Type consistency: `LinkedInScraper` methods use consistent signatures.
- [ ] Cookie path: `~/.job-applicator/cookies/linkedin.json` is consistent across all tasks.
- [ ] Error handling: Cookie load/save failures are logged but don't crash the scraper.
- [ ] Backward compatibility: If no cookies exist, falls back to password login.
