# Indeed / Cloudflare — empirical research findings

_Date: 2026-06-15 · Status: research only (no implementation in this pass)_

## TL;DR

The Indeed wall is a Cloudflare **managed JS challenge** (`cf-mitigated: challenge`),
**not** a rate-limit and **not** TLS/JA3 fingerprinting. After fully isolating every
variable, the fix is far cheaper than the assumed "fingerprint-resistant engine":

> **Run Indeed headed (a real/virtual display) with a clean browser profile.**
> The existing Playwright + playwright-stealth stack then passes and returns real jobs —
> with **no patchright, no `channel="chrome"`, no `curl_cffi`, no cookie import, and zero
> new dependencies.**

patchright/channel=chrome were tested and work too, but the controls prove they are
**not necessary** — they were not the differentiator. The two things that matter are
**headed mode** (mandatory) and a **non-poisoned profile**.

## Question

Two theories were on the table for why Indeed scraping failed:
1. **Rate-limit** — a cooldown + warm session would clear it.
2. **TLS/JA3 fingerprinting** — bundled Chromium's TLS differs from Chrome; fix would be
   `curl_cffi` (TLS impersonation) or a fingerprint-resistant engine (patchright/nodriver).

Both were tested on neutral endpoints + Indeed before writing code. **Both are refuted.**

## Environment

- Host: Linux, display `:1` present (`xvfb-run` also available).
- Real Google **Chrome 149.0.7827.114**; Playwright bundled Chromium **149** (version-matched).
- **patchright 1.60.1** already installed (transitive; undeclared). `curl_cffi`: not installed/needed.
- Imported Indeed session: 91 cookies incl. `cf_clearance` (valid to 2027).

## Experiment 1 — TLS fingerprint, neutral endpoint (0 Indeed hits)

JA3/JA4 via `https://tls.peet.ws/api/all` across four configs:

| Config | JA4 (stable comparator) |
|---|---|
| Playwright bundled Chromium | `t13d1516h2_8daaf6152771_d8a2da3f94cd` |
| Playwright `channel=chrome` (real Chrome 149) | `t13d1516h2_8daaf6152771_d8a2da3f94cd` |
| Patchright bundled Chromium | `t13d1516h2_8daaf6152771_d8a2da3f94cd` |
| Patchright `channel=chrome` | `t13d1516h2_8daaf6152771_d8a2da3f94cd` |

**All JA4 identical**, and identical to real Chrome. (`ja3_hash` differed run-to-run —
expected: Chrome 110+ randomises ClientHello extension order per connection; JA4 sorts
them and is the right comparator.) **TLS/JA3 is not the wall** → `curl_cffi` and
`channel="chrome"`-as-a-TLS-fix are ruled out.

## Experiment 2 — single Indeed diagnostic (1 hit)

App's real stack, capturing the response:

```
HTTP 403 · cf-mitigated: challenge · cf-ray …-YUL (Montreal) · title 'Just a moment...'
```

Not a `429`/`retry-after` → **not rate-limit; waiting doesn't help.** It's an active
Cloudflare **managed JS challenge**, rejecting the replayed `cf_clearance`.

## Experiment 3 — full isolation matrix vs Indeed

Each row a real Indeed hit. "clean" = fresh temp profile, no stealth, no
`--disable-blink-features=AutomationControlled` arg.

| engine | binary | mode | profile | stealth | arg | cookies | result |
|---|---|---|---|---|---|---|---|
| playwright | bundled | headless | app-persistent | ✓ | ✓ | warm | **403 challenge** |
| playwright | bundled | **headed** | app-persistent | ✓ | ✓ | warm | **403 challenge** (control) |
| patchright | chrome | headless | fresh | – | – | warm | 403 block |
| patchright | chrome | **headed** | fresh | – | – | warm | **200 — 16 jobs ✅** |
| patchright | chrome | **headed** | fresh | – | – | cold | **200 — 16 jobs ✅** |
| playwright | chrome | **headed** | fresh | – | – | warm | **200 — 16 jobs ✅** |
| patchright | bundled | **headed** | fresh | – | – | warm | **200 — 16 jobs ✅** |
| playwright | bundled | **headed** | fresh | – | – | warm | **200 — 16 jobs ✅** |
| playwright | bundled | **headed** | fresh | – | – | cold | **200 — 16 jobs ✅** |
| playwright | bundled | **headed** | fresh | – | ✓ | warm | **200 — 16 jobs ✅** |
| playwright | bundled | **headed** | fresh | ✓ | – | warm | **200 — 16 jobs ✅** |
| **real app code path** (stealth+arg+UA) | bundled | **headed** | fresh | ✓ | ✓ | – | **200 — 16 jobs ✅** |
| playwright | bundled | headless | fresh | – | – | warm | 403 block |
| playwright | chrome | headless | fresh | – | – | warm | 403 block |
| playwright | bundled | **headed via `xvfb-run`** | fresh | – | – | warm | **200 — 16 jobs ✅** |

### What the matrix isolates

- **`headed` is the dominant, mandatory, *causal* variable.** Every headless arm fails
  (5/5); every headed arm with a clean profile passes (10/10). The two clinching probes —
  `headless + fresh + clean` (bundled and chrome) — were run **last, after all the headed
  passes**, in the same late time window with identical profile/args; only headed→headless
  changed, and they failed. That late, interleaved A/B rules out "the IP simply cooled over
  the session" and makes `headed` genuinely causal, not a temporal artifact.
- **A clean profile is required; the existing shared profile failed.** The only headed
  failure used `~/.job-applicator/browser-profile/`; the **real app code path** — same
  stealth + arg + UA — **passes** once given headed + a fresh profile (16 live jobs). Note:
  that fail/pass pair is confounded with timing (the persistent-profile run was early), so
  this is an *observation* ("a clean profile reliably passes; the existing one failed once"),
  not a proven poisoning mechanism. Either way, using a clean profile is the fix.
- **A virtual display works:** `xvfb-run` (no real monitor) also passed — so headless
  *servers* can run this via Xvfb, not just desktops with a real display.
- **Not the differentiator (pass with or without):** patchright, `channel="chrome"`, the
  `AutomationControlled` arg, playwright-stealth, and imported cookies (passes cold).

## Diagnosis

The block is a Cloudflare **managed JS challenge** at the browser-environment layer. It is
defeated simply by presenting a real (headed) browser session from a clean profile — the
existing engine already does this once those two conditions hold. The earlier failures came
from (a) running headless and (b) reusing a profile that had been stuck in challenge loops.

## Options, ranked

1. **Headed + clean profile on the existing stack — RECOMMENDED. Zero new deps.**
   - Empirically passes via the real app code path (16 jobs).
   - Requirements to design around:
     - **Headed is mandatory** → needs a display; on a headless box wrap in `xvfb-run`
       (confirmed working here) / Xvfb. Indeed scraping cannot run silently in pure-headless mode.
     - **Use a clean/dedicated profile for Indeed** (the shared persistent profile failed
       in testing). Either a separate Indeed profile dir or a reset; a fresh profile each run
       also works and sidesteps the issue entirely.
     - LinkedIn is unaffected — it uses the persistent session headlessly and keeps working.
       Scope the headed + clean-profile behaviour to the Indeed path only.
     - Cookie import for Indeed becomes **optional** (passed cold). Keep as a warm-start nicety.
     - Not a guarantee — Cloudflare difficulty varies; keep `ScraperError` surfacing + retry.
2. **patchright (channel=chrome), headed** — also passes; keep as a documented fallback if
   the plain stack regresses. Drop-in async-Playwright API, already installed (would need
   declaring). Not needed today.
3. **nodriver** (undetected-chromedriver successor) — strongest per public benchmarks but a
   larger non-Playwright rewrite. Reserve for a future hard regression only.
4. **`curl_cffi`** — **ruled out** (TLS already matches; runs no JS → can't solve a challenge).
5. **External solver/proxy** — works, costs money, sends queries to a third party; out of scope.

## Recommendation

Implement **Option 1**: run the Indeed scraper **headed** (with `xvfb-run`/Xvfb fallback on
headless hosts) using a **clean, Indeed-dedicated profile**, on the existing Playwright stack.
No new dependency. Keep cookie import optional and treat a challenge as a recoverable
`ScraperError`. This is materially cheaper than the assumed patchright/engine path, which the
controls proved unnecessary.

> **Before shipping (reliability check):** the 10 headed passes were spread across *different*
> configs. During implementation, run the **exact** chosen config (plain stack, headed, clean
> profile) ~3× in a row to confirm same-config stability — a managed challenge that's 10/10
> across variants is very likely stable, but the "reliable" bar wants the same-config repeat.

## Secondary finding (separate issue)

The passing runs returned **US** jobs for a `Montreal, QC` query (they hit `www.indeed.com`
and Indeed served US results without redirecting). This is the known region behaviour, not
the Cloudflare wall — pin `target.indeed_domain = "ca.indeed.com"` (or strengthen the region
auto-redirect) when implementing. Track separately from the anti-bot fix.

## Reproduction & cost

Throwaway probes (not committed to `src/`): `/tmp/tls_probe.py` (Exp 1),
`/tmp/indeed_probe.py` (Exp 2 + headed control), `/tmp/patchright_probe.py`,
`/tmp/engine_matrix.py`, `/tmp/app_fresh_probe.py` (Exp 3). ~14 Indeed hits total — all
either `200` or a `403` managed challenge; **no rate-limit (`429`/`retry-after`) was observed
during this research.** (This rules out rate-limiting as the *current* wall; a prior session's
block may have been a real 429 that has since cleared — not claimed here either way.)

## ToS note

Automated Indeed scraping carries the same inherent ToS risk as LinkedIn. For the user's own
low-volume job search; prefer the public search surface and keep volume low.
