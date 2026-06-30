"""The CLI volume options are clamped (a typo guard): `search --max`, `apply --limit`, and
`batch --top-k` reject out-of-range values via typer's min/max. The rejection happens at PARSE
time — before the command body — so a mistyped `--max 1000` can't kick off an oversized run, and
these tests never launch a browser or touch the real account (the body is never reached)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

import job_applicator.cli as cli


@pytest.mark.parametrize(
    "args",
    [
        ["search", "-q", "x", "--max", "1000"],
        ["apply", "-q", "x", "--limit", "1000"],
        ["batch", "--jobs-file", "x.json", "--top-k", "1000"],
    ],
)
def test_volume_option_rejects_out_of_range(args: list[str]) -> None:
    """An out-of-range volume value is rejected at parse time (exit 2 = usage error), not run."""
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 2  # click usage error, NOT a body/browser failure (which is exit 1)
    assert "is not in the range" in (result.output + (result.stderr or ""))


@pytest.mark.parametrize(
    "args",
    [
        ["search", "-q", "x", "--max", "0"],
        ["apply", "-q", "x", "--limit", "0"],
        ["batch", "--jobs-file", "x.json", "--top-k", "0"],
    ],
)
def test_volume_option_rejects_zero_or_negative(args: list[str]) -> None:
    """min=1 also rejects 0 (and negatives) — a positive-int guard, rejected before the body."""
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 2
    assert "is not in the range" in (result.output + (result.stderr or ""))
