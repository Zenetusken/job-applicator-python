#!/usr/bin/env bash
# Agent wrapper for the canonical project green gate.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "${ROOT}" ]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
fi

exec "${ROOT}/scripts/green_gate.sh"
