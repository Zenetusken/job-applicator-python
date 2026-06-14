---
description: "Full doc sync — update README.md, AGENTS.md, and MEMORY.md with current project state."
---

# Doc Sync

Synchronize all documentation with the current codebase state.

## Steps

### 1. Gather Current State
```bash
cd {project_root}
TEST_COUNT=$(pytest tests/unit/ -q 2>&1 | grep -oP '\d+ passed' | grep -oP '\d+')
COMMIT_COUNT=$(git log --oneline | wc -l)
FILES=$(find src/job_applicator/ -name '*.py' -not -name '__init__.py' -not -name '__main__.py' | wc -l)
```

### 2. Update AGENTS.md
- Verify test count matches actual
- Verify architecture tree matches actual files
- Verify CLI commands listed match actual `typer` commands
- Verify gotchas are current (remove fixed ones, add new ones)
- Verify config sub-configs listed match actual

### 3. Update README.md
- Verify features listed match actual implementation
- Verify CLI commands match actual
- Verify installation instructions are correct
- Verify configuration examples match config.py

### 4. Update MEMORY.md
- Update test count, commit count, file count
- Add any new discoveries from recent sessions
- Remove stale entries (fixed bugs, resolved gotchas)
- Update timestamp

### 5. Verify
```bash
ruff check src/ tests/
ruff format --check src/ tests/
pytest tests/unit/ -q
```

### 6. Report
- List all files changed
- List specific updates made
- Confirm tests still pass
