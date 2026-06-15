---
description: "Run full verification suite: tests, lint, format check, typecheck. Use before commits and PRs."
---

# Verify

Run the full verification suite. Exits on first failure.

## Steps

### 1. Unit tests
```bash
cd {project_root} && .venv/bin/pytest tests/unit/ -q
```

### 2. Lint
```bash
cd {project_root} && ruff check src/ tests/
```

### 3. Format check
```bash
cd {project_root} && ruff format --check src/ tests/
```

### 4. Typecheck
```bash
cd {project_root} && .venv/bin/mypy src/job_applicator/ --ignore-missing-imports
```

### 5. Report
- Test count and pass/fail
- Lint status
- Format status
- Mypy status
- Overall: PASS or FAIL with specific failure details
