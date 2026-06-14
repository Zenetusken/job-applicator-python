---
description: "Push to main + full memory/doc sync. Run after commits are ready on main."
---

# Ship: Push + Sync

Push all local commits to origin/main, then perform a full memory and documentation sync.

## Steps

1. **Verify tests pass**
   ```bash
   cd {project_root} && .venv/bin/python -m pytest tests/unit/ -q
   ```

2. **Verify lint/format/mypy clean**
   ```bash
   ruff check src/ tests/
   ruff format --check src/ tests/
   mypy src/job_applicator/ --ignore-missing-imports
   ```

3. **Push to origin/main**
   ```bash
   git push origin main
   ```

4. **Memory sync** — Read MEMORY.md and notes.md. Update with:
   - Any new discoveries from this session
   - New gotchas encountered
   - Updated test count, commit count, or stats
   - New architecture decisions
   - Timestamp update

5. **Doc sync** — Verify AGENTS.md accuracy:
   - Test count matches actual (`pytest tests/unit/ -q`)
   - Architecture tree matches actual files
   - Gotchas are current
   - CLI commands listed match actual

6. **Report** — Summarize what was pushed and synced:
   - Commits pushed
   - Test count
   - MEMORY.md changes (if any)
   - AGENTS.md changes (if any)
