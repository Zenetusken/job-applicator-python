---
description: "Run a dream memory consolidation pass. Reads memory files + trajectory DB, verifies claims, updates MEMORY.md."
---

# Dream: Memory Consolidation

Perform a manual dream memory consolidation pass for the current project.

## Data Sources

Use available evidence in this order:
1. Recent mimocode sessions and their assistant work (trajectory database)
2. Memory files (project MEMORY.md, session checkpoint.md, notes.md)
3. Existing skills, agents, commands

Trajectory database: `<DATA>/mimocode.db` (SQLite, read-only)
Memory files root: `<DATA>/memory/`

## Steps

### Phase 0 — Locate Data
1. Find `<DATA>/mimocode.db` from the resolved memory root
2. List all memory files under `<DATA>/memory/`
3. If no recent activity and memory is empty, report "Nothing to consolidate" and stop

### Phase 1 — Orient
1. Read project MEMORY.md fully
2. Read session checkpoint.md and notes.md
3. Query trajectory DB for recent sessions and message counts

### Phase 2 — Gather From Memory Files
1. Scan checkpoint.md for recurring task shapes
2. Scan notes.md for unresolved questions and cross-session observations
3. Scan MEMORY.md for stale entries that need updating

### Phase 3 — Verify Against Raw Trajectory
1. Query trajectory DB for user statements containing rules, decisions, keywords
2. Verify each MEMORY.md claim against actual codebase (grep, file existence)
3. Check git log for commits not reflected in memory

### Phase 4 — Consolidate
1. Add newly verified facts to appropriate MEMORY.md sections
2. Correct any stale entries found in Phase 3
3. Keep under 200 lines / 10KB — prune low-value entries when adding new ones
4. Add session citation `[ses_xxx]` to new entries

### Phase 5 — Prune And Verify
1. Re-read MEMORY.md — check for duplicates, stale references, broken rules
2. Verify all referenced file paths and function names exist in codebase
3. Remove entries that are no longer relevant

## Output Format

Return a brief summary:
- **Consolidated**: what was added/updated
- **Deleted**: what was removed (and why)
- **Health**: MEMORY.md line count, byte count, any issues
