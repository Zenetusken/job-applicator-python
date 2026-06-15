"""Secure on-disk storage for session secrets (e.g. auth cookies)."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any


def write_secret_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write ``payload`` as JSON to ``path`` with owner-only perms.

    Hardened for credential material (e.g. a LinkedIn ``li_at`` session token):

    - the parent directory is created and forced to ``0700``;
    - the file is created ``0600`` from the start via :func:`tempfile.mkstemp`
      (which ignores the umask), so there is no world-readable window between
      ``create`` and ``chmod``;
    - content is written to a temp file in the same directory and atomically
      :func:`os.replace`-d into place — a crash never leaves a half-written
      token, a pre-existing symlink *at* ``path`` is replaced (not followed),
      and a symlinked *parent* directory is refused outright, so the token write
      cannot be redirected through a planted symlink.
    """
    parent = path.parent
    if parent.is_symlink():
        raise OSError(f"refusing to write a secret into a symlinked directory: {parent}")
    parent.mkdir(parents=True, exist_ok=True)
    parent.chmod(0o700)
    fd, tmp = tempfile.mkstemp(dir=parent, prefix=".cookies-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, path)  # atomic; the file is already 0600 from mkstemp
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise
