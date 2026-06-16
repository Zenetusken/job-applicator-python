"""Minimal local type stub for pyvirtualdisplay (ships no py.typed).

Covers only the surface BrowserManager uses: constructing a virtual display and
start()/stop(). Kept here (on mypy_path) so the dependency is properly typed
rather than silenced with an ignore.
"""

from types import TracebackType

class Display:
    def __init__(
        self,
        backend: str | None = ...,
        visible: bool = ...,
        size: tuple[int, int] = ...,
        color_depth: int = ...,
        use_xauth: bool = ...,
    ) -> None: ...
    def start(self) -> Display: ...
    def stop(self) -> Display: ...
    def __enter__(self) -> Display: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...
