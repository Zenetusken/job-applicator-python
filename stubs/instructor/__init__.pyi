from typing import Any, Awaitable, Callable, overload

class Mode:
    TOOLS: str

class Instructor:
    pass

class AsyncInstructor:
    pass

@overload
def from_litellm(
    completion: Callable[..., Awaitable[Any]],
    mode: Mode = ...,
    **kwargs: Any,
) -> AsyncInstructor: ...
@overload
def from_litellm(
    completion: Callable[..., Any],
    mode: Mode = ...,
    **kwargs: Any,
) -> Instructor: ...
