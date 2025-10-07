from __future__ import annotations

from typing import Any, Callable, Tuple


def safe_component(
    factory: Callable[..., Any],
    *args: Any,
    optional_keys: Tuple[str, ...] = ("live",),
    **kwargs: Any,
) -> Any:
    """Instantiate a Gradio component, ignoring unsupported optional kwargs."""

    attempt_kwargs = dict(kwargs)
    while True:
        try:
            return factory(*args, **attempt_kwargs)
        except TypeError as exc:
            message = str(exc)
            removed = False
            for key in optional_keys:
                if key in attempt_kwargs and f"'{key}'" in message:
                    attempt_kwargs.pop(key)
                    removed = True
                    break
            if not removed:
                raise


__all__ = ["safe_component"]
