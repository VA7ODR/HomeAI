from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Tuple


class ToolRegistry:
    """Simple registry mapping tool names to callables with keyword args."""

    def __init__(self) -> None:
        self.tools: Dict[str, Callable[..., Any]] = {}
        self.aliases: Dict[str, str] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        self.tools[name] = fn

    def alias(self, alias_name: str, target_name: str) -> None:
        if target_name not in self.tools:
            raise ValueError(f"Cannot alias unknown tool '{target_name}'")
        self.aliases[alias_name] = target_name

    def _resolve_name(self, name: str) -> str:
        if name in self.tools:
            return name
        if name in self.aliases:
            return self.aliases[name]
        if "." in name:
            suffix = name.split(".")[-1]
            if suffix in self.tools:
                return suffix
            if suffix in self.aliases:
                return self.aliases[suffix]
        return name

    def run(self, name: str, args: Dict[str, Any] | None = None) -> Any:
        resolved = self._resolve_name(name)
        if resolved not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
        return self.tools[resolved](**(args or {}))


_BLOCK_RE = re.compile(r"\{[\s\S]*?\}")


def parse_tool_call(text: str) -> Tuple[str | None, Dict[str, Any] | None]:
    """Extract {"tool":..., "tool_args":...} from assistant text if present."""

    if not text:
        return None, None
    match = _BLOCK_RE.search(text)
    if not match:
        return None, None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, None
    if not isinstance(obj, dict) or "tool" not in obj:
        return None, None
    args = obj.get("tool_args", {})
    if isinstance(args, str):
        args = {"path": args}
    if not isinstance(args, dict):
        return None, None
    return str(obj["tool"]), args


def parse_structured_tool_call(payload: Any) -> Tuple[str | None, Dict[str, Any] | None]:
    """Extract the first tool call from a structured model response."""

    if not isinstance(payload, dict):
        return None, None

    message = payload.get("message")
    if isinstance(message, dict):
        payload = message

    tool_calls = payload.get("tool_calls") if isinstance(payload, dict) else None
    if not tool_calls or not isinstance(tool_calls, list):
        return None, None

    first_call = tool_calls[0]
    if not isinstance(first_call, dict):
        return None, None

    function_payload = first_call.get("function")
    if not isinstance(function_payload, dict):
        return None, None

    tool_name = function_payload.get("name")
    if not tool_name:
        return None, None

    raw_args = function_payload.get("arguments", {})
    if isinstance(raw_args, str):
        raw_args = raw_args.strip()
        if raw_args:
            try:
                parsed_args = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed_args = {"path": raw_args}
        else:
            parsed_args = {}
    elif isinstance(raw_args, dict):
        parsed_args = raw_args
    else:
        return str(tool_name), {}

    if not isinstance(parsed_args, dict):
        return str(tool_name), {}

    return str(tool_name), parsed_args


__all__ = [
    "ToolRegistry",
    "parse_tool_call",
    "parse_structured_tool_call",
]
