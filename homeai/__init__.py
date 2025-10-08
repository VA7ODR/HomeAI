"""Internal modules that back the HomeAI Gradio application."""

from . import config as _config
from .filesystem import (
    assert_in_allowlist,
    get_file_info,
    list_dir,
    locate_files,
    read_text_file,
    resolve_under_base,
)
from .model_engine import LocalModelEngine
from .pgvector_store import EmbeddingError, PgVectorStore, SupportsEmbed
from .tool_utils import ToolRegistry, parse_tool_call, parse_structured_tool_call
from .ui_utils import safe_component

reload_from_environment = _config.reload_from_environment

__all__ = [
    "assert_in_allowlist",
    "get_file_info",
    "list_dir",
    "locate_files",
    "read_text_file",
    "resolve_under_base",
    "LocalModelEngine",
    "PgVectorStore",
    "EmbeddingError",
    "SupportsEmbed",
    "ToolRegistry",
    "parse_tool_call",
    "parse_structured_tool_call",
    "safe_component",
    "reload_from_environment",
]


def __getattr__(name: str):
    if hasattr(_config, name):
        return getattr(_config, name)
    raise AttributeError(name)
