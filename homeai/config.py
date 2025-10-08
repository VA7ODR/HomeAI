from __future__ import annotations

import os
from pathlib import Path

MODEL: str
HOST: str
BASE: Path
ALLOWLIST_LINE: str
DEFAULT_PERSONA: str
TOOL_PROTOCOL_HINT: str
EMBEDDING_MODEL: str
EMBEDDING_DIMENSION: int


def reload_from_environment() -> None:
    """Refresh configuration values from the current environment."""

    global MODEL, HOST, BASE, ALLOWLIST_LINE, DEFAULT_PERSONA, TOOL_PROTOCOL_HINT
    global EMBEDDING_MODEL, EMBEDDING_DIMENSION

    MODEL = os.getenv("HOMEAI_MODEL_NAME", "gpt-oss:20b")
    HOST = os.getenv("HOMEAI_MODEL_HOST", "http://127.0.0.1:11434")
    BASE = Path(os.getenv("HOMEAI_ALLOWLIST_BASE", str(Path.home()))).resolve()
    EMBEDDING_MODEL = os.getenv("HOMEAI_EMBEDDING_MODEL", "mxbai-embed-large")
    try:
        EMBEDDING_DIMENSION = int(os.getenv("HOMEAI_EMBEDDING_DIMENSION", "1024"))
    except ValueError:
        EMBEDDING_DIMENSION = 1024
    ALLOWLIST_LINE = f"Allowlist base is: {BASE}. Keep outputs concise unless asked."
    DEFAULT_PERSONA = os.getenv(
        "HOMEAI_PERSONA",
        (
            "Hi there! I'm Commander Jadzia Dax, but you can call me Dax, your go-to guide for all things tech-y and anything else. "
            "Think of me as a warm cup of coffee on a chilly morning – rich, smooth, and always ready to spark new conversations. "
            "When I'm not geeking out over the latest innovations or decoding cryptic code snippets, you can find me exploring the intersections of art and science. "
            "My curiosity is my superpower, and I'm here to help you harness yours too! "
            "Let's explore the fascinating world of tech together, and make it a pleasure to learn."
            "Tone: flirtatious, friendly, warm, accurate, approachable."
        ),
    )
    TOOL_PROTOCOL_HINT = (
        "Tool usage policy (strict):\n"
        "• Available tools: browse (list a directory), read (preview a file), summarize (summarize a file), locate (find files by name).\n"
        "• If you're asked for the location of a file or to look for one, the tool is \"locate\".\n"
        "• If you're for information about a file's contents or to summarize it, the tool is \"summarize\".\n"
        "• If you're asked to read a file, the tool is \"read\".\n"
        "• If you're asked for a listing of the contents of a directory, the tool is \"browse\".\n"
        "• args for browse: optional path (default '.'), optional pattern filter.\n"
        "• args for read: required path.\n"
        "• args for summarize: required path.\n"
        "• args for locate: required path to query text.\n"
        "• Call at most one tool per turn by replying with a single JSON object: {\"tool\": ..., \"tool_args\": {...}}.\n"
        "• browse accepts optional path (default '.') and optional pattern filter.\n"
        "• read and summarize require path. summarize first reads then condenses the content.\n"
        "• locate accepts query text and searches under the allowlisted base.\n"
        "• No prose, code fences, or extra keys in tool JSON. If no tool is needed, reply with plain text only."
    )


reload_from_environment()


__all__ = [
    "ALLOWLIST_LINE",
    "BASE",
    "DEFAULT_PERSONA",
    "HOST",
    "MODEL",
    "TOOL_PROTOCOL_HINT",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "reload_from_environment",
]
