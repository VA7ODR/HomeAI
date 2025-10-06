from pathlib import Path

from context_memory import LocalJSONMemoryBackend


def test_load_messages_quarantines_corrupt_file(tmp_path: Path) -> None:
    backend = LocalJSONMemoryBackend(base_dir=tmp_path)
    conversation_id = "conversation"
    path = backend._conversation_path(conversation_id)
    path.write_text("{ invalid json", encoding="utf-8")

    messages = backend._load_messages(conversation_id)

    assert messages == []
    assert not path.exists()
    quarantined = list(tmp_path.glob("conversation.json.corrupt-*"))
    assert len(quarantined) == 1
