
from pathlib import Path

from context_memory import ContextBuilder, LocalJSONMemoryBackend


class DummyVectorStore:
    def __init__(self, message_id: str, distance: float = 0.42) -> None:
        self.message_id = message_id
        self.distance = distance
        self.calls = []

    def search_messages(self, top_k, query_text, filters=None, *, embedder=None):
        self.calls.append((top_k, query_text, filters, embedder))
        return [
            {
                "message_id": self.message_id,
                "distance": self.distance,
                "created_at": "2024-01-01T00:00:00+00:00",
                "updated_at": "2024-01-01T00:00:00+00:00",
            }
        ]

def test_context_builder_deduplicates_latest_user(tmp_path):
    backend = LocalJSONMemoryBackend(base_dir=tmp_path)
    conversation_id = backend.new_conversation_id()

    backend.add_message(conversation_id, "assistant", {"text": "Hello"})
    backend.add_message(conversation_id, "user", {"text": "Earlier question"})

    latest_prompt = "What's the update?"
    backend.add_message(conversation_id, "user", {"text": latest_prompt})

    builder = ContextBuilder(backend)
    messages = builder.build_context(conversation_id, latest_prompt, persona_seed="Test persona")

    occurrences = [
        msg
        for msg in messages
        if msg.get("role") == "user" and msg.get("content") == latest_prompt
    ]

    assert len(occurrences) == 1
    assert messages[-1].get("role") == "user"
    assert messages[-1].get("content") == latest_prompt

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


def test_search_semantic_uses_vector_store_when_available(tmp_path: Path) -> None:
    backend = LocalJSONMemoryBackend(base_dir=tmp_path)
    conversation_id = backend.new_conversation_id()
    stored = backend.add_message(conversation_id, "user", {"text": "Hello world"})

    vector_store = DummyVectorStore(stored.id, distance=0.25)
    backend.configure_vector_search(vector_store)

    results = backend.search_semantic(conversation_id, "hello", limit=3)

    assert vector_store.calls
    top_k, query_text, filters, _ = vector_store.calls[0]
    assert top_k == 3
    assert query_text == "hello"
    assert filters == {"thread_id": conversation_id}

    assert results
    assert results[0].id == stored.id
    assert results[0].score == 0.25
