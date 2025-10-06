from context_memory import ContextBuilder, LocalJSONMemoryBackend


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
