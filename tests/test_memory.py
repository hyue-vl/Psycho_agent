from psycho_agent.memory import AMemMemoryManager


def test_memory_manager_prioritises_recent_and_relevant_notes():
    manager = AMemMemoryManager(max_history=10)
    manager.recall_append(
        "u1",
        [
            {
                "role": "user",
                "content": "I feel intense exam anxiety before the physics test.",
                "metadata": {"tags": ["exam", "anxiety"]},
            },
            {
                "role": "assistant",
                "content": "Let's map the triggers together and slow things down.",
                "metadata": {"tags": ["support"]},
            },
            {
                "role": "user",
                "content": "Also stressed about family expectations.",
                "metadata": {"tags": ["family"]},
            },
        ],
    )
    context = manager.load_context("u1", top_k=2, query="exam anxiety")
    assert len(context["recall"]) == 2
    assert "exam" in context["recall"][0].content.lower()
    assert len(context["working"]) == 2


def test_memory_manager_archival_includes_core_memory():
    manager = AMemMemoryManager()
    manager.recall_append(
        "core-user",
        [
            {
                "role": "user",
                "content": "I panic when Sunday arrives.",
                "metadata": {"tags": ["sunday"]},
            }
        ],
    )
    manager.core_memory_replace("core-user", {"safety_plan": "Call therapist before midnight."})
    context = manager.load_context("core-user", top_k=1)
    assert context["archival"]
    assert "safety_plan" in context["archival"][0]


def test_memory_manager_export_profile_contains_history():
    manager = AMemMemoryManager()
    manager.recall_append(
        "profile-user",
        [
            {"role": "user", "content": "I failed another exam."},
            {"role": "assistant", "content": "Let's build a paced study ritual."},
        ],
    )
    profile = manager.export_profile("profile-user")
    assert profile["history"]
    first = profile["history"][0]
    assert "record_id" in first
    assert first["summary"]
    assert isinstance(first["keywords"], list)
