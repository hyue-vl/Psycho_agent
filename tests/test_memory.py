from psycho_agent.memory import AMemMemoryManager, MemoryGraph


def test_amem_links_related_notes():
    manager = AMemMemoryManager(graph=MemoryGraph(link_threshold=0.1))
    manager.recall_append(
        "u1",
        [
            {
                "role": "user",
                "content": "I feel intense exam anxiety and pressure before my physics test.",
                "metadata": {"context": "academics", "tags": ["anxiety", "exam"]},
            }
        ],
    )
    manager.recall_append(
        "u1",
        [
            {
                "role": "assistant",
                "content": "Let's build a study ritual to ease the exam anxiety you described for physics.",
                "metadata": {
                    "context": "academics",
                    "tags": ["planning", "exam"],
                    "keywords": ["exam", "anxiety", "study"],
                },
            }
        ],
    )
    profile = manager.export_profile("u1")
    notes = profile["notes"]
    assert len(notes) == 2
    by_id = {note["note_id"]: note for note in notes}
    note_ids = list(by_id.keys())
    first_links = set(by_id[note_ids[0]]["links"])
    second_links = set(by_id[note_ids[1]]["links"])
    assert note_ids[1] in first_links
    assert note_ids[0] in second_links


def test_amem_load_context_returns_structured_slices():
    manager = AMemMemoryManager()
    manager.recall_append(
        "demo",
        [
            {"role": "user", "content": "I failed my exam again and I am panicking."},
            {"role": "assistant", "content": "We can map triggers and respond with breathing drills."},
            {"role": "user", "content": "It keeps me awake at night; I dread disappointing my parents."},
        ],
    )
    context = manager.load_context("demo", top_k=3, query="exam panic")
    assert len(context["recall"]) == 3
    assert len(context["working"]) == 2
    assert "Keywords:" in context["recall"][0].content
