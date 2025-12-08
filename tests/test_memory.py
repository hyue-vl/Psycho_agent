import json

from psycho_agent.memory import AMemMemoryManager, MemoryGraph, PersistentHistoryLogger


class _StubTupleEncoder:
    def encode(self, role, content, context=None):
        # deterministic triple for tests without hitting real LLM
        snippet = (context or content or "")[:12] or "context"
        return [(role or "actor", "提及", snippet)]


class _StubHistoryLogger:
    def __init__(self):
        self.events = []

    def append(self, user_id, event):
        self.events.append((user_id, event))


class _StubAnnotator:
    def annotate(self, *, user_message, assistant_message, metadata=None):
        return {
            "intent": "custom_plan",
            "strategy": "supportive_planning",
            "tone": "supportive",
            "risk_level": "low",
            "personalization_tags": ["stub"],
            "follow_up": False,
            "training_notes": "stub annotation",
            "reference": {"user": user_message, "assistant": assistant_message},
        }


def test_amem_links_related_notes():
    logger = _StubHistoryLogger()
    annotator = _StubAnnotator()
    manager = AMemMemoryManager(
        graph=MemoryGraph(link_threshold=0.1),
        tuple_encoder=_StubTupleEncoder(),
        history_logger=logger,
        annotator=annotator,
    )
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
    assert notes[0]["tuples"]


def test_amem_load_context_returns_structured_slices():
    manager = AMemMemoryManager(
        tuple_encoder=_StubTupleEncoder(),
        history_logger=_StubHistoryLogger(),
        annotator=_StubAnnotator(),
    )
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
    assert "Tuples:" in context["recall"][0].content


def test_history_logger_creates_jsonl_and_txt(tmp_path):
    logger = PersistentHistoryLogger(base_dir=tmp_path)
    manager = AMemMemoryManager(
        tuple_encoder=_StubTupleEncoder(),
        history_logger=logger,
        annotator=_StubAnnotator(),
    )
    manager.recall_append(
        "finetune",
        [
            {"role": "user", "content": "I feel stuck before my exam."},
            {
                "role": "assistant",
                "content": "Let's draft a custom plan and ritual that fits your style.",
                "metadata": {"tags": ["plan"], "topic": "study"},
            },
        ],
    )
    jsonl_path = tmp_path / "jsonl" / "finetune.jsonl"
    txt_path = tmp_path / "txt" / "finetune.txt"
    assert jsonl_path.exists()
    assert txt_path.exists()
    payload = json.loads(jsonl_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert payload["role"] == "assistant"
    assert payload["annotation"]["intent"] == "custom_plan"
