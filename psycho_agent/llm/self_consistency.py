"""Utility for running multiple LLM passes and aggregating votes."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from typing import Any, Callable, Dict, List, Sequence


class SelfConsistencyRunner:
    """Executes multiple inference passes and merges their outputs."""

    def __init__(self, passes: int = 3) -> None:
        self.passes = passes

    def run(self, call_fn: Callable[[int], str]) -> Dict[str, Any]:
        outputs: List[Dict[str, Any]] = []
        for idx in range(self.passes):
            raw = call_fn(idx)
            try:
                outputs.append(json.loads(raw))
            except json.JSONDecodeError:
                outputs.append({"raw": raw})
        return self._aggregate(outputs)

    def _aggregate(self, outputs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        distortions = self._vote(outputs, "cognitive_distortions")
        intents = self._vote(outputs, "communicative_intent")
        valence = self._mean(outputs, "emotion_valence")
        arousal = self._mean(outputs, "emotion_arousal")
        risk = self._mean(outputs, "risk_level")
        rationale = "\n".join(o.get("rationale", "") for o in outputs if o.get("rationale"))
        return {
            "cognitive_distortions": distortions,
            "communicative_intent": intents[0] if intents else "unknown",
            "emotion_valence": valence,
            "emotion_arousal": arousal,
            "risk_level": risk,
            "rationale": rationale,
        }

    def _vote(self, outputs: Sequence[Dict[str, Any]], key: str) -> List[str]:
        counter: Counter[str] = Counter()
        for item in outputs:
            value = item.get(key)
            if isinstance(value, list):
                counter.update(value)
            elif isinstance(value, str):
                counter.update([value])
        if not counter:
            return []
        most_common = counter.most_common()
        max_count = most_common[0][1]
        return [label for label, count in most_common if count == max_count]

    def _mean(self, outputs: Sequence[Dict[str, Any]], key: str) -> float:
        values = [float(item[key]) for item in outputs if key in item]
        return statistics.mean(values) if values else 0.0
