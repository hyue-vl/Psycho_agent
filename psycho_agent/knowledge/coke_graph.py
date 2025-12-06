"""Neo4j-backed interface for the COKE CBT knowledge graph."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, List

try:  # pragma: no cover
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover
    GraphDatabase = None

from ..config import settings

LOGGER = logging.getLogger(__name__)


class COKEKGraph:
    """Thin client for retrieving graph-constrained CBT chains."""

    def __init__(self) -> None:
        if GraphDatabase is None:
            LOGGER.warning("neo4j driver not installed; graph queries disabled.")
            self._driver = None
            self._database = None
            return
        try:
            self._driver = GraphDatabase.driver(
                settings.neo4j.uri,
                auth=(settings.neo4j.user, settings.neo4j.password),
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Neo4j driver unavailable: %s", exc)
            self._driver = None
        self._database = settings.neo4j.database

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()

    def fetch_paths(self, situation: str, belief: str, limit: int = 5) -> List[str]:
        query = """
        MATCH (s:Situation {name:$situation})-[:TRIGGERS]->(t:Thought)-[:LEADS_TO]->(e:Emotion)
        OPTIONAL MATCH (t)-[:MODERATED_BY]->(c:CoreBelief {name:$belief})
        RETURN s.name AS situation, t.name AS thought, e.name AS emotion, collect(DISTINCT c.name) AS core_beliefs
        LIMIT $limit
        """
        if self._driver is None:
            return []
        session_args = {"database": self._database} if self._database else {}
        with self._driver.session(**session_args) as session:
            records = session.run(query, situation=situation, belief=belief, limit=limit)
            paths = []
            for record in records:
                chain = f"Situation: {record['situation']} -> Thought: {record['thought']} -> Emotion: {record['emotion']}"
                beliefs = [b for b in record["core_beliefs"] if b]
                if beliefs:
                    chain += f" | Core beliefs: {', '.join(beliefs)}"
                paths.append(chain)
            return paths

    @lru_cache(maxsize=256)
    def interventions_for_distortion(self, distortion: str) -> List[str]:
        query = """
        MATCH (d:Distortion {name:$name})<-[:TARGETS]-(i:Intervention)
        RETURN DISTINCT i.name AS intervention
        """
        if self._driver is None:
            return []
        session_args = {"database": self._database} if self._database else {}
        with self._driver.session(**session_args) as session:
            result = session.run(query, name=distortion)
            return [record["intervention"] for record in result]
