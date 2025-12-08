"""Environment-driven configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class QwenSettings:
    base_url: str = os.getenv("QWEN_BASE_URL", "https://api-inference.modelscope.cn/v1")
    api_key: str = os.getenv("QWEN_API_KEY", "ms-2eebb53e-c102-47c9-b37d-b360a007c5c8")
    model: str = os.getenv("QWEN_MODEL_PATH", "Qwen/Qwen2.5-72B-Instruct")
    small_model: str = os.getenv("QWEN_SMALL_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    planner_model: str = os.getenv(
        "QWEN_PLANNER_MODEL",
        os.getenv("QWEN_SMALL_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    )
    temperature: float = float(os.getenv("QWEN_TEMPERATURE", "0.2"))
    timeout: int = int(os.getenv("QWEN_TIMEOUT_MS", "60000"))
    max_output_tokens: int = int(os.getenv("QWEN_MAX_TOKENS", "1024"))


@dataclass(slots=True)
class BGEConfig:
    model_path: str = os.getenv("BGE_MODEL_PATH", "/data0/hy/models/bge-m3")
    normalize_embeddings: bool = _bool_env("BGE_NORMALIZE", True)
    batch_size: int = int(os.getenv("BGE_BATCH_SIZE", "8"))


@dataclass(slots=True)
class Neo4jConfig:
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "password")
    database: Optional[str] = os.getenv("NEO4J_DATABASE")


@dataclass(slots=True)
class KnowledgeConfig:
    coke_knowledge_path: str = os.getenv(
        "COKE_KB_PATH",
        str((Path(__file__).resolve().parent / "knowledge" / "coke_knowledge.txt")),
    )
    max_hits: int = int(os.getenv("COKE_KB_MAX_HITS", "8"))
    fuzzy_match_threshold: float = float(os.getenv("COKE_KB_MATCH_THRESHOLD", "0.35"))


@dataclass(slots=True)
class LettaConfig:
    base_url: str = os.getenv("LETTA_BASE_URL", "http://localhost:8288")
    api_key: Optional[str] = os.getenv("LETTA_API_KEY")
    enabled: bool = _bool_env("LETTA_ENABLED", False)
    timeout: float = float(os.getenv("LETTA_TIMEOUT_S", "4.0"))


@dataclass(slots=True)
class PlannerConfig:
    max_branches: int = int(os.getenv("PLANNER_BRANCHES", "3"))
    search_depth: int = int(os.getenv("PLANNER_DEPTH", "2"))
    enable_mcts: bool = _bool_env("PLANNER_ENABLE_MCTS", False)
    ucb_exploration_constant: float = float(os.getenv("PLANNER_UCB_C", "1.4"))


@dataclass(slots=True)
class RewardWeights:
    safety: float = float(os.getenv("REWARD_WEIGHT_SAFETY", "0.35"))
    empathy: float = float(os.getenv("REWARD_WEIGHT_EMPATHY", "0.25"))
    adherence: float = float(os.getenv("REWARD_WEIGHT_ADHERENCE", "0.2"))
    improvement: float = float(os.getenv("REWARD_WEIGHT_IMPROVEMENT", "0.2"))


@dataclass(slots=True)
class Settings:
    qwen: QwenSettings = field(default_factory=QwenSettings)
    bge: BGEConfig = field(default_factory=BGEConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    letta: LettaConfig = field(default_factory=LettaConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    risk_threshold: float = float(os.getenv("RISK_THRESHOLD", "0.6"))
    recall_top_k: int = int(os.getenv("RECALL_TOP_K", "5"))
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "3"))


settings = Settings()
