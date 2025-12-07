# Psycho-World Agent

Psycho-World 是一个“纯推理”多智能体心理干预系统，基于语义化重构的 POMDP 框架与 LangGraph 的循环编排，实现 System-2 级别的多步规划与“梦境推演”能力。该实现复现了问题描述中的五大节点：感知、记忆、规划、模拟与行动，并将 Qwen 远程大模型与本地 BGE 语义检索、COKE 图谱、MemGPT 记忆结构整合在同一个推理图中。

## 架构概览

- **语义状态空间 (`GlobalState`)**：包括 MemGPT 管理的工作记忆/回忆/归档切片、显式 JSON 信念态 `BeliefState`、动态图知识 `KnowledgeContext` 以及策略候选。
- **动作空间**：Planner 以 Tree-of-Thoughts 思路输出多策略集合 `<k_t, u_t>`，Action Agent 负责在约束下生成最终语言动作。
- **转移/奖励**：Simulation Agent 通过图约束 LLM 推演 `s_{t+1}`，同时产出多维奖励向量（安全、共情、一致性、状态改善），由 Action Agent 按权重汇总。
- **多智能体编排**：LangGraph 构建 Memory ➜ Perception ➜ Planning ➜ Simulation ➜ Action 图；满足高风险路径触发 System-2（ToT/MCTS）推理的自适应策略。

## 关键模块

| 模块 | 文件 | 说明 |
| --- | --- | --- |
| Qwen 接口 | `psycho_agent/llm/qwen.py` | 复用提供的 OpenAI 兼容客户端，负责所有 LLM 调用 |
| 自洽诊断 | `psycho_agent/agents/perception.py` | DoT Prompt + Self-Consistency 多票合并，输出结构化信念态 |
| 记忆管理 | `psycho_agent/memory.py`, `psycho_agent/vectorstore.py` | MemGPT PKB + BGE/FAISS 回忆检索，支持 fallback |
| 图谱约束 | `psycho_agent/knowledge/coke_graph.py` | Neo4j COKE 查询，策略映射与模拟约束 |
| 规划/推演 | `psycho_agent/agents/planning.py`, `psycho_agent/agents/simulation.py` | Tree-of-Thought 策略生成 + Graph-Constrained Simulation |
| LangGraph 工作流 | `psycho_agent/workflow.py` | 可依赖注入的 MAS 编排，暴露 `PsychoWorldGraph.invoke()` |
| CLI | `psycho_agent/cli.py` | `typer` 命令行触发一次完整对话轮次 |

## 配置

所有运行所需的 URL、API Key、模型路径都集中在 `psycho_agent/config.py`，均支持环境变量覆盖：

- `QWEN_BASE_URL`, `QWEN_API_KEY`, `QWEN_MODEL_PATH`
- `BGE_MODEL_PATH=/data0/hy/models/bge-m3`
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- `ENABLE_SYSTEM2`, `PLANNER_BRANCHES`, `RISK_THRESHOLD` 等控制推理强度

## 快速开始

```bash
pip install -e .
export QWEN_API_KEY=...
python -m psycho_agent.cli chat --user-id u123
# 也可以传入首轮用户输入并开启诊断输出
python -m psycho_agent.cli chat "我最近又梦到考试失败" --user-id u123 --diagnostics
```

CLI 会进入多轮对话循环，按 `exit`（或 `Ctrl+D`）即可结束。默认情况下每一轮都会展示 Workflow trace（记忆抽取、规划树、模拟评分、行动提示等关键节点），并且整个会话期间，用户与 Agent 的发言都会写回 MemGPT/Letta 记忆空间，便于之后的上下文召回与个性化干预；若还需要完整原始 JSON，可开启 `--diagnostics`。

## 开发与测试

- 依赖管理：`pyproject.toml`
- 单元测试：`pytest tests/test_workflow.py`
  - 通过依赖注入（Stub VectorStore/MemGPT/LLM）验证 LangGraph 管线可独立运行
- 建议在开发环境中设置 `ENABLE_SYSTEM2=0`，仅在高风险场景启用 ToT/MCTS，避免不必要的 token 成本。

## 下一步可扩展点

1. **MCTS 升级**：在 `PlanningAgent` 中替换当前解析逻辑为真实的蒙特卡洛树搜索，结合 LLM-as-a-Judge 的价值函数。
2. **GraphRAG 接入**：将 `KnowledgeContext` 与实际的 GraphRAG Service 对接，动态注入局部子图提示。
3. **MemGPT Hooks**：实现 `core_memory_replace`/`recall_append` 等函数调用，与 Letta Runtime 深度结合。
4. **Safety Escalation**：在 `ActionAgent` 中加入自动化报警或人工介入 Hook，当 reward 向量低于阈值时终止自动回复。