# FlowRAG-Agent 测试体系

面向 Agentic RAG 项目的分层 pytest 套件，验证**代码正确性**与**架构约束**。生成质量由 [evaluation/README.md](../evaluation/README.md) 负责。

## 目录结构

```
tests/
├── conftest.py                 # 全局 fixture + marker 注册
├── fixtures/
│   ├── factories.py            # PDF 等测试工厂
│   └── data/                   # 测试数据（与业务代码分离）
│       ├── documents.py        # 样例文档与检索 query
│       ├── router_cases.py     # Router 意图/scope 用例表
│       ├── qa_scenarios.py     # E2E / Skill 场景
│       └── llm_responses.py    # Fake LLM 响应
├── rag/
│   ├── test_retrieval.py       # [1] RAG Retrieval
│   └── test_answer.py          # [2] RAG Answer
├── router/
│   └── test_router.py          # [3] Router
├── tools/
│   └── test_tools.py           # [4] Tool
├── skills/
│   └── test_skills.py          # [5] Skill → Tool
├── runtime/
│   └── test_runtime.py         # [6] Runtime
├── memory/
│   ├── test_isolation.py       # [7] Memory 隔离与生命周期
│   ├── test_history_format.py  # history 截断策略
│   └── test_history_injection.py  # 多轮注入与超长截断
├── e2e/
│   └── test_agent_pipeline.py  # [8] End-to-End
├── unit/
│   ├── test_ingestion.py       # 入库基础
│   ├── test_packaging.py       # editable install 冒烟
│   └── test_mcp_server.py      # MCP 注册
└── examples/
    └── test_rag_usage_examples.py  # API 用法示例（@example）
```

## 测试分层

| 标记 | 目录 | 验证内容 |
|------|------|----------|
| `rag_retrieval` | `rag/test_retrieval.py` | Chroma filter、RAGService、Retriever |
| `rag_answer` | `rag/test_answer.py` | QA 答案、Citation、空库兜底 |
| `router` | `router/test_router.py` | LLM + 规则 Router、scope 构建 |
| `tool` | `tools/test_tools.py` | ToolRegistry、原子 Tool |
| `skill` | `skills/test_skills.py` | Skill 编排、Tool 调用链 |
| `runtime` | `runtime/test_runtime.py` | AgentRuntime、orchestrator 兼容 |
| `memory` | `memory/` | Memory 生命周期、隔离、history 注入 |
| `e2e` | `e2e/test_agent_pipeline.py` | 入库 → Runtime → 答案全链路 |
| `unit` | `unit/` | 入库、packaging、MCP |
| `example` | `examples/` | 可复制 API 示例 |

## 运行方式

```bash
# 全量（推荐，跳过 API 示例）
pytest tests/ -q -m "not example"

# 全量含示例
pytest tests/ -v

# 按层运行
pytest tests/ -m rag_retrieval -v
pytest tests/ -m rag_answer -v
pytest tests/ -m router -v
pytest tests/ -m tool -v
pytest tests/ -m skill -v
pytest tests/ -m runtime -v
pytest tests/ -m memory -v
pytest tests/ -m e2e -v

# API 示例
pytest tests/examples/ -v
```

## 设计原则

1. **数据与逻辑分离** — 用例表在 `fixtures/data/`，测试只写断言
2. **不修改业务代码** — 使用 FakeEmbeddings / FakeListChatModel 注入
3. **Fast & Isolated** — 每个测试使用 `tmp_path` 隔离 Chroma / SQLite
4. **架构约束** — AST 扫描 `tools/`、`skills/` 无 `memory` import

## 各层验证要点

### RAG Retrieval

- `RetrievalScope` → Chroma where filter 正确
- 入库样例 PDF 后检索命中期望页码 / 关键词
- LangChain Retriever 与 RAGService 行为一致

### RAG Answer

- Citation 去重与格式
- 有库 / 空库 QA Skill 行为
- trace 含 `GENERATION` 步骤

### Router

- LLM 结构化输出映射到 QA / SUMMARIZE / ANALYZE
- 低置信度 / 异常时规则回退
- UI doc 选择 → ALL / SINGLE / SELECTED scope

### Tool

- Registry 生命周期与内置 Tool 列表
- 原子 Tool 不越权（不自行检索、不访问 Memory）

### Skill

- 内置 3 个 Skill 已注册
- QA trace：Retriever → ContextBuilder → Generator → Citations
- Analysis：多 sub-query 检索 → 去重 → analyze

### Runtime

- `AgentRuntime.execute()` 完整链路
- Memory 记录 turn
- `orchestrator.run()` 兼容包装

### Memory

- begin / record / end 生命周期
- **history 注入**：多轮对话后 Prompt 含 prior turns
- **超长截断**：`max_turns` / `max_tokens` 优先保留最近轮
- Tool / Skill 层无 `import memory`

### E2E

```
make PDF → ingest_pdf → AgentRuntime.execute(qa/summary/analysis)
    → 断言 intent、citations、trace
```

## 与 Evaluation 的区别

| | pytest（tests/） | evaluation/ |
|--|------------------|-------------|
| 目的 | 代码正确性 + 架构约束 | Agent 效果量化 |
| LLM | FakeLLM（确定性） | 真实 LLM 或 LLM-as-Judge |
| 环境 | `tmp_path` 临时目录 | `reports/.cache/` 隔离目录 |
| 输出 | pass / fail | Markdown 报告 + 指标 |

## 共享 Fixture

| Fixture | 作用 |
|---------|------|
| `rag_settings` | 临时 data/chroma/registry 路径 |
| `fake_embeddings` | 128 维确定性向量 |
| `fake_llm` | 固定回复列表 |
| `chroma_store` | 已连接 ChromaStore |
| `doc_registry` | 临时 SQLite 注册表 |
| `sample_contract_pdf` | 2 页样例 PDF |
| `ingested_contract` | 已完成入库的四元组 |

## pytest 路径说明

`pyproject.toml` 配置了 `pythonpath = ["."]`，pytest 可直接 import 项目模块。Streamlit 启动需 `pip install -e .` 或依赖 `config/env_bootstrap.py` 的路径修复。

## 扩展测试数据

编辑 `tests/fixtures/data/` 下对应文件，**不要**改业务代码：

- `documents.py` — 样例 PDF 文本、检索 query
- `router_cases.py` — Router 表驱动用例
- `qa_scenarios.py` — E2E 场景
- `llm_responses.py` — Fake LLM 固定响应
