# FlowRAG-Agent 系统架构

> 版本 0.1.0 · 单体架构 · Agentic RAG（固定 Skill 编排）

## 1. 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web UI                          │
│  upload_panel │ chat_panel │ agent_trace │ citation_view     │
└──────────────────────────┬──────────────────────────────────┘
                           │ AgentRuntime.execute()
┌──────────────────────────▼──────────────────────────────────┐
│                   Agent Runtime                              │
│  session/memory │ trace │ router │ skill dispatch              │
└──────────┬───────────────────────────────┬──────────────────┘
           │                               │
┌──────────▼──────────┐         ┌──────────▼──────────┐
│   Skills (业务编排)  │         │  MemoryManager      │
│  QA / Summary /     │         │  短期会话 history    │
│  Analysis           │         │  （Runtime 独占访问） │
└──────────┬──────────┘         └─────────────────────┘
           │ invoke_tool()
┌──────────▼──────────┐
│   Tools (原子能力)   │
│  search / generate  │
│  summarize / analyze│
└──────────┬──────────┘
           │
┌──────────▼──────────┐         ┌─────────────────────┐
│   RAGService        │◄────────│  Ingestion Pipeline │
│   Chroma + Citation │         │  pdf→chunk→embed    │
└──────────┬──────────┘         └─────────────────────┘
           │
┌──────────▼──────────┐
│   ChromaDB +        │
│   doc_registry      │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   LLM / Embeddings  │
│   Ollama / OpenAI   │
└─────────────────────┘
```

## 2. 分层职责

| 层 | 目录 | 职责 |
|----|------|------|
| 表现层 | `app/`, `ui/` | 上传、对话、执行轨迹、引用展示 |
| 编排层 | `agent/` | AgentRuntime、Router、Trace、Orchestrator 兼容入口 |
| 业务层 | `skills/` | 按意图编排 Tool 链（QA / Summary / Analysis） |
| 能力层 | `tools/` | 原子 Tool 注册与执行（委托 RAGService / LLM） |
| 记忆层 | `memory/` | Session 级短期对话（仅 Runtime 访问） |
| 入库层 | `ingestion/` | PDF 解析、切分、Embedding、写入 Chroma |
| 检索层 | `retrieval/` | RAGService 统一检索、Chroma 存储、Citation |
| 数据层 | `models/`, `data/` | Schema 契约、文档注册表 |
| 基础设施 | `config/`, `llm/` | 配置、Prompt、模型工厂 |
| 评估 | `evaluation/` | 离线 Hit Rate + LLM-as-Judge |
| 对外集成 | `mcp_server/` | MCP 检索服务（独立于 Streamlit） |

## 3. 执行流程

### 3.1 单次请求（Query → Answer）

```
User Query
  → AgentRuntime.begin_session
  → Router（LLM 结构化输出选 Skill；失败/低置信度 → 规则回退）
  → build_history_for_prompt（Memory → 格式化 history）
  → SkillContext.create(conversation_history=...)
  → Skill.run（固定 Tool 调用链）
  → MemoryManager.record_turn
  → AnswerResult + trace → UI
```

### 3.2 入库（Upload → Index）

```
PDF → pdf_parser（按页 Document）
    → chunker（RecursiveCharacterTextSplitter）
    → embedder（batch embedding）
    → indexer → chroma_store.add + doc_registry.update
```

## 4. Router 与 Skill

### 4.1 Router

- **输入**：用户 query、UI 选中的 doc_ids
- **输出**：`RouterResult`（intent + scope + confidence）
- **策略**：LLM `with_structured_output` 选 Skill；confidence < 0.5 或异常时回退关键词规则
- **不做**：Tool 选择、检索范围推断（scope 来自 UI 文档选择）

### 4.2 Skill 一览

| Skill | Intent | 检索策略 | Tool 链（概要） | 输出 |
|-------|--------|----------|-----------------|------|
| QASkill | QA | Top-K 语义检索 | search → format_context → llm.generate → citations | 短答 + 引用 |
| SummarySkill | SUMMARIZE | 宽检索（top_k≈12） | search → format_context → summarize → citations | 结构化摘要 |
| AnalysisSkill | ANALYZE | 多 sub-query 检索 + 去重 | search×N → format_context → analyze → citations | 分点分析 |

Skill 内通过 `skills/helpers.invoke_tool()` 调用 Tool，并 emit `AgentStep` trace。

## 5. Memory 注入

- **范围**：Session 级短期对话，无长期记忆
- **访问约束**：`memory/` 仅 Runtime import；Skill / Tool 禁止直接访问
- **注入路径**：`build_history_for_prompt()` → `SkillContext.conversation_history` → Prompt 模板
- **截断策略**：优先保留最近轮次；受 `memory_max_turns`、`memory_max_tokens` 限制

Prompt 格式（见 `config/prompts.py`）：

```
Conversation History:
{history}

Retrieved Context:
{context}

Question:
{question}
```

## 6. 执行轨迹（AgentStep）

| step_type | 典型 name | 含义 |
|-----------|-----------|------|
| `routing` | Router | 意图路由 |
| `retrieval` | Retriever / ContextBuilder | 检索或上下文构建 |
| `generation` | Generator / Summarizer / Analyzer | LLM 生成 |
| `postprocess` | Citations | 引用构建 |

Tool 调用时 `metadata.tool` 记录 Tool 名（如 `search_document`）。Skill 层无独立 trace 步骤，通过 Router 的 `intent` 推断。

## 7. 模块依赖

```
streamlit_app
    └── ui/chat_panel
            └── agent.runtime.AgentRuntime.execute()
                    ├── agent/router.route()
                    ├── agent/runtime/history.build_history_for_prompt()
                    ├── memory/manager.MemoryManager
                    ├── skills/registry → skills/*_skill
                    │       └── skills/helpers.invoke_tool()
                    │               └── tools/registry → tools/*
                    │                       └── retrieval/rag_service
                    └── models/schemas.AnswerResult

upload_panel → ingestion/indexer
    ├── ingestion/pdf_parser, chunker, embedder
    ├── retrieval/chroma_store
    └── models/doc_registry

mcp_server/server（独立入口）
    └── tools/search_document → RAGService

evaluation/runner（独立入口）
    └── AgentRuntime.execute() + metrics
```

## 8. 配置与扩展

| 扩展点 | 方式 |
|--------|------|
| 切换 LLM | `.env` → `LLM_PROVIDER` → `llm/factory.py` |
| 切换 Embedding | `.env` → `EMBEDDING_PROVIDER` |
| 新增 Skill | 继承 `skills/base.py`，注册到 `skills/registry.py` |
| 新增 Tool | 继承 `tools/base.py`，注册到 `tools/registry.py` |
| Memory 预算 | `memory_max_turns` / `memory_max_tokens` in `config/settings.py` |
| Metadata 契约 | [METADATA_SCHEMA.md](./METADATA_SCHEMA.md) |

## 9. CLI 入口

| 入口 | 模块 |
|------|------|
| Web UI | `streamlit run app/streamlit_app.py` |
| Agent API | `agent.runtime.get_runtime().execute(ExecuteRequest(...))` |
| 兼容包装 | `agent.orchestrator.run(query)` |
| 入库 | `flowrag-ingest` / `python -m scripts.ingest_cli` |
| 评估 | `flowrag-eval` / `python -m evaluation.runner` |
| MCP | `flowrag-mcp` / `python -m mcp_server.server` |

## 10. 非目标

- ReAct / Planner 自主 Agent（无 tool loop）
- 长期记忆、用户认证、多租户
- 微服务 / 分布式部署
- OCR、表格 / 图片理解

## 11. 相关文档

- [MODULE_MAP.md](./MODULE_MAP.md) — 文件级职责索引
- [METADATA_SCHEMA.md](./METADATA_SCHEMA.md) — 检索 filter 与 citation 约定
- [../tests/README.md](../tests/README.md) — 测试体系
- [../evaluation/README.md](../evaluation/README.md) — 离线评估
