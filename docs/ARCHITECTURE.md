# FlowRAG-Agent 系统架构

> 版本：0.1.0 | 单体架构 | 五天内可交付 MVP

## 1. 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web UI                          │
│  upload_panel │ chat_panel │ agent_trace │ citation_view     │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   Agent Orchestrator                         │
│         router.py → workflows/{qa,summarize,analyze}       │
└──────────┬───────────────────────────────┬──────────────────┘
           │                               │
┌──────────▼──────────┐         ┌──────────▼──────────┐
│   Ingestion Pipeline │         │  Retrieval Layer    │
│  pdf→chunk→embed     │         │  chroma→citation    │
└──────────┬──────────┘         └──────────┬──────────┘
           │                               │
           └──────────────┬────────────────┘
                          │
              ┌───────────▼───────────┐
              │   ChromaDB (local)    │
              │   + doc_registry      │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │   LLM / Embeddings    │
              │   Ollama / OpenAI     │
              └───────────────────────┘
```

## 2. 分层职责

| 层 | 目录 | 职责 |
|----|------|------|
| 表现层 | `app/`, `ui/` | 用户交互、流式输出、执行追踪展示 |
| 编排层 | `agent/` | 意图路由、工作流调度、LangChain Tools |
| 入库层 | `ingestion/` | PDF 解析、切分、Embedding、写入 Chroma |
| 检索层 | `retrieval/` | 向量检索、Filter 构造、Citation 构建 |
| 数据层 | `models/`, `data/` | Schema 契约、文档注册表、持久化 |
| 基础设施 | `config/`, `llm/` | 配置、Prompt、模型工厂 |

## 3. 数据流

### 3.1 入库（Upload → Index）

```
PDF → pdf_parser (按页 Document)
    → chunker (RecursiveCharacterTextSplitter)
    → embedder (batch embedding)
    → indexer → chroma_store.add + doc_registry.update
```

### 3.2 问答（Query → Answer）

```
User Query → router (intent: qa | summarize | analyze)
           → workflow.run(scope, query)
           → retriever (Chroma + where filter)
           → LLM (prompt + context)
           → citation_builder
           → AnswerResult → UI
```

## 4. Agent 工作流

| 工作流 | 意图 | 检索策略 | 输出 |
|--------|------|----------|------|
| QA | 事实查询 | Top-K 语义检索 | 短答 + 引用 |
| Summarize | 总结/概述 | 单文档 Map-Reduce | 结构化摘要 + 关键页 |
| Analyze | 对比/分析 | 多 sub-query 检索 | 分点分析 + 每点引用 |

路由策略：**规则优先 + LLM 结构化分类**（见 `agent/router.py`）。

## 5. 模块依赖关系

```
streamlit_app
    ├── ui/*
    └── agent/orchestrator
            ├── agent/router
            ├── agent/workflows/*
            │       ├── retrieval/retriever_factory
            │       ├── retrieval/citation_builder
            │       ├── config/prompts
            │       └── llm/factory
            └── agent/callbacks/streamlit_callback

upload_panel → ingestion/indexer
    ├── ingestion/pdf_parser
    ├── ingestion/chunker
    ├── ingestion/embedder
    ├── retrieval/chroma_store
    └── models/doc_registry
```

## 6. 配置与扩展点

- **LLM 切换**：`LLM_PROVIDER` → `llm/factory.py`
- **Embedding 切换**：`EMBEDDING_PROVIDER` → `llm/factory.py`
- **新工作流**：继承 `agent/workflows/base.py`，注册到 `orchestrator.py`
- **Metadata 契约**：见 [METADATA_SCHEMA.md](./METADATA_SCHEMA.md)

## 7. 非目标（MVP 不做）

- 微服务 / 分布式部署
- OCR、表格/图片理解
- 用户认证、多租户
- 复杂多轮 ReAct Agent

## 8. 五天开发里程碑

| Day | 交付 |
|-----|------|
| 1 | 入库流水线 + Chroma 可检索 |
| 2 | QA 工作流 + 引用 |
| 3 | Router + Summarize / Analyze |
| 4 | Streamlit UI + 执行追踪 |
| 5 | 打磨、测试、文档、Demo |
