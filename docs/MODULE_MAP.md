# FlowRAG-Agent 模块索引

> 快速查阅每个文件的职责与实现计划日。

## config/

| 文件 | 职责 | 计划日 |
|------|------|--------|
| `settings.py` | 全局配置（路径、Chroma、chunk、LLM） | Day 0 ✅ |
| `prompts.py` | 各工作流 Prompt 模板 | Day 0 ✅ |

## models/

| 文件 | 职责 | 计划日 |
|------|------|--------|
| `schemas.py` | Pydantic 数据契约（ChunkMetadata、Citation、AnswerResult） | Day 0 ✅ |
| `doc_registry.py` | 文档注册表 CRUD（SQLite） | Day 1 |

## ingestion/

| 文件 | 职责 | 计划日 |
|------|------|--------|
| `pdf_parser.py` | PyMuPDF 按页解析 | Day 1 |
| `chunker.py` | 文本切分 + metadata 赋值 | Day 1 |
| `embedder.py` | 批量 Embedding | Day 1 |
| `indexer.py` | 入库编排（parse→chunk→embed→chroma） | Day 1 |

## retrieval/

| 文件 | 职责 | 计划日 |
|------|------|--------|
| `chroma_store.py` | Chroma 客户端、where filter 构造 | Day 1-2 |
| `retriever_factory.py` | LangChain Retriever 工厂 | Day 2 |
| `citation_builder.py` | 检索结果 → Citation 列表 | Day 2 |

## llm/

| 文件 | 职责 | 计划日 |
|------|------|--------|
| `factory.py` | LLM / Embedding 实例化 | Day 2 |

## agent/

| 文件 | 职责 | 计划日 |
|------|------|--------|
| `orchestrator.py` | 总调度入口 `run()` | Day 2-3 |
| `router.py` | 意图分类（规则 + LLM） | Day 3 |
| `workflows/qa.py` | 问答工作流 | Day 2 |
| `workflows/summarize.py` | 总结工作流（Map-Reduce） | Day 3 |
| `workflows/analyze.py` | 分析工作流（多 sub-query） | Day 3 |
| `tools/retriever.py` | 检索 Tool | Day 2 |
| `tools/doc_loader.py` | 按 doc_id 加载全文片段 | Day 3 |
| `callbacks/streamlit_callback.py` | LangChain → AgentStep 桥接 | Day 4 |

## ui/

| 文件 | 职责 | 计划日 |
|------|------|--------|
| `state.py` | st.session_state 封装 | Day 4 |
| `components/upload_panel.py` | PDF 上传与入库进度 | Day 4 |
| `components/chat_panel.py` | 对话与流式输出 | Day 4 |
| `components/agent_trace.py` | 执行步骤可视化 | Day 4 |
| `components/citation_view.py` | 引用来源展示 | Day 4 |

## app/

| 文件 | 职责 | 计划日 |
|------|------|--------|
| `streamlit_app.py` | Streamlit 入口与页面布局 | Day 4 |

## scripts/

| 文件 | 职责 | 计划日 |
|------|------|--------|
| `ingest_cli.py` | 命令行批量入库 | Day 1 |
| `reset_db.py` | 清空知识库 | Day 1 |

## tests/

| 文件 | 覆盖模块 | 计划日 |
|------|----------|--------|
| `test_pdf_parser.py` | ingestion/pdf_parser | Day 1 |
| `test_chunker.py` | ingestion/chunker | Day 1 |
| `test_retriever.py` | retrieval/chroma_store | Day 2 |
| `test_router.py` | agent/router | Day 3 |

## docs/

| 文件 | 内容 |
|------|------|
| `ARCHITECTURE.md` | 系统架构、数据流、依赖关系 |
| `METADATA_SCHEMA.md` | Chroma metadata 与 filter 约定 |
| `MODULE_MAP.md` | 本文件 — 模块索引 |
