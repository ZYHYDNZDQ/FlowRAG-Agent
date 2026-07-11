# FlowRAG-Agent 模块索引

> 快速查阅每个目录 / 文件的职责。架构总览见 [ARCHITECTURE.md](./ARCHITECTURE.md)。

## app/

| 文件 | 职责 |
|------|------|
| `streamlit_app.py` | Streamlit 入口：侧边栏上传 + 主区域对话 |

## ui/

| 文件 | 职责 |
|------|------|
| `state.py` | `st.session_state` 封装（消息、trace、选中文档） |
| `components/upload_panel.py` | PDF 上传与入库进度 |
| `components/chat_panel.py` | 对话输入，调用 `AgentRuntime.execute()` |
| `components/agent_trace.py` | 执行步骤可视化（`AgentStep` 列表） |
| `components/citation_view.py` | 引用来源展示 |

## agent/

| 文件 | 职责 |
|------|------|
| `orchestrator.py` | 兼容入口，委托 `AgentRuntime.execute()` |
| `router.py` | LLM 结构化 Router + 规则回退；构建 `RetrievalScope` |
| `runtime/runtime.py` | **主执行入口**：session、memory、route、skill 调度 |
| `runtime/context.py` | `AgentContext` 依赖容器（settings / store / llm） |
| `runtime/requests.py` | `ExecuteRequest` / `ExecuteResult` |
| `runtime/trace.py` | `TraceCollector`，累积 `AgentStep` |
| `runtime/history.py` | Runtime 桥接：Memory → Prompt history 文本 |
| `runtime/session.py` | Session 状态辅助 |
| `callbacks/streamlit_callback.py` | 预留（当前 trace 走 `skills/helpers.emit_step`） |
| `skills/__init__.py` | 重导出 `skills` 包（兼容 import 路径） |

## skills/

| 文件 | 职责 |
|------|------|
| `base.py` | `BaseSkill` 抽象 |
| `context.py` | `SkillContext`（含 `conversation_history`） |
| `registry.py` | Skill 注册与按 intent 查找 |
| `helpers.py` | `invoke_tool()`、`emit_step()` |
| `qa_skill/skill.py` | QA：search → format → generate → citations |
| `summary_skill/skill.py` | 总结：宽检索 → summarize → citations |
| `analysis_skill/skill.py` | 分析：多 sub-query 检索 → analyze → citations |

## tools/

| 文件 | 职责 |
|------|------|
| `base.py` | `BaseTool` 抽象 |
| `context.py` | `ToolExecutionContext`（settings / vector_store / llm） |
| `registry.py` | Tool 注册、生命周期、`run()` |
| `search_document_tool.py` | 语义检索（委托 RAGService） |
| `rag_tool.py` | `rag.retrieve`、`rag.format_context` |
| `generator_tool.py` | `llm.generate` |
| `summarize_tool.py` | 总结生成（含 history） |
| `analyze_tool.py` | 分析生成（含 history） |
| `citation_tool.py` | `rag.build_citations` |
| `retriever.py` | LangChain Retriever 适配（委托 RAGService） |
| `doc_loader.py` | 按 doc_id 加载片段（辅助） |

## memory/

| 文件 | 职责 |
|------|------|
| `base.py` | `ConversationMemory`、`ConversationTurn` |
| `manager.py` | `MemoryManager` 生命周期（**仅 Runtime 使用**） |
| `in_memory_store.py` | 内存存储实现 |
| `storage_interface.py` | 存储抽象 |
| `history_format.py` | history 格式化与 token / turns 截断 |

## ingestion/

| 文件 | 职责 |
|------|------|
| `pdf_parser.py` | PyMuPDF 按页解析 |
| `chunker.py` | 文本切分 + metadata 赋值 |
| `embedder.py` | 批量 Embedding |
| `indexer.py` | 入库编排（parse → chunk → embed → chroma） |
| `utils.py` | 文件 hash 等工具 |

## retrieval/

| 文件 | 职责 |
|------|------|
| `rag_service.py` | 统一检索入口（query + filter + dedupe + format_context） |
| `chroma_store.py` | Chroma 客户端、`build_where_filter()` |
| `chunk_utils.py` | 去重、分数过滤 |
| `citation_builder.py` | 检索结果 → Citation 列表 |
| `retriever_factory.py` | LangChain Retriever 工厂 |

## models/

| 文件 | 职责 |
|------|------|
| `schemas.py` | Pydantic 契约（ChunkMetadata、AgentStep、AnswerResult 等） |
| `doc_registry.py` | 文档注册表 CRUD（SQLite） |

## llm/

| 文件 | 职责 |
|------|------|
| `factory.py` | LLM / Embedding 实例化（Ollama / OpenAI / HuggingFace） |

## config/

| 文件 | 职责 |
|------|------|
| `settings.py` | 全局配置（路径、Chroma、chunk、LLM、memory 预算） |
| `prompts.py` | Router / Skill / Tool Prompt 模板 |
| `env_bootstrap.py` | 项目根路径与 HuggingFace 环境引导 |

## evaluation/

| 文件 | 职责 |
|------|------|
| `runner.py` | CLI 入口 |
| `harness.py` | 隔离知识库搭建 |
| `datasets/` | Benchmark 定义与加载 |
| `metrics/` | Top-k Hit Rate、LLM-as-Judge、报告生成 |
| `reports/` | Markdown 报告输出 |

## mcp_server/

| 文件 | 职责 |
|------|------|
| `server.py` | MCP 服务入口（stdio） |
| `tools.py` | 注册 `search_document` Tool |
| `resources.py` | 文档索引 / 单文档 Resource |

## scripts/

| 文件 | 职责 |
|------|------|
| `ingest_cli.py` | 命令行批量入库 |
| `reset_db.py` | 清空 Chroma + registry |
| `verify_mcp.py` | MCP 服务验证脚本 |

## tests/

分层测试，详见 [../tests/README.md](../tests/README.md)。

| 目录 | 覆盖 |
|------|------|
| `rag/` | RAGService、Chroma filter、Citation |
| `router/` | LLM + 规则 Router |
| `tools/` | ToolRegistry、原子 Tool |
| `skills/` | Skill 编排链 |
| `runtime/` | AgentRuntime、orchestrator |
| `memory/` | 隔离、history 注入、截断 |
| `e2e/` | 入库 → 执行全链路 |
| `unit/` | 入库、packaging、MCP |
| `examples/` | RAG API 用法示例 |

## docs/

| 文件 | 内容 |
|------|------|
| `ARCHITECTURE.md` | 系统架构与数据流 |
| `METADATA_SCHEMA.md` | Chroma metadata 与 filter 约定 |
| `MODULE_MAP.md` | 本文件 |

## 已移除 / 不再使用

| 路径 | 说明 |
|------|------|
| `agent/workflows/` | 已合并为 `skills/`，文档与代码均不再引用 |
| `agent/tools/` | 已合并为顶层 `tools/` |

## pyproject 入口脚本

| 脚本 | 目标 |
|------|------|
| `flowrag-ingest` | `scripts.ingest_cli:main` |
| `flowrag-reset` | `scripts.reset_db:main` |
| `flowrag-eval` | `evaluation.runner:main` |
| `flowrag-mcp` | `mcp_server.server:main` |
