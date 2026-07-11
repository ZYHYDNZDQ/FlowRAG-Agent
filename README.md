# FlowRAG-Agent

面向企业 PDF 知识库的**本地化 Agentic RAG 系统**。用户上传文档后，系统通过 LLM Router 选择业务 Skill（问答 / 总结 / 分析），编排检索与生成 Tool，输出带页码引用的可信回答。

> **定位**：固定 Skill 编排 + 意图路由，不是 ReAct 自主 Agent。详见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 核心能力


| 能力             | 说明                                               |
| -------------- | ------------------------------------------------ |
| PDF 入库         | 解析、切分、Embedding、写入 Chroma（本地持久化）                 |
| RAG 检索         | 按文档范围过滤、去重、Citation 构建                           |
| Agent 编排       | Runtime → Router → Skill → Tool → RAGService     |
| Session Memory | 多轮对话 history 由 Runtime 注入 Prompt                 |
| 执行轨迹           | Streamlit 可视化 Router / Retriever / Generator 等步骤 |
| 离线评估           | Top-k Hit Rate + LLM-as-Judge                    |
| MCP 服务         | 对外暴露 `search_document` 检索能力                      |


## 技术栈


| 组件        | 选型                                                 |
| --------- | -------------------------------------------------- |
| 语言        | Python 3.10+                                       |
| 框架        | LangChain                                          |
| 向量库       | ChromaDB                                           |
| 界面        | Streamlit                                          |
| PDF       | PyMuPDF                                            |
| Embedding | sentence-transformers（默认 `BAAI/bge-small-zh-v1.5`） |
| LLM       | Ollama / OpenAI 兼容 API                             |
| 配置        | pydantic-settings                                  |
| 测试        | pytest（83+）                                        |
| MCP       | `mcp`（可选，`pip install -e ".[mcp]"`）                |


## 快速开始

### 1. 环境准备

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -e ".[dev,ollama]"
```

### 2. 配置

```bash
copy .env.example .env          # Windows
# cp .env.example .env          # Linux / macOS
```

按需修改 `.env`：LLM 提供商、Embedding 模型、Chroma 路径等。完整变量见 [.env.example](.env.example)。

### 3. 启动 Web UI

```bash
streamlit run app/streamlit_app.py
```

流程：侧边栏上传 PDF → 选择文档范围 → 主区域输入问题（问答 / 总结 / 分析）。

### 4. CLI 工具


| 命令                          | 说明                       |
| --------------------------- | ------------------------ |
| `flowrag-ingest <file.pdf>` | 命令行批量入库                  |
| `flowrag-reset`             | 清空知识库（Chroma + registry） |
| `flowrag-eval --fake`       | 离线评估（无需真实 LLM）           |
| `flowrag-mcp`               | 启动 MCP 检索服务（stdio）       |


等效调用：

```bash
python -m scripts.ingest_cli path/to/file.pdf
python -m evaluation.runner --fake
python -m mcp_server.server
```

### 5. 运行测试

```bash
pytest tests/ -q -m "not example"    # 全量（跳过 API 示例）
pytest tests/ -m router -v           # 按层运行
pytest tests/examples/ -v            # RAG API 用法示例
```

测试说明见 [tests/README.md](tests/README.md)。

## 项目结构

```
FlowRAG-Agent/
├── app/                 # Streamlit 入口
├── ui/                  # UI 组件与 session 状态
├── agent/               # Runtime、Router、Orchestrator
├── skills/              # 业务 Skill（QA / Summary / Analysis）
├── tools/               # 原子 Tool 与 Registry
├── memory/              # Session 级短期记忆
├── ingestion/           # PDF 解析 → 切分 → 入库
├── retrieval/           # RAGService、Chroma、Citation
├── models/              # Pydantic Schema、doc_registry
├── llm/                 # LLM / Embedding 工厂
├── config/              # 配置与 Prompt 模板
├── evaluation/          # 离线评估
├── mcp_server/          # MCP 服务
├── scripts/             # CLI 脚本
├── tests/               # 分层测试
├── docs/                # 架构与设计文档
└── data/                # 运行时数据（gitignore）
```

## 文档索引


| 文档                                                 | 内容                          |
| -------------------------------------------------- | --------------------------- |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)       | 系统架构、数据流、扩展点                |
| [docs/MODULE_MAP.md](docs/MODULE_MAP.md)           | 模块与文件职责索引                   |
| [docs/METADATA_SCHEMA.md](docs/METADATA_SCHEMA.md) | Chroma metadata 与 filter 约定 |
| [tests/README.md](tests/README.md)                 | 测试分层与运行方式                   |
| [evaluation/README.md](evaluation/README.md)       | 离线评估指标与用法                   |


## 开发状态


| 模块                                | 状态  |
| --------------------------------- | --- |
| 配置 / Schema / Prompts             | ✅   |
| PDF 入库流水线                         | ✅   |
| RAGService + Citation             | ✅   |
| LLM Router（结构化输出 + 规则回退）          | ✅   |
| Skill 编排（QA / Summary / Analysis） | ✅   |
| AgentRuntime + Trace              | ✅   |
| Session Memory 注入                 | ✅   |
| Streamlit UI                      | ✅   |
| 分层测试（83+）                         | ✅   |
| 离线 Evaluation                     | ✅   |
| MCP Server                        | ✅   |


## 远期目标

- ReAct / Planner 自主 Agent
- 长期记忆
- 微服务部署、多租户、OCR

## License

MIT