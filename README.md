# FlowRAG-Agent

面向企业知识管理场景的**本地化 AI Agent 系统**。用户上传 PDF 后，Agent 理解任务意图，动态选择检索、总结、分析等工作流，并基于 Chroma 向量库生成带引用来源的可信回答。

## 核心能力

- PDF 上传、解析与文本切分
- Embedding 入库 Chroma（本地持久化）
- RAG 检索增强生成
- Agent 动态工作流：问答 / 总结 / 文档分析
- 答案附带引用来源（文件名 + 页码）
- Streamlit Web 界面，可视化 Agent 执行过程

## 技术栈

| 组件 | 选型 |
|------|------|
| 语言 | Python 3.10+ |
| 框架 | LangChain (LCEL) |
| 向量库 | ChromaDB |
| 界面 | Streamlit |
| PDF | PyMuPDF |
| Embedding | sentence-transformers (bge-small-zh) |
| LLM | Ollama / OpenAI 兼容 API |

## 快速开始

```bash
# 1. 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 2. 安装依赖
pip install -e ".[dev,ollama]"

# 3. 配置环境
copy .env.example .env          # Windows
# cp .env.example .env            # Linux/macOS

# 4. 启动 Streamlit
streamlit run app/streamlit_app.py

# 5. 运行测试（含 RAG 用法示例）
pytest tests/ -v
pytest tests/examples/ -v    # 仅查看可复制的 API 示例
```

测试说明见 [tests/README.md](tests/README.md)。

## 项目结构

```
FlowRAG-Agent/
├── app/                 # Streamlit 入口
├── ui/                  # UI 组件与 session 状态
├── agent/               # Orchestrator、Router、Workflows
├── ingestion/           # PDF 解析 → 切分 → 入库
├── retrieval/           # Chroma 检索与引用构建
├── models/              # Pydantic Schema、文档注册表
├── llm/                 # LLM / Embedding 工厂
├── config/              # 配置与 Prompt 模板
├── scripts/             # CLI 工具
├── tests/               # 单元测试
├── docs/                # 架构与设计文档
└── data/                # 运行时数据（gitignore）
```

详细设计见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 与 [docs/METADATA_SCHEMA.md](docs/METADATA_SCHEMA.md)。

## 开发状态

| 模块 | 状态 |
|------|------|
| 配置 / Schema / Prompts | ✅ 完成 |
| doc_registry (SQLite) | ✅ 完成 |
| Chroma where filter / Citation builder | ✅ 完成 |
| Intent Router (规则层) | ✅ 完成 |
| Streamlit UI 骨架 | ✅ 可运行 |
| PDF 入库流水线 (RAG Core) | ✅ 完成 |
| Retriever + Citation | ✅ 完成 |
| RAG 问答 / Agent Workflows | ✅ 完成 |
| LLM 工厂 | ✅ 完成 |

## License

MIT
