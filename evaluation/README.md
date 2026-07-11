# FlowRAG-Agent Evaluation

离线评估模块，量化 RAG 检索与 Agent 生成效果。**独立运行**，不启动 Streamlit，不写入生产 `data/`。

架构背景见 [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)。

## 目录

```
evaluation/
├── datasets/          # 评估数据集（与业务代码分离）
│   ├── schema.py      # RetrievalCase / GenerationCase
│   ├── loader.py      # Benchmark 加载
│   └── contract_benchmark.py
├── metrics/
│   ├── retrieval.py   # Top-k Hit Rate
│   ├── generation.py  # LLM-as-Judge
│   └── report.py      # Markdown 报告
├── harness.py         # 隔离知识库搭建
├── runner.py          # CLI 入口
└── reports/           # Markdown 报告输出
```

## 指标

| 类型 | 指标 | 说明 |
|------|------|------|
| Retrieval | **Top-k Hit Rate** | 期望页码或关键词出现在 Top-k 检索结果中 |
| Generation | **LLM-as-Judge** | 1–5 分评分，≥4 为 pass |

## 运行

```bash
# 离线模式（FakeEmbeddings + FakeLLM，无需 Ollama）
python -m evaluation.runner --fake

# 等效
flowrag-eval --fake

# 使用真实模型（需配置 .env 中 LLM / Embedding）
python -m evaluation.runner

# 仅检索评估
python -m evaluation.runner --fake --skip-generation

# 指定 benchmark 与输出目录
python -m evaluation.runner --benchmark contract --output-dir evaluation/reports
```

## 评估流程

```
load_benchmark（datasets/）
    → build_eval_environment（隔离 ingest 样例 PDF）
    → evaluate_retrieval（Top-k Hit Rate）
    → evaluate_generation（AgentRuntime + LLM-as-Judge）
    → build_markdown_report → reports/*.md
```

**不启动 Streamlit，不读写生产 `data/`。**

## Retrieval — Top-k Hit Rate

对每个 `RetrievalCase`：

1. 调用 `RAGService.query_chunks(query, scope, top_k=k)`
2. 取返回 chunks 的前 k 条
3. 判定 **hit**（满足任一即可）：
   - `expected_page` 出现在 chunk 的 `metadata.page` 中
   - 或 `keyword_in_chunk` 出现在 chunk 文本中

```
Hit Rate = hit 数 / 总 case 数
```

## Generation — LLM-as-Judge

对每个 `GenerationCase`：

1. 调用 `AgentRuntime.execute(query)` 获取 Agent 答案
2. Judge LLM 对比：问题、参考答案、评估标准、Agent 回答
3. 返回 JSON：`{"score": 1-5, "pass": bool, "reasoning": "..."}`
4. **pass 判定**：`score >= 4`

| 分数 | 含义 |
|------|------|
| 5 | 完全正确、有依据、完整 |
| 4 | 基本正确，少量遗漏 |
| 3 | 部分正确 |
| 2 | 大部分错误或缺乏依据 |
| 1 | 完全错误或无关 |

## 离线模式（`--fake`）

| 组件 | 替换 |
|------|------|
| Embedding | `FakeEmbeddings(size=128)` |
| Agent LLM | `FakeListChatModel`（固定合同相关回复） |
| Judge LLM | `FakeListChatModel`（固定 JSON 评分） |

用于 CI 或无 Ollama / OpenAI 环境，验证**评估流水线本身**可运行。

## 扩展数据集

1. 在 `evaluation/datasets/` 新增 benchmark 文件
2. 在 `loader.py` 注册 benchmark 名
3. 运行 `python -m evaluation.runner --benchmark <name>`

不要修改 Agent 业务代码。

## 设计原则

1. **隔离**：每次运行在 `evaluation/reports/.cache/` 下建临时 Chroma / SQLite
2. **只读调用**：通过 `RAGService` / `AgentRuntime.execute()` 评估
3. **数据分离**：用例在 `datasets/`，指标在 `metrics/`

## 与 pytest 的区别

| | pytest（tests/） | evaluation/ |
|--|------------------|-------------|
| 目的 | 代码正确性 + 架构约束 | Agent 效果量化 |
| 断言 | 结构、intent、trace、隔离 | Hit Rate、Judge Score |
| LLM | FakeLLM（确定性） | 真实 LLM 或 Fake（离线） |
| 输出 | pass / fail | Markdown 报告 |

详见 [tests/README.md](../tests/README.md)。
