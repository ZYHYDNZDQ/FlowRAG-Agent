# FlowRAG-Agent 测试说明

## 运行方式

```bash
# 全部测试
pytest tests/ -v

# 仅 RAG 使用示例（推荐新手阅读）
pytest tests/examples/ -v

# 仅端到端流水线
pytest tests/test_rag_pipeline.py -v

# 单个文件
pytest tests/test_pdf_parser.py -v
```

## 目录结构

```
tests/
├── conftest.py              # 共享 fixture（临时目录、FakeEmbeddings、样例 PDF）
├── fixtures/
│   └── sample_content.py    # 样例合同文本与查询表
├── examples/
│   └── test_rag_usage_examples.py   # ⭐ 可复制的 RAG 用法示例
├── test_rag_pipeline.py   # 端到端入库 + 检索
├── test_pdf_parser.py       # PyMuPDF 解析
├── test_chunker.py          # 切分与 metadata
├── test_retriever.py        # Chroma where filter
├── test_citation_builder.py
├── test_doc_registry.py
└── test_router.py
```

## 示例测试一览

| 测试函数 | 演示内容 |
|----------|----------|
| `test_example_step_by_step_pipeline` | 手动逐步：parse → chunk → embed → store → query |
| `test_example_ingest_pdf_one_liner` | 生产路径：一行 `ingest_pdf` 完成入库 |
| `test_example_retrieve_with_sources` | `retrieve_with_sources` 返回文件名 + 页码 |
| `test_example_langchain_retriever_invoke` | LangChain `retriever.invoke()` |
| `test_example_page_range_filter` | 按页码范围过滤检索 |
| `test_example_build_citations_from_hits` | 检索结果 → Citation 列表 |
| `test_example_parametrized_queries` | 表驱动多查询冒烟测试 |

## 共享 Fixture

| Fixture | 作用 |
|---------|------|
| `rag_settings` | 隔离的临时 data/chroma/registry 路径 |
| `fake_embeddings` | 128 维假向量，无需下载模型 |
| `chroma_store` | 已连接的 `ChromaStore` |
| `doc_registry` | 临时 SQLite 文档注册表 |
| `sample_contract_pdf` | 自动生成的 2 页样例 PDF |
| `ingested_contract` | 已完成入库的 (result, store, registry, embeddings) 元组 |

## 扩展样例数据

编辑 `tests/fixtures/sample_content.py`：

```python
SAMPLE_QUERIES.append({
    "query": "your question",
    "expected_page": 1,
    "keyword_in_chunk": "keyword",
})
```

然后运行：

```bash
pytest tests/examples/test_rag_usage_examples.py::test_example_parametrized_queries -v
```

## 注意事项

- 样例 PDF 页内文本使用 **英文**，避免 Windows 下 PyMuPDF 中文字体提取乱码。
- 测试使用 `FakeEmbeddings`，不依赖 GPU 或 HuggingFace 模型下载。
- 集成真实 Embedding 模型时，可去掉 `embeddings=fake_embeddings` 参数，改用默认 `get_embeddings()`。
