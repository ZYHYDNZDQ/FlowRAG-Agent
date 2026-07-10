# Chroma Metadata Schema 与检索 Filter 约定

> FlowRAG-Agent 数据契约文档。入库、检索、引用构建均须遵守本规范。

## 1. Collection 配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `collection_name` | `flowrag_docs` | 全局单 collection |
| `distance_metric` | `cosine` | 与 bge 系列 embedding 一致 |
| `persist_directory` | `data/chroma/` | 本地持久化路径 |

## 2. Chunk Metadata 字段

### 2.1 必填（Required）

| 字段 | 类型 | 示例 | 说明 |
|------|------|------|------|
| `doc_id` | str | `a1b2c3d4-...` | 文档 UUID，主过滤键 |
| `source_file` | str | `合同_2024.pdf` | 用户可见文件名 |
| `source_path` | str | `uploads/合同_2024.pdf` | 相对 data/ 的路径 |
| `page` | int | `12` | **1-based** 页码 |
| `chunk_index` | int | `3` | 文档内顺序（0-based） |
| `chunk_id` | str | `a1b2c3d4_p0012_c0003` | 与 Chroma id 一致 |
| `ingest_version` | int | `1` | 重入库版本号 |

### 2.2 推荐（Recommended）

| 字段 | 类型 | 说明 |
|------|------|------|
| `page_count` | int | PDF 总页数 |
| `char_start` | int | 页内字符起始偏移 |
| `char_end` | int | 页内字符结束偏移 |
| `token_estimate` | int | 预估 token 数 |
| `section_hint` | str | 章节标题提示 |
| `language` | str | 语言代码，默认 `zh` |
| `file_hash` | str | SHA256，用于去重 |

### 2.3 约束

- metadata 必须为**扁平标量**（str / int / float / bool）
- 禁止嵌套 JSON、禁止在 metadata 存正文
- 字段名全局统一，不可随意更名

## 3. chunk_id 生成规则

```
{doc_id_short}_p{page:04d}_c{chunk_index:04d}
```

- `doc_id_short`：UUID 前 8 位
- 示例：`a1b2c3d4_p0012_c0003`

## 4. doc_registry 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `doc_id` | str | 主键 |
| `source_file` | str | 原始文件名 |
| `source_path` | str | 磁盘路径 |
| `file_hash` | str | SHA256 |
| `page_count` | int | 总页数 |
| `chunk_count` | int | chunk 数量 |
| `status` | enum | `pending` / `indexed` / `failed` |
| `error_message` | str | 失败原因 |
| `ingested_at` | datetime | 入库时间 |
| `ingest_version` | int | 当前版本 |

## 5. 检索 Filter 预设

### F0：全库

```python
where = None
```

### F1：单文档

```python
where = {"doc_id": {"$eq": "<doc_id>"}}
```

### F2：多文档

```python
where = {"doc_id": {"$in": ["<id1>", "<id2>"]}}
```

### F4：页码范围

```python
where = {
    "$and": [
        {"doc_id": {"$eq": "<doc_id>"}},
        {"page": {"$gte": 10}},
        {"page": {"$lte": 25}},
    ]
}
```

## 6. 工作流 × Filter 映射

| 工作流 | 默认 scope | Filter | top_k |
|--------|------------|--------|-------|
| QA | 用户选中 / 全库 | F1 / F2 / F0 | 6 |
| Summarize | 单文档 | F1 | 15~20 |
| Analyze | 单/多文档 | F1 / F2 (+ F4) | 5 × N queries |

## 7. RetrievalScope → where 构造

```python
mode == "all"      → where = None
mode == "single"   → F1(doc_ids[0])
mode == "selected" → F2(doc_ids)
page_start/end     → 与上式 $and 合并为 F4
```

## 8. Citation 构建

引用**直接从检索 metadata 构建**，不依赖 LLM 自报页码。

去重键：`(source_file, page, chunk_id)`，保留最高相似度。

LLM 引用格式：`[来源: {source_file} 第{page}页]`

## 9. 删除与更新

| 操作 | 行为 |
|------|------|
| 删除文档 | `delete(where={"doc_id": {"$eq": id}})` + 更新 registry |
| 覆盖入库 | 先 delete 同 doc_id，再 add，`ingest_version++` |
| 禁止 | 只删 registry 不删 Chroma |

## 10. 页内切分策略（MVP）

- 默认 `PAGE_BREAK_STRATEGY=intra_page`：chunk 不跨页
- `page` 取 chunk 所在页（1-based）
- 空页不生成 chunk

详细配置见 `config/settings.py`。
