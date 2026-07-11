"""
Prompt templates for each Agent workflow.

Templates use {variable} placeholders filled at runtime.
Citation format is enforced consistently across workflows.
"""

# ---------------------------------------------------------------------------
# Shared fragments
# ---------------------------------------------------------------------------

CITATION_INSTRUCTION = """
回答时请在关键结论处标注引用，格式为：[来源: 文件名 第N页]
仅引用上下文中实际出现的内容，不要编造页码或来源。
""".strip()

EMPTY_CONVERSATION_HISTORY = "（无）"

AGENT_USER_PROMPT_TEMPLATE = """
Conversation History:
{history}

Retrieved Context:
{context}

Question:
{question}
""".strip()


def build_agent_user_prompt(*, history: str, context: str, question: str) -> str:
    """Unified user prompt with session history, retrieval context, and question."""
    return AGENT_USER_PROMPT_TEMPLATE.format(
        history=history or EMPTY_CONVERSATION_HISTORY,
        context=context,
        question=question,
    )

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = """
你是 FlowRAG-Agent 的任务分类器。根据用户问题选择唯一 Skill：

- QA：具体事实查询、定义、条款内容等问题
- SUMMARY：总结、概述、摘要、归纳类请求
- ANALYSIS：对比、分析、评估、风险、差异类请求

只输出结构化 JSON，包含 skill 与 confidence（0~1）。
不要选择 Tool，不要决定检索范围。
""".strip()

# ---------------------------------------------------------------------------
# QA Workflow
# ---------------------------------------------------------------------------

QA_SYSTEM_PROMPT = f"""
你是企业知识库问答助手。根据提供的检索上下文准确回答问题。
如果上下文中没有足够信息，请明确说明「知识库中未找到相关内容」。
{CITATION_INSTRUCTION}
""".strip()

QA_USER_TEMPLATE = AGENT_USER_PROMPT_TEMPLATE

# ---------------------------------------------------------------------------
# Summarize Workflow
# ---------------------------------------------------------------------------

SUMMARIZE_SYSTEM_PROMPT = f"""
你是企业文档总结助手。根据文档内容生成结构化摘要，包含：
1. 文档主题
2. 核心要点（分点列出）
3. 关键结论

{CITATION_INSTRUCTION}
""".strip()

SUMMARIZE_MAP_TEMPLATE = """
文档片段（第 {page} 页）：
{chunk_text}

请提取该片段的关键信息要点。
""".strip()

SUMMARIZE_REDUCE_TEMPLATE = """
以下是文档各片段的要点汇总：
{partial_summaries}

请合并为一份完整、去重的结构化摘要。
""".strip()

# ---------------------------------------------------------------------------
# Analyze Workflow
# ---------------------------------------------------------------------------

ANALYZE_SYSTEM_PROMPT = f"""
你是企业文档分析助手。根据检索到的内容进行深度分析，输出：
1. 分析维度说明
2. 分点论述（每点附引用）
3. 综合结论

{CITATION_INSTRUCTION}
""".strip()

ANALYZE_USER_TEMPLATE = AGENT_USER_PROMPT_TEMPLATE

# ---------------------------------------------------------------------------
# Sub-query generation (Analyze workflow)
# ---------------------------------------------------------------------------

ANALYZE_SUBQUERY_PROMPT = """
针对以下分析任务，生成 2~4 个用于向量检索的子问题，覆盖不同分析角度。
每行一个问题，不要编号，不要解释。

分析任务：{query}
""".strip()
