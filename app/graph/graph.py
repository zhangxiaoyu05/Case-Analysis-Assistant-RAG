"""
LangGraph RAG 流程图构建

构建并编译 RAG 管道的 StateGraph。模块级单例 _compiled_graph 在首次调用时编译，
后续所有请求复用同一编译图。

v1.0.0: 从药品问答改造为病例分析，流程:
    START → intent → [条件路由]
      ├─ "clinical" → case_preprocess → multi_retrieve → rank → synthesize → generate → END
      ├─ "chitchat" → chitchat → END（问候白名单，不走检索）
      └─ "not_clinical" → reject → END（统一拦截，不调 LLM）
"""

from typing import Optional

from langgraph.graph import END, START, StateGraph
from loguru import logger

from app.graph.edges import (
    route_after_case_preprocess,
    route_after_intent,
    route_after_retrieve,
)
from app.graph.nodes import (
    case_preprocess_node,
    chitchat_node,
    generate_node,
    intent_node,
    multi_retrieve_node,
    rank_node,
    reject_node,
    synthesize_node,
)
from app.graph.state import RagState


def build_graph() -> StateGraph:
    """
    构建并编译 RAG 管道 StateGraph。

    Returns:
        已编译的 CompiledStateGraph（支持 .invoke(state) 和 .ainvoke(state)）
    """
    logger.info("正在构建 LangGraph RAG 流程（临床病例分析）...")

    builder = StateGraph(RagState)

    # ---- 注册节点（8 个）----
    builder.add_node("intent", intent_node)
    builder.add_node("case_preprocess", case_preprocess_node)
    builder.add_node("multi_retrieve", multi_retrieve_node)
    builder.add_node("rank", rank_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("generate", generate_node)
    builder.add_node("chitchat", chitchat_node)
    builder.add_node("reject", reject_node)

    # ---- 边 ----
    # 入口
    builder.add_edge(START, "intent")

    # 门禁路由: clinical → case_preprocess, chitchat → chitchat, not_clinical → reject
    builder.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "case_preprocess": "case_preprocess",
            "chitchat": "chitchat",
            "reject": "reject",
        },
    )

    # 病例预处理 → 多路检索
    builder.add_conditional_edges(
        "case_preprocess",
        route_after_case_preprocess,
        {
            "multi_retrieve": "multi_retrieve",
        },
    )

    # 检索后 → 重排序
    builder.add_conditional_edges(
        "multi_retrieve",
        route_after_retrieve,
        {
            "rank": "rank",
        },
    )

    # 重排序 → 上下文合成
    builder.add_edge("rank", "synthesize")

    # 上下文合成 → 生成
    builder.add_edge("synthesize", "generate")

    # 终点
    builder.add_edge("generate", END)
    builder.add_edge("chitchat", END)
    builder.add_edge("reject", END)

    compiled = builder.compile()
    logger.info("LangGraph RAG 流程构建完成（8 节点）")
    return compiled


# ============================================================
# 模块级单例
# ============================================================
_compiled_graph: Optional[StateGraph] = None


def get_graph() -> StateGraph:
    """
    获取编译好的 RAG 图单例。

    首次调用时编译，后续调用直接返回缓存的编译图。
    线程安全（Python GIL + 模块级赋值）。
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
