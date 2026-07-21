"""
LangGraph RAG 流程图构建

构建并编译 RAG 管道的 StateGraph。模块级单例 _compiled_graph 在首次调用时编译，
后续所有请求复用同一编译图。

流程:
    START → intent → [条件路由]
      ├─ "drug_inquiry" → retrieve → rank → generate → END
      ├─ "chitchat" → chitchat → END（问候/闲聊，不走检索）
      ├─ "general" → general → END（非药品正常问题，LLM 直接回答）
      ├─ "attack" → attack → END（提示词注入/越狱，安全拒绝）
      └─ intent 出错 → retrieve（降级继续）
"""

from typing import Optional

from langgraph.graph import END, START, StateGraph
from loguru import logger

from app.graph.edges import route_after_intent, route_after_retrieve
from app.graph.nodes import (
    attack_node,
    chitchat_node,
    general_node,
    generate_node,
    intent_node,
    rank_node,
    retrieve_node,
)
from app.graph.state import RagState


def build_graph() -> StateGraph:
    """
    构建并编译 RAG 管道 StateGraph。

    Returns:
        已编译的 CompiledStateGraph（支持 .invoke(state) 和 .ainvoke(state)）
    """
    logger.info("正在构建 LangGraph RAG 流程...")

    builder = StateGraph(RagState)

    # ---- 注册节点 ----
    builder.add_node("intent", intent_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rank", rank_node)
    builder.add_node("generate", generate_node)
    builder.add_node("chitchat", chitchat_node)
    builder.add_node("general", general_node)
    builder.add_node("attack", attack_node)

    # ---- 边 ----
    # 入口
    builder.add_edge(START, "intent")

    # 意图路由: drug_inquiry → retrieve, chitchat → chitchat, general → general, attack → attack
    builder.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "retrieve": "retrieve",
            "chitchat": "chitchat",
            "general": "general",
            "attack": "attack",
        },
    )

    # 检索后 → 重排序
    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "rank": "rank",
        },
    )

    # 重排序 → 生成
    builder.add_edge("rank", "generate")

    # 终点
    builder.add_edge("generate", END)
    builder.add_edge("chitchat", END)
    builder.add_edge("general", END)
    builder.add_edge("attack", END)

    compiled = builder.compile()
    logger.info("LangGraph RAG 流程构建完成")
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
