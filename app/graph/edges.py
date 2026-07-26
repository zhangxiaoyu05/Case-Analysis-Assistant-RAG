"""
LangGraph 条件路由函数

每个路由函数接收 RagState，返回目标节点名称字符串。

v1.0.0: 从药品三元路由改为临床病例路由。
  clinical → case_preprocess（进入病例预处理 + RAG 全流程）
  chitchat → chitchat（问候白名单）
  not_clinical → reject（统一拦截）
"""

from app.graph.state import RagState


def route_after_intent(state: RagState) -> str:
    """
    门禁后的路由。

    - clinical → 病例预处理（RAG 全流程第一步）
    - chitchat → 闲聊回应（问候白名单命中，不走检索）
    - not_clinical → 统一拦截（非临床问题，不调 LLM）
    - 门禁出错 → 降级放行到病例预处理
    """
    if state.get("error_node") == "intent":
        return "case_preprocess"

    intent = state.get("intent", "")
    if intent == "chitchat":
        return "chitchat"
    if intent == "not_clinical":
        return "reject"
    return "case_preprocess"


def route_after_case_preprocess(state: RagState) -> str:
    """
    病例预处理后的路由。

    - 有 error_node 且为 case_preprocess → 仍然进入检索（降级）
    - 否则 → 进入多路检索
    """
    return "multi_retrieve"


def route_after_retrieve(state: RagState) -> str:
    """
    检索后的路由。

    - 有结果或无结果 → 统一走重排序（空列表 rank_node 会跳过）
    - 检索出错但仍可继续 → 走 rank（rank_node 会回退）
    """
    return "rank"
