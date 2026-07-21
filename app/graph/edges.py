"""
LangGraph 条件路由函数

每个路由函数接收 RagState，返回目标节点名称字符串。
"""

from app.graph.state import RagState


def route_after_intent(state: RagState) -> str:
    """
    意图分类后的路由。

    - drug_inquiry → 走正常检索流程
    - chitchat → 闲聊回应（不走检索）
    - general → 通用问答（不走检索，LLM 直接回答）
    - attack → 拒绝回答（安全提示）
    - 意图节点出错 → 降级到检索流程（出兜底回答）
    """
    if state.get("error_node") == "intent":
        # 意图识别失败但默认视为药品问题，继续走正常流程
        return "retrieve"

    intent = state.get("intent", "")
    if intent == "chitchat":
        return "chitchat"
    if intent == "general":
        return "general"
    if intent == "attack":
        return "attack"
    return "retrieve"


def route_after_retrieve(state: RagState) -> str:
    """
    检索后的路由。

    - 有结果或无结果 → 统一走重排序（空列表 rank_node 会跳过）
    - 检索出错但仍可继续 → 走 rank（rank_node 会回退）
    """
    return "rank"
