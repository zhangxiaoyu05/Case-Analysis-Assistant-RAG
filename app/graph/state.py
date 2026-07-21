"""
LangGraph RAG 图状态定义

定义 RagState TypedDict（图中流转的状态字典）和 GraphResult dataclass（调用方友好返回）。
"""

from dataclasses import dataclass, field
from typing import Optional, TypedDict


# ============================================================
# RagState — LangGraph 图内部状态
# ============================================================
class RagState(TypedDict, total=False):
    """RAG 管道的完整状态。每个节点只读写自己关心的字段。"""

    # ---- 输入 ----
    query: str
    history: list[dict]  # [{"role": "user"/"assistant", "content": "..."}]
    memory_summary: str  # 早期对话的累积摘要（由 MemoryManager 生成）

    # ---- 意图分类 ----
    intent: str  # "drug_inquiry" | "chitchat" | "general" | "attack"
    intent_confidence: float  # 0.0 ~ 1.0

    # ---- 混合检索 ----
    search_results: list[dict]  # SearchResult 序列化为 dict 列表
    search_count: int

    # ---- 重排序 ----
    ranked_docs: list[dict]  # RankedDocument 序列化为 dict 列表
    ranked_count: int

    # ---- 答案生成 ----
    answer: str
    sources: list[dict]  # 最终引用来源
    template_used: str  # "default" | "comparison" | "dosage_followup"

    # ---- 错误 ----
    error: Optional[str]
    error_node: Optional[str]


# ============================================================
# GraphResult — 调用方友好返回
# ============================================================
@dataclass
class GraphResult:
    """编译后的图执行完毕后，给调用方的结构化结果。"""

    success: bool
    answer: str = ""
    sources: list[dict] = field(default_factory=list)
    intent: str = ""
    intent_confidence: float = 0.0
    template_used: str = "default"
    error: Optional[str] = None
    search_count: int = 0
    ranked_count: int = 0

    @classmethod
    def from_state(cls, state: RagState) -> "GraphResult":
        """从 RagState 字典构建 GraphResult。"""
        return cls(
            success=state.get("error") is None,
            answer=state.get("answer", ""),
            sources=state.get("sources", []),
            intent=state.get("intent", ""),
            intent_confidence=state.get("intent_confidence", 0.0),
            template_used=state.get("template_used", "default"),
            error=state.get("error"),
            search_count=state.get("search_count", 0),
            ranked_count=state.get("ranked_count", 0),
        )
