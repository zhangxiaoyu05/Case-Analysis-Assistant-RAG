"""
LangGraph RAG 图状态定义

v1.0.0: 从药品问答改造为病例分析，新增 case_profile / search_queries /
search_breakdown / synthesized_context / file_name / analysis_mode 字段。

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
    memory_summary: str  # 早期对话的累积摘要
    user_memories: str  # 跨会话用户中期记忆文本
    user_profile: str  # 用户画像文本
    file_name: str  # 上传的病例文件名（如有）
    analysis_mode: str  # 分析模式: comprehensive/diagnosis/treatment/drug_review

    # ---- 意图分类 ----
    intent: str  # "clinical" | "chitchat" | "not_clinical"
    intent_confidence: float  # 0.0 ~ 1.0

    # ---- 病例预处理 ----
    case_profile: dict  # 结构化病例信息（LLM 提取结果）
    search_queries: list[str]  # 基于病例构造的多路检索查询列表

    # ---- 多路检索 ----
    search_results: list[dict]  # SearchResult 序列化为 dict 列表
    search_count: int
    search_breakdown: dict  # 按 source_type 统计 {"drug": 5, "disease": 3, ...}

    # ---- 重排序 ----
    ranked_docs: list[dict]  # RankedDocument 序列化为 dict 列表
    ranked_count: int

    # ---- 多源上下文合成 ----
    synthesized_context: dict  # 按临床维度组织的上下文 {"disease": [...], "guideline": [...], ...}

    # ---- 答案生成 ----
    answer: str
    sources: list[dict]  # 最终引用来源
    template_used: str  # case_summary/differential_diagnosis/treatment_analysis/drug_review/guideline_lookup

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
    template_used: str = "case_summary"
    error: Optional[str] = None
    search_count: int = 0
    ranked_count: int = 0
    case_profile: dict = field(default_factory=dict)
    search_breakdown: dict = field(default_factory=dict)

    @classmethod
    def from_state(cls, state: RagState) -> "GraphResult":
        """从 RagState 字典构建 GraphResult。"""
        return cls(
            success=state.get("error") is None,
            answer=state.get("answer", ""),
            sources=state.get("sources", []),
            intent=state.get("intent", ""),
            intent_confidence=state.get("intent_confidence", 0.0),
            template_used=state.get("template_used", "case_summary"),
            error=state.get("error"),
            search_count=state.get("search_count", 0),
            ranked_count=state.get("ranked_count", 0),
            case_profile=state.get("case_profile", {}),
            search_breakdown=state.get("search_breakdown", {}),
        )
