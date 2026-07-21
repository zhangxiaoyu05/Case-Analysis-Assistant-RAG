"""
问答相关 Pydantic 模型

定义问答 API 的请求体、响应体、来源引用等数据结构。
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ============================================================
# 请求模型
# ============================================================
class ChatRequest(BaseModel):
    """单轮问答请求"""

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="用户提问内容",
        examples=["阿司匹林的用法用量是什么？"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="会话 ID。不传则自动创建新会话，传入则可继续多轮对话。",
        examples=["sess_abc123"],
    )
    stream: bool = Field(
        default=False,
        description="是否使用 SSE 流式返回（默认 false）",
    )


# ============================================================
# 响应模型
# ============================================================
class SourceDoc(BaseModel):
    """检索到的参考来源文档"""

    drug_name: str = Field(description="药品名称")
    section: Optional[str] = Field(
        default=None,
        description="所属章节（如：用法用量、禁忌、不良反应）",
    )
    chunk_text: str = Field(description="匹配的文本片段内容")
    score: Optional[float] = Field(
        default=None,
        description="相关性得分（RRF 融合后的分数）",
    )
    doc_id: Optional[int] = Field(default=None, description="原始文档数据库 ID")


class ChatResponse(BaseModel):
    """单轮问答响应"""

    answer: str = Field(description="生成的回答文本")
    sources: list[SourceDoc] = Field(
        default_factory=list,
        description="检索到的参考来源列表（用于回答溯源）",
    )
    session_id: str = Field(description="会话 ID（用于后续追问）")
    intent: Optional[str] = Field(
        default=None,
        description="意图识别结果：drug_inquiry / chitchat / general / attack",
    )
    elapsed_ms: Optional[float] = Field(
        default=None,
        description="处理耗时（毫秒）",
    )


class ChatHistoryItem(BaseModel):
    """对话历史中的单条记录"""

    role: str = Field(description="角色: user / assistant")
    content: str = Field(description="消息内容")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="消息时间")
    sources: Optional[list[SourceDoc]] = Field(
        default=None,
        description="如果是 assistant 消息，附带当时的引用来源",
    )


class HistoryResponse(BaseModel):
    """对话历史响应"""

    session_id: str = Field(description="会话 ID")
    history: list[ChatHistoryItem] = Field(
        default_factory=list,
        description="对话历史记录列表",
    )
    turn_count: int = Field(default=0, description="当前会话对话轮数")


class ClearHistoryResponse(BaseModel):
    """清除会话响应"""

    session_id: str = Field(description="已清除的会话 ID")
    cleared: bool = Field(default=True, description="是否成功清除")


# ============================================================
# 流式响应（SSE 事件）
# ============================================================
class StreamEvent(BaseModel):
    """SSE 流式事件（每条事件是一条 JSON）"""

    event: str = Field(description="事件类型: token / sources / done / error")
    data: str = Field(description="事件数据（token 文本 / JSON 序列化的 sources / 结束标记）")
