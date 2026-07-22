"""
会话管理路由

GET    /api/v1/conversations              — 获取当前用户的对话列表
POST   /api/v1/conversations              — 创建新对话窗口
GET    /api/v1/conversations/{session_id}  — 获取指定会话信息
PATCH  /api/v1/conversations/{session_id}  — 更新对话标题
DELETE /api/v1/conversations/{session_id}  — 删除对话窗口
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from loguru import logger

from app.api.dependencies import get_current_user, get_history_manager
from app.services.conversation_manager import ConversationManager

router = APIRouter(prefix="/conversations", tags=["会话"])


# ============================================================
# 请求/响应模型
# ============================================================
class ConversationItem(BaseModel):
    id: int
    session_id: str
    title: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class ConversationListResponse(BaseModel):
    conversations: list[ConversationItem]
    count: int


class CreateConversationResponse(BaseModel):
    id: int
    session_id: str
    title: str | None = None
    created_at: str | None = None


class UpdateTitleRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=100, description="新的对话标题")


class DeleteConversationResponse(BaseModel):
    session_id: str
    deleted: bool


# ============================================================
# GET /api/v1/conversations — 对话列表
# ============================================================
@router.get(
    "",
    response_model=ConversationListResponse,
    summary="获取对话列表",
    description="返回当前用户所有活跃的对话窗口（按更新时间倒序）。",
)
def list_conversations(
    current_user: dict = Depends(get_current_user),
) -> ConversationListResponse:
    manager = ConversationManager()
    try:
        convs = manager.list_active(current_user["user_id"])
        return ConversationListResponse(
            conversations=[ConversationItem(**c) for c in convs],
            count=len(convs),
        )
    finally:
        manager.close()


# ============================================================
# POST /api/v1/conversations — 创建新对话
# ============================================================
@router.post(
    "",
    response_model=CreateConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建新对话",
    description="为当前用户创建一个新的空白对话窗口，返回 session_id 供后续问答使用。",
)
def create_conversation(
    current_user: dict = Depends(get_current_user),
) -> CreateConversationResponse:
    manager = ConversationManager()
    try:
        conv = manager.create(user_id=current_user["user_id"])
        return CreateConversationResponse(**conv)
    finally:
        manager.close()


# ============================================================
# GET /api/v1/conversations/{session_id} — 获取会话信息
# ============================================================
@router.get(
    "/{session_id}",
    response_model=ConversationItem,
    summary="获取会话信息",
    description="根据 session_id 获取指定会话的详细信息。",
    responses={
        404: {"description": "会话不存在"},
    },
)
def get_conversation(
    session_id: str,
    current_user: dict = Depends(get_current_user),
) -> ConversationItem:
    manager = ConversationManager()
    try:
        conv = manager.get_by_session_id(session_id)
        if conv is None or conv.get("user_id") != current_user["user_id"]:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
        return ConversationItem(**conv)
    finally:
        manager.close()


# ============================================================
# PATCH /api/v1/conversations/{session_id} — 更新标题
# ============================================================
@router.patch(
    "/{session_id}",
    response_model=ConversationItem,
    summary="更新对话标题",
    description="手动修改对话窗口的标题。",
    responses={
        404: {"description": "会话不存在"},
    },
)
def update_title(
    session_id: str,
    request: UpdateTitleRequest,
    current_user: dict = Depends(get_current_user),
) -> ConversationItem:
    manager = ConversationManager()
    try:
        conv = manager.get_by_session_id(session_id)
        if conv is None or conv.get("user_id") != current_user["user_id"]:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")

        manager.update_title(session_id, request.title)
        updated = manager.get_by_session_id(session_id)
        return ConversationItem(**updated)
    finally:
        manager.close()


# ============================================================
# DELETE /api/v1/conversations/{session_id} — 删除对话
# ============================================================
@router.delete(
    "/{session_id}",
    response_model=DeleteConversationResponse,
    summary="删除对话窗口",
    description="软删除对话窗口，同时清除对应的 Redis 短期记忆。跨会话的中/长期记忆不受影响。",
    responses={
        404: {"description": "会话不存在"},
    },
)
async def delete_conversation(
    session_id: str,
    current_user: dict = Depends(get_current_user),
) -> DeleteConversationResponse:
    manager = ConversationManager()
    try:
        conv = manager.get_by_session_id(session_id)
        if conv is None or conv.get("user_id") != current_user["user_id"]:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")

        # 软删除 MySQL 记录
        manager.soft_delete(session_id)

        # 清除 Redis 短期记忆
        try:
            history_manager = get_history_manager()
            await history_manager.clear_history(current_user["user_id"], session_id)
        except Exception as e:
            logger.warning(f"清除 Redis 短期记忆失败 [{session_id}]: {e}")

        logger.info(f"对话已删除: user_id={current_user['user_id']}, session_id={session_id}")
        return DeleteConversationResponse(session_id=session_id, deleted=True)
    finally:
        manager.close()
