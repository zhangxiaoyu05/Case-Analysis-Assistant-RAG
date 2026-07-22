"""
用户路由

GET    /api/v1/user/settings          — 获取用户设置（display_name）
PUT    /api/v1/user/settings          — 更新 display_name
GET    /api/v1/user/profile           — 获取所有画像字段 + 有效字段列表
PUT    /api/v1/user/profile           — 批量更新画像字段
DELETE /api/v1/user/profile/{field_name} — 删除单个画像字段
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from loguru import logger

from app.api.dependencies import get_current_user
from app.services.user_manager import UserManager
from app.services.user_profile_manager import UserProfileManager

router = APIRouter(prefix="/user", tags=["用户"])


# ============================================================
# 请求/响应模型
# ============================================================
class UpdateDisplayNameRequest(BaseModel):
    display_name: str | None = Field(
        default=None, max_length=100,
        description="新的显示名称（昵称）。传 null 或空字符串清除昵称。",
    )


class ProfileFieldUpdate(BaseModel):
    field_name: str = Field(..., description="画像字段名（如 medical_history）")
    field_value: str = Field(default="", max_length=500, description="字段值。空字符串表示删除该字段。")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="置信度（手动编辑默认 1.0）")


class UpdateProfileRequest(BaseModel):
    fields: list[ProfileFieldUpdate] = Field(..., min_length=1, max_length=20, description="要更新的画像字段列表")


class UserSettingsResponse(BaseModel):
    user_id: int
    username: str
    display_name: str | None = None
    created_at: str | None = None
    last_login_at: str | None = None


class ProfileFieldValue(BaseModel):
    field_value: str
    confidence: float


class ValidFieldInfo(BaseModel):
    field_name: str
    label: str


class ProfileResponse(BaseModel):
    profile: dict[str, ProfileFieldValue]
    valid_fields: list[ValidFieldInfo]


# ============================================================
# GET /api/v1/user/settings — 用户设置
# ============================================================
@router.get(
    "/settings",
    response_model=UserSettingsResponse,
    summary="获取用户设置",
    description="返回当前登录用户的基本信息（用户名、显示名称等）。",
)
def get_settings(
    current_user: dict = Depends(get_current_user),
) -> UserSettingsResponse:
    manager = UserManager()
    try:
        user = manager.get_user_by_id(current_user["user_id"])
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")
        return UserSettingsResponse(**user)
    finally:
        manager.close()


# ============================================================
# PUT /api/v1/user/settings — 更新显示名称
# ============================================================
@router.put(
    "/settings",
    response_model=UserSettingsResponse,
    summary="更新显示名称",
    description="更新当前用户的显示名称（昵称）。传 null 或空字符串清除昵称，恢复显示用户名。",
)
def update_settings(
    request: UpdateDisplayNameRequest,
    current_user: dict = Depends(get_current_user),
) -> UserSettingsResponse:
    manager = UserManager()
    try:
        ok = manager.update_display_name(current_user["user_id"], request.display_name)
        logger.info(
            f"用户 [{current_user['user_id']}] 更新显示名称: "
            f"{request.display_name} (ok={ok})"
        )
        user = manager.get_user_by_id(current_user["user_id"])
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")
        return UserSettingsResponse(**user)
    finally:
        manager.close()


# ============================================================
# GET /api/v1/user/profile — 获取画像
# ============================================================
@router.get(
    "/profile",
    response_model=ProfileResponse,
    summary="获取用户画像",
    description="返回当前用户的所有画像字段（长期记忆提取的人口属性）及有效字段列表。",
)
def get_profile(
    current_user: dict = Depends(get_current_user),
) -> ProfileResponse:
    manager = UserProfileManager()
    try:
        profile = manager.get_profile(current_user["user_id"])
        valid_fields = UserProfileManager.get_valid_fields()
        return ProfileResponse(profile=profile, valid_fields=valid_fields)
    finally:
        manager.close()


# ============================================================
# PUT /api/v1/user/profile — 批量更新画像
# ============================================================
@router.put(
    "/profile",
    response_model=dict,
    summary="批量更新画像字段",
    description="批量更新用户的画像字段。field_value 为空字符串表示删除该字段。",
    responses={
        422: {"description": "字段验证失败"},
    },
)
def update_profile(
    request: UpdateProfileRequest,
    current_user: dict = Depends(get_current_user),
) -> dict:
    manager = UserProfileManager()
    try:
        fields_payload = [f.model_dump() for f in request.fields]
        results = manager.update_profile_batch(current_user["user_id"], fields_payload)
        logger.info(
            f"用户 [{current_user['user_id']}] 更新了 {len(results)} 个画像字段"
        )
        return {"updated": results}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    finally:
        manager.close()


# ============================================================
# DELETE /api/v1/user/profile/{field_name} — 删除单个画像字段
# ============================================================
@router.delete(
    "/profile/{field_name}",
    response_model=dict,
    summary="删除画像字段",
    description="删除当前用户的单个画像字段。",
    responses={
        404: {"description": "字段不存在或无效"},
    },
)
def delete_profile_field(
    field_name: str,
    current_user: dict = Depends(get_current_user),
) -> dict:
    field_name = field_name.strip().lower()
    manager = UserProfileManager()
    try:
        deleted = manager.delete_field(current_user["user_id"], field_name)
        logger.info(
            f"用户 [{current_user['user_id']}] 删除画像字段: "
            f"{field_name} (deleted={deleted})"
        )
        return {"field_name": field_name, "deleted": deleted}
    finally:
        manager.close()
