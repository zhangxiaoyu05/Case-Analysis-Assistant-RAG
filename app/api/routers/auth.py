"""
认证路由

POST /api/v1/auth/register — 注册新用户
POST /api/v1/auth/login    — 用户登录
GET  /api/v1/auth/me       — 获取当前用户信息
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from loguru import logger

from app.api.dependencies import get_current_user
from app.services.user_manager import UserManager

router = APIRouter(prefix="/auth", tags=["认证"])


# ============================================================
# 请求/响应模型
# ============================================================
class RegisterRequest(BaseModel):
    username: str = Field(
        ..., min_length=2, max_length=30,
        description="用户名（中文/英文/数字/下划线）",
        examples=["zhangsan"],
    )
    password: str = Field(
        ..., min_length=4, max_length=128,
        description="密码（≥4 字符）",
        examples=["1234"],
    )


class LoginRequest(BaseModel):
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")


class AuthResponse(BaseModel):
    token: str = Field(description="JWT token")
    user_id: int = Field(description="用户 ID")
    username: str = Field(description="用户名")


class UserInfoResponse(BaseModel):
    user_id: int
    username: str
    display_name: str | None = None
    created_at: str | None = None
    last_login_at: str | None = None


# ============================================================
# POST /api/v1/auth/register
# ============================================================
@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    summary="用户注册",
    description="注册新用户。用户名 2-30 字符，密码 ≥4 字符。",
    responses={
        201: {"description": "注册成功"},
        409: {"description": "用户名已存在"},
        422: {"description": "参数校验失败"},
    },
)
def register(request: RegisterRequest) -> AuthResponse:
    """注册新用户并返回 JWT token。"""
    manager = UserManager()
    try:
        user = manager.register(request.username, request.password)
        token_result = manager.login(request.username, request.password)
        return AuthResponse(
            token=token_result["token"],
            user_id=user["user_id"],
            username=user["username"],
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    finally:
        manager.close()


# ============================================================
# POST /api/v1/auth/login
# ============================================================
@router.post(
    "/login",
    response_model=AuthResponse,
    summary="用户登录",
    description="使用用户名和密码登录，返回 JWT token（7 天有效）。",
    responses={
        200: {"description": "登录成功"},
        401: {"description": "用户名或密码错误"},
    },
)
def login(request: LoginRequest) -> AuthResponse:
    """用户登录。"""
    manager = UserManager()
    try:
        result = manager.login(request.username, request.password)
        return AuthResponse(
            token=result["token"],
            user_id=result["user_id"],
            username=result["username"],
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    finally:
        manager.close()


# ============================================================
# GET /api/v1/auth/me
# ============================================================
@router.get(
    "/me",
    response_model=UserInfoResponse,
    summary="获取当前用户信息",
    description="通过 JWT token 获取当前登录用户的信息。",
    responses={
        200: {"description": "成功"},
        401: {"description": "未登录或 token 无效/过期"},
    },
)
def get_me(current_user: dict = Depends(get_current_user)) -> UserInfoResponse:
    """获取当前登录用户信息。"""
    manager = UserManager()
    try:
        user = manager.get_user_by_id(current_user["user_id"])
        if user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")
        return UserInfoResponse(**user)
    finally:
        manager.close()
