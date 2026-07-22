"""
FastAPI 依赖注入

提供应用级别单例的获取函数（供路由使用）。
单例在 lifespan 中初始化。

新增 Phase 0:
  - get_current_user(): JWT 鉴权依赖，从 Authorization header 解析用户信息
"""

from fastapi import Depends, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from loguru import logger

from app.graph.graph import get_graph as _get_graph
from app.services.history_manager import AsyncRedisHistoryManager

# ============================================================
# 单例引用（由 lifespan 设置）
# ============================================================
_history_manager: AsyncRedisHistoryManager | None = None


def set_history_manager(hm: AsyncRedisHistoryManager) -> None:
    """由 lifespan startup 调用，设置全局历史管理器。"""
    global _history_manager
    _history_manager = hm


def get_history_manager() -> AsyncRedisHistoryManager:
    """获取异步 Redis 历史管理器单例。"""
    if _history_manager is None:
        raise RuntimeError(
            "HistoryManager 尚未初始化。请确保应用通过 lifespan 启动。"
        )
    return _history_manager


def get_graph():
    """获取编译好的 LangGraph RAG 图单例。"""
    return _get_graph()


# ============================================================
# JWT 鉴权依赖（Phase 0 新增）
# ============================================================
# Authorization: Bearer <token> 头提取
_bearer_scheme = APIKeyHeader(name="Authorization", auto_error=False, description="Bearer <JWT token>")


def get_current_user(
    authorization: str | None = Depends(_bearer_scheme),
) -> dict:
    """
    从 Authorization: Bearer <token> 头解析 JWT，返回当前用户信息。

    Returns:
        {"user_id": int, "username": str}

    Raises:
        HTTPException 401: token 缺失、无效或过期
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未登录，请先登录",
        )

    # 解析 Bearer 格式
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization 头格式错误，应为 Bearer <token>",
        )

    # 验证 JWT
    from app.services.user_manager import decode_token

    try:
        payload = decode_token(token)
    except Exception:
        logger.warning(f"JWT 验证失败: token={token[:20]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录已过期，请重新登录",
        )

    user_id_str = payload.get("sub", "")
    username = payload.get("username", "")

    if not user_id_str or not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 格式无效",
        )

    return {"user_id": int(user_id_str), "username": username}
