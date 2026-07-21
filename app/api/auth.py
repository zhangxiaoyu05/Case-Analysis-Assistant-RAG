"""
API Key 鉴权依赖

通过 FastAPI Depends 注入，支持两种传递方式:
  - X-API-Key: <key>  （推荐）
  - Authorization: Bearer <key>

鉴权开关由 config.yaml 的 security.auth.enabled + 环境变量 APP_API_KEY 共同决定。

使用方式:
    from app.api.auth import verify_api_key

    router = APIRouter(dependencies=[Depends(verify_api_key)])
"""

from fastapi import Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from loguru import logger

from app.config import config

# FastAPI 安全方案定义（自动进入 OpenAPI docs）
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False, description="API 鉴权密钥")
api_key_bearer = APIKeyHeader(
    name="Authorization",
    auto_error=False,
    description='Bearer 令牌（格式: "Bearer <key>"）',
)


def _extract_api_key(
    x_api_key: str | None = Security(api_key_header),
    authorization: str | None = Security(api_key_bearer),
) -> str | None:
    """从请求头中提取 API Key。优先 X-API-Key，其次 Authorization Bearer。"""
    if x_api_key:
        return x_api_key

    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            return token

    return None


def verify_api_key(
    api_key: str | None = Depends(_extract_api_key),
) -> str:
    """
    验证 API Key。

    Raises:
        HTTPException 401: 鉴权启用但未提供 Key 或 Key 不正确。
    Returns:
        str: 验证通过的 API Key。
    """
    # 鉴权开关检查
    if not config.auth_enabled:
        return "anonymous"

    expected_key = config.APP_API_KEY

    # 防御性检查：理论上 auth_enabled 已保证 expected_key 非空
    if not expected_key:
        return "anonymous"

    if not api_key:
        logger.warning("API 鉴权失败: 缺少 API Key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少 API Key。请在请求头中提供 X-API-Key 或 Authorization: Bearer <key>。",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # 常数时间比较（防止时序攻击）
    if not _constant_time_compare(api_key, expected_key):
        logger.warning("API 鉴权失败: API Key 不匹配")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key 无效",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key


def _constant_time_compare(a: str, b: str) -> bool:
    """常数时间字符串比较，防止时序攻击。"""
    if len(a) != len(b):
        return False
    result = 0
    for x, y in zip(a, b):
        result |= ord(x) ^ ord(y)
    return result == 0
