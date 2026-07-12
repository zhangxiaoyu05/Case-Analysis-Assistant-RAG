"""
健康检查路由

GET /health        - 基础健康检查
GET /health/ready  - 就绪检查（含依赖服务状态：Milvus / MySQL / Redis）
"""

from datetime import datetime, timezone

from fastapi import APIRouter
from loguru import logger

from app.schemas.common import ErrorResponse, HealthResponse, ReadinessResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="基础健康检查",
    description="返回服务运行状态和版本信息",
)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc),
        version="0.1.0",
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="就绪检查",
    description="检查服务及所有依赖（Milvus / MySQL / Redis）是否就绪",
)
async def readiness_check() -> ReadinessResponse:
    checks: dict[str, bool] = {}

    # --- Milvus ---
    try:
        from app.db.milvus_client import MilvusClient

        milvus = MilvusClient()
        milvus.connect()
        checks["milvus"] = milvus.collection_exists()
        milvus.disconnect()
    except Exception as e:
        logger.warning(f"Milvus 健康检查失败: {e}")
        checks["milvus"] = False

    # --- MySQL ---
    try:
        from app.db.mysql_client import MySQLClient

        mysql = MySQLClient()
        mysql.connect()
        checks["mysql"] = mysql.is_ready()
        mysql.disconnect()
    except Exception as e:
        logger.warning(f"MySQL 健康检查失败: {e}")
        checks["mysql"] = False

    # --- Redis ---
    try:
        import redis.asyncio as aioredis

        from app.config import config

        r = aioredis.from_url(
            f"redis://{config.REDIS_HOST}:{config.REDIS_PORT}",
            socket_connect_timeout=3,
        )
        checks["redis"] = await r.ping()
        await r.aclose()
    except Exception as e:
        logger.warning(f"Redis 健康检查失败: {e}")
        checks["redis"] = False

    all_ready = all(checks.values())
    return ReadinessResponse(
        status="ready" if all_ready else "not_ready",
        checks=checks,
    )
