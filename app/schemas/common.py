"""
通用响应模型

用于健康检查、错误响应等全局共享的 Pydantic 模型。
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str = Field(default="ok", description="服务状态: ok / degraded / down")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="检查时间")
    version: str = Field(default="0.1.0", description="API 版本号")


class ReadinessResponse(BaseModel):
    """就绪检查响应（含依赖状态）"""

    status: str = Field(description="就绪状态: ready / not_ready")
    checks: dict[str, bool] = Field(
        default_factory=dict,
        description="各依赖服务健康状态，如 {'milvus': True, 'mysql': True, 'redis': True}",
    )


class ErrorResponse(BaseModel):
    """标准错误响应"""

    detail: str = Field(description="错误描述信息")
    error_code: Optional[str] = Field(default=None, description="错误码（如 INTENT_BLOCKED / NO_RESULTS）")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="错误发生时间")
