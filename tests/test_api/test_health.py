"""
测试 API 健康检查端点

覆盖: GET /health, GET /health/ready, Pydantic schemas
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ============================================================
# TestClient fixture
# ============================================================
@pytest.fixture
def client():
    """创建测试客户端。"""
    mock_redis_hm = AsyncMock()
    mock_redis_hm.get_history.return_value = []
    mock_redis_hm.add_turn.return_value = None
    mock_redis_hm.clear_history.return_value = True
    mock_redis_hm.close = AsyncMock()

    with patch("app.api.main.get_graph"), \
         patch("app.api.main.AsyncRedisHistoryManager", return_value=mock_redis_hm):
        from app.api.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ============================================================
# GET /health
# ============================================================
class TestHealthCheck:
    """测试基础健康检查。"""

    def test_health_returns_200(self, client):
        """健康检查返回 200。"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
        assert "timestamp" in data

    def test_root_returns_html_or_fallback(self, client):
        """根路径返回问答页面 HTML（或 fallback JSON）。"""
        response = client.get("/")
        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            # 有前端文件时返回 HTML 页面
            assert "药品" in response.text or "chat" in response.text.lower()
        else:
            # 无前端文件时 fallback 为 JSON
            data = response.json()
            assert "message" in data


# ============================================================
# GET /health/ready
# ============================================================
class TestReadinessCheck:
    """测试就绪检查。"""

    def test_readiness_all_healthy(self, client):
        """所有依赖健康。"""
        with patch("app.db.milvus_client.MilvusClient") as mock_milvus, \
             patch("app.db.mysql_client.MySQLClient") as mock_mysql, \
             patch("redis.asyncio.from_url") as mock_redis_from_url:
            # Mock Milvus
            milvus_instance = MagicMock()
            milvus_instance.collection_exists.return_value = True
            mock_milvus.return_value = milvus_instance

            # Mock MySQL
            mysql_instance = MagicMock()
            mysql_instance.is_ready.return_value = True
            mock_mysql.return_value = mysql_instance

            # Mock Redis
            redis_instance = AsyncMock()
            redis_instance.ping.return_value = True
            mock_redis_from_url.return_value = redis_instance

            response = client.get("/health/ready")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"

    def test_readiness_some_unhealthy(self, client):
        """部分依赖不健康。"""
        with patch("app.db.milvus_client.MilvusClient") as mock_milvus, \
             patch("app.db.mysql_client.MySQLClient") as mock_mysql, \
             patch("redis.asyncio.from_url") as mock_redis_from_url:
            # Milvus 不健康
            milvus_instance = MagicMock()
            milvus_instance.connect.side_effect = Exception("Connection refused")
            mock_milvus.return_value = milvus_instance

            # MySQL 健康
            mysql_instance = MagicMock()
            mysql_instance.is_ready.return_value = True
            mock_mysql.return_value = mysql_instance

            # Redis 健康
            redis_instance = AsyncMock()
            redis_instance.ping.return_value = True
            mock_redis_from_url.return_value = redis_instance

            response = client.get("/health/ready")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "not_ready"

    def test_readiness_all_unhealthy(self, client):
        """所有依赖不健康。"""
        with patch("app.db.milvus_client.MilvusClient") as mock_milvus, \
             patch("app.db.mysql_client.MySQLClient") as mock_mysql, \
             patch("redis.asyncio.from_url") as mock_redis_from_url:
            milvus_instance = MagicMock()
            milvus_instance.connect.side_effect = Exception("fail")
            mock_milvus.return_value = milvus_instance
            mysql_instance = MagicMock()
            mysql_instance.connect.side_effect = Exception("fail")
            mock_mysql.return_value = mysql_instance
            redis_instance = AsyncMock()
            redis_instance.ping.side_effect = Exception("fail")
            mock_redis_from_url.return_value = redis_instance

            response = client.get("/health/ready")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "not_ready"


# ============================================================
# Schemas 验证
# ============================================================
class TestHealthSchemas:
    """测试 Pydantic 模型验证。"""

    def test_health_response_schema(self):
        """HealthResponse 模型。"""
        from datetime import datetime, timezone
        from app.schemas.common import HealthResponse
        resp = HealthResponse(
            status="ok",
            timestamp=datetime.now(timezone.utc),
            version="0.1.0",
        )
        assert resp.status == "ok"

    def test_readiness_response_schema(self):
        """ReadinessResponse 模型。"""
        from app.schemas.common import ReadinessResponse
        resp = ReadinessResponse(
            status="ready",
            checks={"milvus": True, "mysql": True, "redis": True},
        )
        assert resp.status == "ready"
        assert len(resp.checks) == 3

    def test_error_response_schema(self):
        """ErrorResponse 模型。"""
        from app.schemas.common import ErrorResponse
        resp = ErrorResponse(
            detail="Something went wrong",
            error_code="INTERNAL_ERROR",
        )
        assert resp.detail == "Something went wrong"
        assert resp.error_code == "INTERNAL_ERROR"
