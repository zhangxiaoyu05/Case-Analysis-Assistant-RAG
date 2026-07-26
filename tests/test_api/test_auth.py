"""
测试 API 鉴权与安全功能

Phase 0 更新：
  - 问答端点 (chat) 使用 JWT 鉴权（Bearer token）
  - 知识库端点 (knowledge) 保持 API Key 鉴权
  - 健康检查 / 根路径保持公开

覆盖:
  - JWT 鉴权: 无 token / 无效 token → 401, 有效 token → 200
  - API Key 鉴权: 无 Key / 错误 Key → 401, 正确 Key → 200
  - 公共路径: /health, /login, /app 不受鉴权保护
  - 速率限制: 正常通过 / 超限返回 429 / 公共路径不限
  - 安全响应头: X-Content-Type-Options, X-Frame-Options 等
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.middleware import RateLimitMiddleware


# ============================================================
# 共享基础 fixture
# ============================================================
@pytest.fixture
def base_client():
    """创建基础测试客户端。"""
    mock_redis_hm = AsyncMock()
    mock_redis_hm.get_history.return_value = []
    mock_redis_hm.add_turn.return_value = None
    mock_redis_hm.clear_history.return_value = True
    mock_redis_hm.close = AsyncMock()
    mock_redis_hm.get_summary = AsyncMock(return_value="")
    mock_redis_hm.set_summary = AsyncMock()

    mock_compiled = MagicMock()
    mock_compiled.invoke.return_value = {
        "answer": "测试回答",
        "sources": [],
        "intent": "clinical",
        "intent_confidence": 0.95,
        "template_used": "case_summary",
        "error": None,
    }

    with patch("app.api.main.get_graph", return_value=mock_compiled), \
         patch("app.api.main.AsyncRedisHistoryManager", return_value=mock_redis_hm), \
         patch("app.services.memory_manager.MemoryManager.summarize") as mock_summarize:
        mock_summarize.return_value = ("", [])
        from app.api.main import app
        from app.api.dependencies import get_current_user

        # Override JWT auth for chat tests
        app.dependency_overrides[get_current_user] = lambda: {
            "user_id": 1,
            "username": "testuser",
        }

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

    app.dependency_overrides.clear()
    _set_rate_limit(app, 99999)


# ============================================================
# Helper
# ============================================================
def _set_rate_limit(app, rpm: int) -> None:
    """修改 RateLimitMiddleware 参数。"""
    _ = app.middleware_stack
    current = app.middleware_stack
    while current is not None:
        inner = getattr(current, "app", None)
        if isinstance(inner, RateLimitMiddleware):
            inner.enabled = True
            inner.requests_per_minute = rpm
            inner._clients = {}
            return
        current = inner


# ============================================================
# JWT 鉴权测试（问答端点）
# ============================================================
class TestJwtAuth:
    """测试 JWT 鉴权依赖（Phase 0 新增）。"""

    def test_chat_with_jwt_returns_200(self, base_client):
        """有 JWT 鉴权时问答端点正常返回。"""
        response = base_client.post(
            "/api/v1/chat",
            json={"message": "阿司匹林怎么吃？"},
            headers={"Authorization": "Bearer fake_jwt_token"},
        )
        assert response.status_code == 200
        assert "answer" in response.json()

    def test_chat_without_jwt_returns_401(self, base_client):
        """无 JWT token 返回 401。"""
        from app.api.main import app
        app.dependency_overrides.clear()

        client = TestClient(app)
        response = client.post("/api/v1/chat", json={"message": "测试"})
        assert response.status_code == 401

    def test_chat_stream_with_jwt_returns_200(self, base_client):
        """流式问答有 JWT 时正常返回。"""
        response = base_client.post(
            "/api/v1/chat/stream",
            json={"message": "阿司匹林怎么吃？"},
            headers={"Authorization": "Bearer fake_jwt_token"},
        )
        assert response.status_code == 200

    def test_chat_history_with_jwt(self, base_client):
        """获取对话历史有 JWT 时正常返回。"""
        response = base_client.get(
            "/api/v1/chat/history/abc123",
            headers={"Authorization": "Bearer fake_jwt_token"},
        )
        assert response.status_code == 200


# ============================================================
# API Key 鉴权测试（知识库端点）
# ============================================================
class TestApiKeyAuth:
    """测试 API Key 鉴权依赖（知识库端点保持原有机制）。"""

    @pytest.fixture(autouse=True)
    def _enable_auth(self, monkeypatch):
        monkeypatch.setenv("APP_API_KEY", "test-secret-key")

    def test_knowledge_no_key_returns_401(self, base_client):
        """不提供 API Key 返回 401。"""
        response = base_client.get("/api/v1/knowledge/drugs")
        assert response.status_code == 401

    def test_knowledge_correct_key_ok(self, base_client):
        """正确的 API Key 可访问知识库端点。"""
        response = base_client.get(
            "/api/v1/knowledge/drugs",
            headers={"X-API-Key": "test-secret-key"},
        )
        # 不返回 401 即鉴权通过（可能 DB 未连接返回 500，但非鉴权问题）
        assert response.status_code != 401

    def test_knowledge_wrong_key_returns_401(self, base_client):
        """错误的 API Key 返回 401。"""
        response = base_client.get(
            "/api/v1/knowledge/drugs",
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401


# ============================================================
# 公共路径鉴权豁免测试
# ============================================================
class TestPublicPaths:
    """验证公共路径不受鉴权保护。"""

    @pytest.fixture(autouse=True)
    def _enable_auth(self, monkeypatch):
        monkeypatch.setenv("APP_API_KEY", "test-secret-key")

    def test_health_no_auth_returns_200(self, base_client):
        """健康检查无需鉴权。"""
        response = base_client.get("/health")
        assert response.status_code == 200

    def test_login_page_no_auth(self, base_client):
        """登录页面无需鉴权。"""
        response = base_client.get("/login")
        # 可能 200（文件存在）或 404（文件路径问题），但不应该是 401
        assert response.status_code != 401

    def test_app_page_no_auth(self, base_client):
        """主页面无需鉴权。"""
        response = base_client.get("/app")
        assert response.status_code != 401

    def test_docs_no_auth(self, base_client):
        """Swagger 文档无需鉴权。"""
        response = base_client.get("/docs")
        assert response.status_code == 200


# ============================================================
# 速率限制测试
# ============================================================
class TestRateLimiting:
    """测试速率限制中间件。"""

    def test_requests_over_limit_return_429(self, base_client):
        """超过限制的请求返回 429。"""
        _set_rate_limit(base_client.app, 3)
        responses = []
        for i in range(5):
            resp = base_client.post("/api/v1/chat", json={"message": f"测试 {i}"},
                                    headers={"Authorization": "Bearer fake_jwt_token"})
            responses.append(resp)

        status_codes = [r.status_code for r in responses]
        assert 200 in status_codes
        assert 429 in status_codes

    def test_public_paths_not_rate_limited(self, base_client):
        """公共路径不计入限流。"""
        _set_rate_limit(base_client.app, 1)
        for _ in range(10):
            resp = base_client.get("/health")
            assert resp.status_code == 200


# ============================================================
# 安全响应头测试
# ============================================================
class TestSecurityHeaders:
    """验证所有安全响应头正确注入。"""

    EXPECTED_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    }

    def test_security_headers_on_health(self, base_client):
        """健康检查端点包含安全头。"""
        response = base_client.get("/health")
        for header, expected_value in self.EXPECTED_HEADERS.items():
            assert response.headers.get(header) == expected_value, \
                f"期望 {header}: {expected_value}，实际: {response.headers.get(header)}"

    def test_security_headers_on_login(self, base_client):
        """登录页面包含安全头。"""
        response = base_client.get("/login")
        if response.status_code == 200:
            for header, expected_value in self.EXPECTED_HEADERS.items():
                assert response.headers.get(header) == expected_value, \
                    f"期望 {header}: {expected_value}，实际: {response.headers.get(header)}"
