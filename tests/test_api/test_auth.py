"""
测试 API 鉴权与安全功能

覆盖:
  - API Key 鉴权: 无 Key / 错误 Key → 401, 正确 Key → 200
  - 公共路径: /health, /, /docs 不受鉴权保护
  - 速率限制: 正常通过 / 超限返回 429 / 公共路径不限
  - 安全响应头: X-Content-Type-Options, X-Frame-Options 等

重要: 所有 fixture 使用同一个 app 实例（通过共享 base_client fixture），
不删除 sys.modules，避免污染其他测试模块。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.middleware import RateLimitMiddleware


# ============================================================
# 共享基础 fixture（只创建一次 app，所有测试复用）
# ============================================================
@pytest.fixture
def base_client():
    """创建基础测试客户端（鉴权禁用、限流默认配置）。

    每个测试函数获得独立的 TestClient，但底层 app 共享。
    每次测试结束后，重置限流器状态，防止跨测试泄漏。
    """
    mock_redis_hm = AsyncMock()
    mock_redis_hm.get_history.return_value = []
    mock_redis_hm.add_turn.return_value = None
    mock_redis_hm.clear_history.return_value = True
    mock_redis_hm.close = AsyncMock()

    mock_compiled = MagicMock()
    mock_compiled.invoke.return_value = {
        "answer": "测试回答",
        "sources": [],
        "intent": "drug_inquiry",
        "intent_confidence": 0.95,
        "template_used": "default",
        "error": None,
    }

    with patch("app.api.main.get_graph", return_value=mock_compiled), \
         patch("app.api.main.AsyncRedisHistoryManager", return_value=mock_redis_hm):
        from app.api.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

    # teardown: 重置限流器到极宽松状态，防止影响后续测试
    _set_rate_limit(app, 99999)


# ============================================================
# Helper: 查找 RateLimitMiddleware 实例
# ============================================================
def _set_rate_limit(app, rpm: int) -> None:
    """修改已创建 app 上的 RateLimitMiddleware 参数。

    通过触发中间件栈构建，然后遍历找到 RateLimitMiddleware 实例并修改参数。
    """
    # 强制构建中间件栈（通过访问 app.middleware_stack）
    _ = app.middleware_stack

    # 遍历中间件链找到 RateLimitMiddleware 实例
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
# API Key 鉴权测试
# ============================================================
class TestApiKeyAuth:
    """测试 API Key 鉴权依赖。"""

    @pytest.fixture(autouse=True)
    def _enable_auth(self, monkeypatch):
        """设置 APP_API_KEY 以启用鉴权。"""
        monkeypatch.setenv("APP_API_KEY", "test-secret-key")

    def test_no_key_returns_401(self, base_client):
        """不提供 API Key 返回 401。"""
        response = base_client.post("/api/v1/chat", json={
            "message": "阿司匹林怎么吃？",
        })
        assert response.status_code == 401
        assert "API Key" in response.json()["detail"]

    def test_wrong_key_returns_401(self, base_client):
        """错误的 API Key 返回 401。"""
        response = base_client.post(
            "/api/v1/chat",
            json={"message": "阿司匹林怎么吃？"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401
        assert "无效" in response.json()["detail"]

    def test_correct_key_returns_200(self, base_client):
        """正确的 API Key 返回 200。"""
        response = base_client.post(
            "/api/v1/chat",
            json={"message": "阿司匹林怎么吃？"},
            headers={"X-API-Key": "test-secret-key"},
        )
        assert response.status_code == 200
        assert "answer" in response.json()

    def test_correct_key_for_knowledge_endpoint(self, base_client):
        """正确的 API Key 访问知识库端点。"""
        response = base_client.get(
            "/api/v1/knowledge/drugs",
            headers={"X-API-Key": "test-secret-key"},
        )
        # 可能返回 200（正常）或 500（DB 未连接），但不应返回 401
        assert response.status_code != 401

    def test_bearer_token_valid(self, base_client):
        """Bearer Token 鉴权成功。"""
        response = base_client.post(
            "/api/v1/chat",
            json={"message": "阿司匹林怎么吃？"},
            headers={"Authorization": "Bearer test-secret-key"},
        )
        assert response.status_code == 200

    def test_bearer_token_invalid(self, base_client):
        """Bearer Token 鉴权失败。"""
        response = base_client.post(
            "/api/v1/chat",
            json={"message": "阿司匹林怎么吃？"},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert response.status_code == 401

    def test_x_api_key_takes_precedence(self, base_client):
        """X-API-Key 优先于 Authorization Bearer。"""
        response = base_client.post(
            "/api/v1/chat",
            json={"message": "阿司匹林怎么吃？"},
            headers={
                "X-API-Key": "test-secret-key",
                "Authorization": "Bearer wrong-key",
            },
        )
        assert response.status_code == 200

    def test_bearer_no_token(self, base_client):
        """Authorization 头格式错误（无 token）。"""
        response = base_client.post(
            "/api/v1/chat",
            json={"message": "阿司匹林怎么吃？"},
            headers={"Authorization": "Bearer "},
        )
        assert response.status_code == 401

    def test_non_bearer_scheme(self, base_client):
        """非 Bearer 的 Authorization 头被视为无效。"""
        response = base_client.post(
            "/api/v1/chat",
            json={"message": "阿司匹林怎么吃？"},
            headers={"Authorization": "Basic dGVzdDp0ZXN0"},
        )
        assert response.status_code == 401


# ============================================================
# 公共路径鉴权豁免测试
# ============================================================
class TestPublicPaths:
    """验证公共路径不受鉴权保护。"""

    @pytest.fixture(autouse=True)
    def _enable_auth(self, monkeypatch):
        """设置 APP_API_KEY 以启用鉴权。"""
        monkeypatch.setenv("APP_API_KEY", "test-secret-key")

    def test_health_no_key_returns_200(self, base_client):
        """健康检查无需 API Key。"""
        response = base_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_ready_no_key_returns_200(self, base_client):
        """就绪检查无需 API Key。"""
        response = base_client.get("/health/ready")
        assert response.status_code != 401

    def test_root_no_key_returns_200(self, base_client):
        """根路径无需 API Key。"""
        response = base_client.get("/")
        assert response.status_code == 200

    def test_docs_no_key_returns_200(self, base_client):
        """Swagger 文档无需 API Key。"""
        response = base_client.get("/docs")
        assert response.status_code == 200

    def test_openapi_json_no_key_returns_200(self, base_client):
        """OpenAPI schema 无需 API Key。"""
        response = base_client.get("/openapi.json")
        assert response.status_code == 200


# ============================================================
# 速率限制测试
# ============================================================
class TestRateLimiting:
    """测试速率限制中间件。"""

    @pytest.fixture(autouse=True)
    def _disable_auth_for_ratelimit(self, monkeypatch):
        """禁用鉴权，仅测试限流。"""
        monkeypatch.setenv("APP_API_KEY", "")

    def test_requests_under_limit_pass(self, base_client):
        """低于限制的请求正常通过。"""
        _set_rate_limit(base_client.app, 60)
        response = base_client.post("/api/v1/chat", json={"message": "测试"})
        assert response.status_code == 200

    def test_requests_over_limit_return_429(self, base_client):
        """超过限制的请求返回 429。"""
        _set_rate_limit(base_client.app, 3)
        responses = []
        for i in range(5):
            resp = base_client.post("/api/v1/chat", json={"message": f"测试 {i}"})
            responses.append(resp)

        status_codes = [r.status_code for r in responses]
        assert 200 in status_codes, f"期望至少有一次 200，实际: {status_codes}"
        assert 429 in status_codes, f"期望有 429 限流响应，实际: {status_codes}"

    def test_429_response_has_retry_after(self, base_client):
        """429 响应包含 Retry-After 头。"""
        _set_rate_limit(base_client.app, 3)
        for i in range(4):
            base_client.post("/api/v1/chat", json={"message": f"测试 {i}"})

        resp = base_client.post("/api/v1/chat", json={"message": "超额"})
        if resp.status_code == 429:
            assert "Retry-After" in resp.headers
            data = resp.json()
            assert "retry_after_seconds" in data
            assert data["retry_after_seconds"] > 0

    def test_public_paths_not_rate_limited(self, base_client):
        """公共路径不计入限流。"""
        _set_rate_limit(base_client.app, 1)
        for _ in range(10):
            resp = base_client.get("/health")
            assert resp.status_code == 200

    def test_root_not_rate_limited(self, base_client):
        """根路径不计入限流。"""
        _set_rate_limit(base_client.app, 1)
        for _ in range(10):
            resp = base_client.get("/")
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

    @pytest.fixture(autouse=True)
    def _enable_auth(self, monkeypatch):
        """设置 APP_API_KEY 以启用鉴权（使受保护端点可达）。"""
        monkeypatch.setenv("APP_API_KEY", "test-secret-key")

    def test_security_headers_on_chat(self, base_client):
        """问答端点包含安全头。"""
        response = base_client.post(
            "/api/v1/chat",
            json={"message": "测试"},
            headers={"X-API-Key": "test-secret-key"},
        )
        for header, expected_value in self.EXPECTED_HEADERS.items():
            assert response.headers.get(header) == expected_value, \
                f"期望 {header}: {expected_value}，实际: {response.headers.get(header)}"

    def test_security_headers_on_health(self, base_client):
        """健康检查端点也包含安全头。"""
        response = base_client.get("/health")
        for header, expected_value in self.EXPECTED_HEADERS.items():
            assert response.headers.get(header) == expected_value, \
                f"期望 {header}: {expected_value}，实际: {response.headers.get(header)}"

    def test_security_headers_on_root(self, base_client):
        """根路径包含安全头。"""
        response = base_client.get("/")
        for header, expected_value in self.EXPECTED_HEADERS.items():
            assert response.headers.get(header) == expected_value, \
                f"期望 {header}: {expected_value}，实际: {response.headers.get(header)}"

    def test_security_headers_on_404(self, base_client):
        """404 响应也包含安全头。"""
        response = base_client.get("/api/v1/knowledge/nonexistent",
                                   headers={"X-API-Key": "test-secret-key"})
        for header, expected_value in self.EXPECTED_HEADERS.items():
            assert response.headers.get(header) == expected_value, \
                f"404 响应期望 {header}: {expected_value}，实际: {response.headers.get(header)}"


# ============================================================
# 鉴权禁用时的向后兼容测试
# ============================================================
class TestAuthDisabled:
    """鉴权禁用时所有端点正常可访问。"""

    @pytest.fixture(autouse=True)
    def _disable_auth(self, monkeypatch):
        """清空 APP_API_KEY 以禁用鉴权。"""
        monkeypatch.setenv("APP_API_KEY", "")

    def test_chat_without_key_when_auth_disabled(self, base_client):
        """鉴权禁用时无需 Key 即可访问问答端点。"""
        resp = base_client.post("/api/v1/chat", json={"message": "测试"})
        assert resp.status_code == 200
