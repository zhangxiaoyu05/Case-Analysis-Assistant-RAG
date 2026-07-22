"""
测试会话管理 API 端点

覆盖：创建/列表/获取/更新标题/删除会话。
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def conv_client(mock_mysql_for_users):
    """创建带 mock 和 JWT 鉴权的 TestClient。"""
    # Mock ConversationManager
    mock_conv_manager = MagicMock()
    mock_conv_manager.create.return_value = {
        "id": 1,
        "session_id": "abc123def456",
        "title": None,
        "created_at": "2026-07-22T10:00:00",
    }
    mock_conv_manager.list_active.return_value = [
        {
            "id": 1,
            "session_id": "abc123def456",
            "title": "布洛芬用法用量咨询",
            "created_at": "2026-07-22T10:00:00",
            "updated_at": "2026-07-22T10:30:00",
        },
    ]
    mock_conv_manager.get_by_session_id.return_value = {
        "id": 1,
        "user_id": 1,
        "session_id": "abc123def456",
        "title": "布洛芬用法用量咨询",
        "is_active": True,
        "created_at": "2026-07-22T10:00:00",
        "updated_at": "2026-07-22T10:30:00",
    }
    mock_conv_manager.update_title.return_value = True
    mock_conv_manager.soft_delete.return_value = True

    with patch("app.api.routers.conversations.ConversationManager", return_value=mock_conv_manager):
        from app.api.main import app
        from app.api.dependencies import get_current_user

        # Override JWT auth
        app.dependency_overrides[get_current_user] = lambda: {
            "user_id": 1,
            "username": "testuser",
        }

        client = TestClient(app)
        yield client

        app.dependency_overrides.clear()


class TestListConversations:
    """列表查询测试"""

    def test_list_conversations(self, conv_client):
        """列出活跃会话。"""
        response = conv_client.get("/api/v1/conversations", headers={
            "Authorization": "Bearer fake_token",
        })
        assert response.status_code == 200
        data = response.json()
        assert "conversations" in data
        assert len(data["conversations"]) == 1
        assert data["conversations"][0]["title"] == "布洛芬用法用量咨询"

    def test_list_without_auth(self, conv_client):
        """无鉴权返回 401。"""
        from app.api.main import app
        app.dependency_overrides.clear()

        client = TestClient(app)
        response = client.get("/api/v1/conversations")
        assert response.status_code == 401


class TestCreateConversation:
    """创建会话测试"""

    def test_create_conversation(self, conv_client):
        """创建新会话返回 201。"""
        response = conv_client.post("/api/v1/conversations", headers={
            "Authorization": "Bearer fake_token",
        })
        assert response.status_code == 201
        data = response.json()
        assert "session_id" in data
        assert data["session_id"] == "abc123def456"


class TestGetConversation:
    """获取单个会话测试"""

    def test_get_conversation(self, conv_client):
        """按 session_id 获取会话。"""
        response = conv_client.get("/api/v1/conversations/abc123def456", headers={
            "Authorization": "Bearer fake_token",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "abc123def456"

    def test_get_nonexistent(self, conv_client):
        """不存在的会话返回 404。"""
        with patch("app.api.routers.conversations.ConversationManager") as MockCM:
            mock_manager = MagicMock()
            mock_manager.get_by_session_id.return_value = None
            MockCM.return_value = mock_manager

            from app.api.main import app
            from app.api.dependencies import get_current_user
            app.dependency_overrides[get_current_user] = lambda: {
                "user_id": 1, "username": "testuser",
            }

            client = TestClient(app)
            response = client.get("/api/v1/conversations/nonexistent", headers={
                "Authorization": "Bearer fake_token",
            })
            assert response.status_code == 404
            app.dependency_overrides.clear()


class TestUpdateTitle:
    """更新标题测试"""

    def test_update_title(self, conv_client):
        """更新标题成功。"""
        response = conv_client.patch(
            "/api/v1/conversations/abc123def456",
            json={"title": "新标题"},
            headers={"Authorization": "Bearer fake_token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "布洛芬用法用量咨询"  # 来自 mock


class TestDeleteConversation:
    """删除会话测试"""

    def test_delete_conversation(self, conv_client):
        """删除会话成功。"""
        response = conv_client.delete(
            "/api/v1/conversations/abc123def456",
            headers={"Authorization": "Bearer fake_token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] is True
