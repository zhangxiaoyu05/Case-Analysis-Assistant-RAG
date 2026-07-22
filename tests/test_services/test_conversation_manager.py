"""
测试会话管理服务

覆盖：会话 CRUD、标题更新、软删除、列表查询。
"""

import pytest
from unittest.mock import MagicMock

from app.services.conversation_manager import ConversationManager


# ============================================================
# 辅助：设置 conversation 查询的 fetchone 返回值
# ============================================================
_CONV_ROW = {
    "id": 1,
    "user_id": 1,
    "session_id": "abc123def456",
    "title": "布洛芬用法用量咨询",
    "is_active": True,
    "created_at": "2026-07-22T10:00:00",
    "updated_at": "2026-07-22T10:30:00",
}


def _set_conv_fetchone(mock_mysql_client):
    """让 cursor.fetchone() 返回会话记录（而非用户记录）。"""
    cursor = mock_mysql_client.conn.cursor.return_value
    cursor.__enter__.return_value = cursor
    cursor.fetchone.return_value = dict(_CONV_ROW)


class TestConversationManager:
    """ConversationManager 服务测试"""

    def test_create_conversation(self, mock_mysql_for_users):
        """创建新会话。"""
        manager = ConversationManager(mysql_client=mock_mysql_for_users)
        result = manager.create(user_id=1, session_id="abc123")

        assert result["session_id"] == "abc123"
        assert result["id"] == 1

    def test_create_conversation_auto_id(self, mock_mysql_for_users):
        """不传 session_id 时自动生成。"""
        manager = ConversationManager(mysql_client=mock_mysql_for_users)
        result = manager.create(user_id=1)

        assert len(result["session_id"]) == 16
        assert result["id"] == 1

    def test_list_active_conversations(self, mock_mysql_for_users):
        """列出活跃会话。"""
        manager = ConversationManager(mysql_client=mock_mysql_for_users)
        convs = manager.list_active(user_id=1)

        assert len(convs) == 1
        assert convs[0]["session_id"] == "abc123def456"
        assert convs[0]["title"] == "布洛芬用法用量咨询"

    def test_list_active_empty(self, mock_mysql_for_users):
        """没有活跃会话时返回空列表。"""
        cursor = mock_mysql_for_users.conn.cursor.return_value
        cursor.fetchall.return_value = []

        manager = ConversationManager(mysql_client=mock_mysql_for_users)
        convs = manager.list_active(user_id=1)

        assert len(convs) == 0

    def test_get_by_session_id(self, mock_mysql_for_users):
        """按 session_id 获取会话。"""
        _set_conv_fetchone(mock_mysql_for_users)

        manager = ConversationManager(mysql_client=mock_mysql_for_users)
        conv = manager.get_by_session_id("abc123def456")

        assert conv is not None
        assert conv["session_id"] == "abc123def456"
        assert conv["title"] == "布洛芬用法用量咨询"

    def test_get_by_session_id_not_found(self, mock_mysql_for_users):
        """不存在的会话返回 None。"""
        cursor = mock_mysql_for_users.conn.cursor.return_value
        cursor.fetchone.return_value = None

        manager = ConversationManager(mysql_client=mock_mysql_for_users)
        conv = manager.get_by_session_id("nonexistent")

        assert conv is None

    def test_update_title(self, mock_mysql_for_users):
        """更新会话标题。"""
        manager = ConversationManager(mysql_client=mock_mysql_for_users)
        result = manager.update_title("abc123def456", "新标题")

        assert result is True

    def test_update_title_not_found(self, mock_mysql_for_users):
        """更新不存在的会话返回 False。"""
        cursor = mock_mysql_for_users.conn.cursor.return_value
        cursor.rowcount = 0

        manager = ConversationManager(mysql_client=mock_mysql_for_users)
        result = manager.update_title("nonexistent", "新标题")

        assert result is False

    def test_soft_delete(self, mock_mysql_for_users):
        """软删除会话。"""
        manager = ConversationManager(mysql_client=mock_mysql_for_users)
        result = manager.soft_delete("abc123def456")

        assert result is True

    def test_soft_delete_not_found(self, mock_mysql_for_users):
        """软删除不存在的会话返回 False。"""
        cursor = mock_mysql_for_users.conn.cursor.return_value
        cursor.rowcount = 0

        manager = ConversationManager(mysql_client=mock_mysql_for_users)
        result = manager.soft_delete("nonexistent")

        assert result is False
