"""
测试用户管理服务

覆盖：密码哈希/验证、JWT token 创建/解码、注册、登录。
"""

import pytest
from unittest.mock import MagicMock, patch

from app.services.user_manager import (
    UserManager,
    create_token,
    decode_token,
    hash_password,
    verify_password,
)


class TestPasswordHashing:
    """密码哈希与验证测试"""

    def test_hash_and_verify_success(self):
        """密码哈希后可正常验证。"""
        pw = "1234"
        hashed = hash_password(pw)
        assert hashed.startswith("pbkdf2:sha256:")
        assert verify_password(pw, hashed) is True

    def test_verify_wrong_password(self):
        """错误密码验证失败。"""
        hashed = hash_password("correct")
        assert verify_password("wrong", hashed) is False

    def test_hash_is_unique(self):
        """每次哈希结果不同（随机盐）。"""
        h1 = hash_password("1234")
        h2 = hash_password("1234")
        assert h1 != h2

    def test_verify_corrupted_hash(self):
        """损坏的哈希字符串不会报错，返回 False。"""
        assert verify_password("1234", "garbage_string") is False
        assert verify_password("1234", "") is False


class TestJWT:
    """JWT token 测试"""

    def test_create_and_decode_token(self):
        """创建 token 后可正常解码。"""
        token = create_token(user_id=42, username="zhangsan")
        payload = decode_token(token)

        assert payload["sub"] == "42"
        assert payload["username"] == "zhangsan"

    def test_decode_invalid_token(self):
        """无效 token 抛出异常。"""
        with pytest.raises(Exception):
            decode_token("invalid.token.here")

    def test_token_payload_fields(self):
        """Token 包含必要字段。"""
        token = create_token(user_id=1, username="test")
        payload = decode_token(token)

        assert "sub" in payload
        assert "username" in payload
        assert "iat" in payload
        assert "exp" in payload


class TestUserManager:
    """UserManager 服务测试"""

    def test_register_success(self, mock_mysql_for_users):
        """注册新用户成功。"""
        # 模拟用户名不存在
        cursor = mock_mysql_for_users.conn.cursor.return_value
        cursor.fetchone.return_value = None  # 用户不存在

        manager = UserManager(mysql_client=mock_mysql_for_users)
        result = manager.register("newuser", "1234")

        assert result["username"] == "newuser"
        assert result["user_id"] == 1

    def test_register_username_too_short(self, mock_mysql_for_users):
        """用户名太短抛出 ValueError。"""
        manager = UserManager(mysql_client=mock_mysql_for_users)
        with pytest.raises(ValueError, match="用户名长度"):
            manager.register("a", "1234")

    def test_register_username_too_long(self, mock_mysql_for_users):
        """用户名太长抛出 ValueError。"""
        manager = UserManager(mysql_client=mock_mysql_for_users)
        with pytest.raises(ValueError, match="用户名长度"):
            manager.register("a" * 31, "1234")

    def test_register_password_too_short(self, mock_mysql_for_users):
        """密码太短抛出 ValueError。"""
        manager = UserManager(mysql_client=mock_mysql_for_users)
        with pytest.raises(ValueError, match="密码长度"):
            manager.register("validuser", "12")

    def test_register_duplicate_username(self, mock_mysql_for_users):
        """重复用户名抛出 ValueError。"""
        manager = UserManager(mysql_client=mock_mysql_for_users)
        with pytest.raises(ValueError, match="已被注册"):
            manager.register("testuser", "1234")

    def test_login_success(self, mock_mysql_for_users):
        """登录成功返回 token。"""
        # 使用正确的密码哈希
        from app.services.user_manager import hash_password

        pw = "1234"
        hashed = hash_password(pw)

        cursor = mock_mysql_for_users.conn.cursor.return_value
        cursor.fetchone.return_value = {
            "id": 1,
            "username": "testuser",
            "password_hash": hashed,
        }

        manager = UserManager(mysql_client=mock_mysql_for_users)
        result = manager.login("testuser", pw)

        assert "token" in result
        assert result["user_id"] == 1
        assert result["username"] == "testuser"

    def test_login_wrong_password(self, mock_mysql_for_users):
        """错误密码登录失败。"""
        hashed = hash_password("correct")
        cursor = mock_mysql_for_users.conn.cursor.return_value
        cursor.fetchone.return_value = {
            "id": 1,
            "username": "testuser",
            "password_hash": hashed,
        }

        manager = UserManager(mysql_client=mock_mysql_for_users)
        with pytest.raises(ValueError, match="用户名或密码错误"):
            manager.login("testuser", "wrong")

    def test_login_user_not_found(self, mock_mysql_for_users):
        """不存在的用户登录失败。"""
        cursor = mock_mysql_for_users.conn.cursor.return_value
        cursor.fetchone.return_value = None

        manager = UserManager(mysql_client=mock_mysql_for_users)
        with pytest.raises(ValueError, match="用户名或密码错误"):
            manager.login("nouser", "1234")

    def test_get_user_by_id(self, mock_mysql_for_users):
        """按 ID 获取用户信息。"""
        cursor = mock_mysql_for_users.conn.cursor.return_value
        cursor.fetchone.return_value = {
            "id": 1,
            "username": "testuser",
            "display_name": None,
            "created_at": None,
            "last_login_at": None,
        }

        manager = UserManager(mysql_client=mock_mysql_for_users)
        user = manager.get_user_by_id(1)

        assert user is not None
        assert user["user_id"] == 1
        assert user["username"] == "testuser"

    def test_get_user_by_id_not_found(self, mock_mysql_for_users):
        """不存在的用户返回 None。"""
        cursor = mock_mysql_for_users.conn.cursor.return_value
        cursor.fetchone.return_value = None

        manager = UserManager(mysql_client=mock_mysql_for_users)
        user = manager.get_user_by_id(999)

        assert user is None
