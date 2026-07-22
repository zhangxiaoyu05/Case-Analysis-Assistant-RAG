"""
用户管理服务

提供用户注册、登录、JWT 鉴权功能。
密码使用 PBKDF2-HMAC-SHA256 哈希（stdlib，无需额外依赖）。
JWT 使用 PyJWT 库。

使用方式:
    from app.services.user_manager import UserManager

    manager = UserManager()
    user = manager.register("zhangsan", "1234")
    token = manager.login("zhangsan", "1234")
    payload = manager.verify_token(token)
"""

import hashlib
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from loguru import logger

from app.config import config

# ============================================================
# 常量
# ============================================================
# JWT 密钥：优先使用环境变量，否则自动生成（生产环境请务必在 .env 中设置）
_JWT_SECRET = os.getenv("JWT_SECRET", "")
if not _JWT_SECRET:
    _JWT_SECRET = uuid.uuid4().hex
    logger.warning(
        "JWT_SECRET 未设置，已自动生成临时密钥。"
        "生产环境请在 .env 中设置 JWT_SECRET 以保证多实例一致性。"
    )

JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_DAYS = 7

# PBKDF2 参数
PBKDF2_ITERATIONS = 600_000  # OWASP 2023 推荐值
PBKDF2_HASH_NAME = "sha256"
PBKDF2_SALT_LENGTH = 32  # bytes
PBKDF2_KEY_LENGTH = 32  # bytes


# ============================================================
# 密码工具
# ============================================================
def hash_password(password: str) -> str:
    """
    使用 PBKDF2-HMAC-SHA256 对密码进行哈希。

    Returns:
        格式为 "pbkdf2:sha256:<iterations>$<salt_hex>$<hash_hex>" 的字符串
    """
    salt = os.urandom(PBKDF2_SALT_LENGTH)
    dk = hashlib.pbkdf2_hmac(
        PBKDF2_HASH_NAME,
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
        dklen=PBKDF2_KEY_LENGTH,
    )
    return f"pbkdf2:{PBKDF2_HASH_NAME}:{PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    """
    验证密码是否匹配。

    Args:
        password: 明文密码
        password_hash: 由 hash_password() 生成的哈希字符串

    Returns:
        True 表示密码匹配
    """
    try:
        # 解析哈希字符串: pbkdf2:sha256:600000$salt_hex$hash_hex
        algorithm_part, rest = password_hash.split("$", 1)
        _, hash_name, iterations_str = algorithm_part.split(":")
        iterations = int(iterations_str)

        salt_hex, stored_hash_hex = rest.split("$")
        salt = bytes.fromhex(salt_hex)

        dk = hashlib.pbkdf2_hmac(
            hash_name,
            password.encode("utf-8"),
            salt,
            iterations,
            dklen=len(bytes.fromhex(stored_hash_hex)),
        )
        return dk.hex() == stored_hash_hex
    except (ValueError, IndexError, AttributeError):
        return False


# ============================================================
# JWT 工具
# ============================================================
def create_token(user_id: int, username: str) -> str:
    """创建 JWT token（7 天有效期）。"""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": now,
        "exp": now + timedelta(days=JWT_EXPIRATION_DAYS),
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """
    验证并解码 JWT token。

    Args:
        token: JWT 字符串

    Returns:
        解码后的 payload dict

    Raises:
        jwt.ExpiredSignatureError: token 已过期
        jwt.InvalidTokenError: token 无效
    """
    return jwt.decode(token, _JWT_SECRET, algorithms=[JWT_ALGORITHM])


# ============================================================
# UserManager
# ============================================================
class UserManager:
    """
    用户管理服务。

    使用方式:
        manager = UserManager()
        user = manager.register("zhangsan", "1234")
        token = manager.login("zhangsan", "1234")
    """

    def __init__(self, mysql_client=None):
        """
        Args:
            mysql_client: MySQLClient 实例（不传则自动创建）
        """
        self._mysql = mysql_client
        self._own_mysql = mysql_client is None

    # ----------------------------------------------------------
    # 懒加载
    # ----------------------------------------------------------
    @property
    def mysql(self):
        if self._mysql is None:
            from app.db.mysql_client import MySQLClient

            self._mysql = MySQLClient()
            self._mysql.connect()
        return self._mysql

    # ----------------------------------------------------------
    # 公开 API
    # ----------------------------------------------------------
    def register(self, username: str, password: str) -> dict:
        """
        注册新用户。

        Args:
            username: 用户名（2-30 字符，中文/英文/数字/下划线）
            password: 明文密码（≥4 字符）

        Returns:
            {"user_id": int, "username": str}

        Raises:
            ValueError: 用户名或密码不符合要求，或用户名已存在
        """
        username = username.strip()

        # 校验
        if len(username) < 2 or len(username) > 30:
            raise ValueError("用户名长度必须为 2-30 个字符")
        if len(password) < 4:
            raise ValueError("密码长度至少为 4 个字符")

        # 检查用户名是否已存在
        existing = self._get_user_by_username(username)
        if existing:
            raise ValueError(f"用户名 '{username}' 已被注册")

        # 创建用户
        password_hash = hash_password(password)
        conn = self.mysql.conn
        with conn.cursor() as cursor:
            sql = "INSERT INTO users (username, password_hash) VALUES (%s, %s)"
            cursor.execute(sql, (username, password_hash))
            conn.commit()
            user_id = cursor.lastrowid

        logger.info(f"用户注册成功: id={user_id}, username={username}")
        return {"user_id": user_id, "username": username}

    def login(self, username: str, password: str) -> dict:
        """
        用户登录。

        Args:
            username: 用户名
            password: 明文密码

        Returns:
            {"token": str, "user_id": int, "username": str}

        Raises:
            ValueError: 用户名或密码错误
        """
        username = username.strip()

        user = self._get_user_by_username(username)
        if not user:
            raise ValueError("用户名或密码错误")

        if not verify_password(password, user["password_hash"]):
            raise ValueError("用户名或密码错误")

        # 更新最后登录时间
        conn = self.mysql.conn
        with conn.cursor() as cursor:
            sql = "UPDATE users SET last_login_at = NOW() WHERE id = %s"
            cursor.execute(sql, (user["id"],))
            conn.commit()

        token = create_token(user["id"], user["username"])
        logger.info(f"用户登录成功: id={user['id']}, username={username}")

        return {
            "token": token,
            "user_id": user["id"],
            "username": user["username"],
        }

    def get_user_by_id(self, user_id: int) -> Optional[dict]:
        """按 ID 获取用户信息（不含密码哈希）。"""
        conn = self.mysql.conn
        with conn.cursor() as cursor:
            sql = "SELECT id, username, display_name, created_at, last_login_at FROM users WHERE id = %s"
            cursor.execute(sql, (user_id,))
            row = cursor.fetchone()
            if row:
                return {
                    "user_id": row["id"],
                    "username": row["username"],
                    "display_name": row.get("display_name"),
                    "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
                    "last_login_at": row["last_login_at"].isoformat() if row.get("last_login_at") else None,
                }
            return None

    def update_display_name(self, user_id: int, display_name: str | None) -> bool:
        """
        更新用户的显示名称（昵称）。

        Args:
            user_id: 用户 ID
            display_name: 新的显示名称，传 None 或空字符串以清除昵称

        Returns:
            True 表示操作成功（有行被影响）
        """
        # Normalize: empty string -> None
        if display_name is not None and display_name.strip() == "":
            display_name = None
        conn = self.mysql.conn
        with conn.cursor() as cursor:
            sql = "UPDATE users SET display_name = %s WHERE id = %s"
            cursor.execute(sql, (display_name, user_id))
            conn.commit()
            return cursor.rowcount > 0

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------
    def _get_user_by_username(self, username: str) -> Optional[dict]:
        """按用户名获取用户完整记录（含密码哈希）。"""
        conn = self.mysql.conn
        with conn.cursor() as cursor:
            sql = "SELECT id, username, password_hash FROM users WHERE username = %s"
            cursor.execute(sql, (username,))
            return cursor.fetchone()

    def close(self) -> None:
        if self._own_mysql and self._mysql is not None:
            self._mysql.disconnect()
            self._mysql = None
