"""
会话管理服务

管理用户的对话窗口（conversations 表 CRUD），
支持 LLM 自动生成对话标题。

每个对话窗口 = 一个 session_id，绑定到 user_id。
删除对话 = 标记 is_active=false + 清除 Redis 短期记忆。

使用方式:
    from app.services.conversation_manager import ConversationManager

    manager = ConversationManager()
    conv = manager.create(user_id=1, session_id="abc123")
    convs = manager.list_active(user_id=1)
    manager.generate_title(session_id="abc123", first_message="布洛芬的用法用量？")
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from app.config import config

# ============================================================
# 辅助函数
# ============================================================
def _format_datetime(val) -> Optional[str]:
    """将 datetime 对象或字符串统一转为 ISO 格式字符串。"""
    if val is None:
        return None
    if isinstance(val, str):
        return val
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return str(val)


# ============================================================
# 标题生成提示词
# ============================================================
_TITLE_SYSTEM_PROMPT = """你是一个简洁标题生成助手。根据用户的第一条消息，生成一个简短的对话标题。

要求：
1. 使用中文
2. 不超过 15 个字
3. 准确概括用户问题的核心主题
4. 不要加引号、书名号等额外符号
5. 只返回标题文本，不要有其他内容

示例：
- 用户："阿司匹林一天吃几次？饭前还是饭后？" → 阿司匹林用法用量咨询
- 用户："布洛芬和对乙酰氨基酚哪个退烧好？" → 退烧药对比选择
- 用户："高血压180，吃什么降压药？" → 高血压降压药选择
- 用户："感冒了，有点发烧咳嗽怎么办？" → 感冒发烧用药咨询"""


# ============================================================
# ConversationManager
# ============================================================
class ConversationManager:
    """
    会话管理服务。

    使用方式:
        manager = ConversationManager()
        conv = manager.create(user_id=1, session_id="abc123")
    """

    def __init__(self, mysql_client=None):
        """
        Args:
            mysql_client: MySQLClient 实例（不传则自动创建）
        """
        self._mysql = mysql_client
        self._own_mysql = mysql_client is None

    @property
    def mysql(self):
        if self._mysql is None:
            from app.db.mysql_client import MySQLClient

            self._mysql = MySQLClient()
            self._mysql.connect()
        return self._mysql

    # ----------------------------------------------------------
    # CRUD
    # ----------------------------------------------------------
    def create(self, user_id: int, session_id: Optional[str] = None) -> dict:
        """
        创建新的对话窗口。

        Args:
            user_id: 用户 ID
            session_id: 会话标识（不传则自动生成 16 位 hex）

        Returns:
            {"id": int, "session_id": str, "title": None, "created_at": str}
        """
        session_id = session_id or uuid.uuid4().hex[:16]

        conn = self.mysql.conn
        with conn.cursor() as cursor:
            sql = (
                "INSERT INTO conversations (user_id, session_id) VALUES (%s, %s)"
            )
            cursor.execute(sql, (user_id, session_id))
            conn.commit()
            conv_id = cursor.lastrowid

        logger.info(f"会话创建: id={conv_id}, user_id={user_id}, session_id={session_id}")
        return {
            "id": conv_id,
            "session_id": session_id,
            "title": None,
            "created_at": _format_datetime(datetime.now(timezone.utc)),
        }

    def list_active(self, user_id: int) -> list[dict]:
        """
        列出用户所有活跃的对话窗口（按更新时间倒序）。

        Returns:
            [{"id": int, "session_id": str, "title": str|null, "created_at": str, "updated_at": str}, ...]
        """
        conn = self.mysql.conn
        with conn.cursor() as cursor:
            sql = (
                "SELECT id, session_id, title, created_at, updated_at "
                "FROM conversations "
                "WHERE user_id = %s AND is_active = TRUE "
                "ORDER BY updated_at DESC"
            )
            cursor.execute(sql, (user_id,))
            rows = cursor.fetchall()

        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "title": row.get("title"),
                "created_at": _format_datetime(row.get("created_at")),
                "updated_at": _format_datetime(row.get("updated_at")),
            }
            for row in rows
        ]

    def get_by_session_id(self, session_id: str) -> Optional[dict]:
        """按 session_id 获取会话信息。"""
        conn = self.mysql.conn
        with conn.cursor() as cursor:
            sql = (
                "SELECT id, user_id, session_id, title, is_active, created_at, updated_at "
                "FROM conversations WHERE session_id = %s"
            )
            cursor.execute(sql, (session_id,))
            row = cursor.fetchone()

        if not row:
            return None

        return {
            "id": row["id"],
            "user_id": row.get("user_id", 0),
            "session_id": row["session_id"],
            "title": row.get("title"),
            "is_active": row.get("is_active", True),
            "created_at": _format_datetime(row.get("created_at")),
            "updated_at": _format_datetime(row.get("updated_at")),
        }

    def update_title(self, session_id: str, title: str) -> bool:
        """
        更新对话标题（用户手动编辑）。

        Returns:
            True 表示更新成功，False 表示会话不存在
        """
        title = title.strip()[:100]
        conn = self.mysql.conn
        with conn.cursor() as cursor:
            sql = "UPDATE conversations SET title = %s WHERE session_id = %s"
            cursor.execute(sql, (title, session_id))
            conn.commit()
            affected = cursor.rowcount

        if affected:
            logger.info(f"会话标题已更新: {session_id} → {title}")
        return affected > 0

    def soft_delete(self, session_id: str) -> bool:
        """
        软删除对话窗口（标记 is_active=false）。

        注意：调用方需要同步清除 Redis 中的短期记忆。
        中期记忆和长期记忆保留（归属 user_id，不受会话删除影响）。

        Returns:
            True 表示删除成功
        """
        conn = self.mysql.conn
        with conn.cursor() as cursor:
            sql = (
                "UPDATE conversations SET is_active = FALSE WHERE session_id = %s"
            )
            cursor.execute(sql, (session_id,))
            conn.commit()
            affected = cursor.rowcount

        if affected:
            logger.info(f"会话已软删除: {session_id}")
        return affected > 0

    # ----------------------------------------------------------
    # 标题自动生成
    # ----------------------------------------------------------
    def generate_title(self, session_id: str, first_message: str) -> None:
        """
        异步生成对话标题（同步方法，内部调用 LLM）。

        应在首条用户消息发送后调用，不阻塞对话响应。
        一般通过 chat.py 中 asyncio.create_task 调度。

        Args:
            session_id: 会话标识
            first_message: 用户的首条消息
        """
        if not first_message or not first_message.strip():
            return

        try:
            title = self._call_title_generation(first_message[:500])
            if title:
                self.update_title(session_id, title)
                logger.info(f"标题自动生成: {session_id} → {title}")
        except Exception as e:
            logger.warning(f"标题生成失败 [{session_id}]: {e}")

    def _call_title_generation(self, message: str) -> str:
        """调用 DashScope LLM 生成标题（同步方法）。"""
        from dashscope import Generation

        response = Generation.call(
            model="qwen-flash",  # 标题生成用轻量模型即可
            messages=[
                {"role": "system", "content": _TITLE_SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            temperature=0.2,
            max_tokens=30,
            api_key=config.DASHSCOPE_API_KEY,
            result_format="message",
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"标题生成 API 错误: status={response.status_code}, "
                f"message={response.message}"
            )

        output = response.output
        if output and output.choices:
            title = output.choices[0].message.content.strip()
            return title[:15]  # 最多 15 字
        elif output and output.text:
            return output.text.strip()[:15]

        raise RuntimeError("标题生成 API 返回了空结果")

    def close(self) -> None:
        if self._own_mysql and self._mysql is not None:
            self._mysql.disconnect()
            self._mysql = None
