"""
会话历史管理器（基于 Redis）

使用 redis.asyncio 提供异步会话 CRUD，支持：
- 按 session_id 获取/追加/清除对话历史
- 自动 TTL 过期（默认 3600s）
- 自动裁剪到最大轮数（默认 10 轮）

使用方式:
    from app.services.history_manager import AsyncRedisHistoryManager

    manager = AsyncRedisHistoryManager()
    history = await manager.get_history("sess_abc")
    await manager.add_turn("sess_abc", user_msg, assistant_msg, sources)
    await manager.clear_history("sess_abc")
"""

import json
from datetime import datetime, timezone
from typing import Optional

import redis.asyncio as aioredis
from loguru import logger

from app.config import config


class AsyncRedisHistoryManager:
    """异步 Redis 会话历史管理。"""

    def __init__(self, redis_client: Optional[aioredis.Redis] = None) -> None:
        """
        Args:
            redis_client: 可复用的 aioredis.Redis 连接。不传则自动创建。
        """
        self._redis: Optional[aioredis.Redis] = redis_client
        self._own_redis = redis_client is None
        self._ttl = config.redis_session_ttl  # 默认 3600s
        self._max_history = config.redis_max_history  # 默认 10 轮

    # ----------------------------------------------------------
    # 连接管理
    # ----------------------------------------------------------
    async def _get_redis(self) -> aioredis.Redis:
        """懒加载 Redis 连接。"""
        if self._redis is None:
            url = f"redis://{config.REDIS_HOST}:{config.REDIS_PORT}"
            self._redis = aioredis.from_url(
                url,
                decode_responses=True,
                max_connections=10,
            )
            logger.info(f"Redis 会话管理器已连接: {url}")
        return self._redis

    async def close(self) -> None:
        """关闭自管理的 Redis 连接。"""
        if self._own_redis and self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            logger.info("Redis 会话管理器已关闭")

    # ----------------------------------------------------------
    # Key 生成
    # ----------------------------------------------------------
    @staticmethod
    def _key(session_id: str) -> str:
        return f"session:{session_id}:history"

    # ----------------------------------------------------------
    # 公共 API
    # ----------------------------------------------------------
    async def get_history(self, session_id: str) -> list[dict]:
        """
        获取会话对话历史。

        Returns:
            历史记录列表，每项包含 role / content / timestamp / sources
        """
        r = await self._get_redis()
        raw = await r.get(self._key(session_id))
        if raw is None:
            return []
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"会话 {session_id} 历史 JSON 解码失败，视为空")
            return []

    async def add_turn(
        self,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
        sources: Optional[list[dict]] = None,
    ) -> None:
        """
        追加一轮对话（1 user + 1 assistant）。

        Args:
            session_id: 会话 ID
            user_msg: 用户消息
            assistant_msg: 助手回答
            sources: 回答的引用来源（可选）
        """
        r = await self._get_redis()

        # 加载现有历史
        raw = await r.get(self._key(session_id))
        if raw:
            try:
                history: list[dict] = json.loads(raw)
            except json.JSONDecodeError:
                history = []
        else:
            history = []

        # 追加新的一轮
        now = datetime.now(timezone.utc).isoformat()
        history.append({
            "role": "user",
            "content": user_msg,
            "timestamp": now,
        })
        history.append({
            "role": "assistant",
            "content": assistant_msg,
            "timestamp": now,
            "sources": sources or [],
        })

        # 裁剪到 max_history 轮（每轮 2 条 = user + assistant）
        max_entries = self._max_history * 2
        if len(history) > max_entries:
            trimmed = len(history) - max_entries
            history = history[-max_entries:]
            logger.debug(f"会话 {session_id} 历史裁剪 {trimmed} 条")

        # 写回 Redis 并刷新 TTL
        await r.set(
            self._key(session_id),
            json.dumps(history, ensure_ascii=False),
            ex=self._ttl,
        )
        logger.debug(
            f"会话 {session_id} 追加一轮对话 (当前 {len(history) // 2} 轮)"
        )

    async def clear_history(self, session_id: str) -> bool:
        """
        清除指定会话的历史记录。

        Returns:
            True 表示有记录被删除，False 表示不存在该会话。
        """
        r = await self._get_redis()
        deleted = await r.delete(self._key(session_id))
        if deleted:
            logger.info(f"会话 {session_id} 历史已清除")
        else:
            logger.debug(f"会话 {session_id} 不存在，无需清除")
        return deleted > 0
