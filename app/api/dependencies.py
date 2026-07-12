"""
FastAPI 依赖注入

提供应用级别单例的获取函数（供路由使用）。
单例在 lifespan 中初始化。
"""

from app.graph.graph import get_graph as _get_graph
from app.services.history_manager import AsyncRedisHistoryManager

# ============================================================
# 单例引用（由 lifespan 设置）
# ============================================================
_history_manager: AsyncRedisHistoryManager | None = None


def set_history_manager(hm: AsyncRedisHistoryManager) -> None:
    """由 lifespan startup 调用，设置全局历史管理器。"""
    global _history_manager
    _history_manager = hm


def get_history_manager() -> AsyncRedisHistoryManager:
    """获取异步 Redis 历史管理器单例。"""
    if _history_manager is None:
        raise RuntimeError(
            "HistoryManager 尚未初始化。请确保应用通过 lifespan 启动。"
        )
    return _history_manager


def get_graph():
    """获取编译好的 LangGraph RAG 图单例。"""
    return _get_graph()
