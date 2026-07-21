"""
测试 app.services.memory_manager — 对话摘要与短期记忆

覆盖: MemoryManager.summarize, _fallback_summary, _format_turns
"""

import pytest

from app.services.memory_manager import MemoryManager


# ============================================================
# 测试数据
# ============================================================
SHORT_HISTORY = [
    {"role": "user", "content": "阿司匹林怎么吃？"},
    {"role": "assistant", "content": "成人一次0.3～0.6g，一日3次。"},
]

LONG_HISTORY = [
    {"role": "user", "content": "阿司匹林是什么？"},
    {"role": "assistant", "content": "阿司匹林是一种解热镇痛药。"},
    {"role": "user", "content": "一次吃多少？"},
    {"role": "assistant", "content": "成人一次0.3～0.6g。"},
    {"role": "user", "content": "有什么副作用？"},
    {"role": "assistant", "content": "常见副作用包括胃肠道不适。"},
    {"role": "user", "content": "可以长期吃吗？"},
    {"role": "assistant", "content": "不建议长期服用，需遵医嘱。"},
    {"role": "user", "content": "和阿莫西林有什么区别？"},
    {"role": "assistant", "content": "阿司匹林是解热镇痛药，阿莫西林是抗生素。"},
    {"role": "user", "content": "谢谢"},
    {"role": "assistant", "content": "不客气！"},
]


# ============================================================
# MemoryManager 初始化
# ============================================================
class TestMemoryManagerInit:
    """测试 MemoryManager 初始化。"""

    def test_default_recent_turns(self):
        """默认保留 4 轮。"""
        manager = MemoryManager()
        assert manager._recent_turns == 4

    def test_custom_recent_turns(self):
        """自定义保留轮数。"""
        manager = MemoryManager(recent_turns=2)
        assert manager._recent_turns == 2

    def test_default_model(self):
        """默认使用 qwen-flash。"""
        manager = MemoryManager()
        assert manager._model == "qwen-flash"

    def test_custom_model(self):
        """自定义摘要模型。"""
        manager = MemoryManager(model="qwen-plus")
        assert manager._model == "qwen-plus"


# ============================================================
# summarize — 不触发摘要
# ============================================================
class TestSummarizeNoTrigger:
    """历史不足时不应触发摘要。"""

    @pytest.mark.asyncio
    async def test_empty_history(self):
        """空历史返回空摘要。"""
        manager = MemoryManager(recent_turns=4)
        summary, recent = await manager.summarize([], "")
        assert summary == ""
        assert recent == []

    @pytest.mark.asyncio
    async def test_short_history_no_summary(self):
        """2轮历史 ≤ 4轮阈值，不触发摘要。"""
        manager = MemoryManager(recent_turns=4)
        summary, recent = await manager.summarize(SHORT_HISTORY, "")
        assert summary == ""
        assert recent == SHORT_HISTORY  # 原样返回

    @pytest.mark.asyncio
    async def test_exact_threshold_no_summary(self):
        """恰好等于阈值，不触发摘要。"""
        manager = MemoryManager(recent_turns=2)
        # 2轮 = 4条消息，阈值是 2*2=4
        history = SHORT_HISTORY  # 4 条消息
        summary, recent = await manager.summarize(history, "")
        assert summary == ""
        assert recent == history


# ============================================================
# summarize — 触发摘要（使用回退，避免调用真实 API）
# ============================================================
class TestSummarizeTriggered:
    """历史超过阈值时触发摘要。"""

    @pytest.mark.asyncio
    async def test_long_history_triggers_summary(self):
        """12条历史 > 4轮(8条)阈值，触发摘要。"""
        manager = MemoryManager(recent_turns=4)
        summary, recent = await manager.summarize(LONG_HISTORY, "")
        # 回退摘要包含用户问题
        assert summary != ""
        assert "阿司匹林" in summary
        # 最近4轮(8条)被保留
        assert len(recent) == 8
        # 最近一条是最后一条 assistant 消息
        assert recent[-1]["content"] == "不客气！"

    @pytest.mark.asyncio
    async def test_recent_turns_custom(self):
        """自定义保留 2 轮。"""
        manager = MemoryManager(recent_turns=2)
        summary, recent = await manager.summarize(LONG_HISTORY, "")
        assert summary != ""
        assert len(recent) == 4  # 2轮 × 2条

    @pytest.mark.asyncio
    async def test_existing_summary_merged(self):
        """已有摘要应与新摘要合并。"""
        manager = MemoryManager(recent_turns=2)
        existing = "用户曾询问过阿司匹林的用法用量。"
        summary, recent = await manager.summarize(LONG_HISTORY, existing)
        assert existing in summary  # 前序摘要被保留
        assert len(recent) == 4


# ============================================================
# _fallback_summary
# ============================================================
class TestFallbackSummary:
    """测试回退摘要生成。"""

    def test_extracts_user_questions(self):
        """回退摘要提取用户问题。"""
        manager = MemoryManager()
        old = [
            {"role": "user", "content": "阿司匹林怎么吃？"},
            {"role": "assistant", "content": "成人一次0.3～0.6g。"},
            {"role": "user", "content": "有什么副作用？"},
            {"role": "assistant", "content": "胃肠道不适。"},
        ]
        result = manager._fallback_summary(old, "")
        assert "阿司匹林怎么吃" in result
        assert "有什么副作用" in result

    def test_truncates_long_questions(self):
        """长问题被截断。"""
        manager = MemoryManager()
        long_q = "阿司匹林" * 80  # 超过100字
        old = [{"role": "user", "content": long_q}]
        result = manager._fallback_summary(old, "")
        assert "…" in result
        assert len(result) < 500

    def test_merges_with_existing(self):
        """前序摘要被合并保留。"""
        manager = MemoryManager()
        old = [{"role": "user", "content": "布洛芬怎么吃？"}]
        result = manager._fallback_summary(old, "用户曾询问阿司匹林。")
        assert "阿司匹林" in result
        assert "布洛芬" in result

    def test_empty_old_returns_existing(self):
        """空旧轮次返回前序摘要。"""
        manager = MemoryManager()
        result = manager._fallback_summary([], "已有摘要")
        assert result == "已有摘要"

    def test_no_user_questions(self):
        """只有 assistant 消息时返回前序摘要。"""
        manager = MemoryManager()
        old = [
            {"role": "assistant", "content": "不客气！"},
        ]
        result = manager._fallback_summary(old, "前序摘要")
        assert result == "前序摘要"


# ============================================================
# _format_turns
# ============================================================
class TestFormatTurns:
    """测试对话格式化。"""

    def test_basic_format(self):
        """基本格式化。"""
        entries = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
        ]
        result = MemoryManager._format_turns(entries)
        assert "[用户]: 你好" in result
        assert "[助手]: 你好！" in result

    def test_truncates_long_content(self):
        """长内容被截断。"""
        long_content = "X" * 600
        entries = [{"role": "assistant", "content": long_content}]
        result = MemoryManager._format_turns(entries)
        assert "…" in result
        assert len(result) < 600


# ============================================================
# summarize_history 便捷函数
# ============================================================
class TestSummarizeHistoryFunction:
    """测试便捷函数。"""

    @pytest.mark.asyncio
    async def test_returns_tuple(self):
        """返回 (summary, recent) 元组。"""
        from app.services.memory_manager import summarize_history
        summary, recent = await summarize_history(SHORT_HISTORY, "")
        assert isinstance(summary, str)
        assert isinstance(recent, list)
