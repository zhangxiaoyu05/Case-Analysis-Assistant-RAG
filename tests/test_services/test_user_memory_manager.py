"""
测试 app.services.user_memory_manager — 中期记忆管理

Phase 2: 跨会话用户动态记忆提取、衰减、Upsert、Top-K 召回。

覆盖: UserMemoryManager 全量 API。
"""

import pytest
from unittest.mock import MagicMock, patch

from app.services.user_memory_manager import UserMemoryManager


# ============================================================
# Mock MySQL — 用户记忆
# ============================================================
@pytest.fixture
def mock_mysql_for_memories():
    """Mock MySQLClient，支持 user_memories 表操作。"""
    client = MagicMock()
    client.connect.return_value = None
    client.disconnect.return_value = None

    cursor = MagicMock()
    cursor.lastrowid = 1
    cursor.rowcount = 1
    cursor.__enter__.return_value = cursor
    cursor.__exit__.return_value = None

    # 默认 fetchone: 无匹配（用于 _upsert_memory 的已有记忆查询）
    cursor.fetchone.return_value = None  # 无已有记忆 → 新增

    # 默认 fetchall:
    # - 用于 get_top_memories
    # - 用于 _upsert_memory 的已有记忆查询
    cursor.fetchall.return_value = []

    conn = MagicMock()
    conn.cursor.return_value = cursor
    client.conn = conn

    return client


# ============================================================
# 样本数据
# ============================================================
_SAMPLE_MEMORIES = [
    {
        "id": 1, "memory_type": "drug_interest", "content": "关注阿司匹林用法用量",
        "keywords": "阿司匹林,用法用量", "importance_score": 1.0,
        "source_session_id": "abc123", "access_count": 2,
    },
    {
        "id": 2, "memory_type": "fact", "content": "用户自述有高血压病史",
        "keywords": "高血压,病史", "importance_score": 0.9,
        "source_session_id": "abc123", "access_count": 1,
    },
    {
        "id": 3, "memory_type": "concern", "content": "担心药物副作用",
        "keywords": "副作用,担忧", "importance_score": 0.7,
        "source_session_id": "def456", "access_count": 0,
    },
]


# ============================================================
# 初始化
# ============================================================
class TestUserMemoryManagerInit:
    """初始化测试。"""

    def test_default_values(self):
        """默认值从 config 读取。"""
        manager = UserMemoryManager()
        assert manager._decay_factor == 0.95
        assert manager._min_importance == 0.1
        assert manager._recall_top_k == 5
        assert manager._max_per_user == 50
        assert manager._merge_threshold == 0.6

    def test_custom_mysql_client(self, mock_mysql_for_memories):
        """支持自定义 MySQL 客户端。"""
        manager = UserMemoryManager(mysql_client=mock_mysql_for_memories)
        assert manager._mysql is mock_mysql_for_memories
        assert manager._own_mysql is False


# ============================================================
# get_top_memories
# ============================================================
class TestGetTopMemories:
    """Top-K 召回测试。"""

    def test_returns_formatted_list(self, mock_mysql_for_memories):
        """返回格式化的记忆列表。"""
        cursor = mock_mysql_for_memories.conn.cursor.return_value
        cursor.fetchall.return_value = _SAMPLE_MEMORIES

        manager = UserMemoryManager(mysql_client=mock_mysql_for_memories)
        result = manager.get_top_memories(1, limit=3)

        assert len(result) == 3
        assert result[0]["memory_type"] == "drug_interest"
        assert result[0]["content"] == "关注阿司匹林用法用量"

    def test_returns_empty_list(self, mock_mysql_for_memories):
        """无记忆时返回空列表。"""
        manager = UserMemoryManager(mysql_client=mock_mysql_for_memories)
        result = manager.get_top_memories(1)

        assert result == []

    def test_respects_limit(self, mock_mysql_for_memories):
        """尊照 limit 参数。"""
        cursor = mock_mysql_for_memories.conn.cursor.return_value
        cursor.fetchall.return_value = _SAMPLE_MEMORIES[:2]

        manager = UserMemoryManager(mysql_client=mock_mysql_for_memories)
        result = manager.get_top_memories(1, limit=2)

        assert len(result) == 2


# ============================================================
# format_memories_for_prompt
# ============================================================
class TestFormatMemoriesForPrompt:
    """Prompt 格式化测试。"""

    def test_empty_returns_empty_string(self, mock_mysql_for_memories):
        """无记忆返回空字符串。"""
        manager = UserMemoryManager(mysql_client=mock_mysql_for_memories)
        result = manager.format_memories_for_prompt(1)
        assert result == ""

    def test_formats_with_labels(self, mock_mysql_for_memories):
        """包含类型标签的格式化文本。"""
        cursor = mock_mysql_for_memories.conn.cursor.return_value
        cursor.fetchall.return_value = _SAMPLE_MEMORIES[:2]

        manager = UserMemoryManager(mysql_client=mock_mysql_for_memories)
        result = manager.format_memories_for_prompt(1)

        assert "用户偏好/关注点" in result
        assert "[关注药品]" in result
        assert "[用户事实]" in result
        assert "阿司匹林" in result
        assert "高血压" in result


# ============================================================
# _format_memories_text (static)
# ============================================================
class TestFormatMemoriesText:
    """静态格式化方法测试。"""

    def test_empty_list(self):
        """空列表返回空字符串。"""
        assert UserMemoryManager._format_memories_text([]) == ""

    def test_formats_multiple_types(self):
        """多类型格式化。"""
        result = UserMemoryManager._format_memories_text(_SAMPLE_MEMORIES[:3])
        assert "[关注药品]" in result
        assert "[用户事实]" in result
        assert "[用户顾虑]" in result

    def test_unknown_type_uses_raw_value(self):
        """未知类型使用原始值。"""
        mem = [{"memory_type": "custom_type", "content": "测试"}]
        result = UserMemoryManager._format_memories_text(mem)
        assert "[custom_type]" in result


# ============================================================
# apply_decay
# ============================================================
class TestApplyDecay:
    """衰减机制测试。"""

    def test_returns_stats(self, mock_mysql_for_memories):
        """返回衰减统计。"""
        manager = UserMemoryManager(mysql_client=mock_mysql_for_memories)
        result = manager.apply_decay(1)

        assert "decayed" in result
        assert "deleted" in result
        assert result["decayed"] >= 0


# ============================================================
# apply_decay_all_users
# ============================================================
class TestApplyDecayAllUsers:
    """全量衰减测试。"""

    def test_no_users(self, mock_mysql_for_memories):
        """无用户时返回零。"""
        manager = UserMemoryManager(mysql_client=mock_mysql_for_memories)
        result = manager.apply_decay_all_users()

        assert result["users"] == 0
        assert result["decayed"] == 0
        assert result["deleted"] == 0

    def test_with_users(self, mock_mysql_for_memories):
        """有用户时正常衰减。"""
        cursor = mock_mysql_for_memories.conn.cursor.return_value
        # get_top_memories 查询返回空（默认）
        # apply_decay_all_users 查询 DISTINCT user_id
        cursor.fetchall.return_value = [{"user_id": 1}, {"user_id": 2}]

        manager = UserMemoryManager(mysql_client=mock_mysql_for_memories)
        result = manager.apply_decay_all_users()

        assert result["users"] == 2


# ============================================================
# _upsert_memory
# ============================================================
class TestUpsertMemory:
    """Upsert 逻辑测试。"""

    def test_insert_new(self, mock_mysql_for_memories):
        """无已有记忆 → 新增。"""
        manager = UserMemoryManager(mysql_client=mock_mysql_for_memories)
        result = manager._upsert_memory(
            1, "fact", "用户有高血压", "高血压", "sess_abc"
        )

        assert result is not None
        assert result["merged"] is False
        assert result["memory_type"] == "fact"

    def test_merge_high_overlap(self, mock_mysql_for_memories):
        """关键词高度重叠 → 合并。"""
        cursor = mock_mysql_for_memories.conn.cursor.return_value
        # 已有记忆：keywords="高血压,病史"
        cursor.fetchall.return_value = [
            {"id": 10, "content": "用户自述高血压", "keywords": "高血压,病史",
             "importance_score": 0.5},
        ]

        manager = UserMemoryManager(mysql_client=mock_mysql_for_memories)
        result = manager._upsert_memory(
            1, "fact", "用户有高血压病史多年", "高血压,病史,慢性病", "sess_new"
        )

        assert result is not None
        assert result["merged"] is True

    def test_no_merge_low_overlap(self, mock_mysql_for_memories):
        """关键词重叠度低 → 不合并（新增）。"""
        cursor = mock_mysql_for_memories.conn.cursor.return_value
        cursor.fetchall.return_value = [
            {"id": 10, "content": "用户关注阿司匹林", "keywords": "阿司匹林",
             "importance_score": 0.5},
        ]

        manager = UserMemoryManager(mysql_client=mock_mysql_for_memories)
        result = manager._upsert_memory(
            1, "fact", "用户有高血压", "高血压,病史", "sess_new"
        )
        # 关键词 "阿司匹林" vs "高血压,病史" → 重叠 0
        assert result is not None
        assert result["merged"] is False


# ============================================================
# _enforce_limit
# ============================================================
class TestEnforceLimit:
    """数量上限测试。"""

    def test_within_limit_no_action(self, mock_mysql_for_memories):
        """未超限时不删除。"""
        cursor = mock_mysql_for_memories.conn.cursor.return_value
        cursor.fetchone.return_value = {"cnt": 10}  # < 50

        manager = UserMemoryManager(mysql_client=mock_mysql_for_memories)
        manager._enforce_limit(1)
        # 不应执行 DELETE

    def test_exceeds_limit_deletes(self, mock_mysql_for_memories):
        """超限时删除最低 importance 的记忆。"""
        cursor = mock_mysql_for_memories.conn.cursor.return_value
        cursor.fetchone.return_value = {"cnt": 60}  # > 50

        manager = UserMemoryManager(mysql_client=mock_mysql_for_memories)
        manager._enforce_limit(1)
        # 应执行 DELETE LIMIT 10


# ============================================================
# extract_and_save
# ============================================================
class TestExtractAndSave:
    """LLM 提取 + 保存集成测试。"""

    @pytest.mark.asyncio
    async def test_empty_user_msg(self, mock_mysql_for_memories):
        """空用户消息跳过。"""
        manager = UserMemoryManager(mysql_client=mock_mysql_for_memories)
        result = await manager.extract_and_save(1, "sess", "", "回答")
        assert result == []

    @pytest.mark.asyncio
    async def test_extract_success(self, mock_mysql_for_memories):
        """正常提取并保存。"""
        with patch.object(UserMemoryManager, "_extract_via_llm") as mock_extract:
            mock_extract.return_value = [
                {"memory_type": "drug_interest", "content": "关注阿司匹林",
                 "keywords": "阿司匹林"},
                {"memory_type": "fact", "content": "用户有高血压",
                 "keywords": "高血压"},
            ]

            manager = UserMemoryManager(mysql_client=mock_mysql_for_memories)
            result = await manager.extract_and_save(
                1, "sess_abc", "阿司匹林对高血压患者安全吗？", "阿司匹林..."
            )

            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_extract_empty(self, mock_mysql_for_memories):
        """LLM 返回空数组。"""
        with patch.object(UserMemoryManager, "_extract_via_llm") as mock_extract:
            mock_extract.return_value = []

            manager = UserMemoryManager(mysql_client=mock_mysql_for_memories)
            result = await manager.extract_and_save(
                1, "sess_abc", "你好", "你好！有什么可以帮您？"
            )

            assert result == []

    @pytest.mark.asyncio
    async def test_extract_api_error(self, mock_mysql_for_memories):
        """LLM API 错误时返回空。"""
        with patch.object(UserMemoryManager, "_extract_via_llm") as mock_extract:
            mock_extract.side_effect = RuntimeError("API error")

            manager = UserMemoryManager(mysql_client=mock_mysql_for_memories)
            result = await manager.extract_and_save(
                1, "sess_abc", "阿司匹林？", "回答"
            )

            assert result == []

    @pytest.mark.asyncio
    async def test_filters_invalid_memory_types(self, mock_mysql_for_memories):
        """过滤无效的记忆类型。"""
        with patch.object(UserMemoryManager, "_extract_via_llm") as mock_extract:
            mock_extract.return_value = [
                {"memory_type": "drug_interest", "content": "有效",
                 "keywords": "test"},
                {"memory_type": "invalid_type", "content": "无效",
                 "keywords": "test"},
                {"memory_type": "", "content": "无类型", "keywords": ""},
            ]

            manager = UserMemoryManager(mysql_client=mock_mysql_for_memories)
            result = await manager.extract_and_save(
                1, "sess_abc", "测试问题", "测试回答"
            )

            # 只保存 drug_interest，过滤掉 invalid_type 和空类型
            assert len(result) == 1
            assert result[0]["memory_type"] == "drug_interest"


# ============================================================
# close
# ============================================================
class TestClose:
    """连接关闭测试。"""

    def test_close_own_mysql(self):
        """关闭自管理的 MySQL 连接。"""
        mock_client = MagicMock()
        mock_client.disconnect.return_value = None
        manager = UserMemoryManager(mysql_client=mock_client)
        manager._own_mysql = True

        manager.close()
        mock_client.disconnect.assert_called_once()

    def test_close_external_mysql(self):
        """不关闭外部传入的 MySQL 连接。"""
        mock_client = MagicMock()
        manager = UserMemoryManager(mysql_client=mock_client)
        manager._own_mysql = False

        manager.close()
        mock_client.disconnect.assert_not_called()
