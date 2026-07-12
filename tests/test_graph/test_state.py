"""
测试 app.graph.state — 图状态定义

覆盖: RagState TypedDict, GraphResult dataclass
"""

import pytest

from app.graph.state import GraphResult, RagState


# ============================================================
# RagState
# ============================================================
class TestRagState:
    """测试 RagState TypedDict。"""

    def test_minimal_state(self):
        """最小状态（仅 query）。"""
        state: RagState = {"query": "阿司匹林怎么吃？"}
        assert state["query"] == "阿司匹林怎么吃？"
        # total=False 意味着可以不提供其他字段
        assert "history" not in state

    def test_full_state(self):
        """完整状态。"""
        state: RagState = {
            "query": "测试问题",
            "history": [{"role": "user", "content": "历史问题"}],
            "intent": "drug_inquiry",
            "intent_confidence": 0.95,
            "search_results": [{"chunk_text": "结果"}],
            "search_count": 1,
            "ranked_docs": [{"chunk_text": "排序后"}],
            "ranked_count": 1,
            "answer": "生成的回答",
            "sources": [{"drug_name": "阿司匹林"}],
            "template_used": "default",
            "error": None,
            "error_node": None,
        }
        assert state["query"] == "测试问题"
        assert state["intent"] == "drug_inquiry"
        assert state["search_count"] == 1
        assert state["ranked_count"] == 1
        assert state["answer"] == "生成的回答"

    def test_error_state(self):
        """错误状态。"""
        state: RagState = {
            "query": "测试问题",
            "error": "检索失败",
            "error_node": "retriever",
        }
        assert state["error"] == "检索失败"
        assert state["error_node"] == "retriever"


# ============================================================
# GraphResult
# ============================================================
class TestGraphResult:
    """测试 GraphResult 数据类。"""

    def test_default_values(self):
        """默认值。"""
        result = GraphResult(success=True)
        assert result.answer == ""
        assert result.sources == []
        assert result.intent == ""

    def test_successful_result(self):
        """成功结果。"""
        result = GraphResult(
            success=True,
            answer="阿司匹林用于解热镇痛。",
            sources=[{"drug_name": "阿司匹林肠溶片", "chunk_text": "..."}],
            intent="drug_inquiry",
            intent_confidence=0.95,
            template_used="default",
            search_count=4,
            ranked_count=4,
        )
        assert result.success is True
        assert len(result.sources) == 1

    def test_failed_result(self):
        """失败结果。"""
        result = GraphResult(
            success=False,
            error="检索服务不可用",
        )
        assert result.success is False
        assert result.error == "检索服务不可用"

    def test_from_state_success(self):
        """从成功的 RagState 构建。"""
        state: RagState = {
            "query": "测试",
            "answer": "回答内容",
            "sources": [{"drug_name": "药"}],
            "intent": "drug_inquiry",
            "intent_confidence": 0.9,
            "template_used": "default",
            "search_count": 5,
            "ranked_count": 5,
        }
        result = GraphResult.from_state(state)
        assert result.success is True
        assert result.answer == "回答内容"
        assert result.intent == "drug_inquiry"
        assert result.search_count == 5
        assert result.ranked_count == 5

    def test_from_state_with_error(self):
        """从含错误的 RagState 构建。"""
        state: RagState = {
            "query": "测试",
            "answer": "兜底回答",
            "error": "生成失败",
            "error_node": "generator",
        }
        result = GraphResult.from_state(state)
        assert result.success is False
        assert result.error == "生成失败"
        assert result.answer == "兜底回答"

    def test_from_state_default_values(self):
        """缺失字段使用默认值。"""
        state: RagState = {"query": "测试"}
        result = GraphResult.from_state(state)
        assert result.answer == ""
        assert result.sources == []
        assert result.intent == ""
        assert result.search_count == 0
        assert result.ranked_count == 0
