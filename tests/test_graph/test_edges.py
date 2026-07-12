"""
测试 app.graph.edges — 条件路由函数

覆盖: route_after_intent, route_after_retrieve
"""

import pytest

from app.graph.edges import route_after_intent, route_after_retrieve


# ============================================================
# route_after_intent
# ============================================================
class TestRouteAfterIntent:
    """测试意图后路由。"""

    def test_drug_inquiry_routes_to_retrieve(self):
        """药品问题 → retrieve。"""
        state = {
            "query": "阿司匹林怎么吃？",
            "intent": "drug_inquiry",
            "intent_confidence": 0.95,
        }
        target = route_after_intent(state)
        assert target == "retrieve"

    def test_other_routes_to_reject(self):
        """非药品问题 → reject。"""
        state = {
            "query": "今天天气怎么样？",
            "intent": "other",
            "intent_confidence": 0.9,
        }
        target = route_after_intent(state)
        assert target == "reject"

    def test_error_in_intent_routes_to_retrieve(self):
        """意图节点出错 → 降级到 retrieve。"""
        state = {
            "query": "测试问题",
            "intent": "drug_inquiry",
            "error_node": "intent",
            "error": "Intent classification failed",
        }
        target = route_after_intent(state)
        assert target == "retrieve"

    def test_missing_intent_defaults_to_retrieve(self):
        """缺少 intent 字段时默认走 retrieve。"""
        state = {"query": "测试"}
        target = route_after_intent(state)
        assert target == "retrieve"


# ============================================================
# route_after_retrieve
# ============================================================
class TestRouteAfterRetrieve:
    """测试检索后路由。"""

    def test_always_routes_to_rank(self):
        """检索后始终路由到 rank。"""
        state = {
            "query": "测试",
            "search_results": [{"chunk_text": "结果1"}],
            "search_count": 1,
        }
        target = route_after_retrieve(state)
        assert target == "rank"

    def test_empty_results_still_rank(self):
        """空结果也路由到 rank（rank_node 会跳过）。"""
        state = {
            "query": "测试",
            "search_results": [],
            "search_count": 0,
        }
        target = route_after_retrieve(state)
        assert target == "rank"

    def test_error_still_rank(self):
        """检索出错也路由到 rank（rank_node 会回退）。"""
        state = {
            "query": "测试",
            "search_results": [],
            "search_count": 0,
            "error": "检索失败",
            "error_node": "retriever",
        }
        target = route_after_retrieve(state)
        assert target == "rank"
