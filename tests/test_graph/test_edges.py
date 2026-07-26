"""
测试 app.graph.edges — 条件路由函数

v1.0.0: clinical → case_preprocess, chitchat → chitchat, not_clinical → reject
"""

import pytest

from app.graph.edges import (
    route_after_case_preprocess,
    route_after_intent,
    route_after_retrieve,
)


# ============================================================
# route_after_intent
# ============================================================
class TestRouteAfterIntent:
    """测试门禁后路由。"""

    def test_clinical_routes_to_case_preprocess(self):
        """临床问题 → case_preprocess。"""
        state = {
            "query": "患者高血压怎么治疗？",
            "intent": "clinical",
            "intent_confidence": 0.95,
        }
        target = route_after_intent(state)
        assert target == "case_preprocess"

    def test_chitchat_routes_to_chitchat(self):
        """问候 → chitchat。"""
        state = {
            "query": "你好",
            "intent": "chitchat",
            "intent_confidence": 0.99,
        }
        target = route_after_intent(state)
        assert target == "chitchat"

    def test_not_clinical_routes_to_reject(self):
        """非临床 → reject。"""
        state = {
            "query": "今天天气怎么样？",
            "intent": "not_clinical",
            "intent_confidence": 0.98,
        }
        target = route_after_intent(state)
        assert target == "reject"

    def test_error_in_intent_graceful(self):
        """门禁出错 → 降级到 case_preprocess（放行）。"""
        state = {
            "query": "测试",
            "intent": "clinical",
            "error_node": "intent",
            "error": "Gatekeeper failed",
        }
        target = route_after_intent(state)
        assert target == "case_preprocess"

    def test_missing_intent_defaults_to_case_preprocess(self):
        """缺少 intent 字段时默认走 case_preprocess。"""
        state = {"query": "测试"}
        target = route_after_intent(state)
        assert target == "case_preprocess"


# ============================================================
# route_after_case_preprocess
# ============================================================
class TestRouteAfterCasePreprocess:
    """测试病例预处理后路由。"""

    def test_routes_to_multi_retrieve(self):
        """预处理后进入多路检索。"""
        state = {
            "query": "测试",
            "case_profile": {"chief_complaint": "test"},
            "search_queries": ["test"],
        }
        target = route_after_case_preprocess(state)
        assert target == "multi_retrieve"


# ============================================================
# route_after_retrieve
# ============================================================
class TestRouteAfterRetrieve:
    """测试检索后路由。"""

    def test_always_routes_to_rank(self):
        state = {
            "query": "测试",
            "search_results": [{"chunk_text": "结果1"}],
            "search_count": 1,
        }
        target = route_after_retrieve(state)
        assert target == "rank"

    def test_empty_results_still_rank(self):
        state = {
            "query": "测试",
            "search_results": [],
            "search_count": 0,
        }
        target = route_after_retrieve(state)
        assert target == "rank"

    def test_error_still_rank(self):
        state = {
            "query": "测试",
            "search_results": [],
            "search_count": 0,
            "error": "检索失败",
            "error_node": "retriever",
        }
        target = route_after_retrieve(state)
        assert target == "rank"
