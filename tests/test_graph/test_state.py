"""
测试 app.graph.state — 图状态定义

v1.0.0: 新增 case_profile / search_queries / search_breakdown /
        synthesized_context / file_name / analysis_mode 字段。
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
        state: RagState = {"query": "患者高血压怎么治疗？"}
        assert state["query"] == "患者高血压怎么治疗？"

    def test_full_state(self):
        """完整状态（v1.0.0 新字段）。"""
        state: RagState = {
            "query": "测试病例",
            "history": [{"role": "user", "content": "历史"}],
            "memory_summary": "摘要",
            "user_memories": "中期记忆",
            "user_profile": "用户画像",
            "file_name": "case.pdf",
            "analysis_mode": "treatment",
            "intent": "clinical",
            "intent_confidence": 0.95,
            "case_profile": {"chief_complaint": "胸闷"},
            "search_queries": ["高血压 治疗"],
            "search_results": [{"chunk_text": "结果"}],
            "search_count": 1,
            "search_breakdown": {"drug": 3},
            "ranked_docs": [{"chunk_text": "排序后"}],
            "ranked_count": 1,
            "synthesized_context": {"drug": [], "disease": []},
            "answer": "生成的病例分析",
            "sources": [{"drug_name": "阿司匹林", "source_type": "drug"}],
            "template_used": "case_summary",
            "error": None,
            "error_node": None,
        }
        assert state["query"] == "测试病例"
        assert state["intent"] == "clinical"
        assert state["template_used"] == "case_summary"
        assert state["analysis_mode"] == "treatment"
        assert state["case_profile"]["chief_complaint"] == "胸闷"

    def test_v1_fields_optional(self):
        """v1.0.0 新字段可以不提供（total=False）。"""
        state: RagState = {"query": "test"}
        assert "case_profile" not in state
        assert "synthesized_context" not in state


# ============================================================
# GraphResult
# ============================================================
class TestGraphResult:
    """测试 GraphResult 数据类。"""

    def test_default_values(self):
        result = GraphResult(success=True)
        assert result.answer == ""
        assert result.sources == []
        assert result.case_profile == {}
        assert result.search_breakdown == {}

    def test_v1_fields(self):
        """v1.0.0 新增字段。"""
        result = GraphResult(
            success=True,
            answer="SOAP 分析结果",
            sources=[],
            intent="clinical",
            template_used="case_summary",
            case_profile={"chief_complaint": "胸闷"},
            search_breakdown={"drug": 3},
        )
        assert result.case_profile["chief_complaint"] == "胸闷"
        assert result.search_breakdown["drug"] == 3
        assert result.template_used == "case_summary"

    def test_from_state_v1(self):
        """从 v1.0.0 RagState 构建。"""
        state: RagState = {
            "query": "测试病例",
            "answer": "SOAP 分析...",
            "sources": [],
            "intent": "clinical",
            "intent_confidence": 0.9,
            "template_used": "treatment_analysis",
            "search_count": 5,
            "ranked_count": 5,
            "case_profile": {"suspected_diagnosis": ["肺炎"]},
            "search_breakdown": {"drug": 3, "disease": 2},
        }
        result = GraphResult.from_state(state)
        assert result.success is True
        assert result.answer == "SOAP 分析..."
        assert result.template_used == "treatment_analysis"
        assert result.case_profile["suspected_diagnosis"] == ["肺炎"]
        assert result.search_breakdown["drug"] == 3

    def test_from_state_default(self):
        """缺失字段使用默认值。"""
        state: RagState = {"query": "测试"}
        result = GraphResult.from_state(state)
        assert result.answer == ""
        assert result.sources == []
        assert result.case_profile == {}
        assert result.search_breakdown == {}
