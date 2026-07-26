"""
测试 app.graph.graph — 图构建和编译

v1.0.0: 8 节点新流程 — intent → case_preprocess → multi_retrieve → rank → synthesize → generate
"""

from unittest.mock import patch

import pytest

from app.graph.graph import build_graph, get_graph, _compiled_graph


# ============================================================
# build_graph
# ============================================================
class TestBuildGraph:
    """测试图构建。"""

    def test_build_graph_returns_compiled(self):
        """build_graph 返回编译好的图。"""
        with patch("app.graph.graph.intent_node"), \
             patch("app.graph.graph.case_preprocess_node"), \
             patch("app.graph.graph.multi_retrieve_node"), \
             patch("app.graph.graph.rank_node"), \
             patch("app.graph.graph.synthesize_node"), \
             patch("app.graph.graph.generate_node"), \
             patch("app.graph.graph.chitchat_node"), \
             patch("app.graph.graph.reject_node"):
            graph = build_graph()
            assert hasattr(graph, "invoke")
            assert hasattr(graph, "ainvoke")

    def test_build_graph_two_calls_produce_different_instances(self):
        """每次 build_graph 返回不同的编译图实例。"""
        with patch("app.graph.graph.intent_node"), \
             patch("app.graph.graph.case_preprocess_node"), \
             patch("app.graph.graph.multi_retrieve_node"), \
             patch("app.graph.graph.rank_node"), \
             patch("app.graph.graph.synthesize_node"), \
             patch("app.graph.graph.generate_node"), \
             patch("app.graph.graph.chitchat_node"), \
             patch("app.graph.graph.reject_node"):
            g1 = build_graph()
            g2 = build_graph()
            assert g1 is not g2


# ============================================================
# get_graph (单例)
# ============================================================
class TestGetGraph:
    """测试图单例。"""

    def setup_method(self):
        import app.graph.graph as gmod
        gmod._compiled_graph = None

    def teardown_method(self):
        import app.graph.graph as gmod
        gmod._compiled_graph = None

    def test_first_call_compiles(self):
        with patch("app.graph.graph.intent_node"), \
             patch("app.graph.graph.case_preprocess_node"), \
             patch("app.graph.graph.multi_retrieve_node"), \
             patch("app.graph.graph.rank_node"), \
             patch("app.graph.graph.synthesize_node"), \
             patch("app.graph.graph.generate_node"), \
             patch("app.graph.graph.chitchat_node"), \
             patch("app.graph.graph.reject_node"):
            graph = get_graph()
            assert graph is not None
            assert hasattr(graph, "invoke")

    def test_second_call_returns_same_instance(self):
        with patch("app.graph.graph.intent_node"), \
             patch("app.graph.graph.case_preprocess_node"), \
             patch("app.graph.graph.multi_retrieve_node"), \
             patch("app.graph.graph.rank_node"), \
             patch("app.graph.graph.synthesize_node"), \
             patch("app.graph.graph.generate_node"), \
             patch("app.graph.graph.chitchat_node"), \
             patch("app.graph.graph.reject_node"):
            g1 = get_graph()
            g2 = get_graph()
            assert g1 is g2


# ============================================================
# 流程图结构验证
# ============================================================
class TestGraphStructure:
    """测试图的结构正确性。"""

    def test_graph_invoke_clinical_full_flow(self):
        """临床查询走完整 8 节点流程。"""
        with patch("app.graph.graph.intent_node") as mock_intent, \
             patch("app.graph.graph.case_preprocess_node") as mock_preprocess, \
             patch("app.graph.graph.multi_retrieve_node") as mock_retrieve, \
             patch("app.graph.graph.rank_node") as mock_rank, \
             patch("app.graph.graph.synthesize_node") as mock_synth, \
             patch("app.graph.graph.generate_node") as mock_generate, \
             patch("app.graph.graph.chitchat_node") as mock_chitchat, \
             patch("app.graph.graph.reject_node") as mock_reject:
            mock_intent.return_value = {"intent": "clinical", "intent_confidence": 0.9}
            mock_preprocess.return_value = {
                "case_profile": {}, "search_queries": ["test"],
            }
            mock_retrieve.return_value = {"search_results": [], "search_count": 0}
            mock_rank.return_value = {"ranked_docs": [], "ranked_count": 0}
            mock_synth.return_value = {"synthesized_context": {}}
            mock_generate.return_value = {
                "answer": "病例分析结果...", "sources": [],
                "template_used": "case_summary",
            }

            graph = build_graph()
            result = graph.invoke({"query": "患者高血压10年，如何治疗？"})

            mock_intent.assert_called_once()
            mock_preprocess.assert_called_once()
            mock_retrieve.assert_called_once()
            mock_rank.assert_called_once()
            mock_synth.assert_called_once()
            mock_generate.assert_called_once()
            mock_chitchat.assert_not_called()
            mock_reject.assert_not_called()

            assert "answer" in result

    def test_graph_invoke_not_clinical(self):
        """非临床查询走拒绝流程。"""
        with patch("app.graph.graph.intent_node") as mock_intent, \
             patch("app.graph.graph.case_preprocess_node") as mock_preprocess, \
             patch("app.graph.graph.multi_retrieve_node") as mock_retrieve, \
             patch("app.graph.graph.rank_node") as mock_rank, \
             patch("app.graph.graph.synthesize_node") as mock_synth, \
             patch("app.graph.graph.generate_node") as mock_generate, \
             patch("app.graph.graph.chitchat_node") as mock_chitchat, \
             patch("app.graph.graph.reject_node") as mock_reject:
            mock_intent.return_value = {"intent": "not_clinical", "intent_confidence": 0.98}
            mock_reject.return_value = {
                "answer": "抱歉，我是临床病例分析助手...",
                "sources": [], "template_used": "reject",
            }

            graph = build_graph()
            result = graph.invoke({"query": "今天天气怎么样？"})

            mock_intent.assert_called_once()
            mock_preprocess.assert_not_called()
            mock_retrieve.assert_not_called()
            mock_rank.assert_not_called()
            mock_synth.assert_not_called()
            mock_generate.assert_not_called()
            mock_chitchat.assert_not_called()
            mock_reject.assert_called_once()

            assert "临床" in result["answer"]

    def test_graph_invoke_chitchat(self):
        """问候走闲聊流程。"""
        with patch("app.graph.graph.intent_node") as mock_intent, \
             patch("app.graph.graph.case_preprocess_node") as mock_preprocess, \
             patch("app.graph.graph.multi_retrieve_node") as mock_retrieve, \
             patch("app.graph.graph.rank_node") as mock_rank, \
             patch("app.graph.graph.synthesize_node") as mock_synth, \
             patch("app.graph.graph.generate_node") as mock_generate, \
             patch("app.graph.graph.chitchat_node") as mock_chitchat, \
             patch("app.graph.graph.reject_node") as mock_reject:
            mock_intent.return_value = {"intent": "chitchat", "intent_confidence": 0.99}
            mock_chitchat.return_value = {
                "answer": "你好！有什么可以帮您的吗？",
                "sources": [], "template_used": "chitchat",
            }

            graph = build_graph()
            result = graph.invoke({"query": "你好"})

            mock_intent.assert_called_once()
            mock_preprocess.assert_not_called()
            mock_retrieve.assert_not_called()
            mock_rank.assert_not_called()
            mock_synth.assert_not_called()
            mock_generate.assert_not_called()
            mock_chitchat.assert_called_once()
            mock_reject.assert_not_called()
