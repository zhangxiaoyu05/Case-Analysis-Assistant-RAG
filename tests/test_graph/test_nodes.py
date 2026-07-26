"""
测试 app.graph.nodes — LangGraph 节点函数

v1.0.0: 从药品问答改造为病例分析。
  覆盖: intent_node, case_preprocess_node, multi_retrieve_node, rank_node,
        synthesize_node, generate_node, chitchat_node, reject_node
"""

from unittest.mock import MagicMock, patch

import pytest

from app.graph.nodes import (
    case_preprocess_node,
    chitchat_node,
    generate_node,
    intent_node,
    multi_retrieve_node,
    rank_node,
    reject_node,
    synthesize_node,
)


# ============================================================
# intent_node
# ============================================================
class TestIntentNode:
    """测试门禁节点。"""

    def test_greeting_whitelist(self):
        """问候白名单命中 → chitchat。"""
        with patch("app.graph.nodes.is_greeting", return_value=True):
            result = intent_node({"query": "你好"})
            assert result["intent"] == "chitchat"
            assert result["intent_confidence"] == 0.99

    def test_clinical_related(self):
        """门禁判断为临床相关 → clinical。"""
        with patch("app.graph.nodes.is_greeting", return_value=False), \
             patch("app.graph.nodes.Gatekeeper") as mock_gk_cls:
            mock_instance = MagicMock()
            mock_instance.classify.return_value = MagicMock(
                clinical_related=True, confidence=0.95
            )
            mock_gk_cls.return_value = mock_instance

            result = intent_node({"query": "患者高血压怎么治疗？"})
            assert result["intent"] == "clinical"
            assert result["intent_confidence"] == 0.95

    def test_not_clinical_related(self):
        """门禁判断为非临床 → not_clinical。"""
        with patch("app.graph.nodes.is_greeting", return_value=False), \
             patch("app.graph.nodes.Gatekeeper") as mock_gk_cls:
            mock_instance = MagicMock()
            mock_instance.classify.return_value = MagicMock(
                clinical_related=False, confidence=0.98
            )
            mock_gk_cls.return_value = mock_instance

            result = intent_node({"query": "今天天气怎么样？"})
            assert result["intent"] == "not_clinical"
            assert result["intent_confidence"] == 0.98

    def test_empty_query(self):
        """空查询默认视为 clinical。"""
        result = intent_node({"query": ""})
        assert result["intent"] == "clinical"

    def test_gatekeeper_failure_graceful(self):
        """门禁失败时降级放行。"""
        with patch("app.graph.nodes.is_greeting", return_value=False), \
             patch("app.graph.nodes.Gatekeeper") as mock_gk_cls:
            mock_gk_cls.side_effect = RuntimeError("API error")
            result = intent_node({"query": "高血压？"})
            assert result["intent"] == "clinical"
            assert result["intent_confidence"] == 0.5
            assert "error" in result


# ============================================================
# case_preprocess_node
# ============================================================
class TestCasePreprocessNode:
    """测试病例预处理节点。"""

    def test_basic_extraction(self):
        """基本病例提取流程。"""
        with patch("app.graph.nodes._llm_extract_case") as mock_extract:
            mock_extract.return_value = {
                "chief_complaint": "胸闷气短3天",
                "suspected_diagnosis": ["急性心力衰竭"],
                "current_medications": [{"name": "呋塞米", "dosage": "20mg", "frequency": "qd"}],
                "user_questions": [],
                "key_abnormalities": ["BNP升高"],
            }

            result = case_preprocess_node({
                "query": "患者胸闷气短3天，高血压10年",
                "analysis_mode": "comprehensive",
            })

            assert "case_profile" in result
            assert "search_queries" in result
            assert len(result["search_queries"]) > 0
            assert result["case_profile"]["chief_complaint"] == "胸闷气短3天"

    def test_extraction_failure_fallback(self):
        """提取失败时回退到规则方式。"""
        with patch("app.graph.nodes._llm_extract_case") as mock_extract:
            mock_extract.side_effect = RuntimeError("LLM API error")

            result = case_preprocess_node({
                "query": "患者胸闷气短",
                "analysis_mode": "comprehensive",
            })

            assert "case_profile" in result
            assert "search_queries" in result
            # 回退后至少有一个 case_profile
            assert "chief_complaint" in result["case_profile"]

    def test_file_upload_query_parsing(self):
        """文件上传格式的 query 正确解析。"""
        with patch("app.graph.nodes._llm_extract_case") as mock_extract:
            mock_extract.return_value = {
                "chief_complaint": "发热咳嗽",
                "suspected_diagnosis": ["肺炎"],
                "current_medications": [],
                "user_questions": ["如何治疗？"],
                "key_abnormalities": [],
            }

            query = "【病例文档】\n患者发热咳嗽3天\n\n【用户问题】\n如何治疗？"
            result = case_preprocess_node({
                "query": query,
                "analysis_mode": "treatment",
            })

            assert "case_profile" in result
            assert "search_queries" in result

    def test_build_search_queries_limits_five(self):
        """检索查询不超过 5 条。"""
        with patch("app.graph.nodes._llm_extract_case") as mock_extract:
            mock_extract.return_value = {
                "chief_complaint": "test",
                "suspected_diagnosis": ["D1", "D2", "D3"],
                "current_medications": [
                    {"name": "M1"}, {"name": "M2"}, {"name": "M3"}, {"name": "M4"}
                ],
                "user_questions": ["Q1", "Q2"],
                "key_abnormalities": ["A1", "A2", "A3"],
            }

            result = case_preprocess_node({
                "query": "test",
                "analysis_mode": "comprehensive",
            })

            assert len(result["search_queries"]) <= 5


# ============================================================
# multi_retrieve_node
# ============================================================
class TestMultiRetrieveNode:
    """测试多路检索节点。"""

    def test_retrieve_with_queries(self):
        """有 search_queries 时执行多源检索。"""
        from app.online.retriever import SearchResult

        with patch("app.graph.nodes.Retriever") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.multi_source_retrieve.return_value = [
                {
                    "chunk_text": "对症治疗",
                    "drug_name": "test",
                    "section": "治疗",
                    "score": 0.9,
                    "doc_id": 1,
                    "chunk_index": 0,
                    "source": "drug_milvus",
                    "source_type": "drug",
                },
            ]
            mock_cls.return_value = mock_instance

            result = multi_retrieve_node({
                "query": "如何治疗",
                "search_queries": ["肺炎 治疗", "抗生素使用"],
            })
            assert "search_results" in result
            assert result["search_count"] > 0
            assert "search_breakdown" in result

    def test_fallback_to_original_query(self):
        """search_queries 为空时回退到原始 query。"""
        with patch("app.graph.nodes.Retriever") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.multi_source_retrieve.return_value = []
            mock_cls.return_value = mock_instance

            result = multi_retrieve_node({
                "query": "肺炎治疗",
                "search_queries": [],
            })
            # v1.0.0: 空查询回退到 multi_source_retrieve，也是返回空
            assert result["search_count"] == 0

    def test_retrieve_failure_graceful(self):
        """检索失败时回退到空结果。"""
        with patch("app.graph.nodes.Retriever") as mock_cls:
            mock_cls.side_effect = RuntimeError("Milvus unavailable")
            result = multi_retrieve_node({
                "query": "测试",
                "search_queries": ["测试"],
            })
            # v1.0.0: 异常时回退到单源 drug 检索（也失败），最终返回空
            assert result["search_results"] == []
            assert result["search_count"] == 0
            assert "error" in result


# ============================================================
# synthesize_node
# ============================================================
class TestSynthesizeNode:
    """测试多源上下文合成节点。"""

    def test_organizes_by_source_type(self):
        """按 source_type 组织上下文。"""
        ranked_docs = [
            {"chunk_text": "drug info", "source_type": "drug", "drug_name": "A"},
            {"chunk_text": "disease info", "source_type": "disease", "disease_name": "B"},
            {"chunk_text": "guideline info", "source_type": "guideline", "guideline_title": "C"},
            {"chunk_text": "drug info 2", "source_type": "drug", "drug_name": "D"},
        ]

        result = synthesize_node({"ranked_docs": ranked_docs})
        ctx = result["synthesized_context"]

        assert len(ctx["drug"]) == 2
        assert len(ctx["disease"]) == 1
        assert len(ctx["guideline"]) == 1
        assert len(ctx["literature"]) == 0

    def test_empty_ranked_docs(self):
        """空结果返回空字典。"""
        result = synthesize_node({"ranked_docs": []})
        ctx = result["synthesized_context"]
        assert all(len(v) == 0 for v in ctx.values())

    def test_guideline_sorted_by_year(self):
        """指南按年份降序排列。"""
        ranked_docs = [
            {"chunk_text": "g1", "source_type": "guideline", "publish_year": 2020},
            {"chunk_text": "g2", "source_type": "guideline", "publish_year": 2023},
            {"chunk_text": "g3", "source_type": "guideline", "publish_year": 2018},
        ]

        result = synthesize_node({"ranked_docs": ranked_docs})
        years = [d["publish_year"] for d in result["synthesized_context"]["guideline"]]
        assert years == [2023, 2020, 2018]


# ============================================================
# rank_node
# ============================================================
class TestRankNode:
    """测试重排序节点。"""

    def test_rank_success(self):
        """重排序成功。"""
        from app.online.ranker import RankedDocument

        search_results = [
            {"chunk_text": "成人一次0.3～0.6g", "drug_name": "阿司匹林",
             "section": "用法用量", "score": 0.85, "doc_id": 1, "chunk_index": 0},
        ]

        with patch("app.graph.nodes.Ranker") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.rerank.return_value = [
                RankedDocument(**search_results[0], rerank_index=0, original_score=0.85),
            ]
            mock_cls.return_value = mock_instance

            result = rank_node({
                "query": "用法用量",
                "search_results": search_results,
            })
            assert "ranked_docs" in result
            assert result["ranked_count"] > 0

    def test_rank_empty_results(self):
        """空检索结果跳过重排序。"""
        result = rank_node({"query": "测试", "search_results": []})
        assert result["ranked_docs"] == []
        assert result["ranked_count"] == 0

    def test_rank_failure_fallback(self):
        """重排序失败时回退到原始排序。"""
        search_results = [
            {"chunk_text": "text1", "drug_name": "A", "score": 0.5, "doc_id": 1, "chunk_index": 0},
            {"chunk_text": "text2", "drug_name": "B", "score": 0.9, "doc_id": 2, "chunk_index": 0},
        ]

        with patch("app.graph.nodes.Ranker") as mock_cls:
            mock_cls.side_effect = RuntimeError("API error")
            result = rank_node({"query": "测试", "search_results": search_results})
            assert result["ranked_count"] == len(search_results)
            scores = [d["score"] for d in result["ranked_docs"]]
            assert scores == sorted(scores, reverse=True)


# ============================================================
# generate_node
# ============================================================
class TestGenerateNode:
    """测试生成节点。"""

    def test_generate_success(self):
        """生成成功。"""
        from app.online.generator import GeneratedAnswer

        ranked_docs = [
            {"chunk_text": "对症治疗", "drug_name": "test",
             "section": "治疗", "score": 0.95, "doc_id": 1, "chunk_index": 0},
        ]

        with patch("app.graph.nodes.Generator") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.generate.return_value = GeneratedAnswer(
                answer="根据病例分析...",
                sources=ranked_docs,
                template_used="case_summary",
            )
            mock_cls.return_value = mock_instance

            result = generate_node({
                "query": "分析治疗方案",
                "ranked_docs": ranked_docs,
                "case_profile": {},
                "synthesized_context": {},
                "analysis_mode": "comprehensive",
            })
            assert "answer" in result
            assert result["template_used"] == "case_summary"

    def test_generate_no_docs(self):
        """无参考文档时返回兜底回答。"""
        result = generate_node({
            "query": "不存在",
            "ranked_docs": [],
        })
        assert "answer" in result
        assert "未能在知识库中检索到" in result["answer"]
        assert result["sources"] == []

    def test_generate_failure_fallback(self):
        """生成失败时返回检索原文作为兜底。"""
        ranked_docs = [
            {"chunk_text": "text1", "drug_name": "A", "section": "test",
             "score": 0.9, "doc_id": 1, "chunk_index": 0},
        ]

        with patch("app.graph.nodes.Generator") as mock_cls:
            mock_cls.side_effect = RuntimeError("API error")
            result = generate_node({
                "query": "测试",
                "ranked_docs": ranked_docs,
            })
            assert "answer" in result
            assert "error" in result


# ============================================================
# chitchat_node
# ============================================================
class TestChitchatNode:
    """测试闲聊节点。"""

    def test_chitchat_hello(self):
        result = chitchat_node({"query": "你好"})
        assert "answer" in result
        assert result["template_used"] == "chitchat"
        assert result["sources"] == []

    def test_chitchat_unknown(self):
        result = chitchat_node({"query": "嗨！好久不见"})
        assert "answer" in result
        assert "临床" in result["answer"]


# ============================================================
# reject_node
# ============================================================
class TestRejectNode:
    """测试拦截节点。"""

    def test_reject_non_clinical(self):
        result = reject_node({"query": "今天天气怎么样？"})
        assert "answer" in result
        assert "临床病例分析助手" in result["answer"]
        assert result["template_used"] == "reject"

    def test_reject_code_question(self):
        result = reject_node({"query": "用 Python 写一个排序算法"})
        assert "临床病例分析助手" in result["answer"]
