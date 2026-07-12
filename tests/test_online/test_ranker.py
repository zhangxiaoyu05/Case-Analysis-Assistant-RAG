"""
测试 app.online.ranker — 重排序模块

覆盖: RankedDocument, Ranker 类, rerank_documents 便捷函数
"""

from unittest.mock import MagicMock, patch

import pytest

from app.online.ranker import (
    RankedDocument,
    Ranker,
    rerank_documents,
)


# ============================================================
# RankedDocument
# ============================================================
class TestRankedDocument:
    """测试 RankedDocument 数据类。"""

    def test_create(self):
        """创建 RankedDocument。"""
        doc = RankedDocument(
            chunk_text="用于解热镇痛",
            drug_name="阿司匹林肠溶片",
            section="适应症",
            score=0.95,
            doc_id=1,
            chunk_index=0,
            rerank_index=0,
            original_score=0.88,
        )
        assert doc.score == 0.95
        assert doc.rerank_index == 0
        assert doc.original_score == 0.88


# ============================================================
# Ranker.__init__
# ============================================================
class TestRankerInit:
    """测试初始化。"""

    def test_init_with_defaults(self):
        """默认参数初始化。"""
        ranker = Ranker(api_key="test-key")
        assert ranker._api_key == "test-key"
        assert ranker._model is not None

    def test_init_without_api_key_raises(self):
        """无 API Key 抛出异常。"""
        with patch("app.online.ranker.config") as mock_config:
            mock_config.DASHSCOPE_API_KEY = ""
            mock_config.rerank_model = "test-model"
            with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
                Ranker()


# ============================================================
# Ranker.rerank
# ============================================================
class TestRankerRerank:
    """测试重排序方法。"""

    @pytest.fixture
    def sample_docs(self):
        """测试用文档列表。"""
        return [
            {"chunk_text": "用于解热镇痛，缓解轻至中度疼痛", "drug_name": "阿司匹林肠溶片",
             "section": "适应症", "score": 0.88, "doc_id": 1, "chunk_index": 0},
            {"chunk_text": "成人一次0.3～0.6g，一日3次，饭后服用", "drug_name": "阿司匹林肠溶片",
             "section": "用法用量", "score": 0.85, "doc_id": 1, "chunk_index": 1},
            {"chunk_text": "对阿司匹林过敏者禁用", "drug_name": "阿司匹林肠溶片",
             "section": "禁忌", "score": 0.82, "doc_id": 1, "chunk_index": 2},
            {"chunk_text": "胃肠道反应：恶心、呕吐", "drug_name": "阿司匹林肠溶片",
             "section": "不良反应", "score": 0.80, "doc_id": 1, "chunk_index": 3},
        ]

    def test_rerank_success(self, sample_docs):
        """成功重排序。"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # output 必须是 dict-like（代码使用 output.get("results", [])）
        mock_resp.output = {
            "results": [
                {"index": 0, "relevance_score": 0.95},
                {"index": 2, "relevance_score": 0.88},
                {"index": 1, "relevance_score": 0.82},
                {"index": 3, "relevance_score": 0.75},
            ]
        }

        with patch("dashscope.TextReRank") as mock_rerank:
            mock_rerank.call.return_value = mock_resp
            ranker = Ranker(api_key="test-key")
            results = ranker.rerank("阿司匹林怎么吃？", sample_docs)

        assert len(results) > 0
        for r in results:
            assert isinstance(r, RankedDocument)

    def test_rerank_empty_docs(self):
        """空文档列表。"""
        ranker = Ranker(api_key="test-key")
        results = ranker.rerank("查询", [])
        assert results == []

    def test_rerank_api_failure_fallback(self, sample_docs):
        """API 失败时回退到原始排序。"""
        with patch("dashscope.TextReRank") as mock_rerank:
            mock_rerank.call.side_effect = Exception("API error")
            ranker = Ranker(api_key="test-key")
            results = ranker.rerank("查询", sample_docs)

        assert len(results) > 0
        # 应按原始 score 降序排列
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_rerank_without_return_documents(self, sample_docs):
        """return_documents=False 时不需要返回原文。"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.output = {
            "results": [
                {"index": 0, "relevance_score": 0.95},
                {"index": 1, "relevance_score": 0.85},
            ]
        }

        with patch("dashscope.TextReRank") as mock_rerank:
            mock_rerank.call.return_value = mock_resp
            ranker = Ranker(api_key="test-key")
            results = ranker.rerank("查询", sample_docs, return_documents=False)

        assert len(results) > 0
        for r in results:
            assert isinstance(r, RankedDocument)

    def test_rerank_top_n(self, sample_docs):
        """top_n 参数传递给 API（API 负责限制返回数量）。"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # API 返回所有结果，由 ranker 根据 API 响应构建列表
        mock_resp.output = {
            "results": [
                {"index": 0, "relevance_score": 0.95},
                {"index": 1, "relevance_score": 0.85},
            ]
        }

        with patch("dashscope.TextReRank") as mock_rerank:
            mock_rerank.call.return_value = mock_resp
            ranker = Ranker(api_key="test-key")
            results = ranker.rerank("查询", sample_docs, top_n=2)

        # API 返回了 2 条结果，验证 results 数量
        assert len(results) == 2


# ============================================================
# rerank_documents 便捷函数
# ============================================================
class TestRerankDocuments:
    """测试 rerank_documents 便捷函数。"""

    def test_returns_ranked_list(self):
        """返回 RankedDocument 列表。"""
        docs = [{"chunk_text": "测试", "drug_name": "药", "section": "测试",
                 "score": 0.9, "doc_id": 1, "chunk_index": 0}]
        with patch("app.online.ranker.Ranker.rerank") as mock_rerank:
            mock_rerank.return_value = [
                RankedDocument(**docs[0], rerank_index=0, original_score=docs[0]["score"])
            ]
            results = rerank_documents("查询", docs)
            assert len(results) == 1
