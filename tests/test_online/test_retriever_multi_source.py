"""
测试 v1.0.0 Retriever 多源检索方法
"""

from unittest.mock import MagicMock, patch

import pytest

from app.online.retriever import Retriever, SearchResult, _balanced_sample


# ============================================================
# _balanced_sample
# ============================================================
class TestBalancedSample:
    """测试跨源均衡采样。"""

    def test_empty_results(self):
        """空结果返回空列表。"""
        result = _balanced_sample([], per_source_min=2, total_max=15)
        assert result == []

    def test_single_source(self):
        """单一 source，不超过 total_max。"""
        results = [
            {"source_type": "drug", "score": 0.9, "chunk_text": "text1"},
            {"source_type": "drug", "score": 0.8, "chunk_text": "text2"},
            {"source_type": "drug", "score": 0.7, "chunk_text": "text3"},
        ]
        result = _balanced_sample(results, per_source_min=2, total_max=10)
        assert len(result) == 3  # 不超过输入
        # 按 score 降序
        assert result[0]["score"] >= result[-1]["score"]

    def test_per_source_min_enforced(self):
        """每种 source 至少 per_source_min 条。"""
        results = [
            {"source_type": "drug", "score": 0.9, "chunk_text": "d1"},
            {"source_type": "drug", "score": 0.8, "chunk_text": "d2"},
            {"source_type": "drug", "score": 0.7, "chunk_text": "d3"},
            {"source_type": "disease", "score": 0.6, "chunk_text": "di1"},
            {"source_type": "disease", "score": 0.5, "chunk_text": "di2"},
        ]
        result = _balanced_sample(results, per_source_min=2, total_max=15)
        drug_count = sum(1 for r in result if r["source_type"] == "drug")
        disease_count = sum(1 for r in result if r["source_type"] == "disease")
        assert drug_count >= 2
        assert disease_count >= 2

    def test_total_max_respected(self):
        """总数不超过 total_max。"""
        results = [
            {"source_type": st, "score": 0.9 - i * 0.1, "chunk_text": f"t{i}"}
            for i in range(30)
            for st in ["drug", "disease", "guideline", "literature"]
        ]
        result = _balanced_sample(results, per_source_min=2, total_max=15)
        assert len(result) <= 15


# ============================================================
# Retriever.retrieve_from
# ============================================================
class TestRetrieveFrom:
    """测试单源检索。"""

    def test_retrieve_from_drug_with_no_results(self):
        """drug 源检索无结果。"""
        retriever = Retriever()
        retriever._milvus = MagicMock()
        retriever._mysql = MagicMock()
        retriever._embedder = MagicMock()

        # Mock embedder
        mock_embed_result = MagicMock()
        mock_embed_result.embeddings = [None]  # 向量化失败
        retriever._embedder.embed.return_value = mock_embed_result

        # Mock MySQL
        retriever._mysql.bm25_search_generic.return_value = []

        results = retriever.retrieve_from("test query", "drug")
        assert results == []

    def test_retrieve_from_unknown_source_type(self):
        """未知 source_type 返回空。"""
        retriever = Retriever()
        retriever._milvus = MagicMock()
        retriever._mysql = MagicMock()
        retriever._embedder = MagicMock()

        mock_embed_result = MagicMock()
        mock_embed_result.embeddings = [None]
        retriever._embedder.embed.return_value = mock_embed_result
        retriever._mysql.bm25_search_generic.return_value = []

        # 未知 source_type 不会 crash
        results = retriever.retrieve_from("test", "unknown_source")
        assert results == []

    def test_empty_query(self):
        """空查询返回空。"""
        retriever = Retriever()
        results = retriever.retrieve_from("", "drug")
        assert results == []


# ============================================================
# Retriever.multi_source_retrieve
# ============================================================
class TestMultiSourceRetrieve:
    """测试多源并行检索。"""

    def test_multi_source_with_no_results(self):
        """所有 source 均无结果。"""
        retriever = Retriever()
        # Mock retrieve_from to return empty
        with patch.object(retriever, 'retrieve_from', return_value=[]):
            results = retriever.multi_source_retrieve(
                "罕见病查询",
                sources=["drug", "disease", "guideline", "literature"],
                top_n_per_source=3,
                final_top_n=10,
            )
            assert results == []

    def test_default_sources(self):
        """默认检索所有 4 个源。"""
        retriever = Retriever()
        with patch.object(retriever, 'retrieve_from') as mock_rf:
            mock_rf.return_value = []
            retriever.multi_source_retrieve("测试查询", final_top_n=5)
            assert mock_rf.call_count == 4  # drug, disease, guideline, literature

    def test_dedup_by_key(self):
        """相同 (doc_id, source_type, chunk_text[:100]) 去重。"""
        retriever = Retriever()
        dup_result = SearchResult(
            chunk_text="重复内容",
            drug_name="test",
            section="s1",
            score=0.9,
            doc_id=1,
            chunk_index=0,
            source="drug_vector",
        )
        with patch.object(retriever, 'retrieve_from', return_value=[dup_result, dup_result]):
            results = retriever.multi_source_retrieve(
                "测试", sources=["drug"], top_n_per_source=5, final_top_n=10
            )
            assert len(results) <= 1  # 去重

    def test_failure_isolation(self):
        """某个 source 失败不影响其他。"""
        retriever = Retriever()
        original = retriever.retrieve_from

        def mock_retrieve_from(query, source_type, top_n=None):
            if source_type == "disease":
                raise RuntimeError("Collection not found")
            if source_type == "drug":
                return [SearchResult(
                    chunk_text="content", drug_name="drug1", section="s",
                    score=0.9, doc_id=1, chunk_index=0, source="vector")]
            return []

        with patch.object(retriever, 'retrieve_from', side_effect=mock_retrieve_from):
            results = retriever.multi_source_retrieve(
                "测试", sources=["drug", "disease"], top_n_per_source=5, final_top_n=10
            )
            # drug 源的结果应该还在
            assert len(results) > 0
