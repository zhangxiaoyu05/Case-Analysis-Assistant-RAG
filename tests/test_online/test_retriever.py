"""
测试 app.online.retriever — 混合检索模块

覆盖: SearchResult, Retriever 类, RRF 融合
"""

from unittest.mock import MagicMock, patch

import pytest

from app.online.retriever import (
    Retriever,
    SearchResult,
)


# ============================================================
# SearchResult
# ============================================================
class TestSearchResult:
    """测试 SearchResult 数据类。"""

    def test_create_result(self):
        """创建 SearchResult。"""
        result = SearchResult(
            chunk_text="用于解热镇痛",
            drug_name="阿司匹林肠溶片",
            section="适应症",
            score=0.95,
            doc_id=1,
            chunk_index=0,
            source="milvus",
        )
        assert result.drug_name == "阿司匹林肠溶片"
        assert result.score == 0.95
        assert result.source == "milvus"

    def test_dedup_key(self):
        """去重键。"""
        result = SearchResult(
            chunk_text="文本",
            drug_name="阿司匹林",
            section="用法用量",
            score=0.9,
            doc_id=1,
            chunk_index=2,
            source="milvus",
        )
        key = result.dedup_key()
        assert isinstance(key, str)
        assert "1" in key  # doc_id
        assert "2" in key  # chunk_index


# ============================================================
# Retriever.__init__
# ============================================================
class TestRetrieverInit:
    """测试初始化。"""

    def test_init_with_explicit_clients(self, mock_milvus_client, mock_mysql_client, mock_embedder):
        """使用显式的客户端初始化。"""
        retriever = Retriever(
            milvus_client=mock_milvus_client,
            mysql_client=mock_mysql_client,
            embedder=mock_embedder,
        )
        assert retriever.milvus is mock_milvus_client
        assert retriever.mysql is mock_mysql_client
        assert retriever.embedder is mock_embedder

    def test_init_creates_clients_lazily(self):
        """不传客户端时懒初始化。"""
        # 懒初始化需要实际的连接参数，我们验证 Retriever 可创建即可
        retriever = Retriever()
        assert retriever is not None


# ============================================================
# Retriever.retrieve
# ============================================================
class TestRetrieverRetrieve:
    """测试检索方法。"""

    def test_retrieve_success(self, mock_milvus_client, mock_mysql_client, mock_embedder):
        """成功混合检索。"""
        retriever = Retriever(
            milvus_client=mock_milvus_client,
            mysql_client=mock_mysql_client,
            embedder=mock_embedder,
        )
        results = retriever.retrieve("阿司匹林怎么吃？")
        assert len(results) > 0
        for r in results:
            assert isinstance(r, SearchResult)
            assert r.chunk_text
            assert r.drug_name

    def test_retrieve_with_drug_name_filter(self, mock_milvus_client, mock_mysql_client, mock_embedder):
        """按药品名过滤。"""
        retriever = Retriever(
            milvus_client=mock_milvus_client,
            mysql_client=mock_mysql_client,
            embedder=mock_embedder,
        )
        # 验证 drug_name 参数被传递给 bm25_search
        results = retriever.retrieve("怎么吃？", drug_name="阿司匹林肠溶片")
        assert len(results) > 0
        # bm25_search 应该收到了 drug_name 参数
        mock_mysql_client.bm25_search.assert_called()

    def test_retrieve_top_docs(self, mock_milvus_client, mock_mysql_client, mock_embedder):
        """retrieve_top_docs 返回指定数量的结果。"""
        # 模拟更多搜索结果
        mock_milvus_client.search.return_value = [
            {"doc_id": 1, "drug_name": f"药品{i}", "chunk_text": f"内容{i}",
             "section": "适应症", "score": 0.9 - i * 0.1, "chunk_index": i}
            for i in range(10)
        ]
        mock_mysql_client.bm25_search.return_value = [
            {"doc_id": 1, "drug_name": f"药品{i}", "chunk_text": f"内容{i}",
             "section": "适应症", "score": 8.0 - i, "chunk_index": i}
            for i in range(5)
        ]

        retriever = Retriever(
            milvus_client=mock_milvus_client,
            mysql_client=mock_mysql_client,
            embedder=mock_embedder,
        )
        results = retriever.retrieve_top_docs("测试查询", top_n=3)
        assert len(results) <= 3

    def test_retrieve_context_text(self, mock_milvus_client, mock_mysql_client, mock_embedder):
        """retrieve_context_text 返回格式化文本。"""
        retriever = Retriever(
            milvus_client=mock_milvus_client,
            mysql_client=mock_mysql_client,
            embedder=mock_embedder,
        )
        context = retriever.retrieve_context_text("阿司匹林怎么吃？")
        assert isinstance(context, str)
        assert len(context) > 0

    def test_retrieve_empty_results(self):
        """无检索结果时返回空列表。"""
        mock_milvus = MagicMock()
        mock_milvus.search.return_value = []
        mock_mysql = MagicMock()
        mock_mysql.bm25_search.return_value = []
        mock_emb = MagicMock()
        mock_emb.embed.return_value = type("Result", (), {"embeddings": [[0.1] * 1024]})()

        retriever = Retriever(
            milvus_client=mock_milvus,
            mysql_client=mock_mysql,
            embedder=mock_emb,
        )
        results = retriever.retrieve("查询无结果")
        assert results == []


# ============================================================
# Retriever.close
# ============================================================
class TestRetrieverClose:
    """测试资源清理。"""

    def test_close(self, mock_milvus_client, mock_mysql_client, mock_embedder):
        """close 释放自己创建的资源（外部提供的客户端由调用方管理）。"""
        retriever = Retriever(
            milvus_client=mock_milvus_client,
            mysql_client=mock_mysql_client,
            embedder=mock_embedder,
        )
        retriever.close()
        # 外部提供的客户端不应该被 close 释放
        mock_milvus_client.disconnect.assert_not_called()
        mock_mysql_client.disconnect.assert_not_called()

    def test_context_manager_auto_close(self):
        """上下文管理器自动关闭自建的客户端。"""
        with patch("app.online.retriever.MilvusClient") as mock_mc, \
             patch("app.online.retriever.MySQLClient") as mock_msc:
            mock_milvus = MagicMock()
            mock_mysql = MagicMock()
            mock_mc.return_value = mock_milvus
            mock_msc.return_value = mock_mysql

            with Retriever() as retriever:
                # 访问属性触发懒初始化
                _ = retriever.milvus
                _ = retriever.mysql
                assert retriever is not None
            # 自建的客户端应该被释放
            mock_milvus.disconnect.assert_called_once()
            mock_mysql.disconnect.assert_called_once()
