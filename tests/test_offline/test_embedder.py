"""
测试 app.offline.embedder — 向量化模块

覆盖: EmbeddingResult, Embedder 类, embed_texts 便捷函数
"""

from unittest.mock import MagicMock, patch

import pytest

from app.offline.embedder import (
    Embedder,
    EmbeddingResult,
    _is_retryable,
    embed_texts,
)


# ============================================================
# EmbeddingResult
# ============================================================
class TestEmbeddingResult:
    """测试 EmbeddingResult 数据类。"""

    def test_default_values(self):
        """默认值检查。"""
        result = EmbeddingResult()
        assert result.embeddings == []
        assert result.failed_indices == []
        assert result.total_attempted == 0
        assert result.total_succeeded == 0

    def test_full_result(self):
        """完整结果。"""
        result = EmbeddingResult(
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            failed_indices=[2],
            total_attempted=3,
            total_succeeded=2,
        )
        assert len(result.embeddings) == 2
        assert result.failed_indices == [2]
        assert result.total_attempted == 3
        assert result.total_succeeded == 2


# ============================================================
# _is_retryable
# ============================================================
class TestIsRetryable:
    """测试可重试异常判断。"""

    def test_dashscope_exception(self):
        """DashScopeException 可重试。"""
        exc = Exception("DashScopeException: rate limit")
        assert _is_retryable(exc) is True

    def test_rate_limit(self):
        """限流错误可重试。"""
        exc = Exception("rate limit exceeded")
        assert _is_retryable(exc) is True

    def test_timeout(self):
        """超时可重试。"""
        exc = Exception("connection timeout")
        assert _is_retryable(exc) is True

    def test_non_retryable(self):
        """一般错误不可重试。"""
        exc = ValueError("invalid input")
        assert _is_retryable(exc) is False


# ============================================================
# Embedder.__init__
# ============================================================
class TestEmbedderInit:
    """测试 Embedder 初始化。"""

    def test_init_with_api_key(self):
        """使用显式 API Key 初始化。"""
        embedder = Embedder(api_key="test-key")
        assert embedder._api_key == "test-key"
        assert embedder._model is not None
        assert embedder._dimension > 0
        assert embedder._batch_size > 0

    def test_init_without_api_key_raises(self, monkeypatch):
        """无 API Key 时初始化抛出 ValueError。"""
        # 清除环境变量中的 API Key
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        # 需要确保 config 也不会返回 key
        with patch("app.offline.embedder.config") as mock_config:
            mock_config.DASHSCOPE_API_KEY = ""
            mock_config.embedding_model = "test-model"
            mock_config.embedding_dimension = 1024
            mock_config.embedding_batch_size = 25
            with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
                Embedder()


# ============================================================
# Embedder.embed
# ============================================================
class TestEmbedderEmbed:
    """测试向量化方法。"""

    def test_empty_texts(self):
        """空文本列表返回空结果。"""
        embedder = Embedder(api_key="test-key")
        result = embedder.embed([])
        assert result.total_attempted == 0
        assert result.embeddings == []

    def test_embed_success(self, mock_dashscope_response):
        """成功向量化。"""
        mock_resp = mock_dashscope_response(embeddings=[
            {"embedding": [0.1] * 1024},
            {"embedding": [0.2] * 1024},
        ])

        embedder = Embedder(api_key="test-key", batch_size=2)
        with patch.object(embedder, "_call_api_with_retry", return_value=[[0.1] * 1024, [0.2] * 1024]):
            result = embedder.embed(["文本1", "文本2"])
            assert result.total_attempted == 2
            assert result.total_succeeded == 2
            assert len(result.embeddings) == 2
            assert result.failed_indices == []

    def test_embed_partial_failure(self):
        """部分向量化失败。"""
        embedder = Embedder(api_key="test-key", batch_size=1)

        call_count = [0]
        original = embedder._call_api_with_retry

        def side_effect(texts, text_type):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("API error for batch 2")
            return [[0.1] * 1024]

        with patch.object(embedder, "_call_api_with_retry", side_effect=side_effect):
            result = embedder.embed(["文本1", "文本2", "文本3"])
            # 第 2 批失败了，第 3 批也失败因为 batch_size=1 且失败后继续
            assert result.total_attempted == 3


# ============================================================
# Embedder._call_api (static method)
# ============================================================
class TestCallApi:
    """测试底层 API 调用。"""

    def test_call_api_success(self, mock_dashscope_response):
        """API 调用成功返回向量。"""
        mock_resp = mock_dashscope_response(embeddings=[
            {"embedding": [0.1, 0.2, 0.3]},
        ])

        with patch("dashscope.TextEmbedding") as mock_te:
            mock_te.call.return_value = mock_resp
            vectors = Embedder._call_api(
                texts=["测试"],
                model="test-model",
                dimension=1024,
                api_key="test-key",
            )
            assert len(vectors) == 1
            assert vectors[0] == [0.1, 0.2, 0.3]

    def test_call_api_error(self, mock_dashscope_response):
        """API 错误抛出 RuntimeError（经过 retry 装饰器）。"""
        mock_resp = mock_dashscope_response(status_code=500)

        with patch("dashscope.TextEmbedding") as mock_te:
            mock_te.call.return_value = mock_resp
            # _call_api 有 @retry(3) 装饰器，会重试 3 次后仍然失败
            with pytest.raises(RuntimeError):
                # 直接调用静态方法（绕过 retry 的 _call_api_with_retry）
                Embedder._call_api.__wrapped__(
                    texts=["测试"],
                    model="test-model",
                    dimension=1024,
                    api_key="test-key",
                )


# ============================================================
# embed_texts 便捷函数
# ============================================================
class TestEmbedTexts:
    """测试 embed_texts 便捷函数。"""

    def test_embed_texts_returns_embedding_result(self):
        """返回 EmbeddingResult。"""
        with patch("app.offline.embedder.Embedder.embed") as mock_embed:
            mock_embed.return_value = EmbeddingResult(
                embeddings=[[0.1] * 1024],
                total_attempted=1,
                total_succeeded=1,
            )
            result = embed_texts(["测试文本"], api_key="test-key")
            assert isinstance(result, EmbeddingResult)
            assert result.total_succeeded == 1
