"""
向量化模块

调用 DashScope TextEmbedding API 将文本块转为向量，
支持批处理和自动重试。

使用方式:
    from app.offline.embedder import Embedder

    embedder = Embedder()
    result = embedder.embed(["文本1", "文本2", ...])
    vectors = result.embeddings  # list[list[float]]
"""

from dataclasses import dataclass, field
from typing import Optional

from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.config import config


# ============================================================
# 数据类
# ============================================================
@dataclass
class EmbeddingResult:
    """向量化结果"""

    embeddings: list[Optional[list[float]]] = field(default_factory=list)
    failed_indices: list[int] = field(default_factory=list)
    total_attempted: int = 0
    total_succeeded: int = 0


# ============================================================
# 辅助函数
# ============================================================
def _is_retryable(exception: Exception) -> bool:
    """判断异常是否可重试（限流、网络错误等）"""
    # DashScope 异常
    if "DashScopeException" in type(exception).__name__:
        return True
    # 通用网络/连接错误
    error_msg = str(exception).lower()
    retryable_keywords = [
        "rate", "limit", "throttle", "timeout", "connection",
        "network", "unavailable", "busy", "retry",
    ]
    return any(kw in error_msg for kw in retryable_keywords)


# ============================================================
# Embedder
# ============================================================
class Embedder:
    """
    文本向量化器（通过 config.yaml 中 models.embedding 配置的模型）。

    使用方式:
        embedder = Embedder()
        result = embedder.embed(texts)
        for vec in result.embeddings:
            if vec is not None:
                print(len(vec))  # 1536
    """

    def __init__(
        self,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
        batch_size: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        初始化向量化器。

        Args:
            model: 嵌入模型名（默认 config.embedding_model = text-embedding-v4）
            dimension: 向量维度（默认 config.embedding_dimension）
            batch_size: 每批处理的文本数（默认 config.embedding_batch_size）
            api_key: DashScope API Key（默认 config.DASHSCOPE_API_KEY）
        """
        self._model = model or config.embedding_model
        self._dimension = dimension or config.embedding_dimension
        self._batch_size = batch_size or config.embedding_batch_size
        self._api_key = api_key or config.DASHSCOPE_API_KEY

        if not self._model:
            raise ValueError(
                "embedding_model 未配置。请在 config/config.yaml 的 models.embedding.model 中设置。"
            )
        if not self._api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY 未配置。请设置环境变量或在初始化 Embedder 时传入 api_key。"
            )

    def embed(
        self,
        texts: list[str],
        text_type: str = "document",
    ) -> EmbeddingResult:
        """
        对文本列表进行批量向量化。

        Args:
            texts: 待向量化的文本列表
            text_type: 文本类型，\"document\"（离线入库）或 \"query\"（在线检索）

        Returns:
            EmbeddingResult:
                - embeddings: 向量列表（与 texts 同序，失败的为 None）
                - failed_indices: 失败的索引列表
                - total_attempted / total_succeeded: 计数
        """
        if not texts:
            logger.warning("输入文本列表为空，跳过向量化")
            return EmbeddingResult()

        total = len(texts)
        result = EmbeddingResult(total_attempted=total)
        result.embeddings = [None] * total  # 预分配

        num_batches = (total + self._batch_size - 1) // self._batch_size
        logger.info(
            f"开始向量化: {total} 条文本, {num_batches} 批 "
            f"(batch_size={self._batch_size}, model={self._model}, dim={self._dimension})"
        )

        for batch_idx in range(num_batches):
            start = batch_idx * self._batch_size
            end = min(start + self._batch_size, total)
            batch_texts = texts[start:end]

            logger.debug(f"  批次 {batch_idx + 1}/{num_batches}: {start}-{end} ({len(batch_texts)} 条)")

            try:
                batch_vectors = self._call_api_with_retry(batch_texts, text_type)

                if len(batch_vectors) != len(batch_texts):
                    logger.warning(
                        f"API 返回向量数 ({len(batch_vectors)}) 与请求数 ({len(batch_texts)}) 不匹配"
                    )
                    # 尽力匹配
                    for j in range(min(len(batch_vectors), len(batch_texts))):
                        idx = start + j
                        result.embeddings[idx] = batch_vectors[j]
                        result.total_succeeded += 1
                    for j in range(len(batch_vectors), len(batch_texts)):
                        idx = start + j
                        result.failed_indices.append(idx)
                else:
                    for j, vec in enumerate(batch_vectors):
                        idx = start + j
                        result.embeddings[idx] = vec
                        result.total_succeeded += 1

            except Exception as e:
                logger.error(f"批次 {batch_idx + 1} 向量化失败: {e}")
                for j in range(start, end):
                    result.failed_indices.append(j)

        logger.info(
            f"向量化完成: {result.total_succeeded}/{total} 成功"
            + (f", {len(result.failed_indices)} 失败" if result.failed_indices else "")
        )

        return result

    def _call_api_with_retry(
        self, texts: list[str], text_type: str = "document"
    ) -> list[list[float]]:
        """带重试的 API 调用（包装静态方法）"""
        return self._call_api(
            texts=texts,
            model=self._model,
            dimension=self._dimension,
            api_key=self._api_key,
            text_type=text_type,
        )

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        after=lambda retry_state: logger.warning(
            f"向量化重试 {retry_state.attempt_number}/3: {retry_state.outcome.exception() if retry_state.outcome else 'unknown'}"
        ),
    )
    def _call_api(
        texts: list[str],
        model: str,
        dimension: int,
        api_key: str,
        text_type: str = "document",
    ) -> list[list[float]]:
        """
        调用 DashScope TextEmbedding API。

        Args:
            texts: 文本列表
            model: 模型名称
            dimension: 向量维度
            api_key: API Key
            text_type: 文本类型（\"document\" 离线入库 / \"query\" 在线检索）

        Returns:
            向量列表（与 texts 同序）

        Raises:
            RuntimeError: API 返回错误
        """
        from dashscope import TextEmbedding

        response = TextEmbedding.call(
            model=model,
            input=texts,
            text_type=text_type,
            dimension=dimension,
            api_key=api_key,
        )

        if response.status_code != 200:
            error_msg = f"DashScope API 错误: status={response.status_code}, message={response.message}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # 提取嵌入向量
        embeddings_data = response.output.get("embeddings", [])
        if not embeddings_data:
            raise RuntimeError("API 返回了空的 embeddings 列表")

        vectors = [item["embedding"] for item in embeddings_data]
        return vectors


# ============================================================
# 便捷函数
# ============================================================
def embed_texts(
    texts: list[str],
    api_key: Optional[str] = None,
    text_type: str = "document",
) -> EmbeddingResult:
    """
    便捷函数：一行调用完成向量化。

    Args:
        texts: 文本列表
        api_key: API Key（可选，默认从 config 读取）
        text_type: 文本类型（\"document\" 离线入库 / \"query\" 在线检索）

    Returns:
        EmbeddingResult
    """
    embedder = Embedder(api_key=api_key)
    return embedder.embed(texts, text_type=text_type)
