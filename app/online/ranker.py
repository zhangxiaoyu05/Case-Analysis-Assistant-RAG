"""
重排序模块

对混合检索的结果使用 DashScope Rerank API (qwen3-rerank) 进行二次排序，
提升检索结果与查询之间的语义相关性。

使用方式:
    from app.online.ranker import Ranker, RankedDocument

    ranker = Ranker()
    ranked = ranker.rerank(
        query="阿司匹林一天吃几次？",
        documents=["【阿司匹林】【用法用量】口服，成人一次0.3~0.6g...", ...],
    )
    for doc in ranked:
        print(f"score={doc.score:.4f} | {doc.chunk_text[:50]}...")
"""

import uuid
from dataclasses import dataclass, field

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import config


# ============================================================
# 数据类
# ============================================================
@dataclass
class RankedDocument:
    """重排序后的单条文档"""

    chunk_text: str
    drug_name: str
    section: str
    score: float  # 重排序后得分（relevance_score，0~1 左右）
    doc_id: int
    chunk_index: int
    rerank_index: int = -1  # 原始输入中的索引
    original_score: float = 0.0  # RRF 原始得分


# ============================================================
# Ranker
# ============================================================
class Ranker:
    """
    使用 DashScope Rerank API 对检索结果进行二次排序。

    模型: qwen3-rerank（从 config 读取）

    使用方式:
        ranker = Ranker()
        ranked_docs = ranker.rerank(query, documents, top_n=5)
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """
        初始化重排序器。

        Args:
            model: 重排序模型（默认 config.rerank_model = qwen3-rerank）
            api_key: DashScope API Key（默认从 config 读取）
        """
        self._model = model or config.rerank_model
        self._api_key = api_key or config.DASHSCOPE_API_KEY

        if not self._model:
            raise ValueError(
                "rerank_model 未配置。请在 config/config.yaml 的 models.rerank.model 中设置。"
            )
        if not self._api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY 未配置。请设置环境变量或在初始化 Ranker 时传入 api_key。"
            )

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_n: int | None = None,
        return_documents: bool = True,
    ) -> list[RankedDocument]:
        """
        对文档列表进行重排序。

        Args:
            query: 用户查询文本
            documents: 文档列表，每个文档为 dict，需包含:
                - chunk_text: 文本内容
                - drug_name: 药品名
                - section: 章节
                - doc_id: 文档 ID
                - chunk_index: 块序号
                - score (可选): RRF 原始得分
            top_n: 返回 Top-N 结果（默认使用 config.retrieval_rrf_top_n）
            return_documents: 是否在 API 响应中返回原文（默认 True）

        Returns:
            RankedDocument 列表，按 relevance_score 降序排列
        """
        if top_n is None:
            top_n = config.retrieval_rrf_top_n

        if not documents:
            logger.warning("文档列表为空，跳过重排序")
            return []

        if not query or not query.strip():
            logger.warning("查询为空，按原始顺序返回")
            return [
                RankedDocument(
                    chunk_text=d["chunk_text"],
                    drug_name=d.get("drug_name", ""),
                    section=d.get("section", ""),
                    score=d.get("score", 0.0),
                    doc_id=d.get("doc_id", 0),
                    chunk_index=d.get("chunk_index", 0),
                    rerank_index=i,
                    original_score=d.get("score", 0.0),
                )
                for i, d in enumerate(documents)
            ][:top_n]

        request_id = uuid.uuid4().hex[:8]
        logger.info(
            f"[{request_id}] 开始重排序: query={query[:60]}..., "
            f"docs={len(documents)}, top_n={top_n}"
        )

        # 提取纯文本列表
        doc_texts = [d["chunk_text"] for d in documents]

        try:
            results = self._call_rerank_api(query, doc_texts, top_n, return_documents)

            # 构建 RankedDocument 列表
            ranked: list[RankedDocument] = []
            for item in results:
                idx = item.get("index", -1)
                score = item.get("relevance_score", 0.0)

                if 0 <= idx < len(documents):
                    original = documents[idx]
                    ranked.append(RankedDocument(
                        chunk_text=original["chunk_text"],
                        drug_name=original.get("drug_name", ""),
                        section=original.get("section", ""),
                        score=score,
                        doc_id=original.get("doc_id", 0),
                        chunk_index=original.get("chunk_index", 0),
                        rerank_index=idx,
                        original_score=original.get("score", 0.0),
                    ))
                else:
                    logger.warning(f"重排序返回的索引 {idx} 超出范围 (共 {len(documents)} 条)")

            # 按 relevance_score 降序排列
            ranked.sort(key=lambda x: x.score, reverse=True)

            logger.info(
                f"[{request_id}] 重排序完成: {len(ranked)} 条, "
                f"最高分={ranked[0].score:.4f}" if ranked else f"[{request_id}] 重排序完成: 0 条"
            )
            return ranked

        except Exception as e:
            logger.error(f"[{request_id}] 重排序 API 调用失败: {e}，回退到原始排序")
            return self._fallback_sort(documents, top_n)

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        after=lambda retry_state: logger.warning(
            f"重排序重试 {retry_state.attempt_number}/3: "
            f"{retry_state.outcome.exception() if retry_state.outcome else 'unknown'}"
        ),
    )
    def _call_rerank_api(
        self,
        query: str,
        documents: list[str],
        top_n: int,
        return_documents: bool,
    ) -> list[dict]:
        """
        调用 DashScope TextReRank API（带重试）。

        Args:
            query: 查询文本
            documents: 文档文本列表
            top_n: 返回数量
            return_documents: 是否返回文档内容

        Returns:
            API 返回的 results 列表 [{"index": 0, "relevance_score": 0.95, "document": "..."}, ...]
        """
        from dashscope import TextReRank

        response = TextReRank.call(
            model=self._model,
            query=query,
            documents=documents,
            top_n=top_n,
            return_documents=return_documents,
            api_key=self._api_key,
        )

        if response.status_code != 200:
            error_msg = (
                f"DashScope Rerank API 错误: status={response.status_code}, "
                f"message={response.message}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        output = response.output
        if output is None:
            raise RuntimeError("DashScope Rerank API 返回了空的 output")

        results = output.get("results", [])
        if not results:
            logger.warning("DashScope Rerank API 返回了空的 results 列表")

        return results

    @staticmethod
    def _fallback_sort(
        documents: list[dict],
        top_n: int,
    ) -> list[RankedDocument]:
        """
        重排序失败时的回退策略：按原始 RRF 分数排序。
        """
        sorted_docs = sorted(
            enumerate(documents),
            key=lambda x: x[1].get("score", 0.0),
            reverse=True,
        )
        return [
            RankedDocument(
                chunk_text=d["chunk_text"],
                drug_name=d.get("drug_name", ""),
                section=d.get("section", ""),
                score=d.get("score", 0.0),
                doc_id=d.get("doc_id", 0),
                chunk_index=d.get("chunk_index", 0),
                rerank_index=idx,
                original_score=d.get("score", 0.0),
            )
            for idx, d in sorted_docs[:top_n]
        ]


# ============================================================
# 便捷函数
# ============================================================
def rerank_documents(
    query: str,
    documents: list[dict],
    top_n: int | None = None,
) -> list[RankedDocument]:
    """
    便捷函数：一行调用完成重排序。

    Args:
        query: 查询文本
        documents: 文档列表
        top_n: 返回数量

    Returns:
        RankedDocument 列表
    """
    ranker = Ranker()
    return ranker.rerank(query, documents, top_n=top_n)
