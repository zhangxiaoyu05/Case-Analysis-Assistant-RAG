"""
混合检索器 (v1.0.0)

融合 Milvus 向量检索（语义相似度）和 MySQL BM25 全文检索（关键词匹配），
通过 RRF (Reciprocal Rank Fusion) 算法合并结果。

v1.0.0: 支持多源并行检索（drug/disease/guideline/literature）。

使用方式:
    from app.online.retriever import Retriever, SearchResult

    retriever = Retriever(milvus_client, mysql_client)
    results = retriever.retrieve("阿司匹林一天吃几次？")
    multi_results = retriever.multi_source_retrieve("心衰治疗", sources=["drug","disease","guideline","literature"])
"""

import uuid
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from app.config import config
from app.db.milvus_client import MilvusClient
from app.db.mysql_client import MySQLClient
from app.offline.embedder import Embedder


# ============================================================
# 数据类
# ============================================================
@dataclass
class SearchResult:
    """单条检索结果"""

    chunk_text: str
    drug_name: str
    section: str
    score: float  # RRF 融合得分（越高越相关）
    doc_id: int
    chunk_index: int
    source: str = ""  # "vector" / "bm25" / "rrf"（融合来源）

    # 用于去重的唯一键
    def dedup_key(self) -> str:
        """返回去重用的唯一标识"""
        return f"{self.doc_id}_{self.chunk_index}"


# ============================================================
# Retriever
# ============================================================
class Retriever:
    """
    混合检索器：向量检索 + BM25 检索 → RRF 融合。

    使用方式:
        retriever = Retriever()
        results = retriever.retrieve("阿司匹林用法用量")
        top_docs = retriever.retrieve_top_docs("感冒发热", top_n=5)
    """

    def __init__(
        self,
        milvus_client: MilvusClient | None = None,
        mysql_client: MySQLClient | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        """
        初始化检索器。

        Args:
            milvus_client: Milvus 客户端（不传则自动创建）
            mysql_client: MySQL 客户端（不传则自动创建）
            embedder: 向量化器（不传则自动创建）
        """
        self._milvus = milvus_client
        self._mysql = mysql_client
        self._embedder = embedder

        self._own_milvus = milvus_client is None
        self._own_mysql = mysql_client is None
        self._own_embedder = embedder is None

        # 检索参数
        self._vector_top_k = config.retrieval_vector_top_k
        self._bm25_top_k = config.retrieval_bm25_top_k
        self._rrf_k = config.retrieval_rrf_k
        self._rrf_top_n = config.retrieval_rrf_top_n

    # ----------------------------------------------------------
    # 懒加载
    # ----------------------------------------------------------
    @property
    def milvus(self) -> MilvusClient:
        if self._milvus is None:
            self._milvus = MilvusClient()
            self._milvus.connect()
        return self._milvus

    @property
    def mysql(self) -> MySQLClient:
        if self._mysql is None:
            self._mysql = MySQLClient()
            self._mysql.connect()
        return self._mysql

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    # ----------------------------------------------------------
    # 主检索入口
    # ----------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_n: int | None = None,
        drug_name: str | None = None,
    ) -> list[SearchResult]:
        """
        执行混合检索：向量 + BM25 → RRF 融合。

        Args:
            query: 用户查询文本
            top_n: 最终返回的 Top-N 结果数（默认 config.retrieval_rrf_top_n = 5）
            drug_name: 可选，按药品名称过滤（如"阿司匹林"）

        Returns:
            SearchResult 列表，按 RRF score 降序排列
        """
        if top_n is None:
            top_n = self._rrf_top_n

        if not query or not query.strip():
            logger.warning("空查询，返回空结果")
            return []

        request_id = uuid.uuid4().hex[:8]
        logger.info(f"[{request_id}] 开始混合检索: query={query[:60]}..., top_n={top_n}")

        # -------------------------------------------------------
        # 1. 查询向量化
        # -------------------------------------------------------
        try:
            embed_result = self.embedder.embed([query], text_type="query")
            if not embed_result.embeddings or embed_result.embeddings[0] is None:
                logger.warning("查询向量化失败，仅使用 BM25 检索")
                query_vector = None
            else:
                query_vector = embed_result.embeddings[0]
        except Exception as e:
            logger.error(f"查询向量化异常: {e}，仅使用 BM25 检索")
            query_vector = None

        # -------------------------------------------------------
        # 2. 向量检索（Milvus）— v1.0.0 统一 schema 使用 source_name
        # -------------------------------------------------------
        vector_results: list[SearchResult] = []
        if query_vector is not None:
            try:
                # v1.0.0: 兼容新旧 Milvus schema（旧 drug_chunks 用 drug_name，新 collection 用 source_name）
                milvus_results = self.milvus.search(
                    query_vector=query_vector,
                    top_k=self._vector_top_k,
                    filter_expr=None,  # v1.0.0: 不做标量过滤，由 search() 内字段回退处理
                )
                for r in milvus_results:
                    entity = r.get("entity", {})
                    # v1.0.0: 兼容新旧 Milvus schema
                    # 新 schema 用 source_name, 旧 schema 用 drug_name
                    source_name = entity.get("source_name") or entity.get("drug_name", "")
                    vector_results.append(SearchResult(
                        chunk_text=entity.get("chunk_text", ""),
                        drug_name=source_name,
                        section=entity.get("section", ""),
                        score=0.0,  # RRF 阶段再计算
                        doc_id=entity.get("doc_id", 0),
                        chunk_index=entity.get("chunk_index", 0),
                        source="vector",
                    ))
                logger.info(
                    f"[{request_id}] 向量检索: {len(vector_results)} 条 (top_k={self._vector_top_k})"
                )
            except Exception as e:
                logger.error(f"[{request_id}] 向量检索失败: {e}")
        else:
            logger.warning(f"[{request_id}] 无查询向量，跳过向量检索")

        # -------------------------------------------------------
        # 3. BM25 检索（MySQL）
        # -------------------------------------------------------
        bm25_results: list[SearchResult] = []
        try:
            bm25_raw = self.mysql.bm25_search(
                query=query,
                top_k=self._bm25_top_k,
                drug_name=drug_name,
            )
            for r in bm25_raw:
                bm25_results.append(SearchResult(
                    chunk_text=r.get("chunk_text", ""),
                    drug_name=r.get("drug_name", ""),
                    section=r.get("section", ""),
                    score=0.0,  # RRF 阶段再计算
                    doc_id=r.get("doc_id", 0),
                    chunk_index=r.get("chunk_index", 0),
                    source="bm25",
                ))
            logger.info(
                f"[{request_id}] BM25 检索: {len(bm25_results)} 条 (top_k={self._bm25_top_k})"
            )
        except Exception as e:
            logger.error(f"[{request_id}] BM25 检索失败: {e}")

        # -------------------------------------------------------
        # 4. RRF 融合
        # -------------------------------------------------------
        if not vector_results and not bm25_results:
            logger.warning(f"[{request_id}] 两种检索均无结果")
            return []

        fused = self._rrf_fusion(vector_results, bm25_results, top_n)
        logger.info(
            f"[{request_id}] RRF 融合: {len(vector_results)}向量 + {len(bm25_results)}BM25 "
            f"→ {len(fused)} 条最终结果"
        )

        return fused

    # ----------------------------------------------------------
    # RRF 融合算法
    # ----------------------------------------------------------
    def _rrf_fusion(
        self,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
        top_n: int,
    ) -> list[SearchResult]:
        """
        Reciprocal Rank Fusion 融合多个排序列表。

        公式: RRF_score(d) = Σ 1 / (k + rank_r(d))
        其中 k = 60，rank_r(d) 是文档 d 在第 r 个排序列表中的排名（从 1 开始）。

        对两个列表中相同的 chunk（doc_id + chunk_index 相同），RRF 分数相加。
        """
        k = self._rrf_k

        # 聚合 RRF 分数
        rrf_scores: dict[str, tuple[float, SearchResult]] = {}  # dedup_key -> (score, SearchResult)

        # 处理向量检索排名
        for rank, result in enumerate(vector_results, start=1):
            key = result.dedup_key()
            rrf = 1.0 / (k + rank)
            if key in rrf_scores:
                # 同一 chunk 在两个列表中都出现，分数相加
                prev_score, prev_result = rrf_scores[key]
                rrf_scores[key] = (prev_score + rrf, prev_result)
                rrf_scores[key][1].source = "rrf"  # 标记为融合来源
            else:
                rrf_scores[key] = (rrf, result)

        # 处理 BM25 排名
        for rank, result in enumerate(bm25_results, start=1):
            key = result.dedup_key()
            rrf = 1.0 / (k + rank)
            if key in rrf_scores:
                prev_score, prev_result = rrf_scores[key]
                rrf_scores[key] = (prev_score + rrf, prev_result)
                # source 保持 rrf（如果之前已融合）或更新为 rrf
                rrf_scores[key][1].source = "rrf"
            else:
                rrf_scores[key] = (rrf, result)

        # 按 RRF 分数降序排序
        sorted_items = sorted(rrf_scores.values(), key=lambda x: x[0], reverse=True)

        # 取 top_n 并设置最终分数
        results: list[SearchResult] = []
        for score, result in sorted_items[:top_n]:
            result.score = score
            results.append(result)

        return results

    # ----------------------------------------------------------
    # v1.0.0: 多源检索方法
    # ----------------------------------------------------------
    def retrieve_from(
        self,
        query: str,
        source_type: str,
        top_n: int | None = None,
    ) -> list[SearchResult]:
        """
        从指定 source_type 的 collection 检索。

        Args:
            query: 查询文本
            source_type: drug / disease / guideline / literature
            top_n: 返回 Top-K，默认使用 RRF 融合数

        Returns:
            SearchResult 列表，带 source_type 标记
        """
        if top_n is None:
            top_n = self._rrf_top_n

        if not query or not query.strip():
            return []

        request_id = uuid.uuid4().hex[:8]
        logger.info(f"[{request_id}] 单源检索: source={source_type}, query={query[:60]}...")

        # 1. 查询向量化
        try:
            embed_result = self.embedder.embed([query], text_type="query")
            query_vector = embed_result.embeddings[0] if embed_result.embeddings else None
        except Exception as e:
            logger.error(f"[{request_id}] 向量化失败: {e}")
            query_vector = None

        # 2. 向量检索（Milvus）
        vector_results: list[SearchResult] = []
        if query_vector is not None:
            try:
                from app.db.milvus_client import MilvusClient
                collection_name = f"{source_type}_chunks"
                mc = MilvusClient(collection_name=collection_name)
                mc.connect()
                milvus_results = mc.search(
                    query_vector=query_vector,
                    top_k=self._vector_top_k,
                )
                mc.disconnect()
                for r in milvus_results:
                    entity = r.get("entity", {})
                    vector_results.append(SearchResult(
                        chunk_text=entity.get("chunk_text", ""),
                        drug_name=entity.get("source_name", entity.get("drug_name", "")),
                        section=entity.get("section", ""),
                        score=0.0,
                        doc_id=entity.get("doc_id", 0),
                        chunk_index=entity.get("chunk_index", 0),
                        source="vector",
                    ))
            except Exception as e:
                logger.error(f"[{request_id}] {source_type} 向量检索失败: {e}")

        # 3. BM25 检索（MySQL）
        bm25_results: list[SearchResult] = []
        try:
            bm25_raw = self.mysql.bm25_search_generic(
                source_type=source_type,
                query=query,
                top_k=self._bm25_top_k,
            )
            for r in bm25_raw:
                bm25_results.append(SearchResult(
                    chunk_text=r.get("chunk_text", ""),
                    drug_name=r.get("drug_name", r.get("disease_name",
                               r.get("guideline_title", r.get("title", "")))),
                    section=r.get("section", ""),
                    score=0.0,
                    doc_id=r.get("doc_id", 0),
                    chunk_index=r.get("chunk_index", 0),
                    source="bm25",
                ))
        except Exception as e:
            logger.error(f"[{request_id}] {source_type} BM25 检索失败: {e}")

        # 4. RRF 融合
        if not vector_results and not bm25_results:
            return []

        fused = self._rrf_fusion(vector_results, bm25_results, top_n)
        # 标记 source_type
        for r in fused:
            r.source = f"{source_type}_{r.source}"

        return fused

    def multi_source_retrieve(
        self,
        query: str,
        sources: list[str] | None = None,
        top_n_per_source: int = 5,
        final_top_n: int = 15,
    ) -> list[dict]:
        """
        多源并行检索 + 跨源 RRF 融合。

        Args:
            query: 查询文本
            sources: 要检索的 source 列表，默认全部 4 个
                     ["drug", "disease", "guideline", "literature"]
            top_n_per_source: 每个源取 Top-N
            final_top_n: 跨源融合后最终取 N 条

        Returns:
            带 source_type 标记的检索结果字典列表
        """
        if sources is None:
            sources = ["drug", "disease", "guideline", "literature"]

        request_id = uuid.uuid4().hex[:8]
        logger.info(
            f"[{request_id}] 多源检索开始: sources={sources}, "
            f"query={query[:80]}..."
        )

        all_results: list[SearchResult] = []

        for source_type in sources:
            try:
                results = self.retrieve_from(query, source_type, top_n=top_n_per_source)
                # 在 chunk_text 层面标记 source_type（后续 dedup 保留）
                all_results.extend(results)
                logger.info(
                    f"[{request_id}] {source_type}: {len(results)} 条"
                )
            except Exception as e:
                logger.error(f"[{request_id}] {source_type} 检索失败（跳过）: {e}")
                continue

        if not all_results:
            logger.warning(f"[{request_id}] 所有 source 均无检索结果")
            return []

        # 按 (doc_id, source_type, chunk_text[:100]) 去重
        from dataclasses import asdict

        seen = set()
        unique_results: list[dict] = []
        for r in all_results:
            r_dict = asdict(r)
            # 提取 source_type 从 source 字段
            st = "unknown"
            for s in sources:
                if r.source.startswith(s):
                    st = s
                    break
            r_dict["source_type"] = st
            key = (r_dict.get("doc_id"), st, r_dict.get("chunk_text", "")[:100])
            if key not in seen:
                seen.add(key)
                unique_results.append(r_dict)

        # Phase 1 风格：按 score 降序排列，每种 source 至少保留 2 条
        balanced = _balanced_sample(unique_results, per_source_min=2, total_max=final_top_n)

        logger.info(
            f"[{request_id}] 多源检索完成: {len(balanced)} 条 "
            f"(原始 {len(unique_results)} 条去重后)"
        )

        return balanced

    # ----------------------------------------------------------
    # 便捷方法
    # ----------------------------------------------------------
    def retrieve_top_docs(
        self,
        query: str,
        top_n: int = 5,
        drug_name: str | None = None,
    ) -> list[SearchResult]:
        """
        便捷方法：直接检索并返回指定数量的 Top-N 文档。

        Args:
            query: 用户查询
            top_n: 返回数量
            drug_name: 可选，按药品名过滤

        Returns:
            SearchResult 列表
        """
        return self.retrieve(query, top_n=top_n, drug_name=drug_name)

    def retrieve_context_text(
        self,
        query: str,
        top_n: int | None = None,
        drug_name: str | None = None,
    ) -> str:
        """
        检索并拼接为上下文文本，可直接注入到 LLM prompt 中。

        Args:
            query: 用户查询
            top_n: 返回数量
            drug_name: 可选，按药品名过滤

        Returns:
            格式化的上下文字符串
        """
        results = self.retrieve(query, top_n=top_n, drug_name=drug_name)
        return self._format_context(results)

    @staticmethod
    def _format_context(results: list[SearchResult]) -> str:
        """将检索结果格式化为上下文文本"""
        if not results:
            return "（未检索到相关参考资料）"

        parts: list[str] = []
        for i, r in enumerate(results, start=1):
            parts.append(
                f"[{i}] 【{r.drug_name}】{r.section or ''}\n"
                f"{r.chunk_text}\n"
                f"(来源: {r.source}, 得分: {r.score:.4f})"
            )
        return "\n\n".join(parts)

    # ----------------------------------------------------------
    # 资源清理
    # ----------------------------------------------------------
    def close(self) -> None:
        """释放自管理的数据库连接"""
        if self._own_milvus and self._milvus is not None:
            self._milvus.disconnect()
        if self._own_mysql and self._mysql is not None:
            self._mysql.disconnect()

    def __enter__(self) -> "Retriever":
        return self

    def __exit__(self, *args) -> None:
        self.close()


# ============================================================
# v1.0.0: 均衡采样辅助函数
# ============================================================
def _balanced_sample(
    results: list[dict],
    per_source_min: int = 2,
    total_max: int = 15,
) -> list[dict]:
    """
    跨源均衡采样：每种 source_type 至少保留 per_source_min 条，总数不超过 total_max。

    策略:
    1. 先按 source_type 分组
    2. 每组取 top per_source_min 条（按 score 降序）
    3. 剩余名额按各组剩余结果数比例分配
    """
    from collections import defaultdict

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        st = r.get("source_type", "unknown")
        groups[st].append(r)

    # 各组内按 score 降序
    for st in groups:
        groups[st].sort(key=lambda x: x.get("score", 0.0), reverse=True)

    selected: list[dict] = []

    # 第一轮：每种至少 per_source_min 条
    for st, items in groups.items():
        selected.extend(items[:per_source_min])

    # 如果已经超过 total_max，截断
    if len(selected) >= total_max:
        selected.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return selected[:total_max]

    # 第二轮：从各组取剩余
    remaining_slots = total_max - len(selected)
    remaining_pool = []
    for st, items in groups.items():
        remaining_pool.extend(items[per_source_min:])

    remaining_pool.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    selected.extend(remaining_pool[:remaining_slots])

    selected.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return selected
