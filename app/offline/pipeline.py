"""
离线流程编排器

串联文档加载 → 清洗 → 切分 → 向量化 → MySQL + Milvus 入库的完整流程。

使用方式:
    from app.offline.pipeline import run_pipeline, PipelineResult

    result = run_pipeline("data/raw/阿司匹林说明书.pdf")
    print(f"处理完成: {result.total_chunks} 个 chunk, {result.indexed_chunks} 个已索引")
"""

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

from app.config import config
from app.db.milvus_client import MilvusClient
from app.db.mysql_client import MySQLClient
from app.offline.cleaner import clean_text
from app.offline.embedder import Embedder, EmbeddingResult
from app.offline.loader import LoadedDocument, load_document
from app.offline.splitter import Chunk, split_document


# ============================================================
# 数据类
# ============================================================
@dataclass
class PipelineResult:
    """单文档离线处理结果"""

    batch_id: str
    doc_id: int
    drug_name: str
    source_file: str
    total_chunks: int
    indexed_chunks: int  # Milvus 入库成功数
    failed_chunks: int  # Milvus 入库失败数
    status: str  # "completed" | "partial" | "failed"
    error_message: Optional[str] = None
    elapsed_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)


# ============================================================
# 核心流程
# ============================================================
def run_pipeline(
    file_path: Path,
    drug_name: Optional[str] = None,
    drug_manufacturer: Optional[str] = None,
    drug_category: Optional[str] = None,
    desensitize: bool = False,
    batch_id: Optional[str] = None,
    mysql_client: Optional[MySQLClient] = None,
    milvus_client: Optional[MilvusClient] = None,
    embedder: Optional[Embedder] = None,
) -> PipelineResult:
    """
    对单个文档执行完整的离线处理流程。

    流程:
        1. load_document() — 加载文档原文
        2. clean_text() — 清洗文本
        3. split_document() — 章节感知切分
        4. insert_raw_doc() — 存 MySQL 原始文档
        5. insert_chunks_batch() — 存 MySQL 文本块
        6. embedder.embed() — 生成向量
        7. milvus_client.insert_embeddings() — 存 Milvus 向量
        8. update_index_record() — 更新批次状态

    Args:
        file_path: 文档路径
        drug_name: 药品名称（不传则从文件名推断）
        drug_manufacturer: 生产厂家
        drug_category: 药品分类
        desensitize: 是否启用 LLM 脱敏
        batch_id: 批次 ID（自动生成 UUID）
        mysql_client: 现有 MySQL 客户端（不传则自动创建）
        milvus_client: 现有 Milvus 客户端（不传则自动创建）
        embedder: 现有 Embedder（不传则自动创建）

    Returns:
        PipelineResult — 含状态、计数、耗时
    """
    t_start = time.time()
    batch_id = batch_id or uuid.uuid4().hex[:12]
    file_path = Path(file_path)

    # 资源管理
    _own_mysql = mysql_client is None
    _own_milvus = milvus_client is None
    _own_embedder = embedder is None

    if mysql_client is None:
        mysql_client = MySQLClient()
        mysql_client.connect()
    if milvus_client is None:
        milvus_client = MilvusClient()
        milvus_client.connect()
    if embedder is None:
        embedder = Embedder()

    warnings: list[str] = []

    try:
        # ============================================================
        # 步骤 1: 加载文档
        # ============================================================
        logger.info("=" * 60)
        logger.info(f"📄 开始处理: {file_path.name}")
        logger.info(f"   batch_id: {batch_id}")
        logger.info("=" * 60)

        try:
            doc: LoadedDocument = load_document(file_path)
        except Exception as e:
            return PipelineResult(
                batch_id=batch_id,
                doc_id=-1,
                drug_name=drug_name or "unknown",
                source_file=str(file_path),
                total_chunks=0,
                indexed_chunks=0,
                failed_chunks=0,
                status="failed",
                error_message=f"文档加载失败: {e}",
                elapsed_seconds=time.time() - t_start,
            )

        if not doc.raw_text.strip():
            return PipelineResult(
                batch_id=batch_id,
                doc_id=-1,
                drug_name=drug_name or doc.inferred_drug_name or "unknown",
                source_file=str(file_path),
                total_chunks=0,
                indexed_chunks=0,
                failed_chunks=0,
                status="failed",
                error_message="文档内容为空",
                elapsed_seconds=time.time() - t_start,
            )

        # 确定药名
        resolved_drug_name = drug_name or doc.inferred_drug_name or file_path.stem
        logger.info(f"药品名称: {resolved_drug_name}")

        # 创建索引批次记录
        mysql_client.insert_index_record(
            batch_id=batch_id,
            drug_name=resolved_drug_name,
            index_status="running",
        )

        # ============================================================
        # 步骤 2: 清洗
        # ============================================================
        logger.info("🧹 文本清洗...")
        cleaned_text = clean_text(doc.raw_text, desensitize=desensitize)

        if not cleaned_text.strip():
            _finalize_batch(
                mysql_client, batch_id, "failed",
                error_message="清洗后文本为空",
            )
            return PipelineResult(
                batch_id=batch_id,
                doc_id=-1,
                drug_name=resolved_drug_name,
                source_file=str(file_path),
                total_chunks=0,
                indexed_chunks=0,
                failed_chunks=0,
                status="failed",
                error_message="清洗后文本为空",
                elapsed_seconds=time.time() - t_start,
            )

        # ============================================================
        # 步骤 3: 切分
        # ============================================================
        logger.info("✂️ 文本切分...")
        chunks: list[Chunk] = split_document(cleaned_text)

        if not chunks:
            _finalize_batch(
                mysql_client, batch_id, "failed",
                error_message="切分后无有效文本块",
            )
            return PipelineResult(
                batch_id=batch_id,
                doc_id=-1,
                drug_name=resolved_drug_name,
                source_file=str(file_path),
                total_chunks=0,
                indexed_chunks=0,
                failed_chunks=0,
                status="failed",
                error_message="切分后无有效文本块",
                elapsed_seconds=time.time() - t_start,
            )

        # ============================================================
        # 步骤 4: 存储原始文档到 MySQL
        # ============================================================
        logger.info("💾 存储原始文档到 MySQL...")
        try:
            doc_id = mysql_client.insert_raw_doc(
                drug_name=resolved_drug_name,
                raw_content=doc.raw_text,
                drug_manufacturer=drug_manufacturer,
                drug_category=drug_category,
                source_file=str(file_path),
            )
            logger.info(f"原始文档已存储: doc_id={doc_id}")
        except Exception as e:
            _finalize_batch(
                mysql_client, batch_id, "failed",
                error_message=f"MySQL 原始文档存储失败: {e}",
            )
            return PipelineResult(
                batch_id=batch_id,
                doc_id=-1,
                drug_name=resolved_drug_name,
                source_file=str(file_path),
                total_chunks=len(chunks),
                indexed_chunks=0,
                failed_chunks=0,
                status="failed",
                error_message=f"MySQL 原始文档存储失败: {e}",
                elapsed_seconds=time.time() - t_start,
            )

        # 更新索引记录
        mysql_client.update_index_record(
            batch_id=batch_id,
            index_status="running",
            total_chunks=len(chunks),
        )

        # ============================================================
        # 步骤 5: 存储文本块到 MySQL（BM25 全文索引）
        # ============================================================
        logger.info(f"💾 存储 {len(chunks)} 个文本块到 MySQL...")
        try:
            chunk_records = []
            for chunk in chunks:
                chunk_records.append({
                    "doc_id": doc_id,
                    "drug_name": resolved_drug_name,
                    "section": chunk.section[:50] if chunk.section else None,  # VARCHAR 50 限制
                    "chunk_index": chunk.chunk_index,
                    "chunk_text": chunk.chunk_text,
                    "char_count": chunk.char_count,
                })

            mysql_client.insert_chunks_batch(chunk_records)
            logger.info(f"文本块已存储: {len(chunk_records)} 条")
        except Exception as e:
            # MySQL chunk 存储失败，raw_doc 已入库（孤立但无害）
            _finalize_batch(
                mysql_client, batch_id, "failed",
                error_message=f"MySQL 文本块存储失败: {e}",
                total_chunks=len(chunks),
            )
            return PipelineResult(
                batch_id=batch_id,
                doc_id=doc_id,
                drug_name=resolved_drug_name,
                source_file=str(file_path),
                total_chunks=len(chunks),
                indexed_chunks=0,
                failed_chunks=len(chunks),
                status="partial",
                error_message=f"MySQL 文本块存储失败 (raw_doc 已入库): {e}",
                elapsed_seconds=time.time() - t_start,
            )

        # ============================================================
        # 步骤 6: 向量化
        # ============================================================
        logger.info("🧮 生成向量嵌入...")
        chunk_texts = [c.chunk_text for c in chunks]
        emb_result: EmbeddingResult = embedder.embed(chunk_texts)

        # ============================================================
        # 步骤 7: 存储向量到 Milvus
        # ============================================================
        indexed_chunks = 0
        failed_chunks = 0

        if not emb_result.embeddings or all(v is None for v in emb_result.embeddings):
            logger.warning("所有向量化均失败，跳过 Milvus 入库")
            status = "partial"
            error_msg = "所有向量化均失败（BM25 检索仍可用）"
            failed_chunks = len(chunks)
        else:
            logger.info("💾 存储向量到 Milvus...")

            # 确保 Milvus Collection 存在（否则插入会报错）
            if not milvus_client.collection_exists():
                error_msg = (
                    "Milvus Collection 不存在，请先运行: python scripts/init_milvus.py"
                )
                logger.error(error_msg)
                _finalize_batch(
                    mysql_client, batch_id, "partial",
                    total_chunks=len(chunks),
                    indexed_chunks=0,
                    failed_chunks=len(chunks),
                    error_message=error_msg,
                )
                return PipelineResult(
                    batch_id=batch_id,
                    doc_id=doc_id,
                    drug_name=resolved_drug_name,
                    source_file=str(file_path),
                    total_chunks=len(chunks),
                    indexed_chunks=0,
                    failed_chunks=len(chunks),
                    status="partial",
                    error_message=error_msg,
                    elapsed_seconds=time.time() - t_start,
                )

            # 收集有效的（向量化成功的）chunk
            valid_vectors = []
            valid_metadata = []
            for i, vec in enumerate(emb_result.embeddings):
                if vec is not None:
                    valid_vectors.append(vec)
                    valid_metadata.append({
                        "doc_id": doc_id,
                        "chunk_index": chunks[i].chunk_index,
                        "drug_name": resolved_drug_name,
                        "section": chunks[i].section[:50] if chunks[i].section else "",
                        "chunk_text": chunks[i].chunk_text,
                    })
                else:
                    failed_chunks += 1

            if valid_vectors:
                try:
                    insert_result = milvus_client.insert_embeddings(valid_vectors, valid_metadata)
                    indexed_chunks = insert_result.get("insert_count", len(valid_vectors))
                    logger.info(f"向量已存储到 Milvus: {indexed_chunks} 条")
                except Exception as e:
                    logger.error(f"Milvus 插入失败: {e}")
                    failed_chunks += len(valid_vectors)
                    indexed_chunks = 0
                    if not warnings:
                        warnings.append(f"Milvus 插入失败: {e}")
            else:
                logger.warning("无有效向量可插入 Milvus")

            # 确定最终状态
            if failed_chunks == 0:
                status = "completed"
                error_msg = None
            elif indexed_chunks > 0:
                status = "partial"
                error_msg = f"{failed_chunks} 个 chunk 向量化/入库失败（BM25 仍可用）"
            else:
                status = "partial"
                error_msg = (
                    f"所有 {failed_chunks} 个 chunk 的 Milvus 入库失败（BM25 检索仍可用）"
                )

        # ============================================================
        # 步骤 8: 更新索引批次状态
        # ============================================================
        _finalize_batch(
            mysql_client, batch_id, status,
            total_chunks=len(chunks),
            indexed_chunks=indexed_chunks,
            failed_chunks=failed_chunks,
            error_message=error_msg,
        )

        elapsed = time.time() - t_start
        logger.info("=" * 60)
        logger.info(
            f"✅ 处理完成: {resolved_drug_name} "
            f"({len(chunks)} chunks, {indexed_chunks} 索引, {elapsed:.1f}s)"
        )
        logger.info("=" * 60)

        return PipelineResult(
            batch_id=batch_id,
            doc_id=doc_id,
            drug_name=resolved_drug_name,
            source_file=str(file_path),
            total_chunks=len(chunks),
            indexed_chunks=indexed_chunks,
            failed_chunks=failed_chunks,
            status=status,
            error_message=error_msg,
            elapsed_seconds=elapsed,
            warnings=warnings,
        )

    except Exception as e:
        # 未捕获的异常
        elapsed = time.time() - t_start
        logger.exception(f"处理异常: {e}")
        try:
            _finalize_batch(
                mysql_client, batch_id, "failed",
                error_message=str(e),
            )
        except Exception:
            pass

        return PipelineResult(
            batch_id=batch_id,
            doc_id=-1,
            drug_name="unknown",
            source_file=str(file_path),
            total_chunks=0,
            indexed_chunks=0,
            failed_chunks=0,
            status="failed",
            error_message=str(e),
            elapsed_seconds=elapsed,
        )

    finally:
        # 清理自己创建的资源
        if _own_mysql:
            mysql_client.disconnect()
        if _own_milvus:
            milvus_client.disconnect()


# ============================================================
# 批量处理
# ============================================================
def run_pipeline_batch(
    file_paths: list[Path],
    **kwargs,
) -> list[PipelineResult]:
    """
    批量处理多个文档。每个文档独立处理，共享数据库连接。

    Args:
        file_paths: 文档路径列表
        **kwargs: 传递给 run_pipeline() 的其他参数

    Returns:
        PipelineResult 列表（与输入同序）
    """
    logger.info(f"批量处理 {len(file_paths)} 个文档")

    # 共享数据库连接
    mysql_client = MySQLClient()
    mysql_client.connect()
    milvus_client = MilvusClient()
    milvus_client.connect()
    embedder = Embedder()

    try:
        results = []
        for i, fp in enumerate(file_paths, start=1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"📦 进度: {i}/{len(file_paths)}")
            logger.info(f"{'=' * 60}")
            result = run_pipeline(
                file_path=fp,
                mysql_client=mysql_client,
                milvus_client=milvus_client,
                embedder=embedder,
                **kwargs,
            )
            results.append(result)

        # 汇总
        completed = sum(1 for r in results if r.status == "completed")
        partial = sum(1 for r in results if r.status == "partial")
        failed = sum(1 for r in results if r.status == "failed")
        total_chunks = sum(r.total_chunks for r in results)
        total_indexed = sum(r.indexed_chunks for r in results)

        logger.info("\n" + "=" * 60)
        logger.info("📊 批量处理汇总")
        logger.info("=" * 60)
        logger.info(f"  文档数: {len(results)} (成功={completed}, 部分={partial}, 失败={failed})")
        logger.info(f"  总 chunks: {total_chunks}, 总索引: {total_indexed}")
        logger.info(f"  总耗时: {sum(r.elapsed_seconds for r in results):.1f}s")

        return results

    finally:
        mysql_client.disconnect()
        milvus_client.disconnect()


# ============================================================
# 辅助函数
# ============================================================
def _finalize_batch(
    mysql_client: MySQLClient,
    batch_id: str,
    status: str,
    total_chunks: int = 0,
    indexed_chunks: int = 0,
    failed_chunks: int = 0,
    error_message: Optional[str] = None,
) -> None:
    """更新索引批次状态（容错：日志记录但不抛异常）"""
    try:
        mysql_client.update_index_record(
            batch_id=batch_id,
            index_status=status,
            total_chunks=total_chunks if total_chunks > 0 else None,
            indexed_chunks=indexed_chunks if indexed_chunks > 0 else None,
            failed_chunks=failed_chunks if failed_chunks > 0 else None,
            error_message=error_message,
        )
    except Exception as e:
        logger.warning(f"更新索引批次状态失败: {e}")
