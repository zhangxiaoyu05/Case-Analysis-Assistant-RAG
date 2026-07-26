"""
离线流程编排器

串联文档加载 → 清洗 → 切分 → 向量化 → MySQL + Milvus 入库的完整流程。

v1.0.0: 支持多源文档（drug/disease/guideline/literature）。

支持:
- 单药品文档: 一个文件 = 一种药品（常规流程）
- 多药品合集: 一个文件包含多种药品 → 智能拆分后每种独立入库
- 疾病/指南/文献: source_type 路由 + 专用切分器

使用方式:
    from app.offline.pipeline import run_pipeline, PipelineResult

    result = run_pipeline("data/raw/阿司匹林说明书.pdf")
    result = run_pipeline("data/raw/心衰指南2024.pdf", source_type="guideline")
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
from app.offline.multi_drug_splitter import (
    SubDocument,
    detect_multi_drug,
    split_multi_drug,
)
from app.offline.splitter import Chunk, split_document
# v1.0.0: 新切分器
from app.offline.splitter_disease import split_disease_document
from app.offline.splitter_guideline import split_guideline_document
from app.offline.splitter_literature import split_literature_document


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
    # 多药品合集拆分结果（仅合集文档有值）
    sub_results: list["PipelineResult"] = field(default_factory=list)


# ============================================================
# 核心流程 — 单药品处理
# ============================================================
def _process_single_drug(
    raw_text: str,
    drug_name: str,
    source_file: str,
    drug_manufacturer: Optional[str] = None,
    drug_category: Optional[str] = None,
    desensitize: bool = False,
    batch_id: Optional[str] = None,
    mysql_client: Optional[MySQLClient] = None,
    milvus_client: Optional[MilvusClient] = None,
    embedder: Optional[Embedder] = None,
    # v1.0.0
    source_type: str = "drug",
    **extra_fields,
) -> PipelineResult:
    """
    处理单个文档的核心逻辑（步骤 2-8）。v1.0.0 支持多源路由。

    对给定的文本执行：清洗 → 切分 → MySQL 入库 → 向量化 → Milvus 入库。

    Args:
        raw_text: 原始文本
        drug_name: 来源名称（药品名/疾病名/指南标题/文献标题）
        source_file: 来源文件名
        desensitize: 是否启用 LLM 脱敏
        batch_id: 批次 ID（自动生成）
        mysql_client: MySQL 客户端（必传）
        milvus_client: Milvus 客户端（必传）
        embedder: Embedder 实例（必传）
        source_type: drug / disease / guideline / literature

    Returns:
        PipelineResult — 含状态、计数、耗时
    """
    t_start = time.time()
    batch_id = batch_id or uuid.uuid4().hex[:12]

    assert mysql_client is not None, "mysql_client is required"
    assert milvus_client is not None, "milvus_client is required"
    assert embedder is not None, "embedder is required"

    warnings: list[str] = []

    source_emoji = {"drug": "💊", "disease": "🦠", "guideline": "📋", "literature": "📄"}.get(source_type, "📄")
    logger.info("=" * 60)
    logger.info(f"{source_emoji} 处理 {source_type}: {drug_name}")
    logger.info(f"   batch_id: {batch_id}, 来源: {source_file}")
    logger.info(f"   文本长度: {len(raw_text)} 字符")
    logger.info("=" * 60)

    # 创建索引批次记录
    mysql_client.insert_index_record(
        batch_id=batch_id,
        drug_name=drug_name,
        index_status="running",
    )

    try:
        # ============================================================
        # 步骤 2: 清洗
        # ============================================================
        logger.info("🧹 文本清洗...")
        cleaned_text = clean_text(raw_text, desensitize=desensitize)

        if not cleaned_text.strip():
            _finalize_batch(
                mysql_client, batch_id, "failed",
                error_message="清洗后文本为空",
            )
            return PipelineResult(
                batch_id=batch_id,
                doc_id=-1,
                drug_name=drug_name,
                source_file=source_file,
                total_chunks=0,
                indexed_chunks=0,
                failed_chunks=0,
                status="failed",
                error_message="清洗后文本为空",
                elapsed_seconds=time.time() - t_start,
            )

        # ============================================================
        # 步骤 3: 切分（v1.0.0: 按 source_type 选择切分器）
        # ============================================================
        logger.info(f"✂️ 文本切分（{source_type}）...")
        if source_type == "disease":
            chunks: list[Chunk] = split_disease_document(cleaned_text)
        elif source_type == "guideline":
            chunks: list[Chunk] = split_guideline_document(cleaned_text)
        elif source_type == "literature":
            chunks: list[Chunk] = split_literature_document(cleaned_text)
        else:
            chunks: list[Chunk] = split_document(cleaned_text)

        if not chunks:
            _finalize_batch(
                mysql_client, batch_id, "failed",
                error_message="切分后无有效文本块",
            )
            return PipelineResult(
                batch_id=batch_id,
                doc_id=-1,
                drug_name=drug_name,
                source_file=source_file,
                total_chunks=0,
                indexed_chunks=0,
                failed_chunks=0,
                status="failed",
                error_message="切分后无有效文本块",
                elapsed_seconds=time.time() - t_start,
            )

        # ============================================================
        # 步骤 4: 存储原始文档到 MySQL（v1.0.0: 按 source_type 路由）
        # ============================================================
        logger.info(f"💾 存储原始文档到 MySQL（{source_type}）...")
        try:
            if source_type == "drug":
                doc_id = mysql_client.insert_raw_doc(
                    drug_name=drug_name,
                    raw_content=raw_text,
                    drug_manufacturer=drug_manufacturer,
                    drug_category=drug_category,
                    source_file=source_file,
                )
            else:
                fields = _build_raw_doc_fields(source_type, drug_name, raw_text,
                                               source_file, **extra_fields)
                doc_id = mysql_client.insert_raw_doc_generic(source_type, fields)
            logger.info(f"原始文档已存储: doc_id={doc_id}")
        except Exception as e:
            _finalize_batch(
                mysql_client, batch_id, "failed",
                error_message=f"MySQL 原始文档存储失败: {e}",
            )
            return PipelineResult(
                batch_id=batch_id,
                doc_id=-1,
                drug_name=drug_name,
                source_file=source_file,
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
        logger.info(f"💾 存储 {len(chunks)} 个文本块到 MySQL（{source_type}）...")
        try:
            chunk_records = []
            for chunk in chunks:
                record = _build_chunk_record(source_type, doc_id, drug_name, chunk, **extra_fields)
                chunk_records.append(record)

            if source_type == "drug":
                mysql_client.insert_chunks_batch(chunk_records)
            else:
                mysql_client.insert_chunks_batch_generic(source_type, chunk_records)
            logger.info(f"文本块已存储: {len(chunk_records)} 条")
        except Exception as e:
            _finalize_batch(
                mysql_client, batch_id, "failed",
                error_message=f"MySQL 文本块存储失败: {e}",
                total_chunks=len(chunks),
            )
            return PipelineResult(
                batch_id=batch_id,
                doc_id=doc_id,
                drug_name=drug_name,
                source_file=source_file,
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
        error_msg: Optional[str] = None
        status = "completed"

        if not emb_result.embeddings or all(v is None for v in emb_result.embeddings):
            logger.warning("所有向量化均失败，跳过 Milvus 入库")
            status = "partial"
            error_msg = "所有向量化均失败（BM25 检索仍可用）"
            failed_chunks = len(chunks)
        else:
            logger.info("💾 存储向量到 Milvus...")

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
                    drug_name=drug_name,
                    source_file=source_file,
                    total_chunks=len(chunks),
                    indexed_chunks=0,
                    failed_chunks=len(chunks),
                    status="partial",
                    error_message=error_msg,
                    elapsed_seconds=time.time() - t_start,
                )

            valid_vectors = []
            valid_metadata = []
            for i, vec in enumerate(emb_result.embeddings):
                if vec is not None:
                    valid_vectors.append(vec)
                    valid_metadata.append({
                        "doc_id": doc_id,
                        "chunk_index": chunks[i].chunk_index,
                        "source_name": drug_name,
                        "source_type": source_type,
                        "section": chunks[i].section[:100] if chunks[i].section else "",
                        "chunk_text": chunks[i].chunk_text,
                        "extra_field_1": extra_fields.get("evidence_level", ""),
                        "extra_field_2": _get_extra_field_2(source_type, **extra_fields),
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
        logger.info(
            f"✅ 药品处理完成: {drug_name} "
            f"({len(chunks)} chunks, {indexed_chunks} 索引, {elapsed:.1f}s)"
        )

        return PipelineResult(
            batch_id=batch_id,
            doc_id=doc_id,
            drug_name=drug_name,
            source_file=source_file,
            total_chunks=len(chunks),
            indexed_chunks=indexed_chunks,
            failed_chunks=failed_chunks,
            status=status,
            error_message=error_msg,
            elapsed_seconds=elapsed,
            warnings=warnings,
        )

    except Exception as e:
        elapsed = time.time() - t_start
        logger.exception(f"处理异常 [{drug_name}]: {e}")
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
            drug_name=drug_name,
            source_file=source_file,
            total_chunks=0,
            indexed_chunks=0,
            failed_chunks=0,
            status="failed",
            error_message=str(e),
            elapsed_seconds=elapsed,
        )


# ============================================================
# 多药品结果聚合
# ============================================================
def _aggregate_results(
    results: list[PipelineResult],
    source_file: str,
    batch_id: str,
    elapsed: float,
) -> PipelineResult:
    """
    将多个子药品的处理结果聚合为一个汇总结果。

    Args:
        results: 各子药品的 PipelineResult 列表
        source_file: 来源文件
        batch_id: 父批次 ID
        elapsed: 总耗时（秒）

    Returns:
        聚合后的 PipelineResult
    """
    if not results:
        return PipelineResult(
            batch_id=batch_id,
            doc_id=-1,
            drug_name="多药品合集(0种)",
            source_file=source_file,
            total_chunks=0,
            indexed_chunks=0,
            failed_chunks=0,
            status="failed",
            error_message="拆分后无有效药品文档",
            elapsed_seconds=elapsed,
        )

    total_chunks = sum(r.total_chunks for r in results)
    indexed_chunks = sum(r.indexed_chunks for r in results)
    failed_chunks = sum(r.failed_chunks for r in results)

    # 状态判定：全部 completed → completed；否则 partial
    all_completed = all(r.status == "completed" for r in results)
    any_failed = any(r.status == "failed" for r in results)

    if all_completed:
        status = "completed"
    elif any_failed:
        status = "partial"
    else:
        status = "partial"

    # 汇总警告和错误
    warnings: list[str] = []
    for r in results:
        if r.status == "failed":
            warnings.append(f"[{r.drug_name}] 失败: {r.error_message}")
        elif r.status == "partial":
            warnings.append(f"[{r.drug_name}] 部分成功: {r.indexed_chunks}/{r.total_chunks} chunks")
        else:
            warnings.append(f"[{r.drug_name}] 完成: {r.total_chunks} chunks")

    drug_names = [r.drug_name for r in results]
    drug_name_summary = f"多药品合集({len(results)}种: {', '.join(drug_names[:5])}{'...' if len(drug_names) > 5 else ''})"

    logger.info("=" * 60)
    logger.info(f"📊 合集处理汇总: {len(results)} 种药品")
    logger.info(f"   总 chunks: {total_chunks}, 已索引: {indexed_chunks}, 失败: {failed_chunks}")
    logger.info(f"   状态: {status}")
    logger.info("=" * 60)

    return PipelineResult(
        batch_id=batch_id,
        doc_id=-1,  # 合集没有单一 doc_id
        drug_name=drug_name_summary,
        source_file=source_file,
        total_chunks=total_chunks,
        indexed_chunks=indexed_chunks,
        failed_chunks=failed_chunks,
        status=status,
        error_message=None if all_completed else f"{sum(1 for r in results if r.status == 'failed')} 种药品处理失败",
        elapsed_seconds=elapsed,
        warnings=warnings,
        sub_results=results,
    )


# ============================================================
# 公共 API — run_pipeline
# ============================================================
def run_pipeline(
    file_path: Path,
    drug_name: Optional[str] = None,
    drug_manufacturer: Optional[str] = None,
    drug_category: Optional[str] = None,
    desensitize: bool = False,
    overwrite: bool = False,
    batch_id: Optional[str] = None,
    mysql_client: Optional[MySQLClient] = None,
    milvus_client: Optional[MilvusClient] = None,
    embedder: Optional[Embedder] = None,
    # v1.0.0: 多源支持
    source_type: str = "drug",
    disease_name: Optional[str] = None,
    guideline_title: Optional[str] = None,
    **extra_fields,
) -> PipelineResult:
    """
    对单个文档执行完整的离线处理流程。

    支持：
    - 单药品文档：按常规流程处理（一个文件 → 一种药品）
    - 多药品合集文档：智能检测并拆分为多个独立药品，每种独立入库
    - v1.0.0: 多源文档（disease/guideline/literature）使用专用切分器

    流程:
        1. load_document() — 加载文档原文
        1.5 (新增) detect_multi_drug() — 检测是否为多药品合集
            如果是合集 → split_multi_drug() → 每种药品独立执行步骤 2-8
        2. clean_text() — 清洗文本
        3. 按 source_type 选择切分器
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
        overwrite: 已存在时是否覆盖旧数据（默认 False）
        batch_id: 批次 ID（自动生成 UUID）
        mysql_client: 现有 MySQL 客户端（不传则自动创建）
        milvus_client: 现有 Milvus 客户端（不传则自动创建）
        embedder: 现有 Embedder（不传则自动创建）
        source_type: drug / disease / guideline / literature（默认 drug）
        disease_name: source_type=disease 时使用
        guideline_title: source_type=guideline 时使用

    Returns:
        PipelineResult — 含状态、计数、耗时。
        多药品合集文档的 sub_results 字段包含每种药品的独立结果。
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
        collection_name = f"{source_type}_chunks"
        milvus_client = MilvusClient(collection_name=collection_name)
        milvus_client.connect()
    if embedder is None:
        embedder = Embedder()

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

        # ============================================================
        # 步骤 1.5: 多药品文档检测与拆分 (NEW)
        # ============================================================
        if detect_multi_drug(doc.raw_text):
            logger.info("🔍 检测到多药品合集文档，启动智能拆分...")
            sub_docs: list[SubDocument] = split_multi_drug(doc.raw_text)
            logger.info(f"拆分完成: {len(sub_docs)} 种药品将独立入库")

            # 逐一处理每种药品
            all_results: list[PipelineResult] = []
            for i, sub in enumerate(sub_docs):
                logger.info(f"\n--- 处理子文档 {i + 1}/{len(sub_docs)}: {sub.drug_name} ---")
                sub_batch_id = f"{batch_id}_{i}"
                resolved_sub_name = drug_name if i == 0 and drug_name else sub.drug_name

                # 检查子药品是否已存在
                if mysql_client.drug_exists(resolved_sub_name):
                    if overwrite:
                        logger.info(f"药品 '{resolved_sub_name}' 已存在，覆盖模式：先删除旧数据")
                        _delete_drug_data(mysql_client, milvus_client, resolved_sub_name)
                    else:
                        logger.info(f"药品 '{resolved_sub_name}' 已存在，跳过")
                        all_results.append(PipelineResult(
                            batch_id=sub_batch_id,
                            doc_id=-1,
                            drug_name=resolved_sub_name,
                            source_file=str(file_path),
                            total_chunks=0,
                            indexed_chunks=0,
                            failed_chunks=0,
                            status="skipped",
                            error_message=f"药品 '{resolved_sub_name}' 已存在",
                            elapsed_seconds=0,
                        ))
                        continue

                try:
                    sub_result = _process_single_drug(
                        raw_text=sub.text,
                        drug_name=resolved_sub_name,
                        source_file=f"{file_path}#{sub.drug_name}",
                        drug_manufacturer=drug_manufacturer,
                        drug_category=drug_category,
                        desensitize=desensitize,
                        batch_id=sub_batch_id,
                        mysql_client=mysql_client,
                        milvus_client=milvus_client,
                        embedder=embedder,
                    )
                except Exception as e:
                    logger.error(f"子文档处理异常 [{resolved_sub_name}]: {e}")
                    sub_result = PipelineResult(
                        batch_id=sub_batch_id,
                        doc_id=-1,
                        drug_name=resolved_sub_name,
                        source_file=str(file_path),
                        total_chunks=0,
                        indexed_chunks=0,
                        failed_chunks=0,
                        status="failed",
                        error_message=str(e),
                        elapsed_seconds=0,
                    )
                all_results.append(sub_result)

            elapsed = time.time() - t_start
            return _aggregate_results(all_results, str(file_path), batch_id, elapsed)

        # ============================================================
        # 单药品文档: 去重检查 + 正常流程
        # ============================================================
        resolved_drug_name = drug_name or doc.inferred_drug_name or file_path.stem
        logger.info(f"药品名称: {resolved_drug_name}")

        # 检查药品是否已存在
        if mysql_client.drug_exists(resolved_drug_name):
            if overwrite:
                logger.info(f"药品 '{resolved_drug_name}' 已存在，覆盖模式：先删除旧数据")
                _delete_drug_data(mysql_client, milvus_client, resolved_drug_name)
            else:
                elapsed = time.time() - t_start
                logger.info(f"药品 '{resolved_drug_name}' 已存在，跳过（使用 --overwrite 覆盖）")
                return PipelineResult(
                    batch_id=batch_id,
                    doc_id=-1,
                    drug_name=resolved_drug_name,
                    source_file=str(file_path),
                    total_chunks=0,
                    indexed_chunks=0,
                    failed_chunks=0,
                    status="skipped",
                    error_message=f"药品 '{resolved_drug_name}' 已存在，使用 overwrite=True 覆盖",
                    elapsed_seconds=elapsed,
                )

        result = _process_single_drug(
            raw_text=doc.raw_text,
            drug_name=resolved_drug_name,
            source_file=str(file_path),
            drug_manufacturer=drug_manufacturer,
            drug_category=drug_category,
            desensitize=desensitize,
            batch_id=batch_id,
            mysql_client=mysql_client,
            milvus_client=milvus_client,
            embedder=embedder,
            source_type=source_type,
            disease_name=disease_name,
            guideline_title=guideline_title,
            **extra_fields,
        )

        elapsed = time.time() - t_start
        logger.info("=" * 60)
        logger.info(
            f"✅ 处理完成: {result.drug_name} "
            f"({result.total_chunks} chunks, {result.indexed_chunks} 索引, {elapsed:.1f}s)"
        )
        logger.info("=" * 60)

        return result

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
    overwrite: bool = False,
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
                overwrite=overwrite,
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
def _delete_drug_data(
    mysql_client: MySQLClient,
    milvus_client: MilvusClient,
    drug_name: str,
) -> int:
    """
    删除指定药品在 MySQL 和 Milvus 中的全部数据。

    用于 overwrite 模式：先清理旧数据，再入库新数据。
    容错设计：Milvus 删除失败不影响 MySQL 删除结果。

    Args:
        mysql_client: MySQL 客户端
        milvus_client: Milvus 客户端
        drug_name: 药品名称

    Returns:
        MySQL 中删除的 doc 数量
    """
    # 1. MySQL 删除
    deleted_doc_ids = mysql_client.delete_drug_by_name(drug_name)

    # 2. Milvus 向量删除（容错：Collection 可能不存在或为空）
    if deleted_doc_ids:
        try:
            milvus_client.delete_by_drug_name(drug_name)
        except Exception as e:
            logger.warning(f"Milvus 删除药品 '{drug_name}' 向量失败（不影响 MySQL 数据）: {e}")

    return len(deleted_doc_ids)


# ============================================================
# v1.0.0: 多源辅助函数
# ============================================================
def _build_raw_doc_fields(
    source_type: str,
    name: str,
    raw_content: str,
    source_file: str,
    **extra,
) -> dict:
    """根据 source_type 构建原始文档插入字段。"""
    if source_type == "disease":
        return {
            "disease_name": name,
            "disease_category": extra.get("disease_category", ""),
            "department": extra.get("department", ""),
            "raw_content": raw_content,
            "source_type": extra.get("source_subtype", "textbook"),
            "source_file": source_file,
        }
    elif source_type == "guideline":
        return {
            "guideline_title": name,
            "issuing_body": extra.get("issuing_body", ""),
            "publish_year": extra.get("publish_year", 0),
            "disease_name": extra.get("disease_name", ""),
            "department": extra.get("department", ""),
            "raw_content": raw_content,
            "source_file": source_file,
            "url": extra.get("url", ""),
        }
    elif source_type == "literature":
        return {
            "title": name,
            "authors": extra.get("authors", ""),
            "journal": extra.get("journal", ""),
            "publish_year": extra.get("publish_year", 0),
            "doi": extra.get("doi", ""),
            "pmid": extra.get("pmid", ""),
            "abstract_text": extra.get("abstract_text", ""),
            "full_text": raw_content,
            "study_type": extra.get("study_type", ""),
            "disease_name": extra.get("disease_name", ""),
            "keywords": extra.get("keywords", ""),
            "source_file": source_file,
        }
    else:
        return {"unknown": raw_content}


def _build_chunk_record(
    source_type: str,
    doc_id: int,
    name: str,
    chunk,
    **extra,
) -> dict:
    """根据 source_type 构建文本块插入记录。"""
    # 公共字段
    record = {
        "doc_id": doc_id,
        "section": chunk.section[:100] if chunk.section else None,
        "chunk_index": chunk.chunk_index,
        "chunk_text": chunk.chunk_text,
        "char_count": chunk.char_count,
    }

    if source_type == "drug":
        record["drug_name"] = name
    elif source_type == "disease":
        record["disease_name"] = name
        record["source_type"] = extra.get("source_subtype", "textbook")
        record["evidence_level"] = extra.get("evidence_level", "")
    elif source_type == "guideline":
        record["guideline_title"] = name
        record["disease_name"] = extra.get("disease_name", "")
        record["evidence_level"] = extra.get("evidence_level", "")
        record["recommendation_grade"] = extra.get("recommendation_grade", "")
        record["issuing_body"] = extra.get("issuing_body", "")
        record["publish_year"] = extra.get("publish_year", 0)
    elif source_type == "literature":
        record["title"] = name
        record["disease_name"] = extra.get("disease_name", "")
        record["study_type"] = extra.get("study_type", "")
        record["evidence_level"] = extra.get("evidence_level", "")
        record["publish_year"] = extra.get("publish_year", 0)
        record["doi"] = extra.get("doi", "")

    return record


def _get_extra_field_2(source_type: str, **extra) -> str:
    """根据 source_type 返回 extra_field_2 的值。"""
    if source_type == "drug":
        return ""
    elif source_type == "disease":
        return extra.get("source_subtype", "")
    elif source_type == "guideline":
        return extra.get("recommendation_grade", "")
    elif source_type == "literature":
        return extra.get("study_type", "")
    return ""


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
