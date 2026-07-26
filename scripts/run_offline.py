#!/usr/bin/env python
"""
离线处理 CLI 入口 (v1.0.0)

对文档执行完整的离线处理流程：
加载 → 清洗 → 切分 → 向量化 → MySQL + Milvus 入库。

v1.0.0: 支持多源文档（drug/disease/guideline/literature）。

使用方式:
    python scripts/run_offline.py --file data/raw/阿司匹林说明书.pdf
    python scripts/run_offline.py --file doc.pdf --drug-name "阿司匹林" --desensitize
    python scripts/run_offline.py --dir data/raw/ --dry-run
    python scripts/run_offline.py --file doc.pdf --batch-id my-batch-001

    # v1.0.0: 多源入库
    python scripts/run_offline.py --file guideline.pdf --source-type guideline \
        --guideline-title "中国心力衰竭诊疗指南2024" --publish-year 2024
    python scripts/run_offline.py --file disease.txt --source-type disease \
        --disease-name "2型糖尿病"
    python scripts/run_offline.py --file paper.pdf --source-type literature \
        --title "RCT of SGLT2i in HF" --study-type rct
"""

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from app.config import config
from app.offline import (
    Chunk,
    clean_text,
    load_document,
    run_pipeline,
    run_pipeline_batch,
    split_document,
)
# v1.0.0: 新切分器
from app.offline.splitter_disease import split_disease_document
from app.offline.splitter_guideline import split_guideline_document
from app.offline.splitter_literature import split_literature_document


def cmd_dry_run(
    file_path: Path,
    desensitize: bool = False,
    source_type: str = "drug",
) -> None:
    """
    干跑模式：只执行加载 → 清洗 → 切分，不写入数据库。
    用于验证文档处理结果。
    """
    logger.info("=" * 60)
    logger.info("🔍 干跑模式（不写入数据库）")
    logger.info("=" * 60)

    # 1. 加载
    doc = load_document(file_path)
    logger.info(f"\n📄 文档: {doc.source_file}")
    logger.info(f"   格式: {doc.file_type}")
    logger.info(f"   推断名称: {doc.inferred_drug_name}")
    logger.info(f"   原文长度: {len(doc.raw_text)} 字符")

    # 2. 清洗
    cleaned = clean_text(doc.raw_text, desensitize=desensitize)
    logger.info(f"   清洗后: {len(cleaned)} 字符")

    # 3. 切分（按 source_type 选择切分器）
    if source_type == "disease":
        chunks: list[Chunk] = split_disease_document(cleaned)
    elif source_type == "guideline":
        chunks: list[Chunk] = split_guideline_document(cleaned)
    elif source_type == "literature":
        chunks: list[Chunk] = split_literature_document(cleaned)
    else:
        chunks: list[Chunk] = split_document(cleaned)
    logger.info(f"   切分结果: {len(chunks)} 个 chunk")

    # 打印 chunks 预览
    if chunks:
        logger.info("\n" + "─" * 60)
        logger.info("📋 Chunk 预览")
        logger.info("─" * 60)
        for i, chunk in enumerate(chunks):
            preview = chunk.chunk_text[:80].replace("\n", "\\n")
            logger.info(
                f"  [{i:3d}] [{chunk.section:12s}] "
                f"{chunk.char_count:4d}字符 | {preview}..."
            )

    # 章节分布
    sections = {}
    for c in chunks:
        sections[c.section] = sections.get(c.section, 0) + 1
    logger.info("\n📊 章节分布:")
    for sec, count in sorted(sections.items(), key=lambda x: -x[1]):
        logger.info(f"  [{sec}]: {count} chunks")


def cmd_process_file(
    file_path: Path,
    drug_name: str | None = None,
    manufacturer: str | None = None,
    category: str | None = None,
    desensitize: bool = False,
    overwrite: bool = False,
    batch_id: str | None = None,
    source_type: str = "drug",
    **extra_fields,
) -> None:
    """处理单个文件"""
    result = run_pipeline(
        file_path=file_path,
        drug_name=drug_name,
        drug_manufacturer=manufacturer,
        drug_category=category,
        desensitize=desensitize,
        overwrite=overwrite,
        batch_id=batch_id,
        source_type=source_type,
        **extra_fields,
    )

    # 打印结果
    logger.info("\n" + "=" * 60)
    logger.info("📊 处理结果")
    logger.info("=" * 60)
    logger.info(f"  批次 ID:     {result.batch_id}")
    logger.info(f"  文档 ID:     {result.doc_id}")
    logger.info(f"  药品名称:    {result.drug_name}")
    logger.info(f"  源文件:      {result.source_file}")
    logger.info(f"  状态:        {result.status}")
    logger.info(f"  总 chunks:   {result.total_chunks}")
    logger.info(f"  已索引:      {result.indexed_chunks}")
    logger.info(f"  失败:        {result.failed_chunks}")
    logger.info(f"  耗时:        {result.elapsed_seconds:.1f}s")

    if result.error_message:
        logger.warning(f"  错误信息:    {result.error_message}")

    if result.warnings:
        for w in result.warnings:
            logger.warning(f"  ⚠️ {w}")

    if result.status == "failed":
        sys.exit(1)


def cmd_process_dir(
    dir_path: Path,
    drug_name: str | None = None,
    manufacturer: str | None = None,
    category: str | None = None,
    desensitize: bool = False,
    overwrite: bool = False,
    source_type: str = "drug",
    **extra_fields,
) -> None:
    """处理目录中所有文档（v1.0.0: 支持多 source_type）"""
    results = run_pipeline_batch(
        file_paths=sorted(dir_path.glob("*")),
        drug_name=drug_name,
        drug_manufacturer=manufacturer,
        drug_category=category,
        desensitize=desensitize,
        overwrite=overwrite,
        source_type=source_type,
        **extra_fields,
    )

    # 汇总
    failed = [r for r in results if r.status == "failed"]
    if failed:
        logger.warning(f"\n{len(failed)} 个文档处理失败:")
        for r in failed:
            logger.warning(f"  - {r.source_file}: {r.error_message}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG 药品问答系统 — 离线文档处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/run_offline.py --file data/raw/阿司匹林说明书.pdf
  python scripts/run_offline.py --file doc.pdf --drug-name "布洛芬"
  python scripts/run_offline.py --file doc.pdf --dry-run
  python scripts/run_offline.py --dir data/raw/
  python scripts/run_offline.py --file doc.pdf --desensitize
  python scripts/run_offline.py --file doc.pdf --overwrite
        """,
    )

    # 输入源（二选一）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--file",
        type=Path,
        help="单个文档文件路径",
    )
    input_group.add_argument(
        "--dir",
        type=Path,
        help="文档目录路径（处理目录下所有支持的文档）",
    )

    # 可选参数
    parser.add_argument(
        "--drug-name",
        type=str,
        default=None,
        help="药品名称（不传则从文件名推断）",
    )
    parser.add_argument(
        "--manufacturer",
        type=str,
        default=None,
        help="生产厂家",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="药品分类（如：处方药 / OTC）",
    )
    parser.add_argument(
        "--desensitize",
        action="store_true",
        help="启用 LLM 脱敏处理（会消耗 API 额度）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="当药品已存在时覆盖旧数据（默认跳过）",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="自定义批次 ID（用于追踪，默认自动生成）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="干跑模式：只做加载/清洗/切分，不入库",
    )
    # v1.0.0: 多源支持参数
    parser.add_argument(
        "--source-type",
        type=str,
        default="drug",
        choices=["drug", "disease", "guideline", "literature"],
        help="文档来源类型（默认 drug）",
    )
    parser.add_argument(
        "--disease-name",
        type=str,
        default=None,
        help="疾病名称（source-type=disease 时使用）",
    )
    parser.add_argument(
        "--guideline-title",
        type=str,
        default=None,
        help="指南标题（source-type=guideline 时使用）",
    )
    parser.add_argument(
        "--publish-year",
        type=int,
        default=None,
        help="发布年份（source-type=guideline/literature 时使用）",
    )
    parser.add_argument(
        "--issuing-body",
        type=str,
        default=None,
        help="发布机构（source-type=guideline 时使用）",
    )
    parser.add_argument(
        "--study-type",
        type=str,
        default=None,
        help="研究类型（source-type=literature 时使用，如 RCT/meta-analysis/cohort）",
    )
    parser.add_argument(
        "--evidence-level",
        type=str,
        default=None,
        help="证据级别（如 IA/IB/IIA/IIB/1a/1b/2a 等）",
    )
    parser.add_argument(
        "--doi",
        type=str,
        default=None,
        help="DOI（source-type=literature 时使用）",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("RAG 药品问答系统 — 离线文档处理")
    logger.info("=" * 60)
    logger.info(f"Milvus: {config.MILVUS_HOST}:{config.MILVUS_PORT}")
    logger.info(f"MySQL:  {config.MYSQL_HOST}:{config.MYSQL_PORT}/{config.MYSQL_DATABASE}")
    logger.info(f"Redis: {config.REDIS_HOST}:{config.REDIS_PORT}")
    logger.info(f"Embedding: {config.embedding_model} ({config.embedding_dimension}d)")

    # v1.0.0: 构建 extra_fields
    extra_fields = {}
    source_type = args.source_type
    if source_type == "disease":
        if args.disease_name:
            extra_fields["disease_name"] = args.disease_name
        if args.evidence_level:
            extra_fields["evidence_level"] = args.evidence_level
    elif source_type == "guideline":
        if args.guideline_title:
            extra_fields["guideline_title"] = args.guideline_title
        if args.disease_name:
            extra_fields["disease_name"] = args.disease_name
        if args.publish_year:
            extra_fields["publish_year"] = args.publish_year
        if args.issuing_body:
            extra_fields["issuing_body"] = args.issuing_body
        if args.evidence_level:
            extra_fields["evidence_level"] = args.evidence_level
    elif source_type == "literature":
        if args.drug_name:
            extra_fields["title"] = args.drug_name  # drug_name 字段复用为 title
        if args.disease_name:
            extra_fields["disease_name"] = args.disease_name
        if args.study_type:
            extra_fields["study_type"] = args.study_type
        if args.publish_year:
            extra_fields["publish_year"] = args.publish_year
        if args.evidence_level:
            extra_fields["evidence_level"] = args.evidence_level
        if args.doi:
            extra_fields["doi"] = args.doi

    # Resolve name: drug_name is the primary name, but for non-drug types use specific fields
    resolved_name = args.drug_name
    if not resolved_name and source_type == "disease":
        resolved_name = args.disease_name
    elif not resolved_name and source_type == "guideline":
        resolved_name = args.guideline_title

    if args.file:
        if args.dry_run:
            cmd_dry_run(args.file, desensitize=args.desensitize, source_type=source_type)
        else:
            cmd_process_file(
                file_path=args.file,
                drug_name=resolved_name,
                manufacturer=args.manufacturer,
                category=args.category,
                desensitize=args.desensitize,
                overwrite=args.overwrite,
                batch_id=args.batch_id,
                source_type=source_type,
                **extra_fields,
            )
    elif args.dir:
        if args.dry_run:
            logger.warning("--dir 暂不支持 --dry-run，请对单个文件使用 --dry-run")
            sys.exit(1)
        else:
            cmd_process_dir(
                dir_path=args.dir,
                drug_name=args.drug_name,
                manufacturer=args.manufacturer,
                category=args.category,
                desensitize=args.desensitize,
                overwrite=args.overwrite,
                source_type=source_type,
                **extra_fields,
            )


if __name__ == "__main__":
    main()
