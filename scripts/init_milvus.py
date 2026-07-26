#!/usr/bin/env python

"""
Milvus 初始化脚本 (v1.0.0)

创建 4 个 Collection（drug_chunks / disease_chunks / guideline_chunks / literature_chunks）
并构建 IVF_FLAT 索引。使用统一 schema。

使用方式:
    python scripts/init_milvus.py                        # 首次创建全部 4 个（已存在则跳过）
    python scripts/init_milvus.py --force                 # 强制重建全部 4 个
    python scripts/init_milvus.py --force --collections drug,disease  # 只重建指定
    python scripts/init_milvus.py --collections drug       # 只创建 drug
"""

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中，便于直接运行此脚本
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from app.config import config
from app.db.milvus_client import MilvusClient


ALL_COLLECTIONS = ["drug_chunks", "disease_chunks", "guideline_chunks", "literature_chunks"]


def init_milvus(force: bool = False, collections: list[str] | None = None) -> bool:
    """
    初始化 Milvus Collection 和索引。

    Args:
        force: 如果为 True，已有 Collection 时先删除再重建
        collections: 要初始化的 collection 列表，None 表示全部 4 个

    Returns:
        True 表示初始化成功
    """
    targets = collections or ALL_COLLECTIONS

    logger.info("=" * 60)
    logger.info("Milvus 初始化开始 (v1.0.0)")
    logger.info(f"连接地址: {config.MILVUS_HOST}:{config.MILVUS_PORT}")
    logger.info(f"目标 Collection: {targets}")
    logger.info(f"维度: {config.milvus_dimension}")
    logger.info(f"索引类型: {config.milvus_index_type} (nlist={config.milvus_nlist})")
    logger.info(f"度量类型: {config.milvus_metric_type}")
    logger.info("=" * 60)

    all_ok = True
    for collection_name in targets:
        try:
            ok = _init_one_collection(collection_name, force)
            if not ok:
                all_ok = False
        except Exception as e:
            logger.error(f"❌ {collection_name} 初始化失败: {e}")
            all_ok = False

    if all_ok:
        logger.info("\n✅ 所有 Milvus Collection 初始化完成")
    else:
        logger.warning("\n⚠️ 部分 Collection 初始化失败，请检查上述错误信息")

    return all_ok


def _init_one_collection(collection_name: str, force: bool) -> bool:
    """初始化单个 Collection。"""
    client = MilvusClient(collection_name=collection_name)
    client.connect()

    try:
        if force and client.collection_exists():
            logger.warning(f"--force 模式：删除已有 Collection '{collection_name}'")
            client.drop_collection()

        if client.collection_exists():
            logger.info(f"Collection '{collection_name}' 已存在，跳过创建")
        else:
            client.create_collection(drop_if_exists=False)

        # 确保已加载到内存
        client.load_collection()

        # 打印 Collection 信息
        info = client.get_collection_info()
        logger.info(f"  {collection_name}: {info['row_count']} 行, {len(info.get('fields', []))} 字段")

        return True
    except Exception as e:
        logger.error(f"  {collection_name}: ❌ {e}")
        return False
    finally:
        client.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="初始化 Milvus 向量数据库（创建 Collection + 索引）"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重建：如果 Collection 已存在则先删除",
    )
    parser.add_argument(
        "--collections",
        type=str,
        default=None,
        help="要初始化的 Collection，逗号分隔。默认全部。"
             "可选: drug,disease,guideline,literature",
    )
    args = parser.parse_args()

    collections = None
    if args.collections:
        collections = [f"{c.strip()}_chunks" for c in args.collections.split(",")]

    success = init_milvus(force=args.force, collections=collections)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
