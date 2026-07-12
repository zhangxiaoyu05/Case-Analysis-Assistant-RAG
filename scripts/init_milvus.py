#!/usr/bin/env python

"""
Milvus 初始化脚本

创建 drug_chunks Collection 并构建 IVF_FLAT 索引。

使用方式:
    python scripts/init_milvus.py                # 首次创建（已存在则跳过）
    python scripts/init_milvus.py --force        # 强制重建（删除已有 Collection）
"""

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中，便于直接运行此脚本
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from app.config import config
from app.db.milvus_client import MilvusClient


def init_milvus(force: bool = False) -> bool:
    """
    初始化 Milvus Collection 和索引

    Args:
        force: 如果为 True，已有 Collection 时先删除再重建

    Returns:
        True 表示初始化成功
    """
    logger.info("=" * 60)
    logger.info("Milvus 初始化开始")
    logger.info(f"连接地址: {config.MILVUS_HOST}:{config.MILVUS_PORT}")
    logger.info(f"Collection: {config.milvus_collection_name}")
    logger.info(f"维度: {config.milvus_dimension}")
    logger.info(f"索引类型: {config.milvus_index_type} (nlist={config.milvus_nlist})")
    logger.info(f"度量类型: {config.milvus_metric_type}")
    logger.info("=" * 60)

    client = MilvusClient()
    client.connect()

    try:
        if force and client.collection_exists():
            logger.warning("--force 模式：删除已有 Collection")
            client.drop_collection()

        if client.collection_exists():
            logger.info(f"Collection '{config.milvus_collection_name}' 已存在，跳过创建")
        else:
            client.create_collection(drop_if_exists=False)

        # 确保已加载到内存
        client.load_collection()

        # 打印 Collection 信息
        info = client.get_collection_info()
        logger.info(f"\nCollection 详情:")
        logger.info(f"  名称: {info['name']}")
        logger.info(f"  行数: {info['row_count']}")
        logger.info(f"  字段:")
        for field in info.get("fields", []):
            logger.info(f"    - {field['name']} ({field['type']})")
        logger.info(f"  描述: {info['description']}")

        logger.info("\n✅ Milvus 初始化完成")
        return True

    except Exception as e:
        logger.error(f"❌ Milvus 初始化失败: {e}")
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
    args = parser.parse_args()

    success = init_milvus(force=args.force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
