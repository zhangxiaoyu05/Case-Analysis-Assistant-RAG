#!/usr/bin/env python
"""
统一初始化脚本 (v1.0.0)

一键初始化所有存储层：
1. MySQL — 检查业务表状态（含 v1.0.0 新增 6 张表）
2. Milvus — 创建 4 个 Collection + 索引
3. Redis — 连通性检查（PING）
4. 健康检查汇总

使用方式:
    python scripts/init_collection.py                  # 标准初始化
    python scripts/init_collection.py --force           # 强制重建 Milvus Collection
    python scripts/init_collection.py --skip-mysql      # 跳过 MySQL 检查
    python scripts/init_collection.py --skip-milvus     # 跳过 Milvus 初始化
    python scripts/init_collection.py --skip-redis      # 跳过 Redis 检查
"""

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from app.config import config
from scripts.init_milvus import init_milvus


def check_mysql() -> dict:
    """检查 MySQL 表状态（含 v1.0.0 新增表）。"""
    from app.db.mysql_client import MySQLClient

    logger.info("\n" + "─" * 50)
    logger.info("📋 MySQL 检查")
    logger.info("─" * 50)

    try:
        client = MySQLClient()
        client.connect()

        stats = client.get_table_stats()
        is_ready = client.is_ready()
        is_v1_ready = client.is_v1_ready()

        logger.info(f"连接: {client._host}:{client._port}/{client._database}")
        for table, count in stats.items():
            if count is None:
                logger.warning(f"  {table}: ❌ 表不存在")
            else:
                logger.info(f"  {table}: ✅ ({count} 行)")

        if is_v1_ready:
            logger.info("所有 v1.0.0 表就绪 ✅")
        elif is_ready:
            logger.info("核心表就绪，部分 v1.0.0 新表缺失（可执行 migration_v3.sql 创建）")

        client.disconnect()
        return {"success": True, "ready": is_ready, "v1_ready": is_v1_ready, "stats": stats}
    except Exception as e:
        logger.error(f"MySQL 连接失败: {e}")
        return {"success": False, "ready": False, "v1_ready": False, "stats": {}, "error": str(e)}


def check_redis() -> dict:
    """检查 Redis 连通性"""
    import redis

    logger.info("\n" + "─" * 50)
    logger.info("📋 Redis 检查")
    logger.info("─" * 50)

    try:
        r = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            socket_connect_timeout=5,
            decode_responses=True,
        )
        pong = r.ping()
        logger.info(f"PING → {pong}")
        logger.info(f"连接: {config.REDIS_HOST}:{config.REDIS_PORT} ✅")
        r.close()
        return {"success": True, "ready": True}
    except Exception as e:
        logger.error(f"Redis 连接失败: {e}")
        return {"success": False, "ready": False, "error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG 临床病例分析助手 — 一键初始化所有存储层"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重建所有 Milvus Collection",
    )
    parser.add_argument(
        "--skip-mysql",
        action="store_true",
        help="跳过 MySQL 检查",
    )
    parser.add_argument(
        "--skip-milvus",
        action="store_true",
        help="跳过 Milvus 初始化",
    )
    parser.add_argument(
        "--skip-redis",
        action="store_true",
        help="跳过 Redis 检查",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("RAG 临床病例分析助手 — 存储层初始化 (v1.0.0)")
    logger.info("=" * 60)
    logger.info(f"Milvus: {config.MILVUS_HOST}:{config.MILVUS_PORT}")
    logger.info(f"MySQL:  {config.MYSQL_HOST}:{config.MYSQL_PORT}/{config.MYSQL_DATABASE}")
    logger.info(f"Redis: {config.REDIS_HOST}:{config.REDIS_PORT}")

    results = {}

    # 1. MySQL
    if not args.skip_mysql:
        results["mysql"] = check_mysql()
    else:
        logger.info("⏭️ 跳过 MySQL 检查")

    # 2. Milvus
    if not args.skip_milvus:
        logger.info("\n" + "─" * 50)
        logger.info("📋 Milvus 初始化")
        logger.info("─" * 50)
        milvus_ok = init_milvus(force=args.force)
        results["milvus"] = {"success": milvus_ok, "ready": milvus_ok}
    else:
        logger.info("⏭️ 跳过 Milvus 初始化")

    # 3. Redis
    if not args.skip_redis:
        results["redis"] = check_redis()
    else:
        logger.info("⏭️ 跳过 Redis 检查")

    # 4. 汇总
    logger.info("\n" + "=" * 60)
    logger.info("📊 健康检查汇总")
    logger.info("=" * 60)

    all_ready = True
    for name, result in results.items():
        status = "✅ 就绪" if result.get("ready") else "❌ 未就绪"
        logger.info(f"  {name}: {status}")
        if not result.get("ready"):
            all_ready = False
            if "error" in result:
                logger.error(f"    错误: {result['error']}")

    if all_ready:
        logger.info("\n🎉 所有存储层初始化完成！")
    else:
        logger.warning("\n⚠️ 部分存储层未就绪，请检查上述错误信息")
        if results.get("mysql", {}).get("ready") and not results.get("mysql", {}).get("v1_ready"):
            logger.info("💡 提示: 执行 mysql -u root -p rag_pharma < scripts/migration_v3.sql 创建 v1.0.0 新表")

    sys.exit(0 if all_ready else 1)


if __name__ == "__main__":
    main()
