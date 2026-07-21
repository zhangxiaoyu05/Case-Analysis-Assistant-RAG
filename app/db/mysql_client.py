"""
MySQL 数据库连接模块

封装 pymysql 连接池和业务表 CRUD 操作，
供离线入库流程和在线检索流程共用。

表结构参见 scripts/mysql_init.sql:
- drug_raw_docs: 原始药品说明书全文
- drug_chunks: 切分后的文本块（含 BM25 全文索引）
- drug_metadata: 药品结构化元数据
- index_records: 索引批次记录
"""

from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator, Optional

import pymysql
from loguru import logger

from app.config import config


class MySQLClient:
    """
    MySQL 数据库客户端

    使用方式:
        from app.db.mysql_client import MySQLClient

        client = MySQLClient()
        client.connect()

        doc_id = client.insert_raw_doc(drug_name="阿司匹林", ...)
        results = client.bm25_search("阿司匹林用法用量", top_k=20)

        client.disconnect()
    """

    def __init__(self) -> None:
        self._conn: Optional[pymysql.Connection] = None
        self._host = config.MYSQL_HOST
        self._port = config.MYSQL_PORT
        self._user = config.MYSQL_USER
        self._password = config.MYSQL_PASSWORD
        self._database = config.MYSQL_DATABASE

        # 表名
        self._raw_docs_table = config.mysql_raw_docs_table
        self._chunks_table = config.mysql_chunks_table
        self._metadata_table = config.mysql_metadata_table
        self._index_records_table = config.mysql_index_records_table

    # ============================================================
    # 连接管理
    # ============================================================
    def connect(self) -> "MySQLClient":
        """建立 MySQL 连接"""
        if self._conn is None:
            logger.info(f"正在连接 MySQL: {self._host}:{self._port}/{self._database}")
            self._conn = pymysql.connect(
                host=self._host,
                port=self._port,
                user=self._user,
                password=self._password,
                database=self._database,
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor,
                autocommit=False,
            )
            logger.info("MySQL 连接成功")
        return self

    def disconnect(self) -> None:
        """断开 MySQL 连接"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info("MySQL 连接已断开")

    @property
    def conn(self) -> pymysql.Connection:
        """获取底层连接（未连接时自动连接）"""
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        return self._conn

    def is_connected(self) -> bool:
        """检查连接是否有效"""
        if self._conn is None:
            return False
        try:
            self._conn.ping(reconnect=False)
            return True
        except Exception:
            return False

    def ping(self) -> bool:
        """尝试 ping MySQL，返回 True 表示可达"""
        try:
            self.conn.ping(reconnect=True)
            return True
        except pymysql.MySQLError as e:
            logger.error(f"MySQL ping 失败: {e}")
            return False

    @contextmanager
    def transaction(self) -> Generator[pymysql.Connection, None, None]:
        """事务上下文管理器，自动 commit/rollback"""
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ============================================================
    # drug_raw_docs — 原始文档操作
    # ============================================================
    def insert_raw_doc(
        self,
        drug_name: str,
        raw_content: str,
        drug_manufacturer: Optional[str] = None,
        drug_category: Optional[str] = None,
        source_file: Optional[str] = None,
    ) -> int:
        """
        插入一条原始药品说明书

        Returns:
            新插入记录的 doc_id
        """
        sql = f"""
            INSERT INTO {self._raw_docs_table}
                (drug_name, drug_manufacturer, drug_category, raw_content, source_file)
            VALUES (%s, %s, %s, %s, %s)
        """
        with self.conn.cursor() as cursor:
            cursor.execute(sql, (drug_name, drug_manufacturer, drug_category, raw_content, source_file))
            self.conn.commit()
            doc_id = cursor.lastrowid
            logger.info(f"插入原始文档: drug_name={drug_name}, doc_id={doc_id}")
            return doc_id

    def get_raw_doc(self, doc_id: int) -> Optional[dict]:
        """按 ID 获取原始文档"""
        sql = f"SELECT * FROM {self._raw_docs_table} WHERE id = %s"
        with self.conn.cursor() as cursor:
            cursor.execute(sql, (doc_id,))
            return cursor.fetchone()

    def list_raw_docs(self, drug_name: Optional[str] = None) -> list[dict]:
        """列出原始文档，可按药品名过滤"""
        if drug_name:
            sql = f"SELECT id, drug_name, drug_manufacturer, drug_category, source_file, created_at FROM {self._raw_docs_table} WHERE drug_name = %s ORDER BY id DESC"
            with self.conn.cursor() as cursor:
                cursor.execute(sql, (drug_name,))
                return cursor.fetchall()
        else:
            sql = f"SELECT id, drug_name, drug_manufacturer, drug_category, source_file, created_at FROM {self._raw_docs_table} ORDER BY id DESC"
            with self.conn.cursor() as cursor:
                cursor.execute(sql)
                return cursor.fetchall()

    def get_all_drug_names(self) -> list[str]:
        """获取所有已入库的药品名称（去重）"""
        sql = f"SELECT DISTINCT drug_name FROM {self._raw_docs_table} ORDER BY drug_name"
        with self.conn.cursor() as cursor:
            cursor.execute(sql)
            return [row["drug_name"] for row in cursor.fetchall()]

    def drug_exists(self, drug_name: str) -> bool:
        """
        检查指定药品名称是否已存在于知识库中。

        Args:
            drug_name: 药品名称

        Returns:
            True 表示该药品已有入库记录
        """
        sql = f"SELECT COUNT(*) AS cnt FROM {self._raw_docs_table} WHERE drug_name = %s"
        with self.conn.cursor() as cursor:
            cursor.execute(sql, (drug_name,))
            result = cursor.fetchone()
            return (result["cnt"] if result else 0) > 0

    def delete_drug_by_name(self, drug_name: str) -> list[int]:
        """
        删除指定药品名称的所有相关数据（raw_docs + chunks + metadata）。

        删除顺序:
        1. drug_chunks（CASCADE 外键会自动级联删除，此处显式删除确保安全）
        2. drug_raw_docs
        3. drug_metadata

        Args:
            drug_name: 药品名称

        Returns:
            被删除的 doc_id 列表（用于后续 Milvus 向量清理）
        """
        # 先查出所有关联的 doc_id
        sql_select = f"SELECT id FROM {self._raw_docs_table} WHERE drug_name = %s"
        with self.conn.cursor() as cursor:
            cursor.execute(sql_select, (drug_name,))
            rows = cursor.fetchall()
            doc_ids = [row["id"] for row in rows]

        if not doc_ids:
            logger.debug(f"药品 '{drug_name}' 在数据库中无记录，跳过删除")
            return []

        # 删除 chunks（按 doc_id）
        placeholders = ", ".join(["%s"] * len(doc_ids))
        sql_chunks = f"DELETE FROM {self._chunks_table} WHERE doc_id IN ({placeholders})"
        with self.conn.cursor() as cursor:
            cursor.execute(sql_chunks, doc_ids)

        # 删除 raw_docs
        sql_raw = f"DELETE FROM {self._raw_docs_table} WHERE drug_name = %s"
        with self.conn.cursor() as cursor:
            cursor.execute(sql_raw, (drug_name,))

        # 删除 metadata（drug_metadata 有 UNIQUE 约束，单条删除）
        sql_meta = f"DELETE FROM {self._metadata_table} WHERE drug_name = %s"
        with self.conn.cursor() as cursor:
            cursor.execute(sql_meta, (drug_name,))

        self.conn.commit()
        logger.info(f"药品 '{drug_name}' 已从 MySQL 删除: {len(doc_ids)} 条 raw_doc, chunks + metadata 已清理")
        return doc_ids

    # ============================================================
    # drug_chunks — 文本块操作
    # ============================================================
    def insert_chunk(
        self,
        doc_id: int,
        drug_name: str,
        section: Optional[str],
        chunk_index: int,
        chunk_text: str,
    ) -> int:
        """
        插入一条文本块

        Returns:
            新插入记录的 chunk_id
        """
        char_count = len(chunk_text)
        sql = f"""
            INSERT INTO {self._chunks_table}
                (doc_id, drug_name, section, chunk_index, chunk_text, char_count)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        with self.conn.cursor() as cursor:
            cursor.execute(sql, (doc_id, drug_name, section, chunk_index, chunk_text, char_count))
            self.conn.commit()
            return cursor.lastrowid

    def insert_chunks_batch(self, chunks: list[dict]) -> int:
        """
        批量插入文本块

        Args:
            chunks: [{"doc_id":, "drug_name":, "section":, "chunk_index":, "chunk_text":}, ...]

        Returns:
            插入的行数
        """
        sql = f"""
            INSERT INTO {self._chunks_table}
                (doc_id, drug_name, section, chunk_index, chunk_text, char_count)
            VALUES (%(doc_id)s, %(drug_name)s, %(section)s, %(chunk_index)s, %(chunk_text)s, %(char_count)s)
        """
        # 补充 char_count
        for c in chunks:
            c.setdefault("char_count", len(c.get("chunk_text", "")))

        with self.conn.cursor() as cursor:
            cursor.executemany(sql, chunks)
            self.conn.commit()
            count = cursor.rowcount
            logger.info(f"批量插入 {count} 条文本块")
            return count

    def get_chunks_by_doc_id(self, doc_id: int) -> list[dict]:
        """按文档 ID 获取所有文本块"""
        sql = f"""
            SELECT id, doc_id, drug_name, section, chunk_index, chunk_text, char_count
            FROM {self._chunks_table}
            WHERE doc_id = %s
            ORDER BY chunk_index
        """
        with self.conn.cursor() as cursor:
            cursor.execute(sql, (doc_id,))
            return cursor.fetchall()

    def bm25_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        drug_name: Optional[str] = None,
    ) -> list[dict]:
        """
        MySQL 全文检索（BM25 算法）

        使用 MATCH ... AGAINST 配合 ngram parser
        返回按 BM25 相关性得分排序的结果

        Args:
            query: 搜索查询文本
            top_k: 返回 Top-K，默认 config.retrieval_bm25_top_k
            drug_name: 可选，按药品名称过滤

        Returns:
            [{id, doc_id, drug_name, section, chunk_text, bm25_score, ...}]
        """
        if top_k is None:
            top_k = config.retrieval_bm25_top_k

        # 对查询进行简单处理，适配 ngram 全文搜索
        # BOOLEAN MODE 支持 +must -not 等操作符
        query_escaped = pymysql.converters.escape_string(query)

        if drug_name:
            sql = f"""
                SELECT
                    id, doc_id, drug_name, section, chunk_text, chunk_index,
                    MATCH(chunk_text) AGAINST(%s IN BOOLEAN MODE) AS bm25_score
                FROM {self._chunks_table}
                WHERE MATCH(chunk_text) AGAINST(%s IN BOOLEAN MODE)
                  AND drug_name = %s
                ORDER BY bm25_score DESC
                LIMIT %s
            """
            with self.conn.cursor() as cursor:
                cursor.execute(sql, (query, query, drug_name, top_k))
                return cursor.fetchall()
        else:
            sql = f"""
                SELECT
                    id, doc_id, drug_name, section, chunk_text, chunk_index,
                    MATCH(chunk_text) AGAINST(%s IN BOOLEAN MODE) AS bm25_score
                FROM {self._chunks_table}
                WHERE MATCH(chunk_text) AGAINST(%s IN BOOLEAN MODE)
                ORDER BY bm25_score DESC
                LIMIT %s
            """
            with self.conn.cursor() as cursor:
                cursor.execute(sql, (query, query, top_k))
                return cursor.fetchall()

    # ============================================================
    # drug_metadata — 药品元数据操作
    # ============================================================
    def upsert_drug_metadata(
        self,
        drug_name: str,
        generic_name: Optional[str] = None,
        brand_name: Optional[str] = None,
        manufacturer: Optional[str] = None,
        specification: Optional[str] = None,
        dosage_form: Optional[str] = None,
        category: Optional[str] = None,
        approval_number: Optional[str] = None,
    ) -> None:
        """
        插入或更新药品元数据（基于 drug_name 唯一键）
        """
        sql = f"""
            INSERT INTO {self._metadata_table}
                (drug_name, generic_name, brand_name, manufacturer, specification,
                 dosage_form, category, approval_number)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                generic_name = VALUES(generic_name),
                brand_name = VALUES(brand_name),
                manufacturer = VALUES(manufacturer),
                specification = VALUES(specification),
                dosage_form = VALUES(dosage_form),
                category = VALUES(category),
                approval_number = VALUES(approval_number)
        """
        with self.conn.cursor() as cursor:
            cursor.execute(sql, (
                drug_name, generic_name, brand_name, manufacturer,
                specification, dosage_form, category, approval_number,
            ))
            self.conn.commit()
            logger.info(f"药品元数据已更新: {drug_name}")

    def get_drug_metadata(self, drug_name: str) -> Optional[dict]:
        """按药品名获取元数据"""
        sql = f"SELECT * FROM {self._metadata_table} WHERE drug_name = %s"
        with self.conn.cursor() as cursor:
            cursor.execute(sql, (drug_name,))
            return cursor.fetchone()

    def search_drug_by_name(self, keyword: str) -> list[dict]:
        """按药品名称模糊搜索元数据"""
        sql = f"""
            SELECT * FROM {self._metadata_table}
            WHERE drug_name LIKE %s OR generic_name LIKE %s OR brand_name LIKE %s
            LIMIT 20
        """
        pattern = f"%{keyword}%"
        with self.conn.cursor() as cursor:
            cursor.execute(sql, (pattern, pattern, pattern))
            return cursor.fetchall()

    # ============================================================
    # index_records — 索引批次记录操作
    # ============================================================
    def insert_index_record(
        self,
        batch_id: str,
        doc_id: Optional[int] = None,
        drug_name: Optional[str] = None,
        total_chunks: int = 0,
        indexed_chunks: int = 0,
        failed_chunks: int = 0,
        index_status: str = "pending",
    ) -> int:
        """创建索引批次记录"""
        sql = f"""
            INSERT INTO {self._index_records_table}
                (batch_id, doc_id, drug_name, total_chunks, indexed_chunks,
                 failed_chunks, index_status)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        with self.conn.cursor() as cursor:
            cursor.execute(sql, (
                batch_id, doc_id, drug_name, total_chunks,
                indexed_chunks, failed_chunks, index_status,
            ))
            self.conn.commit()
            return cursor.lastrowid

    def update_index_record(
        self,
        batch_id: str,
        index_status: str,
        total_chunks: Optional[int] = None,
        indexed_chunks: Optional[int] = None,
        failed_chunks: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """更新索引批次状态"""
        fields = ["index_status = %s"]
        params: list[Any] = [index_status]

        if total_chunks is not None:
            fields.append("total_chunks = %s")
            params.append(total_chunks)
        if indexed_chunks is not None:
            fields.append("indexed_chunks = %s")
            params.append(indexed_chunks)
        if failed_chunks is not None:
            fields.append("failed_chunks = %s")
            params.append(failed_chunks)
        if error_message:
            fields.append("error_message = %s")
            params.append(error_message)
        if index_status == "completed":
            fields.append("finished_at = NOW()")

        params.append(batch_id)
        sql = f"""
            UPDATE {self._index_records_table}
            SET {', '.join(fields)}
            WHERE batch_id = %s
        """
        with self.conn.cursor() as cursor:
            cursor.execute(sql, params)
            self.conn.commit()
            logger.info(f"索引批次状态更新: {batch_id} -> {index_status}")

    def get_index_record(self, batch_id: str) -> Optional[dict]:
        """按批次 ID 获取索引记录"""
        sql = f"SELECT * FROM {self._index_records_table} WHERE batch_id = %s"
        with self.conn.cursor() as cursor:
            cursor.execute(sql, (batch_id,))
            return cursor.fetchone()

    # ============================================================
    # 表状态检查
    # ============================================================
    def table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        sql = """
            SELECT COUNT(*) AS cnt FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
        """
        with self.conn.cursor() as cursor:
            cursor.execute(sql, (self._database, table_name))
            result = cursor.fetchone()
            return result["cnt"] > 0 if result else False

    def get_table_stats(self) -> dict:
        """获取所有业务表的行数汇总"""
        tables = [
            self._raw_docs_table,
            self._chunks_table,
            self._metadata_table,
            self._index_records_table,
        ]
        stats = {}
        for table in tables:
            if self.table_exists(table):
                with self.conn.cursor() as cursor:
                    cursor.execute(f"SELECT COUNT(*) AS cnt FROM {table}")
                    result = cursor.fetchone()
                    stats[table] = result["cnt"] if result else 0
            else:
                stats[table] = None  # 表不存在
        return stats

    def is_ready(self) -> bool:
        """检查 MySQL 是否就绪（4 张业务表都存在）"""
        expected_tables = [
            self._raw_docs_table,
            self._chunks_table,
            self._metadata_table,
            self._index_records_table,
        ]
        return all(self.table_exists(t) for t in expected_tables)
