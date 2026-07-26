"""
MySQL 数据库连接模块

封装 pymysql 连接池和业务表 CRUD 操作，
供离线入库流程和在线检索流程共用。

v1.0.0: 新增 6 张表通用操作方法 + 多源 BM25 检索。

表结构参见 scripts/mysql_init.sql:
- drug_raw_docs / drug_chunks: 药品说明书
- disease_raw_docs / disease_chunks: 疾病知识
- guideline_raw_docs / guideline_chunks: 临床指南
- literature_raw_docs / literature_chunks: 学术文献
- drug_metadata: 药品结构化元数据
- index_records: 索引批次记录
"""

# v1.0.0: 表名到 source_type 的映射
_TABLE_SOURCE_TYPE_MAP = {
    "drug_raw_docs": "drug",
    "drug_chunks": "drug",
    "disease_raw_docs": "disease",
    "disease_chunks": "disease",
    "guideline_raw_docs": "guideline",
    "guideline_chunks": "guideline",
    "literature_raw_docs": "literature",
    "literature_chunks": "literature",
}

# MySQL BOOLEAN MODE 运算符，需转义避免被误解析
_BOOLEAN_MODE_OPERATORS = str.maketrans({
    '+': r'\+',
    '-': r'\-',
    '>': r'\>',
    '<': r'\<',
    '(': r'\(',
    ')': r'\)',
    '~': r'\~',
    '*': r'\*',
    '"': r'\"',
    '@': r'\@',
})


def _escape_boolean_mode(query: str) -> str:
    """转义 MySQL BOOLEAN MODE 中的特殊运算符字符。"""
    return query.translate(_BOOLEAN_MODE_OPERATORS)

# v1.0.0: 每个 source_type 的 chunks 表 + raw_docs 表
_SOURCE_CHUNKS_TABLE = {
    "drug": "drug_chunks",
    "disease": "disease_chunks",
    "guideline": "guideline_chunks",
    "literature": "literature_chunks",
}

_SOURCE_RAW_TABLE = {
    "drug": "drug_raw_docs",
    "disease": "disease_raw_docs",
    "guideline": "guideline_raw_docs",
    "literature": "literature_raw_docs",
}

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
        返回按 BM25 相关性得分排序的结果。
        优先 BOOLEAN MODE（支持精确匹配），返回 0 条时回退 NATURAL LANGUAGE MODE。

        Args:
            query: 搜索查询文本
            top_k: 返回 Top-K，默认 config.retrieval_bm25_top_k
            drug_name: 可选，按药品名称过滤

        Returns:
            [{id, doc_id, drug_name, section, chunk_text, bm25_score, ...}]
        """
        if top_k is None:
            top_k = config.retrieval_bm25_top_k

        # 转义 BOOLEAN MODE 特殊字符（+ - > < ( ) ~ * " @）
        escaped_query = _escape_boolean_mode(query)

        # 优先 BOOLEAN MODE
        results = self._bm25_search_internal(
            table=self._chunks_table,
            query=escaped_query,
            top_k=top_k,
            mode="BOOLEAN",
            filter_field="drug_name" if drug_name else None,
            filter_value=drug_name,
        )

        # BOOLEAN MODE 返回 0 条 → 回退 NATURAL LANGUAGE MODE
        if not results:
            logger.info(
                f"BM25 BOOLEAN MODE 无结果 (query={query[:60]}...)，"
                f"回退 NATURAL LANGUAGE MODE"
            )
            results = self._bm25_search_internal(
                table=self._chunks_table,
                query=query,  # NATURAL LANGUAGE MODE 不需要转义
                top_k=top_k,
                mode="NATURAL",
                filter_field="drug_name" if drug_name else None,
                filter_value=drug_name,
            )

        return results

    # ============================================================
    # v1.0.0: 通用多源操作方法
    # ============================================================
    def _bm25_search_internal(
        self,
        table: str,
        query: str,
        top_k: int,
        mode: str = "BOOLEAN",
        filter_field: Optional[str] = None,
        filter_value: Optional[str] = None,
    ) -> list[dict]:
        """
        BM25 检索内部实现，支持 BOOLEAN / NATURAL LANGUAGE 两种模式。

        Args:
            table: 表名
            query: 搜索查询文本
            top_k: 返回 Top-K
            mode: "BOOLEAN" 或 "NATURAL"
            filter_field: 可选过滤字段
            filter_value: 可选过滤值

        Returns:
            检索结果列表
        """
        mode_clause = f"IN {mode} LANGUAGE MODE" if mode == "NATURAL" else "IN BOOLEAN MODE"

        if filter_field and filter_value:
            sql = f"""
                SELECT *, MATCH(chunk_text) AGAINST(%s {mode_clause}) AS bm25_score
                FROM {table}
                WHERE MATCH(chunk_text) AGAINST(%s {mode_clause})
                  AND {filter_field} = %s
                ORDER BY bm25_score DESC
                LIMIT %s
            """
            with self.conn.cursor() as cursor:
                cursor.execute(sql, (query, query, filter_value, top_k))
                return cursor.fetchall()
        else:
            sql = f"""
                SELECT *, MATCH(chunk_text) AGAINST(%s {mode_clause}) AS bm25_score
                FROM {table}
                WHERE MATCH(chunk_text) AGAINST(%s {mode_clause})
                ORDER BY bm25_score DESC
                LIMIT %s
            """
            with self.conn.cursor() as cursor:
                cursor.execute(sql, (query, query, top_k))
                return cursor.fetchall()

    def insert_raw_doc_generic(
        self,
        source_type: str,
        fields: dict,
    ) -> int:
        """
        通用原始文档插入，根据 source_type 路由到对应表。

        Args:
            source_type: "drug" / "disease" / "guideline" / "literature"
            fields: 字段名→值的映射（不含 id 和 created_at）

        Returns:
            新插入记录的 doc_id
        """
        table = _SOURCE_RAW_TABLE.get(source_type)
        if not table:
            raise ValueError(f"未知的 source_type: {source_type}")

        columns = ", ".join(fields.keys())
        placeholders = ", ".join(["%s"] * len(fields))
        values = list(fields.values())

        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        with self.conn.cursor() as cursor:
            cursor.execute(sql, values)
            self.conn.commit()
            doc_id = cursor.lastrowid
            logger.info(f"通用插入: {table}, doc_id={doc_id}, source_type={source_type}")
            return doc_id

    def insert_chunks_batch_generic(
        self,
        source_type: str,
        chunk_records: list[dict],
    ) -> int:
        """
        通用文本块批量插入，根据 source_type 路由到对应 chunks 表。

        Args:
            source_type: "drug" / "disease" / "guideline" / "literature"
            chunk_records: dict 列表，每个 dict 包含该 chunks 表需要的字段

        Returns:
            插入的行数
        """
        table = _SOURCE_CHUNKS_TABLE.get(source_type)
        if not table:
            raise ValueError(f"未知的 source_type: {source_type}")

        if not chunk_records:
            return 0

        # 补充 char_count
        for c in chunk_records:
            c.setdefault("char_count", len(c.get("chunk_text", "")))

        columns = list(chunk_records[0].keys())
        col_str = ", ".join(columns)
        placeholder_str = ", ".join([f"%({col})s" for col in columns])

        sql = f"INSERT INTO {table} ({col_str}) VALUES ({placeholder_str})"
        with self.conn.cursor() as cursor:
            cursor.executemany(sql, chunk_records)
            self.conn.commit()
            count = cursor.rowcount
            logger.info(f"通用批量插入: {count} 条 → {table}")
            return count

    def bm25_search_generic(
        self,
        source_type: str,
        query: str,
        top_k: int = 20,
        filter_field: Optional[str] = None,
        filter_value: Optional[str] = None,
    ) -> list[dict]:
        """
        通用 BM25 检索，根据 source_type 路由到对应 chunks 表。

        优先 BOOLEAN MODE，返回 0 条时回退 NATURAL LANGUAGE MODE。

        Args:
            source_type: "drug" / "disease" / "guideline" / "literature"
            query: 搜索查询文本
            top_k: 返回 Top-K
            filter_field: 可选，按字段过滤的字段名
            filter_value: 可选，过滤字段的值

        Returns:
            [{id, doc_id, ..., chunk_text, bm25_score}]
        """
        table = _SOURCE_CHUNKS_TABLE.get(source_type)
        if not table:
            raise ValueError(f"未知的 source_type: {source_type}")

        # 转义 BOOLEAN MODE 特殊字符
        escaped_query = _escape_boolean_mode(query)

        # 优先 BOOLEAN MODE
        results = self._bm25_search_internal(
            table=table,
            query=escaped_query,
            top_k=top_k,
            mode="BOOLEAN",
            filter_field=filter_field,
            filter_value=filter_value,
        )

        # BOOLEAN MODE 返回 0 条 → 回退 NATURAL LANGUAGE MODE
        if not results:
            logger.info(
                f"BM25[{source_type}] BOOLEAN MODE 无结果 "
                f"(query={query[:60]}...)，回退 NATURAL LANGUAGE MODE"
            )
            results = self._bm25_search_internal(
                table=table,
                query=query,  # NATURAL LANGUAGE MODE 不需要转义
                top_k=top_k,
                mode="NATURAL",
                filter_field=filter_field,
                filter_value=filter_value,
            )

        return results

    def delete_by_id_generic(
        self,
        source_type: str,
        doc_id: int,
    ) -> bool:
        """
        通用按 ID 级联删除（raw_docs + chunks）。

        Args:
            source_type: "drug" / "disease" / "guideline" / "literature"
            doc_id: 原始文档 ID

        Returns:
            是否成功删除
        """
        raw_table = _SOURCE_RAW_TABLE.get(source_type)
        chunks_table = _SOURCE_CHUNKS_TABLE.get(source_type)
        if not raw_table or not chunks_table:
            raise ValueError(f"未知的 source_type: {source_type}")

        # chunks 有 CASCADE 外键，但显式删除更安全
        sql_chunks = f"DELETE FROM {chunks_table} WHERE doc_id = %s"
        with self.conn.cursor() as cursor:
            cursor.execute(sql_chunks, (doc_id,))

        sql_raw = f"DELETE FROM {raw_table} WHERE id = %s"
        with self.conn.cursor() as cursor:
            cursor.execute(sql_raw, (doc_id,))

        self.conn.commit()
        logger.info(f"通用删除: {source_type} doc_id={doc_id}")
        return True

    def list_source_docs(
        self,
        source_type: str,
        limit: int = 100,
    ) -> list[dict]:
        """
        列出指定 source_type 已入库的文档摘要。

        Args:
            source_type: "drug" / "disease" / "guideline" / "literature"
            limit: 最大返回数

        Returns:
            文档摘要列表
        """
        raw_table = _SOURCE_RAW_TABLE.get(source_type)
        chunks_table = _SOURCE_CHUNKS_TABLE.get(source_type)
        if not raw_table:
            return []

        # 根据 source_type 选择显示标题的列
        title_col = {
            "drug": "drug_name",
            "disease": "disease_name",
            "guideline": "guideline_title",
            "literature": "title",
        }.get(source_type, "id")

        sql = f"""
            SELECT r.id, r.{title_col} AS title,
                   (SELECT COUNT(*) FROM {chunks_table} WHERE doc_id = r.id) AS chunk_count,
                   r.created_at
            FROM {raw_table} r
            ORDER BY r.id DESC
            LIMIT %s
        """
        with self.conn.cursor() as cursor:
            cursor.execute(sql, (limit,))
            return cursor.fetchall()

    def get_source_counts(self) -> dict:
        """获取各 source_type 的入库数量统计。"""
        counts = {}
        for source_type in ["drug", "disease", "guideline", "literature"]:
            raw_table = _SOURCE_RAW_TABLE.get(source_type)
            if raw_table and self.table_exists(raw_table):
                with self.conn.cursor() as cursor:
                    cursor.execute(f"SELECT COUNT(*) AS cnt FROM {raw_table}")
                    result = cursor.fetchone()
                    counts[source_type] = result["cnt"] if result else 0
            else:
                counts[source_type] = 0
        return counts

    @classmethod
    def get_source_chunks_table(cls, source_type: str) -> str:
        """根据 source_type 返回对应的 chunks 表名（类方法，无需实例化）。"""
        return _SOURCE_CHUNKS_TABLE.get(source_type, "drug_chunks")

    @classmethod
    def get_source_raw_table(cls, source_type: str) -> str:
        """根据 source_type 返回对应的 raw_docs 表名（类方法，无需实例化）。"""
        return _SOURCE_RAW_TABLE.get(source_type, "drug_raw_docs")

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
        """获取所有业务表的行数汇总（含 v1.0.0 新增表）"""
        tables = [
            self._raw_docs_table,
            self._chunks_table,
            self._metadata_table,
            self._index_records_table,
            # v1.0.0 新增
            "disease_raw_docs",
            "disease_chunks",
            "guideline_raw_docs",
            "guideline_chunks",
            "literature_raw_docs",
            "literature_chunks",
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
        """检查 MySQL 核心表是否就绪（至少 drug 4 张表存在）。

        v1.0.0: 不要求新增的 disease/guideline/literature 表必须存在，
        只检查核心的 drug 4 张表。其他表按需检查。
        """
        expected_tables = [
            self._raw_docs_table,
            self._chunks_table,
            self._metadata_table,
            self._index_records_table,
        ]
        return all(self.table_exists(t) for t in expected_tables)

    def is_v1_ready(self) -> bool:
        """检查所有 v1.0.0 表是否就绪（含 6 张新表）。"""
        all_tables = [
            self._raw_docs_table,
            self._chunks_table,
            self._metadata_table,
            self._index_records_table,
            "disease_raw_docs",
            "disease_chunks",
            "guideline_raw_docs",
            "guideline_chunks",
            "literature_raw_docs",
            "literature_chunks",
        ]
        return all(self.table_exists(t) for t in all_tables)
