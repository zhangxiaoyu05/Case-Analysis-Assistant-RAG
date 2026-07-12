"""
Milvus 向量数据库连接模块

封装 pymilvus 的连接管理和常用 CRUD 操作，
供离线入库流程和在线检索流程共用。

使用 MilvusClient (pymilvus 3.x 推荐 API)。
"""

from typing import Any, Optional

from loguru import logger
from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient as _MilvusClient,
)
from pymilvus.exceptions import MilvusException

from app.config import config


class MilvusClient:
    """
    Milvus 向量数据库客户端

    使用方式:
        from app.db.milvus_client import MilvusClient

        async with MilvusClient() as client:
            client.insert_embeddings(vectors, metadata_list)
            results = client.search(query_vector, top_k=10)
    """

    def __init__(self) -> None:
        self._client: Optional[_MilvusClient] = None
        self._uri = f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}"
        self._collection_name = config.milvus_collection_name
        self._dimension = config.milvus_dimension
        self._metric_type = config.milvus_metric_type
        self._index_type = config.milvus_index_type
        self._nlist = config.milvus_nlist
        self._nprobe = config.milvus_nprobe

    # ============================================================
    # 连接管理
    # ============================================================
    def connect(self) -> "_MilvusClient":
        """建立 Milvus 连接，返回底层 client"""
        if self._client is None:
            logger.info(f"正在连接 Milvus: {self._uri}")
            self._client = _MilvusClient(uri=self._uri)
            logger.info(f"Milvus 连接成功")
        return self._client

    def disconnect(self) -> None:
        """断开 Milvus 连接"""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Milvus 连接已断开")

    @property
    def client(self) -> _MilvusClient:
        """获取底层 pymilvus.MilvusClient 实例"""
        if self._client is None:
            self.connect()
        assert self._client is not None
        return self._client

    def __enter__(self) -> "MilvusClient":
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.disconnect()

    # ============================================================
    # Collection 管理
    # ============================================================
    def collection_exists(self) -> bool:
        """检查 Collection 是否已存在"""
        return self.client.has_collection(self._collection_name)

    def create_collection(self, drop_if_exists: bool = False) -> None:
        """
        创建 drug_chunks Collection

        Schema 字段:
        - id: 主键 (INT64, auto_id)
        - doc_id: 原始文档 ID (INT64)
        - chunk_index: 块序号 (INT64)
        - drug_name: 药品名称 (VARCHAR 200)
        - section: 章节名 (VARCHAR 50)
        - chunk_text: 文本内容 (VARCHAR 5000)
        - embedding: 向量 (FLOAT_VECTOR, 1536维)
        """
        if self.collection_exists():
            if drop_if_exists:
                logger.warning(f"删除已有 Collection: {self._collection_name}")
                self.drop_collection()
            else:
                logger.info(f"Collection 已存在，跳过创建: {self._collection_name}")
                return

        # 定义 Schema
        schema = CollectionSchema(
            fields=[
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=True,
                ),
                FieldSchema(
                    name="doc_id",
                    dtype=DataType.INT64,
                ),
                FieldSchema(
                    name="chunk_index",
                    dtype=DataType.INT64,
                ),
                FieldSchema(
                    name="drug_name",
                    dtype=DataType.VARCHAR,
                    max_length=200,
                ),
                FieldSchema(
                    name="section",
                    dtype=DataType.VARCHAR,
                    max_length=50,
                ),
                FieldSchema(
                    name="chunk_text",
                    dtype=DataType.VARCHAR,
                    max_length=5000,
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self._dimension,
                ),
            ],
            description="药品说明书切分文本块向量库",
        )

        # 准备索引参数
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type=self._index_type,
            metric_type=self._metric_type,
            params={"nlist": self._nlist},
        )

        # 创建 Collection
        logger.info(
            f"正在创建 Collection: {self._collection_name} "
            f"(维度={self._dimension}, 索引={self._index_type}, 度量={self._metric_type})"
        )
        self.client.create_collection(
            collection_name=self._collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"Collection 创建成功: {self._collection_name}")

        # 加载到内存
        self.load_collection()

    def drop_collection(self) -> None:
        """删除 Collection"""
        if self.collection_exists():
            self.client.drop_collection(self._collection_name)
            logger.info(f"Collection 已删除: {self._collection_name}")

    def load_collection(self) -> None:
        """加载 Collection 到内存（查询前必须执行）"""
        logger.info(f"正在加载 Collection: {self._collection_name}")
        self.client.load_collection(self._collection_name)
        logger.info(f"Collection 加载完成: {self._collection_name}")

    def get_collection_info(self) -> dict:
        """获取 Collection 信息"""
        if not self.collection_exists():
            return {"exists": False}

        stats = self.client.get_collection_stats(self._collection_name)
        description = self.client.describe_collection(self._collection_name)
        return {
            "exists": True,
            "name": self._collection_name,
            "row_count": stats.get("row_count", 0),
            "description": description.get("description", ""),
            "fields": [
                {"name": f["name"], "type": f["type"]}
                for f in description.get("fields", [])
            ],
        }

    # ============================================================
    # 数据操作
    # ============================================================
    def insert_embeddings(
        self,
        vectors: list[list[float]],
        metadata_list: list[dict],
    ) -> dict:
        """
        批量插入向量和元数据

        Args:
            vectors: 向量列表，每个向量是 1536 维 float 列表
            metadata_list: 元数据列表，每个 dict 需包含：
                doc_id, chunk_index, drug_name, section, chunk_text

        Returns:
            pymilvus 插入结果字典 (含 insert_count, ids)
        """
        if len(vectors) != len(metadata_list):
            raise ValueError(
                f"向量数量 ({len(vectors)}) 与元数据数量 ({len(metadata_list)}) 不匹配"
            )

        # 组装数据
        data = []
        for vec, meta in zip(vectors, metadata_list):
            data.append({
                "doc_id": meta["doc_id"],
                "chunk_index": meta["chunk_index"],
                "drug_name": meta.get("drug_name", ""),
                "section": meta.get("section", ""),
                "chunk_text": meta.get("chunk_text", ""),
                "embedding": vec,
            })

        logger.info(f"正在插入 {len(data)} 条向量到 {self._collection_name}")
        result = self.client.insert(
            collection_name=self._collection_name,
            data=data,
        )
        count = result.get("insert_count", 0)
        logger.info(f"向量插入完成: {count} 条")
        return result

    def search(
        self,
        query_vector: list[float],
        top_k: Optional[int] = None,
        filter_expr: Optional[str] = None,
        output_fields: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        向量相似度检索

        Args:
            query_vector: 查询向量 (1536 维 float 列表)
            top_k: 返回 Top-K 结果，默认使用 config.retrieval_vector_top_k
            filter_expr: 标量过滤表达式，如 'drug_name == "阿司匹林"'
            output_fields: 返回的标量字段列表

        Returns:
            检索结果列表 [{id, distance, entity: {...}}, ...]
        """
        if top_k is None:
            top_k = config.retrieval_vector_top_k

        if output_fields is None:
            output_fields = ["doc_id", "drug_name", "section", "chunk_text", "chunk_index"]

        search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": self._nprobe},
        }

        results = self.client.search(
            collection_name=self._collection_name,
            data=[query_vector],
            filter=filter_expr or "",
            limit=top_k,
            output_fields=output_fields,
            search_params=search_params,
        )

        # results[0] 是第一个查询向量的结果列表
        if results and len(results) > 0:
            return results[0]
        return []

    def query(
        self,
        filter_expr: str,
        output_fields: Optional[list[str]] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        按条件查询（非向量检索，纯标量过滤）

        Args:
            filter_expr: 标量过滤表达式，如 'drug_name == "阿司匹林"'
            output_fields: 返回字段
            limit: 最大返回数

        Returns:
            匹配的记录列表
        """
        if output_fields is None:
            output_fields = ["id", "doc_id", "drug_name", "section", "chunk_text"]

        return self.client.query(
            collection_name=self._collection_name,
            filter=filter_expr,
            output_fields=output_fields,
            limit=limit,
        )

    def count(self) -> int:
        """返回 Collection 中的向量总数"""
        info = self.get_collection_info()
        return info.get("row_count", 0)
