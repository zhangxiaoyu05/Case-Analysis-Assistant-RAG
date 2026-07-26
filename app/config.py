"""
RAG 药品问答系统 - 配置加载模块

加载顺序:
1. 读取 .env 环境变量（通过 python-dotenv）
2. 读取 config/config.yaml 业务参数
3. 合并为统一 Config 对象，提供属性式访问
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

# ============================================================
# 项目根路径 & 基础配置加载
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 1. 加载 .env 文件（覆盖系统环境变量）
load_dotenv(PROJECT_ROOT / ".env")

# 2. 加载 config.yaml 业务配置
_yaml_path = PROJECT_ROOT / "config" / "config.yaml"
with open(_yaml_path, "r", encoding="utf-8") as _f:
    _yaml = yaml.safe_load(_f)


# ============================================================
# 辅助函数
# ============================================================
def _env(key: str, default: Any = None, coerce: type = str) -> Any:
    """从环境变量读取值，支持类型转换和默认值"""
    val = os.getenv(key)
    if val is None:
        return default
    if coerce is bool:
        return val.lower() in ("true", "1", "yes")
    return coerce(val)


def _yaml_section(path: str, default: Any = None) -> Any:
    """从 config.yaml 读取嵌套路径值，如 'models.embedding.dimension'"""
    keys = path.split(".")
    node = _yaml
    for k in keys:
        if isinstance(node, dict) and k in node:
            node = node[k]
        else:
            return default
    return node


# ============================================================
# 配置对象
# ============================================================
class Config:
    """
    RAG 药品问答系统统一配置

    同时读取 .env（敏感信息/基础设施连接）和 config.yaml（业务参数）。

    使用方式:
        from app.config import config

        api_key = config.DASHSCOPE_API_KEY
        chunk_size = config.splitter_chunk_size
        top_k = config.retrieval_vector_top_k

    命名规则:
        - .env 变量: 保持大写蛇形 → config.DASHSCOPE_API_KEY
        - config.yaml 变量: 小写点号路径替换为下划线 → config.models_splitter_chunk_size
    """

    # ============================================================
    # API Keys
    # ============================================================
    @property
    def DASHSCOPE_API_KEY(self) -> str:
        return _env("DASHSCOPE_API_KEY", "")

    # ============================================================
    # Milvus 向量数据库（基础设施连接）
    # ============================================================
    @property
    def MILVUS_HOST(self) -> str:
        return _env("MILVUS_HOST", "localhost")

    @property
    def MILVUS_PORT(self) -> int:
        return _env("MILVUS_PORT", 19530, int)

    # ============================================================
    # MySQL 数据库（基础设施连接）
    # ============================================================
    @property
    def MYSQL_HOST(self) -> str:
        return _env("MYSQL_HOST", "localhost")

    @property
    def MYSQL_PORT(self) -> int:
        return _env("MYSQL_PORT", 3306, int)

    @property
    def MYSQL_USER(self) -> str:
        return _env("MYSQL_USER", "root")

    @property
    def MYSQL_PASSWORD(self) -> str:
        return _env("MYSQL_PASSWORD", "")

    @property
    def MYSQL_DATABASE(self) -> str:
        return _env("MYSQL_DATABASE", "rag_pharma")

    # ============================================================
    # Redis 缓存 & 会话（基础设施连接）
    # ============================================================
    @property
    def REDIS_HOST(self) -> str:
        return _env("REDIS_HOST", "localhost")

    @property
    def REDIS_PORT(self) -> int:
        return _env("REDIS_PORT", 6379, int)

    # ============================================================
    # 应用服务（基础设施连接）
    # ============================================================
    @property
    def APP_HOST(self) -> str:
        return _env("APP_HOST", "0.0.0.0")

    @property
    def APP_PORT(self) -> int:
        return _env("APP_PORT", 8000, int)

    @property
    def APP_API_KEY(self) -> str:
        """应用 API Key（用于外部调用鉴权）。为空字符串时鉴权自动禁用。"""
        return _env("APP_API_KEY", "")

    @property
    def LOG_LEVEL(self) -> str:
        return _env("LOG_LEVEL", "INFO")

    # ============================================================
    # 模型配置 —— 文本切分
    # ============================================================
    @property
    def splitter_chunk_size(self) -> int:
        return _yaml_section("models.splitter.chunk_size", 500)

    @property
    def splitter_chunk_overlap(self) -> int:
        return _yaml_section("models.splitter.chunk_overlap", 50)

    @property
    def splitter_min_chunk_size(self) -> int:
        return _yaml_section("models.splitter.min_chunk_size", 100)

    @property
    def splitter_separator(self) -> str:
        return _yaml_section("models.splitter.separator", "\n\n")

    # ============================================================
    # 模型配置 —— 嵌入模型
    # ============================================================
    @property
    def embedding_provider(self) -> str:
        return _yaml_section("models.embedding.provider", "")

    @property
    def embedding_model(self) -> str:
        return _yaml_section("models.embedding.model", "")

    @property
    def embedding_dimension(self) -> int:
        return _yaml_section("models.embedding.dimension", 0)

    @property
    def embedding_batch_size(self) -> int:
        return _yaml_section("models.embedding.batch_size", 25)

    # ============================================================
    # 模型配置 —— 意图识别
    # ============================================================
    @property
    def intent_model(self) -> str:
        return _yaml_section("models.intent.model", "")

    @property
    def intent_temperature(self) -> float:
        return _yaml_section("models.intent.temperature", 0.1)

    @property
    def intent_max_tokens(self) -> int:
        return _yaml_section("models.intent.max_tokens", 200)

    # ============================================================
    # 模型配置 —— 病例结构化提取
    # ============================================================
    @property
    def case_extraction_model(self) -> str:
        return _yaml_section("models.case_extraction.model", "qwen-flash")

    @property
    def case_extraction_temperature(self) -> float:
        return _yaml_section("models.case_extraction.temperature", 0.1)

    @property
    def case_extraction_max_tokens(self) -> int:
        return _yaml_section("models.case_extraction.max_tokens", 800)

    # ============================================================
    # 模型配置 —— 文档分类
    # ============================================================
    @property
    def classifier_model(self) -> str:
        return _yaml_section("models.classifier.model", "qwen-flash")

    @property
    def classifier_temperature(self) -> float:
        return _yaml_section("models.classifier.temperature", 0.1)

    @property
    def classifier_max_tokens(self) -> int:
        return _yaml_section("models.classifier.max_tokens", 300)

    # ============================================================
    # 模型配置 —— 问答生成
    # ============================================================
    @property
    def chat_model(self) -> str:
        return _yaml_section("models.chat.model", "")

    @property
    def chat_temperature(self) -> float:
        return _yaml_section("models.chat.temperature", 0.3)

    @property
    def chat_max_tokens(self) -> int:
        return _yaml_section("models.chat.max_tokens", 2000)

    @property
    def chat_top_p(self) -> float:
        return _yaml_section("models.chat.top_p", 0.95)

    @property
    def chat_top_k(self) -> int:
        return _yaml_section("models.chat.top_k", 50)

    # ============================================================
    # 模型配置 —— 重排序
    # ============================================================
    @property
    def rerank_model(self) -> str:
        return _yaml_section("models.rerank.model", "")

    # ============================================================
    # v1.0.0: 多源检索配置
    # ============================================================
    @property
    def multi_source_enabled(self) -> bool:
        return _yaml_section("database.multi_source.enabled", True)

    @property
    def multi_source_default_sources(self) -> list:
        return _yaml_section("database.multi_source.default_sources",
                           ["drug", "disease", "guideline", "literature"])

    @property
    def multi_source_top_n_per_source(self) -> int:
        return _yaml_section("database.multi_source.top_n_per_source", 5)

    @property
    def multi_source_final_top_n(self) -> int:
        return _yaml_section("database.multi_source.final_top_n", 15)

    @property
    def multi_source_per_source_min(self) -> int:
        return _yaml_section("database.multi_source.per_source_min", 2)

    # ============================================================
    # 检索配置
    # ============================================================
    @property
    def retrieval_vector_top_k(self) -> int:
        return _yaml_section("retrieval.vector_top_k", 20)

    @property
    def retrieval_bm25_top_k(self) -> int:
        return _yaml_section("retrieval.bm25_top_k", 20)

    @property
    def retrieval_rrf_k(self) -> int:
        return _yaml_section("retrieval.rrf_k", 60)

    @property
    def retrieval_rrf_top_n(self) -> int:
        return _yaml_section("retrieval.rrf_top_n", 5)

    @property
    def retrieval_enable_keyword_filter(self) -> bool:
        return _yaml_section("retrieval.enable_keyword_filter", True)

    # ============================================================
    # Milvus 业务配置
    # ============================================================
    @property
    def milvus_collection_name(self) -> str:
        return _yaml_section("database.milvus.collection_name", "drug_chunks")

    @property
    def milvus_dimension(self) -> int:
        return _yaml_section("database.milvus.dimension", 1536)

    @property
    def milvus_index_type(self) -> str:
        return _yaml_section("database.milvus.index_type", "IVF_FLAT")

    @property
    def milvus_metric_type(self) -> str:
        return _yaml_section("database.milvus.metric_type", "IP")

    @property
    def milvus_nlist(self) -> int:
        return _yaml_section("database.milvus.nlist", 1024)

    @property
    def milvus_nprobe(self) -> int:
        return _yaml_section("database.milvus.nprobe", 16)

    # ============================================================
    # MySQL 业务配置
    # ============================================================
    @property
    def mysql_raw_docs_table(self) -> str:
        return _yaml_section("database.mysql.raw_docs_table", "drug_raw_docs")

    @property
    def mysql_chunks_table(self) -> str:
        return _yaml_section("database.mysql.chunks_table", "drug_chunks")

    @property
    def mysql_metadata_table(self) -> str:
        return _yaml_section("database.mysql.metadata_table", "drug_metadata")

    @property
    def mysql_index_records_table(self) -> str:
        return _yaml_section("database.mysql.index_records_table", "index_records")

    @property
    def mysql_pool_size(self) -> int:
        return _yaml_section("database.mysql.pool_size", 5)

    @property
    def mysql_pool_recycle(self) -> int:
        return _yaml_section("database.mysql.pool_recycle", 3600)

    # ============================================================
    # 短期记忆配置
    # ============================================================
    @property
    def memory_enabled(self) -> bool:
        return _yaml_section("memory.enabled", True)

    @property
    def memory_summarize_model(self) -> str:
        return _yaml_section("memory.summarize_model", "qwen-flash")

    @property
    def memory_recent_turns(self) -> int:
        return _yaml_section("memory.recent_turns", 4)

    @property
    def memory_summary_max_tokens(self) -> int:
        return _yaml_section("memory.summary_max_tokens", 600)

    @property
    def memory_max_summary_chars(self) -> int:
        return _yaml_section("memory.max_summary_chars", 800)

    @property
    def memory_token_threshold_ratio(self) -> float:
        """Token 阈值触发比例：超过上下文窗口的此比例时触发摘要压缩。"""
        return _yaml_section("memory.token_threshold_ratio", 0.7)

    @property
    def memory_context_window_tokens(self) -> int:
        """模型上下文窗口大小（token 数），用于计算摘要触发阈值。"""
        return _yaml_section("memory.context_window_tokens", 8192)

    # ============================================================
    # 中期记忆配置
    # ============================================================
    @property
    def user_memory_extract_model(self) -> str:
        """记忆提取模型（轻量级即可）。"""
        return _yaml_section("user_memory.extract_model", "qwen-flash")

    @property
    def user_memory_max_per_user(self) -> int:
        """每个用户最多保留的记忆条数。"""
        return _yaml_section("user_memory.max_memories_per_user", 50)

    @property
    def user_memory_decay_factor(self) -> float:
        """每日衰减系数（每天 × decay_factor）。"""
        return _yaml_section("user_memory.decay_factor", 0.95)

    @property
    def user_memory_min_importance(self) -> float:
        """最低重要性阈值（低于此值自动清理）。"""
        return _yaml_section("user_memory.min_importance", 0.1)

    @property
    def user_memory_recall_top_k(self) -> int:
        """每次请求召回的记忆条数。"""
        return _yaml_section("user_memory.recall_top_k", 5)

    @property
    def user_memory_merge_threshold(self) -> float:
        """关键词重叠度阈值（超过则合并而非新增）。"""
        return _yaml_section("user_memory.merge_threshold", 0.6)

    @property
    def user_memory_max_tokens_in_prompt(self) -> int:
        """中期记忆注入 prompt 的最大 token 数（超出截断）。"""
        return _yaml_section("user_memory.max_tokens_in_prompt", 600)

    # ============================================================
    # 长期记忆配置（用户画像）
    # ============================================================
    @property
    def user_profile_extract_model(self) -> str:
        """画像提取模型（轻量级即可）。"""
        return _yaml_section("user_profile.extract_model", "qwen-flash")

    @property
    def user_profile_min_confidence(self) -> float:
        """最低置信度阈值（低于此值不保存）。"""
        return _yaml_section("user_profile.min_confidence", 0.5)

    @property
    def user_profile_max_tokens_in_prompt(self) -> int:
        """用户画像注入 prompt 的最大 token 数（超出截断）。"""
        return _yaml_section("user_profile.max_tokens_in_prompt", 300)

    # ============================================================
    # Redis 业务配置
    # ============================================================
    @property
    def redis_session_ttl(self) -> int:
        return _yaml_section("redis.session_ttl", 3600)

    @property
    def redis_max_history(self) -> int:
        return _yaml_section("redis.max_history", 10)

    @property
    def redis_cache_ttl(self) -> int:
        return _yaml_section("redis.cache_ttl", 300)

    # ============================================================
    # 安全配置
    # ============================================================
    @property
    def auth_enabled(self) -> bool:
        """API Key 鉴权开关。需要同时满足 config.yaml 启用 + APP_API_KEY 已配置。"""
        yaml_enabled = _yaml_section("security.auth.enabled", False)
        return bool(yaml_enabled and self.APP_API_KEY)

    @property
    def rate_limit_enabled(self) -> bool:
        return _yaml_section("security.rate_limit.enabled", True)

    @property
    def rate_limit_requests_per_minute(self) -> int:
        return _yaml_section("security.rate_limit.requests_per_minute", 60)

    # ============================================================
    # 应用配置
    # ============================================================
    @property
    def app_debug(self) -> bool:
        return _yaml_section("app.debug", False)

    @property
    def app_cors_origins(self) -> list:
        return _yaml_section("app.cors_origins", ["*"])

    @property
    def app_api_prefix(self) -> str:
        return _yaml_section("app.api_prefix", "/api/v1")

    @property
    def app_max_conversation_turns(self) -> int:
        return _yaml_section("app.max_conversation_turns", 10)

    # ============================================================
    # 日志配置
    # ============================================================
    @property
    def logging_level(self) -> str:
        return _yaml_section("logging.level", "INFO")

    @property
    def logging_file(self) -> str:
        return _yaml_section("logging.file", "logs/app.log")

    @property
    def logging_max_bytes(self) -> int:
        return _yaml_section("logging.max_bytes", 10485760)

    @property
    def logging_backup_count(self) -> int:
        return _yaml_section("logging.backup_count", 5)

    # ============================================================
    # 便捷方法
    # ============================================================
    def get_milvus_connection(self) -> dict:
        """返回 Milvus 连接参数字典"""
        return {
            "host": self.MILVUS_HOST,
            "port": self.MILVUS_PORT,
        }

    def get_mysql_connection(self) -> dict:
        """返回 MySQL 连接参数字典（pymysql 格式）"""
        return {
            "host": self.MYSQL_HOST,
            "port": self.MYSQL_PORT,
            "user": self.MYSQL_USER,
            "password": self.MYSQL_PASSWORD,
            "database": self.MYSQL_DATABASE,
        }

    def get_redis_connection(self) -> dict:
        """返回 Redis 连接参数字典"""
        return {
            "host": self.REDIS_HOST,
            "port": self.REDIS_PORT,
            "decode_responses": True,
        }

    def __repr__(self) -> str:
        return (
            f"Config(\n"
            f"  Milvus: {self.MILVUS_HOST}:{self.MILVUS_PORT}\n"
            f"  MySQL:  {self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}\n"
            f"  Redis:  {self.REDIS_HOST}:{self.REDIS_PORT}\n"
            f"  Embedding: {self.embedding_model} ({self.embedding_dimension}d)\n"
            f"  Chat:   {self.chat_model}\n"
            f"  Rerank: {self.rerank_model}\n"
            f")"
        )


# ============================================================
# 全局单例 —— 应用中唯一导入入口
# ============================================================
config = Config()
