# app/offline/__init__.py
"""
离线流程模块

提供药品说明书的离线数据处理能力：
1. loader              — 文档加载（PDF/DOCX/TXT）
2. cleaner             — 文本清洗（去伪影 / 规范化 / 可选脱敏）
3. splitter            — 章节感知切分（识别【章节名】标记）
4. multi_drug_splitter — 多药品合集文档智能检测与拆分
5. embedder            — 向量化（DashScope text-embedding-v4）
6. pipeline            — 完整流程编排（load → detect/split → clean → split → embed → MySQL + Milvus）

使用方式:
    from app.offline import run_pipeline, load_document, clean_text, split_document, Embedder

    result = run_pipeline("data/raw/阿司匹林说明书.pdf")
"""

from app.offline.cleaner import clean_text
from app.offline.embedder import Embedder, EmbeddingResult, embed_texts
from app.offline.loader import LoadedDocument, LoaderError, load_document, load_documents_from_dir
from app.offline.multi_drug_splitter import (
    SubDocument,
    detect_multi_drug,
    extract_drug_name,
    split_multi_drug,
)
from app.offline.pipeline import PipelineResult, run_pipeline, run_pipeline_batch
from app.offline.splitter import Chunk, split_document

__all__ = [
    # loader
    "load_document",
    "load_documents_from_dir",
    "LoadedDocument",
    "LoaderError",
    # cleaner
    "clean_text",
    # splitter
    "split_document",
    "Chunk",
    # multi_drug_splitter
    "detect_multi_drug",
    "split_multi_drug",
    "extract_drug_name",
    "SubDocument",
    # embedder
    "Embedder",
    "EmbeddingResult",
    "embed_texts",
    # pipeline
    "run_pipeline",
    "run_pipeline_batch",
    "PipelineResult",
]
