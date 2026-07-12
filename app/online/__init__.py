# app/online/__init__.py
"""
在线流程模块

提供 RAG 问答系统的在线检索与生成能力：
1. intent     — 意图识别（判断用户问题是否药品相关）
2. retriever  — 混合检索（向量 + BM25 → RRF 融合）
3. ranker     — 重排序（DashScope qwen3-rerank 二次排序）
4. generator  — 答案生成（基于检索结果 + 场景化提示词模板）

使用方式:
    from app.online import IntentClassifier, Retriever, Ranker, Generator

    # 意图识别
    classifier = IntentClassifier()
    intent = classifier.classify("阿司匹林一天吃几次？")

    # 混合检索
    retriever = Retriever()
    results = retriever.retrieve("阿司匹林一天吃几次？")

    # 重排序
    ranker = Ranker()
    ranked = ranker.rerank("阿司匹林一天吃几次？", [r.__dict__ for r in results])

    # 答案生成
    generator = Generator()
    answer = generator.generate("阿司匹林一天吃几次？", [r.__dict__ for r in ranked])
"""

from app.online.generator import GeneratedAnswer, Generator, generate_answer
from app.online.intent import IntentClassifier, IntentResult, classify_intent
from app.online.ranker import RankedDocument, Ranker, rerank_documents
from app.online.retriever import Retriever, SearchResult

__all__ = [
    # intent
    "IntentClassifier",
    "IntentResult",
    "classify_intent",
    # retriever
    "Retriever",
    "SearchResult",
    # ranker
    "Ranker",
    "RankedDocument",
    "rerank_documents",
    # generator
    "Generator",
    "GeneratedAnswer",
    "generate_answer",
]
