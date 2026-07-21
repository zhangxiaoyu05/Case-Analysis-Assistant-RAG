"""
问答路由

POST   /api/v1/chat                    - 单轮问答（LangGraph 编排）
POST   /api/v1/chat/stream              - 流式问答（SSE）
GET    /api/v1/chat/history/{session_id} - 获取对话历史
DELETE /api/v1/chat/history/{session_id} - 清除会话
"""

import asyncio
import json
import time
import uuid
from dataclasses import asdict

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from app.api.dependencies import get_graph, get_history_manager
from app.graph.state import GraphResult, RagState
from app.online.generator import Generator
from app.online.intent import IntentClassifier
from app.online.ranker import Ranker
from app.online.retriever import Retriever
from app.schemas.chat import (
    ChatHistoryItem,
    ChatRequest,
    ChatResponse,
    ClearHistoryResponse,
    HistoryResponse,
    SourceDoc,
)
from app.schemas.common import ErrorResponse
from app.services.memory_manager import MemoryManager

router = APIRouter()


# ============================================================
# 辅助函数
# ============================================================
def _search_dicts_to_source_docs(sources: list[dict]) -> list[SourceDoc]:
    """将检索结果的 dict 列表转为 Pydantic SourceDoc 列表。"""
    return [
        SourceDoc(
            drug_name=s.get("drug_name", ""),
            section=s.get("section"),
            chunk_text=s.get("chunk_text", ""),
            score=s.get("score"),
            doc_id=s.get("doc_id"),
        )
        for s in sources
    ]


def _build_history_for_llm(history: list[dict]) -> list[dict]:
    """从 Redis 历史中提取 role+content 供 LLM 使用。"""
    return [{"role": h["role"], "content": h["content"]} for h in history]


# ============================================================
# POST /api/v1/chat — 单轮问答
# ============================================================
@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "请求参数错误"},
        503: {"model": ErrorResponse, "description": "依赖服务不可用"},
    },
    summary="单轮问答",
    description="提交药品相关问题，返回 RAG 生成的回答及参考来源。支持通过 session_id 维持多轮对话。",
)
async def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or uuid.uuid4().hex[:16]
    t_start = time.perf_counter()

    # 1. 加载对话历史与记忆摘要
    history_manager = get_history_manager()
    history = await history_manager.get_history(session_id)
    history_for_llm = _build_history_for_llm(history)

    # 2. 短期记忆压缩（摘要旧轮次，保留最近轮次完整）
    memory_manager = MemoryManager()
    existing_summary = await history_manager.get_summary(session_id)
    memory_summary, recent_history = await memory_manager.summarize(
        history_for_llm, existing_summary
    )
    # 更新摘要到 Redis（异步保存，不影响本次响应）
    if memory_summary != existing_summary:
        await history_manager.set_summary(session_id, memory_summary)

    logger.info(
        f"[{session_id}] 单轮问答: {request.message[:60]}... "
        f"(历史={len(history_for_llm)} 条, 摘要={len(memory_summary)} 字, "
        f"最近={len(recent_history)} 条)"
    )

    # 3. 执行 LangGraph RAG 流程（同步图在 asyncio.to_thread 中运行）
    graph = get_graph()
    initial_state: RagState = {
        "query": request.message,
        "history": recent_history,
        "memory_summary": memory_summary,
    }

    try:
        result_state = await asyncio.to_thread(graph.invoke, initial_state)
    except Exception as e:
        logger.error(f"[{session_id}] 图执行异常: {e}")
        # 图级别的兜底
        result_state = RagState(
            query=request.message,
            answer=f"系统处理异常，请稍后重试。错误: {e}",
            sources=[],
            intent="drug_inquiry",
            intent_confidence=0.0,
            template_used="error",
            error=str(e),
            error_node="graph",
        )

    # 4. 构建响应
    elapsed_ms = (time.perf_counter() - t_start) * 1000
    sources = _search_dicts_to_source_docs(result_state.get("sources", []))

    # 5. 保存本轮对话到 Redis
    try:
        await history_manager.add_turn(
            session_id=session_id,
            user_msg=request.message,
            assistant_msg=result_state.get("answer", ""),
            sources=result_state.get("sources"),
        )
    except Exception as e:
        logger.warning(f"[{session_id}] 保存历史失败: {e}（不影响响应）")

    logger.info(
        f"[{session_id}] 问答完成: intent={result_state.get('intent')}, "
        f"sources={len(sources)}, elapsed={elapsed_ms:.0f}ms"
    )

    return ChatResponse(
        answer=result_state.get("answer", ""),
        sources=sources,
        session_id=session_id,
        intent=result_state.get("intent"),
        elapsed_ms=round(elapsed_ms, 2),
    )


# ============================================================
# POST /api/v1/chat/stream — 流式问答（SSE）
# ============================================================
@router.post(
    "/chat/stream",
    response_model=None,
    responses={
        400: {"model": ErrorResponse, "description": "请求参数错误"},
    },
    summary="流式问答（SSE）",
    description="与 /chat 相同，但通过 Server-Sent Events 流式返回生成结果。",
)
async def chat_stream(request: ChatRequest):
    session_id = request.session_id or uuid.uuid4().hex[:16]
    history_manager = get_history_manager()

    # 预加载历史与记忆摘要
    history = await history_manager.get_history(session_id)
    history_for_llm = _build_history_for_llm(history)

    # 短期记忆压缩
    memory_manager = MemoryManager()
    existing_summary = await history_manager.get_summary(session_id)
    memory_summary, recent_history = await memory_manager.summarize(
        history_for_llm, existing_summary
    )
    if memory_summary != existing_summary:
        await history_manager.set_summary(session_id, memory_summary)

    logger.info(
        f"[{session_id}] 流式问答: {request.message[:60]}... "
        f"(摘要={len(memory_summary)} 字, 最近={len(recent_history)} 条)"
    )

    async def event_generator():
        """SSE 事件生成器。每个事件格式: event: <type>\ndata: <json>\n\n"""
        full_answer = ""
        ranked_dicts: list[dict] = []

        try:
            # ---- 阶段 1: 意图识别 ----
            classifier = IntentClassifier()
            intent_result = classifier.classify(request.message)

            if intent_result.intent == "chitchat":
                # 闲聊：返回简单问候，不走检索
                from app.graph.nodes import chitchat_node
                chitchat_state = chitchat_node({"query": request.message})
                chitchat_msg = chitchat_state.get("answer", "你好！有什么可以帮您的吗？")
                yield f"event: token\ndata: {json.dumps({'event': 'token', 'data': chitchat_msg}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'event': 'done', 'data': ''})}\n\n"
                full_answer = chitchat_msg
                return

            if intent_result.intent == "attack":
                # 攻击：返回统一安全拒绝消息，不透露检测细节
                attack_msg = "抱歉，您的请求包含不安全的输入，无法处理。如果您有药品相关的正常问题，请重新表述后提问。"
                yield f"event: token\ndata: {json.dumps({'event': 'token', 'data': attack_msg}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'event': 'done', 'data': ''})}\n\n"
                full_answer = attack_msg
                return

            if intent_result.intent == "general":
                # 通用问题：LLM 直接回答（不走 RAG 检索），末尾附加非专长声明
                generator = Generator()
                full_answer = ""
                for token in generator.generate_stream(
                    query=request.message,
                    context_docs=[],  # 不传检索结果
                    history=recent_history,
                    template="general",
                    memory_summary=memory_summary,
                ):
                    full_answer += token
                    yield f"data: {json.dumps({'event': 'token', 'data': token}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'event': 'done', 'data': ''})}\n\n"
                return

            # ---- 阶段 2: 混合检索 ----
            retriever = Retriever()
            search_results = retriever.retrieve(request.message)
            search_dicts = [asdict(r) for r in search_results]
            logger.info(f"[{session_id}] 流式-检索: {len(search_dicts)} 条")

            # ---- 阶段 3: 重排序 ----
            if search_dicts:
                ranker = Ranker()
                try:
                    ranked = ranker.rerank(request.message, search_dicts)
                    ranked_dicts = [asdict(r) for r in ranked]
                except Exception:
                    ranked_dicts = search_dicts  # 回退
            else:
                ranked_dicts = []

            # 发送 sources 事件
            sources_payload = [
                {
                    "drug_name": r.get("drug_name", ""),
                    "section": r.get("section", ""),
                    "chunk_text": r.get("chunk_text", ""),
                    "score": r.get("score", 0.0),
                    "doc_id": r.get("doc_id", 0),
                }
                for r in ranked_dicts
            ]
            yield f"event: sources\ndata: {json.dumps(sources_payload, ensure_ascii=False)}\n\n"

            # ---- 阶段 4: 流式生成 ----
            if ranked_dicts:
                generator = Generator()
                for token in generator.generate_stream(
                    query=request.message,
                    context_docs=ranked_dicts,
                    history=recent_history,
                    memory_summary=memory_summary,
                ):
                    full_answer += token
                    yield f"data: {json.dumps({'event': 'token', 'data': token}, ensure_ascii=False)}\n\n"
            else:
                no_result_msg = "抱歉，未能在知识库中检索到与您问题相关的药品信息。"
                full_answer = no_result_msg
                yield f"data: {json.dumps({'event': 'token', 'data': no_result_msg}, ensure_ascii=False)}\n\n"

            # ---- 阶段 5: 完成 ----
            yield f"data: {json.dumps({'event': 'done', 'data': ''})}\n\n"

        except Exception as e:
            logger.error(f"[{session_id}] 流式处理异常: {e}")
            yield f"data: {json.dumps({'event': 'error', 'data': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            # 保存到 Redis（无论成功与否）
            if full_answer:
                try:
                    await history_manager.add_turn(
                        session_id=session_id,
                        user_msg=request.message,
                        assistant_msg=full_answer,
                        sources=ranked_dicts,
                    )
                except Exception as e:
                    logger.warning(f"[{session_id}] 保存流式历史失败: {e}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        },
    )


# ============================================================
# GET /api/v1/chat/history/{session_id} — 获取对话历史
# ============================================================
@router.get(
    "/chat/history/{session_id}",
    response_model=HistoryResponse,
    responses={
        404: {"model": ErrorResponse, "description": "会话不存在"},
    },
    summary="获取对话历史",
    description="根据 session_id 从 Redis 中读取多轮对话历史记录。",
)
async def get_history(session_id: str) -> HistoryResponse:
    history_manager = get_history_manager()
    raw_history = await history_manager.get_history(session_id)

    items: list[ChatHistoryItem] = []
    for h in raw_history:
        # 解析 timestamp
        from datetime import datetime

        ts_str = h.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            ts = datetime.utcnow()

        # 解析 sources（仅 assistant 消息有）
        sources = None
        if h.get("sources"):
            sources = _search_dicts_to_source_docs(h["sources"])

        items.append(ChatHistoryItem(
            role=h["role"],
            content=h["content"],
            timestamp=ts,
            sources=sources,
        ))

    turn_count = len([h for h in raw_history if h.get("role") == "user"])

    return HistoryResponse(
        session_id=session_id,
        history=items,
        turn_count=turn_count,
    )


# ============================================================
# DELETE /api/v1/chat/history/{session_id} — 清除会话
# ============================================================
@router.delete(
    "/chat/history/{session_id}",
    response_model=ClearHistoryResponse,
    summary="清除会话",
    description="清除指定 session_id 在 Redis 中的对话历史。",
)
async def clear_history(session_id: str) -> ClearHistoryResponse:
    history_manager = get_history_manager()
    cleared = await history_manager.clear_history(session_id)
    return ClearHistoryResponse(
        session_id=session_id,
        cleared=cleared,
    )
