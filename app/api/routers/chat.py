"""
问答路由

POST   /api/v1/chat                    - 单轮问答（LangGraph 编排）
POST   /api/v1/chat/stream              - 流式问答（SSE）
GET    /api/v1/chat/history/{session_id} - 获取对话历史
DELETE /api/v1/chat/history/{session_id} - 清除会话

Phase 0: 集成 JWT 用户鉴权 + 首条消息触发标题自动生成。
Phase 1: 按 token 阈值触发摘要 + enable_memory 开关 + user_id 绑定。
Phase 2: 跨会话中期记忆加载 + 对话完成后异步提取新记忆。
"""

import asyncio
import json
import time
import uuid
from dataclasses import asdict

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from app.api.dependencies import get_current_user, get_graph, get_history_manager
from app.config import config
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


async def _load_context(
    user_id: int,
    session_id: str,
    query: str,
    enable_memory: bool,
) -> tuple[list[dict], str, list[dict], str, str]:
    """
    加载全部对话上下文：短期记忆 + 中期记忆 + 长期画像。

    Returns:
        (raw_history, memory_summary, recent_history_for_llm, user_memories_text, user_profile_text)
    """
    history_manager = get_history_manager()

    if not enable_memory:
        logger.debug(f"[{session_id}] 短期记忆已禁用，跳过历史加载")
        raw_history, memory_summary, recent_history, history_for_llm = [], "", [], []
    else:
        # 加载 Redis 中的历史
        raw_history = await history_manager.get_history(user_id, session_id)
        history_for_llm = _build_history_for_llm(raw_history)

        # 加载已有摘要
        existing_summary = await history_manager.get_summary(user_id, session_id)

        # Token 阈值摘要
        memory_manager = MemoryManager()
        memory_summary, recent_history = await memory_manager.summarize(
            history_for_llm, existing_summary, query, enable_memory
        )

        # 如果摘要更新了，存回 Redis
        if memory_summary != existing_summary:
            await history_manager.set_summary(user_id, session_id, memory_summary)

    # Phase 2: 加载跨会话中期记忆
    user_memories_text = ""
    try:
        from app.services.user_memory_manager import UserMemoryManager
        umm = UserMemoryManager()
        try:
            user_memories_text = umm.format_memories_for_prompt(user_id)
            if user_memories_text:
                logger.debug(
                    f"[{session_id}] 中期记忆已加载: {len(user_memories_text)} 字符"
                )
        finally:
            umm.close()
    except Exception as e:
        logger.warning(f"[{session_id}] 中期记忆加载失败: {e}")

    # Phase 3: 加载用户长期画像
    user_profile_text = ""
    try:
        from app.services.user_profile_manager import UserProfileManager
        upm = UserProfileManager()
        try:
            user_profile_text = upm.format_profile_for_prompt(user_id)
            if user_profile_text:
                logger.debug(
                    f"[{session_id}] 用户画像已加载: {len(user_profile_text)} 字符"
                )
        finally:
            upm.close()
    except Exception as e:
        logger.warning(f"[{session_id}] 用户画像加载失败: {e}")

    logger.info(
        f"[{session_id}] user_id={user_id} "
        f"历史={len(history_for_llm)} 条, 摘要={len(memory_summary)} 字, "
        f"最近={len(recent_history)} 条, 中期记忆={'有' if user_memories_text else '无'}, "
        f"用户画像={'有' if user_profile_text else '无'}"
    )

    return raw_history, memory_summary, recent_history, user_memories_text, user_profile_text


async def _extract_memories_async(
    user_id: int,
    session_id: str,
    user_msg: str,
    assistant_msg: str,
) -> None:
    """异步提取中期记忆（不阻塞主流程）。"""
    try:
        from app.services.user_memory_manager import UserMemoryManager
        umm = UserMemoryManager()
        try:
            await umm.extract_and_save(user_id, session_id, user_msg, assistant_msg)
        finally:
            umm.close()
    except Exception as e:
        logger.warning(f"中期记忆提取失败 [user={user_id}]: {e}")


async def _extract_profile_async(
    user_id: int,
    session_id: str,
    user_msg: str,
    assistant_msg: str,
) -> None:
    """异步提取用户画像（不阻塞主流程）。"""
    try:
        from app.services.user_profile_manager import UserProfileManager
        upm = UserProfileManager()
        try:
            await upm.extract_and_save(user_id, session_id, user_msg, assistant_msg)
        finally:
            upm.close()
    except Exception as e:
        logger.warning(f"用户画像提取失败 [user={user_id}]: {e}")


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
async def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
) -> ChatResponse:
    user_id = current_user["user_id"]
    session_id = request.session_id or uuid.uuid4().hex[:16]
    t_start = time.perf_counter()

    # 1. 加载全部上下文（短期 + 中期记忆 + 长期画像）
    raw_history, memory_summary, recent_history, user_memories, user_profile = await _load_context(
        user_id, session_id, request.message, request.enable_memory
    )

    logger.info(
        f"[{session_id}] user={current_user['username']} "
        f"单轮问答: {request.message[:60]}..."
    )

    # 2. 执行 LangGraph RAG 流程
    graph = get_graph()
    initial_state: RagState = {
        "query": request.message,
        "history": recent_history,
        "memory_summary": memory_summary,
        "user_memories": user_memories,
        "user_profile": user_profile,
    }

    try:
        result_state = await asyncio.to_thread(graph.invoke, initial_state)
    except Exception as e:
        logger.error(f"[{session_id}] 图执行异常: {e}")
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

    # 3. 构建响应
    elapsed_ms = (time.perf_counter() - t_start) * 1000
    sources = _search_dicts_to_source_docs(result_state.get("sources", []))

    # 4. 保存本轮对话到 Redis（只在启用记忆时）
    if request.enable_memory:
        try:
            history_manager = get_history_manager()
            await history_manager.add_turn(
                user_id=user_id,
                session_id=session_id,
                user_msg=request.message,
                assistant_msg=result_state.get("answer", ""),
                sources=result_state.get("sources"),
            )
        except Exception as e:
            logger.warning(f"[{session_id}] 保存历史失败: {e}（不影响响应）")

    # 5. 首条消息 → 异步生成对话标题
    if len(raw_history) == 0:
        asyncio.create_task(_generate_title_async(session_id, request.message))

    # 6. 异步提取中期记忆 + 用户画像（独立于 enable_memory，跨会话持久化）
    assistant_answer = result_state.get("answer", "")
    if assistant_answer:
        asyncio.create_task(
            _extract_memories_async(
                user_id, session_id, request.message, assistant_answer
            )
        )
        asyncio.create_task(
            _extract_profile_async(
                user_id, session_id, request.message, assistant_answer
            )
        )

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
async def chat_stream(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    session_id = request.session_id or uuid.uuid4().hex[:16]

    # 加载全部上下文（短期 + 中期记忆 + 长期画像）
    raw_history, memory_summary, recent_history, user_memories, user_profile = await _load_context(
        user_id, session_id, request.message, request.enable_memory
    )

    logger.info(
        f"[{session_id}] user={current_user['username']} "
        f"流式问答: {request.message[:60]}..."
    )

    # 首条消息标记（用于标题生成）
    is_first_message = len(raw_history) == 0

    # 首条消息 → 异步生成对话标题
    if is_first_message:
        asyncio.create_task(_generate_title_async(session_id, request.message))

    async def event_generator():
        full_answer = ""
        ranked_dicts: list[dict] = []

        try:
            # ---- 阶段 1: 意图识别 ----
            classifier = IntentClassifier()
            intent_result = classifier.classify(request.message)

            if intent_result.intent == "chitchat":
                from app.graph.nodes import chitchat_node
                chitchat_state = chitchat_node({"query": request.message})
                chitchat_msg = chitchat_state.get("answer", "你好！有什么可以帮您的吗？")
                yield f"event: token\ndata: {json.dumps({'event': 'token', 'data': chitchat_msg}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'event': 'done', 'data': ''})}\n\n"
                full_answer = chitchat_msg
                return

            if intent_result.intent == "attack":
                attack_msg = "抱歉，您的请求包含不安全的输入，无法处理。如果您有药品相关的正常问题，请重新表述后提问。"
                yield f"event: token\ndata: {json.dumps({'event': 'token', 'data': attack_msg}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'event': 'done', 'data': ''})}\n\n"
                full_answer = attack_msg
                return

            if intent_result.intent == "general":
                generator = Generator()
                full_answer = ""
                for token in generator.generate_stream(
                    query=request.message,
                    context_docs=[],
                    history=recent_history,
                    template="general",
                    memory_summary=memory_summary,
                    user_memories=user_memories,
                    user_profile=user_profile,
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
                    ranked_dicts = search_dicts
            else:
                ranked_dicts = []

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
                    user_memories=user_memories,
                    user_profile=user_profile,
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
            if full_answer:
                # 保存短期历史（仅在启用记忆时）
                if request.enable_memory:
                    try:
                        history_manager = get_history_manager()
                        await history_manager.add_turn(
                            user_id=user_id,
                            session_id=session_id,
                            user_msg=request.message,
                            assistant_msg=full_answer,
                            sources=ranked_dicts,
                        )
                    except Exception as e:
                        logger.warning(f"[{session_id}] 保存流式历史失败: {e}")

                # 中期记忆 + 用户画像（独立于 enable_memory，跨会话持久化）
                asyncio.create_task(
                    _extract_memories_async(
                        user_id, session_id, request.message, full_answer
                    )
                )
                asyncio.create_task(
                    _extract_profile_async(
                        user_id, session_id, request.message, full_answer
                    )
                )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
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
async def get_history(
    session_id: str,
    current_user: dict = Depends(get_current_user),
) -> HistoryResponse:
    user_id = current_user["user_id"]
    history_manager = get_history_manager()
    raw_history = await history_manager.get_history(user_id, session_id)

    items: list[ChatHistoryItem] = []
    from datetime import datetime, timezone

    for h in raw_history:
        ts_str = h.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            ts = datetime.now(timezone.utc)

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
async def clear_history(
    session_id: str,
    current_user: dict = Depends(get_current_user),
) -> ClearHistoryResponse:
    user_id = current_user["user_id"]
    history_manager = get_history_manager()
    cleared = await history_manager.clear_history(user_id, session_id)
    return ClearHistoryResponse(
        session_id=session_id,
        cleared=cleared,
    )


# ============================================================
# 辅助：异步标题生成
# ============================================================
async def _generate_title_async(session_id: str, first_message: str) -> None:
    """异步调用 ConversationManager 生成对话标题（不阻塞事件循环）。"""
    try:
        from app.services.conversation_manager import ConversationManager

        def _sync_generate():
            manager = ConversationManager()
            try:
                manager.generate_title(session_id, first_message)
            finally:
                manager.close()

        await asyncio.to_thread(_sync_generate)
    except Exception as e:
        logger.warning(f"标题生成失败 [{session_id}]: {e}")
