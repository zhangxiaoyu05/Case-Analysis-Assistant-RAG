"""
问答路由

POST   /api/v1/chat                    - 单轮问答（LangGraph 编排，支持文件上传）
POST   /api/v1/chat/stream              - 流式问答（SSE，支持文件上传）
GET    /api/v1/chat/history/{session_id} - 获取对话历史
DELETE /api/v1/chat/history/{session_id} - 清除会话

v1.0.0: 从药品问答改造为病例分析，支持 multipart/form-data 文件上传。
"""

import asyncio
import json
import time
import uuid
from dataclasses import asdict

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger

from app.api.dependencies import get_current_user, get_graph, get_history_manager
from app.config import config
from app.graph.state import GraphResult, RagState
from app.online.generator import Generator
from app.online.intent import Gatekeeper, is_greeting
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

# 文件上传限制
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


# ============================================================
# 辅助函数
# ============================================================
def _search_dicts_to_source_docs(sources: list[dict]) -> list[SourceDoc]:
    """将检索结果的 dict 列表转为 Pydantic SourceDoc 列表。"""
    return [
        SourceDoc(
            drug_name=s.get("drug_name", ""),
            disease_name=s.get("disease_name"),
            guideline_title=s.get("guideline_title"),
            section=s.get("section"),
            chunk_text=s.get("chunk_text", ""),
            score=s.get("score"),
            doc_id=s.get("doc_id"),
            source_type=s.get("source_type", "drug"),
            evidence_level=s.get("evidence_level"),
        )
        for s in sources
    ]


def _build_history_for_llm(history: list[dict]) -> list[dict]:
    """从 Redis 历史中提取 role+content 供 LLM 使用。"""
    return [{"role": h["role"], "content": h["content"]} for h in history]


def _parse_uploaded_file(file: UploadFile) -> tuple[str, str]:
    """
    解析上传的病例文件，返回 (file_content_text, error_message)。

    使用现有的 loader.py 基础设施加载 PDF/DOCX/TXT。
    """
    import tempfile
    from pathlib import Path

    ext = Path(file.filename or "unknown").suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        return "", f"不支持的文件格式: {ext}。支持的格式: PDF, DOCX, TXT"

    # 检查文件大小
    if file.size and file.size > MAX_FILE_SIZE:
        return "", f"文件过大（{file.size / 1024 / 1024:.1f}MB），请上传小于 20MB 的文件"

    try:
        # 读取文件内容到临时文件
        content_bytes = file.file.read()
        if not content_bytes:
            return "", "文件为空，请上传有效的病例文档"

        # 使用临时文件调用现有的 loader
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(content_bytes)

        try:
            from app.offline.loader import load_document
            doc = load_document(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        text = doc.raw_text if hasattr(doc, 'raw_text') else str(doc)

        if not text or not text.strip():
            return "", "文件内容为空或无法识别文本内容"

        # 截断过长文本（超过 50000 字符取前 50000）
        if len(text) > 50000:
            text = text[:50000] + "\n\n... (文档过长，已截断前 50000 字符)"
            logger.warning(f"文件过长，截断至 50000 字符: {file.filename}")

        return text, ""

    except Exception as e:
        logger.error(f"文件解析失败 [{file.filename}]: {e}")
        return "", f"文件解析失败: {e}"


def _build_case_query(case_text: str, user_message: str) -> str:
    """
    构造包含病例信息和用户问题的完整 query。

    格式：
    【病例文档】
    {case_text}

    【用户问题】
    {user_message}

    case_preprocess_node 通过 【病例文档】/【用户问题】 标记解析。
    """
    parts = []
    if case_text:
        parts.append(f"【病例文档】\n{case_text}")
    if user_message:
        parts.append(f"【用户问题】\n{user_message}")

    return "\n\n".join(parts)


async def _load_context(
    user_id: int,
    session_id: str,
    query: str,
    enable_memory: bool,
) -> tuple[list[dict], str, list[dict], str, str]:
    """
    加载全部对话上下文：短期记忆 + 中期记忆 + 长期画像。
    """
    history_manager = get_history_manager()

    if not enable_memory:
        logger.debug(f"[{session_id}] 短期记忆已禁用，跳过历史加载")
        raw_history, memory_summary, recent_history, history_for_llm = [], "", [], []
    else:
        raw_history = await history_manager.get_history(user_id, session_id)
        history_for_llm = _build_history_for_llm(raw_history)

        existing_summary = await history_manager.get_summary(user_id, session_id)

        memory_manager = MemoryManager()
        memory_summary, recent_history = await memory_manager.summarize(
            history_for_llm, existing_summary, query, enable_memory
        )

        if memory_summary != existing_summary:
            await history_manager.set_summary(user_id, session_id, memory_summary)

    # 加载跨会话中期记忆
    user_memories_text = ""
    try:
        from app.services.user_memory_manager import UserMemoryManager
        umm = UserMemoryManager()
        try:
            user_memories_text = umm.format_memories_for_prompt(user_id)
            if user_memories_text:
                logger.debug(f"[{session_id}] 中期记忆已加载: {len(user_memories_text)} 字符")
        finally:
            umm.close()
    except Exception as e:
        logger.warning(f"[{session_id}] 中期记忆加载失败: {e}")

    # 加载用户长期画像
    user_profile_text = ""
    try:
        from app.services.user_profile_manager import UserProfileManager
        upm = UserProfileManager()
        try:
            user_profile_text = upm.format_profile_for_prompt(user_id)
            if user_profile_text:
                logger.debug(f"[{session_id}] 用户画像已加载: {len(user_profile_text)} 字符")
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
# POST /api/v1/chat — 单轮问答（支持文件上传）
# ============================================================
@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "请求参数错误"},
        413: {"model": ErrorResponse, "description": "文件过大"},
        503: {"model": ErrorResponse, "description": "依赖服务不可用"},
    },
    summary="单轮问答（支持病例文件上传）",
    description="提交病例文本和/或上传病例文档，返回 SOAP 格式的临床分析及参考来源。",
)
async def chat(
    message: str = Form(default="", description="用户问题/病例文本（至少提供 message 或 file 之一）"),
    file: UploadFile | None = File(default=None, description="病例文档（PDF/DOCX/TXT，可选）"),
    session_id: str = Form(default="", description="会话 ID"),
    analysis_mode: str = Form(default="comprehensive", description="分析模式: comprehensive/diagnosis/treatment/drug_review"),
    enable_memory: bool = Form(default=True, description="是否启用短期记忆"),
    current_user: dict = Depends(get_current_user),
) -> ChatResponse:
    user_id = current_user["user_id"]
    sid = session_id or uuid.uuid4().hex[:16]
    t_start = time.perf_counter()

    # 1. 处理文件上传
    case_text = ""
    file_name = ""
    if file and file.filename:
        case_text, file_error = _parse_uploaded_file(file)
        if file_error:
            from app.schemas.common import ErrorResponse as Err
            return ChatResponse(
                answer=f"文件处理失败: {file_error}",
                sources=[],
                session_id=sid,
                intent="error",
                elapsed_ms=0,
            )
        file_name = file.filename

    # 2. 构造完整 query
    query = _build_case_query(case_text, message)
    if not query.strip():
        return ChatResponse(
            answer="请提供病例信息（文本或上传文件）以便进行分析。",
            sources=[],
            session_id=sid,
            intent="error",
            elapsed_ms=0,
        )

    # 3. 加载上下文
    raw_history, memory_summary, recent_history, user_memories, user_profile = await _load_context(
        user_id, sid, query, enable_memory
    )

    logger.info(
        f"[{sid}] user={current_user['username']} "
        f"单轮问答: query_len={len(query)}, file={file_name}, mode={analysis_mode}"
    )

    # 4. 执行 LangGraph RAG 流程
    graph = get_graph()
    initial_state: RagState = {
        "query": query,
        "history": recent_history,
        "memory_summary": memory_summary,
        "user_memories": user_memories,
        "user_profile": user_profile,
        "file_name": file_name,
        "analysis_mode": analysis_mode,
    }

    try:
        result_state = await asyncio.to_thread(graph.invoke, initial_state)
    except Exception as e:
        logger.error(f"[{sid}] 图执行异常: {e}")
        result_state = {
            "query": query,
            "answer": f"系统处理异常，请稍后重试。错误: {e}",
            "sources": [],
            "intent": "clinical",
            "intent_confidence": 0.0,
            "template_used": "error",
            "error": str(e),
            "error_node": "graph",
            "case_profile": {},
            "search_breakdown": {},
        }

    # 5. 构建响应
    elapsed_ms = (time.perf_counter() - t_start) * 1000
    sources = _search_dicts_to_source_docs(result_state.get("sources", []))

    # 6. 保存本轮对话到 Redis
    if enable_memory:
        try:
            history_manager = get_history_manager()
            await history_manager.add_turn(
                user_id=user_id,
                session_id=sid,
                user_msg=query[:2000],  # 截断很长的 query
                assistant_msg=result_state.get("answer", ""),
                sources=result_state.get("sources"),
            )
        except Exception as e:
            logger.warning(f"[{sid}] 保存历史失败: {e}（不影响响应）")

    # 7. 首条消息 → 异步生成对话标题
    if len(raw_history) == 0:
        short_msg = (message or case_text)[:100]
        asyncio.create_task(_generate_title_async(sid, short_msg))

    # 8. 异步提取中期记忆 + 用户画像
    assistant_answer = result_state.get("answer", "")
    if assistant_answer:
        short_user_msg = (message or case_text)[:500]
        asyncio.create_task(
            _extract_memories_async(user_id, sid, short_user_msg, assistant_answer)
        )
        asyncio.create_task(
            _extract_profile_async(user_id, sid, short_user_msg, assistant_answer)
        )

    logger.info(
        f"[{sid}] 问答完成: intent={result_state.get('intent')}, "
        f"sources={len(sources)}, elapsed={elapsed_ms:.0f}ms"
    )

    return ChatResponse(
        answer=result_state.get("answer", ""),
        sources=sources,
        session_id=sid,
        intent=result_state.get("intent"),
        elapsed_ms=round(elapsed_ms, 2),
        case_profile=result_state.get("case_profile"),
        search_breakdown=result_state.get("search_breakdown"),
    )


# ============================================================
# POST /api/v1/chat/stream — 流式问答（SSE，支持文件上传）
# ============================================================
@router.post(
    "/chat/stream",
    response_model=None,
    responses={
        400: {"model": ErrorResponse, "description": "请求参数错误"},
    },
    summary="流式问答（SSE，支持病例文件上传）",
    description="与 /chat 相同，但通过 Server-Sent Events 流式返回生成结果。",
)
async def chat_stream(
    message: str = Form(default="", description="用户问题/病例文本"),
    file: UploadFile | None = File(default=None, description="病例文档（PDF/DOCX/TXT，可选）"),
    session_id: str = Form(default="", description="会话 ID"),
    analysis_mode: str = Form(default="comprehensive", description="分析模式"),
    enable_memory: bool = Form(default=True, description="是否启用短期记忆"),
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    sid = session_id or uuid.uuid4().hex[:16]

    # 1. 处理文件上传
    case_text = ""
    file_name = ""
    if file and file.filename:
        case_text, file_error = _parse_uploaded_file(file)
        if file_error:
            async def error_gen():
                yield f"event: token\ndata: {json.dumps({'event': 'token', 'data': f'文件处理失败: {file_error}'}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'event': 'done', 'data': ''})}\n\n"
            return StreamingResponse(error_gen(), media_type="text/event-stream")
        file_name = file.filename

    # 2. 构造完整 query
    query = _build_case_query(case_text, message)
    if not query.strip():
        async def empty_gen():
            yield f"event: token\ndata: {json.dumps({'event': 'token', 'data': '请提供病例信息以便进行分析。'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'event': 'done', 'data': ''})}\n\n"
        return StreamingResponse(empty_gen(), media_type="text/event-stream")

    # 3. 加载上下文
    raw_history, memory_summary, recent_history, user_memories, user_profile = await _load_context(
        user_id, sid, query, enable_memory
    )

    logger.info(
        f"[{sid}] user={current_user['username']} "
        f"流式问答: query_len={len(query)}, file={file_name}, mode={analysis_mode}"
    )

    is_first_message = len(raw_history) == 0

    async def event_generator():
        full_answer = ""
        ranked_dicts: list[dict] = []
        search_dicts: list[dict] = []

        try:
            # ---- 阶段 1: 门禁判断 ----
            if is_greeting(query):
                from app.graph.nodes import chitchat_node
                chitchat_state = chitchat_node({"query": query})
                chitchat_msg = chitchat_state.get("answer", "你好！有什么可以帮您的吗？")
                yield f"event: token\ndata: {json.dumps({'event': 'token', 'data': chitchat_msg}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'event': 'done', 'data': ''})}\n\n"
                full_answer = chitchat_msg
                return

            gk = Gatekeeper()
            gate_result = gk.classify(query)

            if not gate_result.clinical_related:
                reject_msg = (
                    "抱歉，我是临床病例分析助手，只能回答临床医学相关的问题。\n\n"
                    "您可以：\n"
                    "- 📋 提交病例进行分析\n"
                    "- 🔬 询问鉴别诊断思路\n"
                    "- 💊 咨询治疗方案或用药审查\n"
                    "- 📜 查询临床指南\n\n"
                    "请尝试重新表述您的临床医学相关问题。"
                )
                yield f"event: token\ndata: {json.dumps({'event': 'token', 'data': reject_msg}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'event': 'done', 'data': ''})}\n\n"
                full_answer = reject_msg
                return

            # ---- 阶段 2: 病例预处理 ----
            from app.graph.nodes import (
                _build_search_queries,
                _extract_key_sections,
                _llm_extract_case,
                _parse_case_query,
            )
            case_t, user_q = _parse_case_query(query)
            if len(case_t) > 3000:
                case_t = _extract_key_sections(case_t)

            case_profile = {}
            try:
                case_profile = _llm_extract_case(case_t)
            except Exception:
                case_profile = {"chief_complaint": case_t[:500] if case_t else None}

            search_queries = _build_search_queries(case_profile, user_q, analysis_mode)
            effective_query = user_q or query

            # ---- 阶段 3: 多路检索 ----
            retriever = Retriever()
            all_search_results = []
            for q in search_queries:
                results = retriever.retrieve(q)
                all_search_results.extend(results)

            seen = set()
            unique_results = []
            for r in all_search_results:
                r_dict = asdict(r)
                key = (r_dict.get("doc_id"), r_dict.get("chunk_text", "")[:100])
                if key not in seen:
                    seen.add(key)
                    r_dict.setdefault("source_type", "drug")
                    unique_results.append(r_dict)
            search_dicts = unique_results[:15]

            logger.info(f"[{sid}] 流式-检索: {len(search_dicts)} 条")

            # ---- 阶段 4: 重排序 ----
            if search_dicts:
                ranker = Ranker()
                try:
                    ranked = ranker.rerank(effective_query, search_dicts)
                    ranked_dicts = [asdict(r) for r in ranked]
                except Exception:
                    ranked_dicts = search_dicts
            else:
                ranked_dicts = []

            # ---- 阶段 5: 上下文合成 ----
            from app.graph.nodes import synthesize_node
            synth_state = synthesize_node({"ranked_docs": ranked_dicts})
            synthesized_context = synth_state.get("synthesized_context", {})

            # 发送 case_profile（病例提取结果）
            if case_profile:
                yield f"event: case_profile\ndata: {json.dumps(case_profile, ensure_ascii=False)}\n\n"

            # 发送 sources
            sources_payload = [
                {
                    "drug_name": r.get("drug_name", ""),
                    "disease_name": r.get("disease_name"),
                    "guideline_title": r.get("guideline_title"),
                    "section": r.get("section", ""),
                    "chunk_text": r.get("chunk_text", ""),
                    "score": r.get("score", 0.0),
                    "doc_id": r.get("doc_id", 0),
                    "source_type": r.get("source_type", "drug"),
                    "evidence_level": r.get("evidence_level"),
                }
                for r in ranked_dicts
            ]
            yield f"event: sources\ndata: {json.dumps(sources_payload, ensure_ascii=False)}\n\n"

            # ---- 阶段 6: 流式生成 ----
            if ranked_dicts:
                generator = Generator()
                for token in generator.generate_stream(
                    query=effective_query,
                    context_docs=ranked_dicts,
                    history=recent_history,
                    memory_summary=memory_summary,
                    user_memories=user_memories,
                    user_profile=user_profile,
                    case_profile=case_profile,
                    synthesized_context=synthesized_context,
                    analysis_mode=analysis_mode,
                ):
                    full_answer += token
                    yield f"data: {json.dumps({'event': 'token', 'data': token}, ensure_ascii=False)}\n\n"
            else:
                no_result_msg = "抱歉，未能在知识库中检索到与您病例相关的临床资料。\n\n请尝试提供更详细的病情描述，或明确您希望分析的具体方面。"
                full_answer = no_result_msg
                yield f"data: {json.dumps({'event': 'token', 'data': no_result_msg}, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'event': 'done', 'data': ''})}\n\n"

        except Exception as e:
            logger.error(f"[{sid}] 流式处理异常: {e}")
            yield f"data: {json.dumps({'event': 'error', 'data': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            if full_answer:
                if enable_memory:
                    try:
                        history_manager = get_history_manager()
                        await history_manager.add_turn(
                            user_id=user_id,
                            session_id=sid,
                            user_msg=query[:2000],
                            assistant_msg=full_answer,
                            sources=ranked_dicts,
                        )
                    except Exception as e:
                        logger.warning(f"[{sid}] 保存流式历史失败: {e}")

                short_msg = (message or case_text)[:500]
                asyncio.create_task(
                    _extract_memories_async(user_id, sid, short_msg, full_answer)
                )
                asyncio.create_task(
                    _extract_profile_async(user_id, sid, short_msg, full_answer)
                )

    if is_first_message:
        short_msg = (message or case_text)[:100]
        asyncio.create_task(_generate_title_async(sid, short_msg))

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
    """异步调用 ConversationManager 生成对话标题。"""
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
