"""
LangGraph 图节点函数

每个节点都是纯函数: (RagState) -> dict（返回需要更新的字段）。
所有节点保持同步（现有在线模块均为同步），图在 asyncio.to_thread 中调用。
"""

from dataclasses import asdict

from loguru import logger

from app.graph.state import RagState
from app.online.generator import GeneratedAnswer, Generator
from app.online.intent import IntentClassifier, IntentResult
from app.online.ranker import RankedDocument, Ranker
from app.online.retriever import Retriever, SearchResult


# ============================================================
# 意图识别节点
# ============================================================
def intent_node(state: RagState) -> dict:
    """
    调用 IntentClassifier 判断用户问题是否药品相关。

    Returns:
        {"intent": str, "intent_confidence": float}
    失败时: intent 默认为 "drug_inquiry"，宁可误判不拒绝用户。
    """
    query = state.get("query", "")
    if not query.strip():
        return {"intent": "drug_inquiry", "intent_confidence": 0.3}

    try:
        classifier = IntentClassifier()
        result: IntentResult = classifier.classify(query)
        logger.info(
            f"意图识别: intent={result.intent}, confidence={result.confidence:.3f}"
        )
        return {
            "intent": result.intent,
            "intent_confidence": result.confidence,
        }
    except Exception as e:
        logger.error(f"意图识别失败: {e}，默认视为药品问题")
        return {
            "intent": "drug_inquiry",
            "intent_confidence": 0.5,
            "error": f"意图识别失败: {e}",
            "error_node": "intent",
        }


# ============================================================
# 混合检索节点
# ============================================================
def retrieve_node(state: RagState) -> dict:
    """
    执行混合检索: 向量检索 + BM25 检索 → RRF 融合。

    Returns:
        {"search_results": list[dict], "search_count": int}
    失败时: 返回空结果，后续节点处理。
    """
    query = state.get("query", "")

    try:
        retriever = Retriever()
        results: list[SearchResult] = retriever.retrieve(query)
        search_dicts = [asdict(r) for r in results]
        logger.info(f"混合检索完成: {len(search_dicts)} 条结果")
        return {
            "search_results": search_dicts,
            "search_count": len(search_dicts),
        }
    except Exception as e:
        logger.error(f"混合检索失败: {e}")
        return {
            "search_results": [],
            "search_count": 0,
            "error": f"检索失败: {e}",
            "error_node": "retriever",
        }


# ============================================================
# 重排序节点
# ============================================================
def rank_node(state: RagState) -> dict:
    """
    调用 DashScope qwen3-rerank 对检索结果二次排序。

    Returns:
        {"ranked_docs": list[dict], "ranked_count": int}
    失败时: 回退到原始检索排序。
    """
    search_results = state.get("search_results", [])

    if not search_results:
        logger.warning("无检索结果，跳过重排序")
        return {"ranked_docs": [], "ranked_count": 0}

    query = state.get("query", "")

    try:
        ranker = Ranker()
        ranked: list[RankedDocument] = ranker.rerank(query, search_results)
        ranked_dicts = [asdict(r) for r in ranked]
        logger.info(
            f"重排序完成: {len(ranked_dicts)} 条, "
            f"最高分={ranked_dicts[0]['score']:.4f}" if ranked_dicts else "重排序完成: 0 条"
        )
        return {"ranked_docs": ranked_dicts, "ranked_count": len(ranked_dicts)}
    except Exception as e:
        logger.error(f"重排序失败: {e}，回退到原始排序")
        # 回退：按原始 RRF score 排序
        fallback = sorted(
            search_results,
            key=lambda x: x.get("score", 0.0),
            reverse=True,
        )
        return {
            "ranked_docs": fallback,
            "ranked_count": len(fallback),
            "error": f"重排序失败，已回退: {e}",
            "error_node": "ranker",
        }


# ============================================================
# 答案生成节点
# ============================================================
def generate_node(state: RagState) -> dict:
    """
    基于重排序后的文档生成回答。

    Returns:
        {"answer": str, "sources": list[dict], "template_used": str}
    失败时: 返回检索结果原文作为兜底回答。
    """
    query = state.get("query", "")
    ranked_docs = state.get("ranked_docs", [])
    history = state.get("history")
    memory_summary = state.get("memory_summary", "")
    user_memories = state.get("user_memories", "")
    user_profile = state.get("user_profile", "")

    if not ranked_docs:
        logger.warning("无参考文档，生成兜底回答")
        return {
            "answer": (
                "抱歉，未能在知识库中检索到与您问题相关的药品信息。\n\n"
                "建议：\n"
                "1. 尝试使用药品的通用名或商品名进行查询\n"
                "2. 简化问题描述，聚焦于单一药品的查询\n"
                "3. 确认您查询的药品说明书已录入系统"
            ),
            "sources": [],
            "template_used": "default",
        }

    try:
        generator = Generator()
        result: GeneratedAnswer = generator.generate(
            query=query,
            context_docs=ranked_docs,
            history=history,
            memory_summary=memory_summary,
            user_memories=user_memories,
            user_profile=user_profile,
        )
        logger.info(
            f"答案生成完成: len={len(result.answer)}, template={result.template_used}"
        )
        return {
            "answer": result.answer,
            "sources": ranked_docs,
            "template_used": result.template_used,
        }
    except Exception as e:
        logger.error(f"答案生成失败: {e}，返回检索原文")
        # 兜底：把检索到的前几条文档原文作为回答
        context_parts: list[str] = []
        for i, doc in enumerate(ranked_docs[:3], start=1):
            drug = doc.get("drug_name", "未知药品")
            section = doc.get("section", "")
            text = doc.get("chunk_text", "")
            section_str = f"（{section}）" if section else ""
            context_parts.append(f"[{i}] {drug}{section_str}\n{text}")

        fallback_answer = (
            "回答生成服务暂时不可用。以下是为您检索到的相关参考资料，供参考：\n\n"
            + "\n\n".join(context_parts)
            + "\n\n⚠️ 以上信息仅供参考，具体用药请咨询医生或药师。"
        )
        return {
            "answer": fallback_answer,
            "sources": ranked_docs,
            "template_used": "default",
            "error": f"生成失败: {e}",
            "error_node": "generator",
        }


# ============================================================
# 闲聊节点（问候、寒暄等，不走检索）
# ============================================================
def chitchat_node(state: RagState) -> dict:
    """
    对日常问候/闲聊返回简单的友好回应，不触发检索流程。

    Returns:
        {"answer": str, "sources": [], "template_used": "chitchat"}
    """
    query = state.get("query", "").strip()
    logger.info(f"闲聊: {query[:60]}")

    # 简单的问候回应映射
    greeting_responses = {
        "你好": "你好！👋 我是药品知识问答助手，有什么药品相关的问题可以随时问我。",
        "您好": "您好！有什么药品相关的问题需要我帮忙吗？",
        "hi": "Hi！有什么可以帮您的吗？",
        "hello": "Hello！有什么药品问题想咨询？",
        "在吗": "在的！有什么药品相关的问题可以随时问我。",
        "谢谢": "不客气！如果还有其他药品问题，随时问我。😊",
        "感谢": "不客气！很高兴能帮到您。",
        "早上好": "早上好！有什么药品问题需要咨询吗？",
        "下午好": "下午好！有什么可以帮您的？",
        "晚上好": "晚上好！有什么药品问题需要咨询吗？",
        "晚安": "晚安！有需要随时回来。🌙",
        "好的": "好的，有什么需要再问我。",
        "ok": "好的！",
        "嗯": "嗯嗯，有什么问题随时说~",
    }

    # 精确匹配
    answer = greeting_responses.get(query.lower())
    if not answer:
        # 模糊匹配（包含关键词）
        for key, resp in greeting_responses.items():
            if key in query:
                answer = resp
                break

    if not answer:
        # 默认友好回应
        answer = (
            "你好！😊 我是药品知识问答助手。\n\n"
            "可以帮你解答药品的适应症、用法用量、禁忌、不良反应、药物相互作用等问题。\n"
            "有什么想了解的吗？"
        )

    return {
        "answer": answer,
        "sources": [],
        "template_used": "chitchat",
    }


# ============================================================
# 通用问答节点（非药品问题，但不涉及攻击）
# ============================================================
def general_node(state: RagState) -> dict:
    """
    对非药品但正常的问题，直接使用 LLM 回答（不走检索），
    并在回答中说明这并非专长领域。

    Returns:
        {"answer": str, "sources": [], "template_used": "general"}
    """
    query = state.get("query", "").strip()
    memory_summary = state.get("memory_summary", "")
    history = state.get("history")
    user_memories = state.get("user_memories", "")
    user_profile = state.get("user_profile", "")
    logger.info(f"通用问答: {query[:60]}")

    try:
        generator = Generator()
        result: GeneratedAnswer = generator.generate(
            query=query,
            context_docs=[],  # 不传检索结果
            template="general",
            history=history,
            memory_summary=memory_summary,
            user_memories=user_memories,
            user_profile=user_profile,
        )
        return {
            "answer": result.answer,
            "sources": [],
            "template_used": "general",
        }
    except Exception as e:
        logger.error(f"通用问答生成失败: {e}")
        return {
            "answer": (
                f"关于「{query[:50]}」...\n\n"
                "我主要擅长药品知识问答，这个问题不是我的专长领域。"
                "建议你通过其他专业渠道获取更准确的信息。"
            ),
            "sources": [],
            "template_used": "general",
            "error": f"生成失败: {e}",
            "error_node": "general",
        }


# ============================================================
# 攻击拒绝节点（提示词注入、越狱等）
# ============================================================
def attack_node(state: RagState) -> dict:
    """
    检测到提示词注入攻击或越狱尝试时，返回统一的安全拒绝消息。
    不透露具体检测细节（安全考量）。

    Returns:
        {"answer": str, "sources": [], "template_used": "attack"}
    """
    logger.warning(f"检测到攻击: {state.get('query', '')[:80]}")
    return {
        "answer": (
            "抱歉，您的请求包含不安全的输入，无法处理。\n\n"
            "如果您有药品相关的正常问题，请重新表述后提问。"
        ),
        "sources": [],
        "template_used": "attack",
    }
