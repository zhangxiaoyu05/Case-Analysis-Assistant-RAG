"""
LangGraph 图节点函数

每个节点都是纯函数: (RagState) -> dict（返回需要更新的字段）。
所有节点保持同步（现有在线模块均为同步），图在 asyncio.to_thread 中调用。

v1.0.0: 从药品问答改造为病例分析。
  流程: intent → case_preprocess → multi_retrieve → rank → synthesize → generate
  新增 case_preprocess_node + synthesize_node + multi_retrieve_node
  intent 从 drug/not_drug 改为 clinical/not_clinical
"""

from dataclasses import asdict

from loguru import logger

from app.config import config
from app.graph.state import RagState
from app.online.generator import GeneratedAnswer, Generator
from app.online.intent import GateResult, Gatekeeper, is_greeting
from app.online.ranker import RankedDocument, Ranker
from app.online.retriever import Retriever, SearchResult


# ============================================================
# 门禁节点（二元判断：临床医学 → 放行 / 非临床 → 拦截）
# ============================================================
def intent_node(state: RagState) -> dict:
    """
    门禁判断：先检查问候白名单，再调用 Gatekeeper 二元分类。

    路由:
    - 问候 → chitchat（友好回应，不走检索）
    - 临床相关 → case_preprocess（病例预处理 + RAG 全流程）
    - 非临床 → reject（统一拦截消息）

    失败时: 默认放行（保证可用性）。
    """
    query = state.get("query", "")
    if not query.strip():
        return {"intent": "clinical", "intent_confidence": 0.3}

    # 1. 问候白名单（零 token，直接返回友好回应）
    if is_greeting(query):
        logger.info(f"门禁: 问候白名单命中 → chitchat")
        return {"intent": "chitchat", "intent_confidence": 0.99}

    # 2. 门禁 LLM 判断
    try:
        gk = Gatekeeper()
        result: GateResult = gk.classify(query)
        intent = "clinical" if result.clinical_related else "not_clinical"
        logger.info(
            f"门禁: clinical_related={result.clinical_related}, "
            f"confidence={result.confidence:.3f} → {intent}"
        )
        return {
            "intent": intent,
            "intent_confidence": result.confidence,
        }
    except Exception as e:
        logger.error(f"门禁失败: {e}，默认放行")
        return {
            "intent": "clinical",
            "intent_confidence": 0.5,
            "error": f"门禁失败: {e}",
            "error_node": "intent",
        }


# ============================================================
# 病例预处理节点（新增 ⭐）
# ============================================================
def case_preprocess_node(state: RagState) -> dict:
    """
    病例预处理：解析查询、提取关键段落、LLM 结构化提取、构造多路检索查询。

    流程:
    1. 分离病例文本和用户问题
    2. 超长文本（>3000字）先做关键段落提取（零 token 正则）
    3. 调用 LLM（qwen-flash）做结构化提取
    4. 基于提取结果构造 3-5 条多路检索查询

    Returns:
        case_profile, search_queries, analysis_mode
    失败时: 跳过结构化提取，直接用原始 query 检索。
    """
    import json
    import re

    query = state.get("query", "")
    analysis_mode = state.get("analysis_mode", "comprehensive")
    file_name = state.get("file_name", "")

    logger.info(f"病例预处理开始: query_len={len(query)}, mode={analysis_mode}")

    # 步骤 1: 分离病例文本和用户问题
    case_text, user_question = _parse_case_query(query)

    # 步骤 2: 超长文本关键段落提取（零 token）
    if len(case_text) > 3000:
        case_text_for_llm = _extract_key_sections(case_text)
        logger.info(
            f"超长文本预处理: {len(query)} → {len(case_text_for_llm)} 字符（关键段落提取）"
        )
    else:
        case_text_for_llm = case_text

    # 步骤 3: LLM 结构化提取
    case_profile = {}
    try:
        case_profile = _llm_extract_case(case_text_for_llm)
        logger.info(
            f"LLM 病例提取完成: "
            f"diagnosis={case_profile.get('suspected_diagnosis', [])}, "
            f"meds={len(case_profile.get('current_medications', []))}, "
            f"questions={case_profile.get('user_questions', [])}"
        )
    except Exception as e:
        logger.warning(f"LLM 病例提取失败: {e}，回退到规则提取")
        # 回退：直接用 case_text 作为上下文，不结构化
        case_profile = {
            "chief_complaint": case_text[:500] if case_text else None,
            "user_questions": [user_question] if user_question else [],
        }

    # 步骤 4: 构造多路检索查询
    search_queries = _build_search_queries(case_profile, user_question, analysis_mode)
    logger.info(f"构造检索查询: {len(search_queries)} 条 → {search_queries}")

    return {
        "case_profile": case_profile,
        "search_queries": search_queries,
        "query": user_question or query,  # 优先使用分离后的问题
    }


# ============================================================
# 病例预处理辅助函数
# ============================================================
def _parse_case_query(query: str) -> tuple[str, str]:
    """
    从完整 query 中分离病例文本和用户问题。

    文件上传格式:  "【病例文档】\n{case_text}\n\n【用户问题】\n{question}"
    纯文本格式:   "{case_text_and_possible_question}"

    Returns:
        (case_text, user_question)
    """
    if "【病例文档】" in query:
        parts = query.split("【用户问题】")
        case_text = parts[0].replace("【病例文档】", "").strip()
        user_question = parts[1].strip() if len(parts) > 1 else ""
        return case_text, user_question

    # 纯文本输入：整体作为病例文本，问题为空
    return query, query


def _extract_key_sections(text: str, max_chars: int = 5000) -> str:
    """
    从超长病例文本中提取关键段落，减少送 LLM 的 token 量。

    识别策略：匹配临床文档的关键章节标题。
    如果提取结果太短（<50行），回退到原始文本前 max_chars 字。
    """
    import re

    KEY_MARKERS = [
        r"(主\s*诉|chief\s*complaint|CC)",
        r"(现\s*病\s*史|present\s*illness|HPI)",
        r"(既\s*往\s*史|past\s*medical\s*history|PMH)",
        r"(个\s*人\s*史|家\s*族\s*史|社\s*会\s*史)",
        r"(体\s*格\s*检\s*查|查\s*体|physical\s*exam|PE)",
        r"(辅\s*助\s*检\s*查|实\s*验\s*室|lab|影\s*像\s*学|超\s*声|CT|MRI|X\s*线|心\s*电\s*图)",
        r"(初\s*步\s*诊\s*断|诊\s*断|impression|diagnosis|assessment)",
        r"(治\s*疗\s*方\s*案|用\s*药|处\s*方|medication|treatment\s*plan)",
        r"(出\s*院\s*小\s*结|discharge\s*summary)",
        r"(手\s*术|operation|surgery)",
    ]

    lines = text.split("\n")
    key_lines = []
    capture = False

    for line in lines:
        for marker in KEY_MARKERS:
            if re.search(marker, line, re.IGNORECASE):
                capture = True
                key_lines.append(f"--- {line.strip()} ---")
                break
        else:
            if capture and len(line.strip()) > 10:
                key_lines.append(line.strip())

    if len(key_lines) < 50:
        return text[:max_chars]

    extracted = "\n".join(key_lines)
    if len(extracted) > max_chars:
        return extracted[:max_chars]
    return extracted


def _llm_extract_case(case_text: str) -> dict:
    """调用 qwen-flash 做病例结构化提取。"""
    import json
    from pathlib import Path

    import yaml
    from dashscope import Generation

    # 加载 case_extraction prompt
    _prompts_path = Path(__file__).resolve().parent.parent.parent / "config" / "prompts.yaml"
    with open(_prompts_path, "r", encoding="utf-8") as _f:
        _prompts = yaml.safe_load(_f)

    _case_extraction = _prompts["case_extraction"]
    system_prompt = _case_extraction["system"]
    user_template = _case_extraction["user"]

    user_prompt = user_template.format(case_text=case_text)

    response = Generation.call(
        model=config.case_extraction_model if hasattr(config, 'case_extraction_model') else "qwen-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=800,
        api_key=config.DASHSCOPE_API_KEY,
        result_format="message",
    )

    if response.status_code != 200:
        raise RuntimeError(f"病例提取 API 错误: {response.status_code} {response.message}")

    output = response.output
    if output.choices:
        text = output.choices[0].message.content
    elif output.text:
        text = output.text
    else:
        raise RuntimeError("病例提取 API 返回空结果")

    # 解析 JSON
    text = text.strip()
    json_match = __import__('re').search(r"\{[\s\S]*\}", text)
    if json_match:
        text = json_match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"病例提取 JSON 解析失败，原始: {text[:200]}")
        return {"raw_extraction": text}


def _build_search_queries(
    case_profile: dict,
    user_question: str,
    analysis_mode: str,
) -> list[str]:
    """
    基于结构化病例 + 用户问题 + 分析模式，构造 3-5 条检索查询。

    查询覆盖四个维度：疾病 / 指南 / 药品 / 文献

    analysis_mode 影响查询侧重：
    - comprehensive: 四维均衡
    - diagnosis: 侧重疾病+指南
    - treatment: 侧重指南+药品
    - drug_review: 侧重药品+文献
    """
    queries = []

    # 1. 疾病维度 — 基于疑似诊断
    for diag in case_profile.get("suspected_diagnosis", [])[:2]:
        if diag:
            queries.append(f"{diag} 诊断标准 临床表现 治疗原则")
            queries.append(f"{diag} 临床指南 诊疗路径")

    # 2. 药品维度 — 基于当前用药
    for med in case_profile.get("current_medications", [])[:3]:
        name = med.get("name", "") if isinstance(med, dict) else str(med)
        if name:
            queries.append(f"{name} 适应症 禁忌 不良反应 药物相互作用")

    # 3. 异常发现维度
    for ab in case_profile.get("key_abnormalities", [])[:2]:
        if ab:
            queries.append(f"{ab} 临床意义 鉴别诊断")

    # 4. 用户问题维度 — 直接作为检索查询
    if user_question and len(user_question) > 5:
        queries.append(user_question)

    # 5. 分析模式补充查询
    if analysis_mode == "drug_review":
        med_names = []
        for m in case_profile.get("current_medications", [])[:3]:
            name = m.get("name", "") if isinstance(m, dict) else str(m)
            if name:
                med_names.append(name)
        if len(med_names) >= 2:
            queries.append(f"{' '.join(med_names[:3])} 药物相互作用")

    # 6. 如果没有任何查询，用 user_question 兜底
    if not queries and case_profile.get("chief_complaint"):
        queries.append(str(case_profile["chief_complaint"])[:200])

    # 去重 + 限 5 条
    seen = set()
    unique = []
    for q in queries:
        if q and q not in seen:
            seen.add(q)
            unique.append(q)

    return unique[:5]


# ============================================================
# 多路检索节点（改造：Phase 1 只用 drug_chunks）
# ============================================================
def multi_retrieve_node(state: RagState) -> dict:
    """
    对每条 search_query 在 4 个 collection 中并行检索，跨源 RRF 融合。

    v1.0.0: 使用 Retriever.multi_source_retrieve 进行真正的多 collection 检索。

    如果 search_queries 为空，回退到原始 query。

    Returns:
        {"search_results": list[dict], "search_count": int, "search_breakdown": dict}
    失败时: 回退到单源 drug_chunks 检索。
    """
    queries = state.get("search_queries", [])
    original_query = state.get("query", "")
    case_profile = state.get("case_profile", {})

    if not queries:
        queries = [original_query]

    logger.info(f"多路检索开始: {len(queries)} 条查询 → 4 个 collection")

    try:
        retriever = Retriever()
        all_results: list[dict] = []

        # v1.0.0: 确定检索源
        # 根据 case_profile 的疑似诊断和用药决定是否检索 disease/guideline/literature
        has_diagnosis = bool(case_profile.get("suspected_diagnosis", []))
        has_medications = bool(case_profile.get("current_medications", []))
        has_abnormalities = bool(case_profile.get("key_abnormalities", []))

        # 总是检索 drug + disease，有更多信息时扩展
        sources = ["drug", "disease"]
        if has_diagnosis or has_medications:
            sources.append("guideline")
        if has_medications or has_abnormalities:
            sources.append("literature")

        for q in queries:
            # 每条 query 做多源检索
            try:
                results = retriever.multi_source_retrieve(
                    query=q,
                    sources=sources,
                    top_n_per_source=config.multi_source_top_n_per_source,
                    final_top_n=config.multi_source_final_top_n,
                )
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"多源检索子查询失败: {q[:80]}... → {e}")
                # 回退到单源检索
                fallback = retriever.retrieve(q, top_n=5)
                for r in fallback:
                    r_dict = asdict(r)
                    r_dict["source_type"] = "drug"
                    all_results.append(r_dict)

        # 按 (doc_id, source_type, chunk_text[:100]) 去重
        seen = set()
        unique_results = []
        for r in all_results:
            key = (r.get("doc_id"), r.get("source_type", "drug"), r.get("chunk_text", "")[:100])
            if key not in seen:
                seen.add(key)
                r.setdefault("source_type", "drug")
                unique_results.append(r)

        # 按 score 降序排序 + 均衡采样
        from app.online.retriever import _balanced_sample

        search_dicts = _balanced_sample(
            unique_results,
            per_source_min=config.multi_source_per_source_min,
            total_max=config.multi_source_final_top_n,
        )

        # 统计 breakdown
        from collections import Counter
        breakdown = Counter(r.get("source_type", "drug") for r in search_dicts)

        logger.info(
            f"多路检索完成: {len(search_dicts)} 条 (来自 {len(queries)} 条查询), "
            f"breakdown={dict(breakdown)}"
        )
        return {
            "search_results": search_dicts,
            "search_count": len(search_dicts),
            "search_breakdown": dict(breakdown),
        }
    except Exception as e:
        logger.error(f"多路检索失败: {e}，回退到单源 drug 检索")
        # 回退
        try:
            retriever = Retriever()
            fallback_results = []
            for q in queries:
                results = retriever.retrieve(q, top_n=5)
                for r in results:
                    r_dict = asdict(r)
                    r_dict["source_type"] = "drug"
                    fallback_results.append(r_dict)
            seen = set()
            unique = []
            for r in fallback_results:
                key = (r.get("doc_id"), r.get("chunk_text", "")[:100])
                if key not in seen:
                    seen.add(key)
                    unique.append(r)
            return {
                "search_results": unique[:15],
                "search_count": len(unique[:15]),
                "search_breakdown": {"drug": len(unique[:15])},
            }
        except Exception:
            return {
                "search_results": [],
                "search_count": 0,
                "search_breakdown": {},
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
# 多源上下文合成节点（新增 ⭐）
# ============================================================
def synthesize_node(state: RagState) -> dict:
    """
    将分散的多源检索结果按临床决策维度组织，为生成准备结构化上下文。

    组织维度：
    - disease_context: 疾病相关（病因、诊断标准...）
    - guideline_context: 指南推荐（按发布年份降序）
    - drug_context: 药品信息
    - evidence_context: 循证文献（按证据级别排序）

    每个维度的上下文都包含完整引用信息供 generator 使用。
    """
    ranked = state.get("ranked_docs", [])

    organized = {
        "disease": [],
        "guideline": [],
        "drug": [],
        "literature": [],
    }

    for doc in ranked:
        st = doc.get("source_type", "drug")
        if st in organized:
            organized[st].append(doc)

    # 指南按年份降序
    organized["guideline"].sort(
        key=lambda x: x.get("publish_year", 0) or 0,
        reverse=True,
    )

    # 文献按证据级别排序
    evidence_order = {
        "1a": 0, "1b": 1, "2a": 2, "2b": 3,
        "3a": 4, "3b": 5, "4": 6, "5": 7,
        "IA": 0, "IB": 1, "IIA": 2, "IIB": 3, "III": 4, "IV": 5,
    }
    organized["literature"].sort(
        key=lambda x: evidence_order.get(
            str(x.get("evidence_level", "5")).upper(), 99
        )
    )

    total = sum(len(v) for v in organized.values())
    logger.info(
        f"上下文合成完成: 总 {total} 条 → "
        f"drug={len(organized['drug'])}, "
        f"disease={len(organized['disease'])}, "
        f"guideline={len(organized['guideline'])}, "
        f"literature={len(organized['literature'])}"
    )

    return {"synthesized_context": organized}


# ============================================================
# 答案生成节点
# ============================================================
def generate_node(state: RagState) -> dict:
    """
    基于重排序后的文档生成 SOAP 格式回答。

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
    case_profile = state.get("case_profile", {})
    synthesized_context = state.get("synthesized_context", {})
    analysis_mode = state.get("analysis_mode", "comprehensive")

    if not ranked_docs:
        logger.warning("无参考文档，生成兜底回答")
        return {
            "answer": (
                "抱歉，未能在知识库中检索到与您病例相关的临床资料。\n\n"
                "建议：\n"
                "1. 尝试提供更详细的病情描述（主诉、现病史、检查结果等）\n"
                "2. 明确您希望分析的具体方面（诊断、治疗、用药审查等）\n"
                "3. 确认知识库中已收录相关疾病指南和药品信息"
            ),
            "sources": [],
            "template_used": "case_summary",
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
            case_profile=case_profile,
            synthesized_context=synthesized_context,
            analysis_mode=analysis_mode,
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
        context_parts: list[str] = []
        for i, doc in enumerate(ranked_docs[:3], start=1):
            source_type = doc.get("source_type", "drug")
            drug = doc.get("drug_name", "")
            disease = doc.get("disease_name", "")
            section = doc.get("section", "")
            text = doc.get("chunk_text", "")
            name = drug or disease or "未知来源"
            section_str = f"（{section}）" if section else ""
            context_parts.append(f"[{i}] [{source_type}] {name}{section_str}\n{text}")

        fallback_answer = (
            "回答生成服务暂时不可用。以下是为您检索到的相关参考资料，供参考：\n\n"
            + "\n\n".join(context_parts)
            + "\n\n⚠️ 以上信息仅供参考，具体诊疗请咨询执业医师。"
        )
        return {
            "answer": fallback_answer,
            "sources": ranked_docs,
            "template_used": "case_summary",
            "error": f"生成失败: {e}",
            "error_node": "generator",
        }


# ============================================================
# 闲聊节点（问候白名单命中，不走检索）
# ============================================================
def chitchat_node(state: RagState) -> dict:
    """
    对日常问候/闲聊返回简单的友好回应，不触发检索流程。

    Returns:
        {"answer": str, "sources": [], "template_used": "chitchat"}
    """
    query = state.get("query", "").strip()
    logger.info(f"闲聊: {query[:60]}")

    greeting_responses = {
        "你好": "你好！👋 我是临床病例分析助手，可以帮您分析病例、查阅指南、评估诊疗方案。有什么可以帮您的？",
        "您好": "您好！有什么临床问题需要我帮忙分析吗？",
        "hi": "Hi！有什么临床病例需要分析的吗？",
        "hello": "Hello！有什么临床问题想咨询？",
        "在吗": "在的！有什么临床病例相关的问题可以随时问我。",
        "谢谢": "不客气！如果还有其他临床问题，随时问我。😊",
        "感谢": "不客气！很高兴能帮到您。",
        "早上好": "早上好！有什么临床问题需要咨询吗？",
        "下午好": "下午好！有什么可以帮您的？",
        "晚上好": "晚上好！有什么临床问题需要咨询吗？",
        "晚安": "晚安！有需要随时回来。🌙",
        "好的": "好的，有什么需要再问我。",
        "ok": "好的！",
        "嗯": "嗯嗯，有什么问题随时说~",
    }

    answer = greeting_responses.get(query.lower())
    if not answer:
        for key, resp in greeting_responses.items():
            if key in query:
                answer = resp
                break

    if not answer:
        answer = (
            "你好！😊 我是临床病例分析助手。\n\n"
            "可以帮你：\n"
            "- 📋 分析病例，生成 SOAP 格式报告\n"
            "- 🔬 鉴别诊断分析\n"
            "- 💊 诊疗方案评估与用药审查\n"
            "- 📜 临床指南查询\n\n"
            "您可以直接输入病例文本，或上传病例文档（PDF/DOCX/TXT）。\n"
            "有什么需要分析的吗？"
        )

    return {
        "answer": answer,
        "sources": [],
        "template_used": "chitchat",
    }


# ============================================================
# 拦截节点（非临床医学问题统一拦截）
# ============================================================
def reject_node(state: RagState) -> dict:
    """
    对非临床医学相关问题返回统一拦截消息。
    不调用任何 LLM，零 token 消耗。

    Returns:
        {"answer": str, "sources": [], "template_used": "reject"}
    """
    query = state.get("query", "").strip()
    logger.info(f"拦截非临床问题: {query[:80]}")

    return {
        "answer": (
            "抱歉，我是临床病例分析助手，只能回答临床医学相关的问题。\n\n"
            "您可以：\n"
            "- 📋 提交病例进行分析（主诉、现病史、检查结果等）\n"
            "- 🔬 询问鉴别诊断思路\n"
            "- 💊 咨询治疗方案或用药审查\n"
            "- 📜 查询临床指南和循证建议\n"
            "- 📤 上传病例文档（PDF/DOCX/TXT）\n\n"
            "请尝试重新表述您的临床医学相关问题。"
        ),
        "sources": [],
        "template_used": "reject",
    }
