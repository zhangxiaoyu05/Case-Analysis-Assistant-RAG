"""
答案生成模块

v1.0.0: 从药品问答改造为病例分析，支持 5 种临床模板 + case_profile + synthesized_context。

使用方式:
    from app.online.generator import Generator, GeneratedAnswer

    generator = Generator()
    result = generator.generate(
        query="请分析这个病例的诊疗方案",
        context_docs=[...],
        case_profile={...},
        synthesized_context={...},
        analysis_mode="treatment",
    )
    print(result.answer)
"""

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import config

# 加载 prompts.yaml 中的问答模板
_PROMPTS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "prompts.yaml"
with open(_PROMPTS_PATH, "r", encoding="utf-8") as _f:
    _PROMPTS = yaml.safe_load(_f)

_CHAT_PROMPTS = _PROMPTS["chat"]


# ============================================================
# 数据类
# ============================================================
@dataclass
class GeneratedAnswer:
    """生成的回答"""

    answer: str
    sources: list[dict] = field(default_factory=list)
    template_used: str = "case_summary"  # 使用的提示词模板
    token_count: Optional[int] = None  # 实际消耗 token（API 返回时填充）


# ============================================================
# Generator
# ============================================================
class Generator:
    """
    基于检索结果生成临床病例分析回答。

    使用 DashScope Generation API (qwen3-max) + 场景化提示词模板。
    支持 5 种模板: case_summary / differential_diagnosis / treatment_analysis /
                 drug_review / guideline_lookup

    使用方式:
        generator = Generator()
        result = generator.generate(
            query="请分析诊疗方案",
            context_docs=[...],
            case_profile={...},
            synthesized_context={...},
            analysis_mode="treatment",
        )
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        api_key: str | None = None,
    ) -> None:
        """
        初始化答案生成器。

        Args:
            model: 模型名（默认 config.chat_model = qwen3-max）
            temperature: 温度（默认 config.chat_temperature = 0.3）
            max_tokens: 最大 token（默认 config.chat_max_tokens = 2000）
            top_p: nucleus sampling（默认 config.chat_top_p = 0.95）
            api_key: DashScope API Key（默认从 config 读取）
        """
        self._model = model or config.chat_model
        self._temperature = temperature if temperature is not None else config.chat_temperature
        self._max_tokens = max_tokens or config.chat_max_tokens
        self._top_p = top_p if top_p is not None else config.chat_top_p
        self._api_key = api_key or config.DASHSCOPE_API_KEY

        if not self._model:
            raise ValueError(
                "chat_model 未配置。请在 config/config.yaml 的 models.chat.model 中设置。"
            )
        if not self._api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY 未配置。请设置环境变量或在初始化 Generator 时传入 api_key。"
            )

    def generate(
        self,
        query: str,
        context_docs: list[dict],
        history: list[dict] | None = None,
        template: str | None = None,
        memory_summary: str = "",
        user_memories: str = "",
        user_profile: str = "",
        case_profile: dict | None = None,
        synthesized_context: dict | None = None,
        analysis_mode: str = "",
    ) -> GeneratedAnswer:
        """
        生成临床病例分析回答。

        Args:
            query: 用户问题
            context_docs: 检索到的参考文档列表
            history: 对话历史
            template: 指定提示词模板（不传则自动检测）
            memory_summary: 早期对话的累积摘要
            user_memories: 跨会话用户中期记忆文本
            user_profile: 用户画像文本
            case_profile: 结构化病例信息 dict
            synthesized_context: 按维度组织的多源上下文 dict
            analysis_mode: 分析模式（comprehensive/diagnosis/treatment/drug_review）

        Returns:
            GeneratedAnswer
        """
        request_id = uuid.uuid4().hex[:8]

        if not query or not query.strip():
            logger.warning("空查询，无法生成回答")
            return GeneratedAnswer(
                answer="请提出一个具体的临床病例相关问题。",
                template_used=template or "case_summary",
            )

        case_profile = case_profile or {}
        synthesized_context = synthesized_context or {}

        # 自动检测合适的模板
        if template is None:
            template = self._detect_template(query, case_profile, analysis_mode)

        # 格式化上下文文本
        context_text = self._format_context(context_docs)
        case_profile_text = self._format_case_profile(case_profile)
        synthesized_text = self._format_synthesized_context(synthesized_context)

        # 构建 system + user messages
        system_prompt = self._get_system_prompt(template)
        user_prompt = self._get_user_prompt(
            template, context_text, query, history,
            memory_summary, user_memories, user_profile,
            case_profile_text, synthesized_text,
        )

        logger.info(
            f"[{request_id}] 开始生成回答: query={query[:60]}..., "
            f"template={template}, docs={len(context_docs)}, "
            f"context_len={len(context_text)}, mode={analysis_mode}"
        )

        try:
            answer = self._call_generation_with_retry(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            logger.info(
                f"[{request_id}] 回答生成完成: len={len(answer)}, template={template}"
            )

            return GeneratedAnswer(
                answer=answer,
                sources=context_docs,
                template_used=template,
            )

        except Exception as e:
            logger.error(f"[{request_id}] 回答生成失败: {e}")
            return GeneratedAnswer(
                answer=(
                    "抱歉，回答生成服务暂时不可用。请稍后重试。\n\n"
                    "以下是为您检索到的相关参考资料，供参考：\n\n"
                    + context_text[:1500]
                ),
                sources=context_docs,
                template_used=template,
            )

    # ----------------------------------------------------------
    # 模板检测
    # ----------------------------------------------------------
    @staticmethod
    def _detect_template(
        query: str,
        case_profile: dict | None = None,
        analysis_mode: str = "",
    ) -> str:
        """
        自动检测最合适的提示词模板。

        优先级：analysis_mode > 关键词检测 > 默认
        """
        # 分析模式优先
        mode_map = {
            "comprehensive": "case_summary",
            "diagnosis": "differential_diagnosis",
            "treatment": "treatment_analysis",
            "drug_review": "drug_review",
        }
        if analysis_mode in mode_map:
            return mode_map[analysis_mode]

        # 关键词检测 — 鉴别诊断
        if any(kw in query for kw in ["鉴别诊断", "可能是什么病", "诊断是什么", "鉴别", "区别"]):
            return "differential_diagnosis"

        # 关键词检测 — 治疗方案
        if any(kw in query for kw in ["治疗", "方案", "用药", "怎么治", "如何处理", "管理"]):
            return "treatment_analysis"

        # 关键词检测 — 用药审查
        if any(kw in query for kw in ["药物", "审查", "相互作用", "剂量", "不良反应", "副作用"]):
            return "drug_review"

        # 关键词检测 — 指南查询
        if any(kw in query for kw in ["指南", "推荐", "共识", "循证", "证据"]):
            return "guideline_lookup"

        # 默认：综合分析
        return "case_summary"

    # ----------------------------------------------------------
    # 上下文格式化
    # ----------------------------------------------------------
    @staticmethod
    def _format_context(docs: list[dict]) -> str:
        """
        将检索到的文档列表格式化为 prompt 中的参考资料文本。

        格式:
            [来源1] 类型: drug
            药品名称: 阿司匹林
            章节: 用法用量
            内容: ...
        """
        if not docs:
            return "（未检索到相关参考资料）"

        parts: list[str] = []
        for i, doc in enumerate(docs, start=1):
            source_type = doc.get("source_type", "drug")
            disease_name = doc.get("disease_name", "")
            drug_name = doc.get("drug_name", "")
            guideline_title = doc.get("guideline_title", "")
            section = doc.get("section", "")
            text = doc.get("chunk_text", "")
            evidence_level = doc.get("evidence_level", "")

            # 构建标题行
            name_parts = []
            if drug_name:
                name_parts.append(f"药品: {drug_name}")
            if disease_name:
                name_parts.append(f"疾病: {disease_name}")
            if guideline_title:
                name_parts.append(f"指南: {guideline_title}")
            name_str = " / ".join(name_parts) if name_parts else "未知来源"

            section_str = f"\n章节: {section}" if section else ""
            evidence_str = f"\n证据级别: {evidence_level}" if evidence_level else ""
            type_str = f"\n来源类型: {source_type}"

            parts.append(
                f"[来源{i}]\n"
                f"{name_str}{section_str}{type_str}{evidence_str}\n"
                f"内容: {text}"
            )

        return "\n\n".join(parts)

    @staticmethod
    def _format_case_profile(profile: dict) -> str:
        """将结构化病例信息格式化为可读文本。"""
        if not profile:
            return "（未提供病例信息）"

        lines = []
        if profile.get("chief_complaint"):
            lines.append(f"主诉: {profile['chief_complaint']}")
        if profile.get("present_illness"):
            lines.append(f"现病史: {profile['present_illness']}")
        if profile.get("past_history"):
            lines.append(f"既往史: {profile['past_history']}")
        if profile.get("family_history"):
            lines.append(f"家族史: {profile['family_history']}")
        if profile.get("physical_exam"):
            lines.append(f"体格检查: {profile['physical_exam']}")

        # 实验室检查
        lab_results = profile.get("lab_results", [])
        if lab_results:
            lines.append("辅助检查:")
            for lab in lab_results:
                if isinstance(lab, dict):
                    name = lab.get("name", "")
                    value = lab.get("value", "")
                    ref = lab.get("reference", "")
                    ref_str = f"（参考范围: {ref}）" if ref else ""
                    lines.append(f"  - {name}: {value} {ref_str}")

        # 当前用药
        meds = profile.get("current_medications", [])
        if meds:
            lines.append("当前用药:")
            for med in meds:
                if isinstance(med, dict):
                    details = f"{med.get('dosage','')} {med.get('frequency','')} {med.get('route','')}"
                    lines.append(f"  - {med.get('name','')}: {details.strip()}")

        if profile.get("suspected_diagnosis"):
            diags = profile["suspected_diagnosis"]
            if isinstance(diags, list):
                lines.append(f"疑似诊断: {', '.join(diags)}")
            else:
                lines.append(f"疑似诊断: {diags}")

        if profile.get("key_abnormalities"):
            lines.append(f"关键异常: {', '.join(profile['key_abnormalities'])}")

        return "\n".join(lines) if lines else "（未提供病例信息）"

    @staticmethod
    def _format_synthesized_context(ctx: dict) -> str:
        """将多源合成的上下文格式化为 prompt 可用的文本。"""
        if not ctx:
            return "（无可用的参考资料）"

        parts = []
        labels = {
            "disease": "📋 疾病相关知识",
            "guideline": "📜 临床指南",
            "drug": "💊 药品信息",
            "literature": "📄 循证文献",
        }

        for source_type in ["disease", "guideline", "drug", "literature"]:
            docs = ctx.get(source_type, [])
            if not docs:
                continue
            label = labels.get(source_type, source_type)
            parts.append(f"\n### {label}")
            for i, doc in enumerate(docs[:5], start=1):  # 每种最多 5 条
                name = (
                    doc.get("drug_name")
                    or doc.get("disease_name")
                    or doc.get("guideline_title")
                    or ""
                )
                section = doc.get("section", "")
                text = doc.get("chunk_text", "")
                evidence = doc.get("evidence_level", "")
                name_str = f" ({name})" if name else ""
                section_str = f" [{section}]" if section else ""
                evidence_str = f" [证据级别: {evidence}]" if evidence else ""
                parts.append(
                    f"{i}.{name_str}{section_str}{evidence_str}\n"
                    f"   {text[:500]}"
                )

        return "\n".join(parts) if parts else "（无可用的参考资料）"

    # ----------------------------------------------------------
    # 提示词构建
    # ----------------------------------------------------------
    @staticmethod
    def _get_system_prompt(template: str) -> str:
        """获取指定模板的 system prompt"""
        template_map = {
            "case_summary": "case_summary",
            "differential_diagnosis": "differential_diagnosis",
            "treatment_analysis": "treatment_analysis",
            "drug_review": "drug_review",
            "guideline_lookup": "guideline_lookup",
        }
        key = template_map.get(template, "case_summary")
        prompt_config = _CHAT_PROMPTS.get(key, _CHAT_PROMPTS["case_summary"])
        return prompt_config["system"]

    def _get_user_prompt(
        self,
        template: str,
        context: str,
        query: str,
        history: list[dict] | None = None,
        memory_summary: str = "",
        user_memories: str = "",
        user_profile: str = "",
        case_profile_text: str = "",
        synthesized_text: str = "",
    ) -> str:
        """获取指定模板的 user prompt（填充变量）"""
        template_map = {
            "case_summary": "case_summary",
            "differential_diagnosis": "differential_diagnosis",
            "treatment_analysis": "treatment_analysis",
            "drug_review": "drug_review",
            "guideline_lookup": "guideline_lookup",
        }
        key = template_map.get(template, "case_summary")
        prompt_config = _CHAT_PROMPTS.get(key, _CHAT_PROMPTS["case_summary"])
        template_text = prompt_config["user"]

        # 格式化近期对话历史
        history_text = ""
        if history:
            history_parts: list[str] = []
            for turn in history[-6:]:
                role = "用户" if turn.get("role") == "user" else "助手"
                content = turn.get("content", "")
                history_parts.append(f"{role}: {content}")
            history_text = "近期对话：\n" + "\n".join(history_parts)

        # 格式化记忆摘要
        memory_text = ""
        if memory_summary:
            memory_text = f"前序对话摘要：\n{memory_summary}\n"

        # 中期记忆文本
        user_memories_text = user_memories or ""

        # 用户画像文本
        user_profile_text = user_profile or ""

        # 病例信息文本
        case_text = case_profile_text or "（未提供病例信息）"

        # 合成上下文文本
        synth_text = synthesized_text or context

        # 填充模板变量
        return template_text.format(
            context=context,
            question=query,
            history=history_text,
            memory_summary=memory_text,
            user_memories=user_memories_text,
            user_profile=user_profile_text,
            case_profile=case_text,
            synthesized_context=synth_text,
        )

    # ----------------------------------------------------------
    # API 调用
    # ----------------------------------------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        after=lambda retry_state: logger.warning(
            f"生成重试 {retry_state.attempt_number}/3: "
            f"{retry_state.outcome.exception() if retry_state.outcome else 'unknown'}"
        ),
    )
    def _call_generation_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """调用 DashScope Generation API（带重试）"""
        from dashscope import Generation

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = Generation.call(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            top_p=self._top_p,
            api_key=self._api_key,
            result_format="message",
        )

        if response.status_code != 200:
            error_msg = (
                f"DashScope Generation API 错误: status={response.status_code}, "
                f"message={response.message}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        output = response.output
        if output is None:
            raise RuntimeError("DashScope Generation API 返回了空的 output")

        if output.choices:
            content = output.choices[0].message.content
        elif output.text:
            content = output.text
        else:
            raise RuntimeError("DashScope Generation API 返回了空的 choices 和 text")

        if not content:
            raise RuntimeError("DashScope Generation API 返回了空的消息内容")

        return content

    # ----------------------------------------------------------
    # 流式生成
    # ----------------------------------------------------------
    def generate_stream(
        self,
        query: str,
        context_docs: list[dict],
        history: list[dict] | None = None,
        template: str | None = None,
        memory_summary: str = "",
        user_memories: str = "",
        user_profile: str = "",
        case_profile: dict | None = None,
        synthesized_context: dict | None = None,
        analysis_mode: str = "",
    ):
        """
        流式版本：逐 token yield 生成结果（用于 SSE）。

        Args:
            query: 用户问题
            context_docs: 检索到的参考文档列表
            history: 对话历史
            template: 提示词模板（不传则自动检测）
            memory_summary: 早期对话的累积摘要
            user_memories: 跨会话用户中期记忆文本
            user_profile: 用户画像文本
            case_profile: 结构化病例信息 dict
            synthesized_context: 按维度组织的多源上下文 dict
            analysis_mode: 分析模式

        Yields:
            str — 每次产出一个增量 token 文本
        """
        from dashscope import Generation

        case_profile = case_profile or {}
        synthesized_context = synthesized_context or {}

        if template is None:
            template = self._detect_template(query, case_profile, analysis_mode)

        context_text = self._format_context(context_docs)
        case_profile_text = self._format_case_profile(case_profile)
        synthesized_text = self._format_synthesized_context(synthesized_context)

        system_prompt = self._get_system_prompt(template)
        user_prompt = self._get_user_prompt(
            template, context_text, query, history,
            memory_summary, user_memories, user_profile,
            case_profile_text, synthesized_text,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.info(
            f"开始流式生成: query={query[:60]}..., template={template}, "
            f"docs={len(context_docs)}, mode={analysis_mode}"
        )

        response = Generation.call(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            top_p=self._top_p,
            api_key=self._api_key,
            stream=True,
            incremental_output=True,
            result_format="message",
        )

        token_count = 0
        for chunk in response:
            if chunk.status_code != 200:
                logger.error(
                    f"流式生成错误: status={chunk.status_code}, message={chunk.message}"
                )
                break

            output = chunk.output
            if output is None:
                continue

            if output.choices:
                content = output.choices[0].message.content
            elif output.text:
                content = output.text
            else:
                continue

            if content:
                token_count += 1
                yield content

        logger.info(f"流式生成完成: {token_count} 个 token")


# ============================================================
# 便捷函数
# ============================================================
def generate_answer(
    query: str,
    context_docs: list[dict],
    history: list[dict] | None = None,
    template: str | None = None,
    memory_summary: str = "",
    case_profile: dict | None = None,
    synthesized_context: dict | None = None,
) -> GeneratedAnswer:
    """
    便捷函数：一行调用生成回答。

    Args:
        query: 用户问题
        context_docs: 检索到的参考文档
        history: 对话历史（可选）
        template: 提示词模板（可选，默认自动检测）
        memory_summary: 早期对话的累积摘要（可选）
        case_profile: 结构化病例信息（可选）
        synthesized_context: 多源合成上下文（可选）

    Returns:
        GeneratedAnswer
    """
    generator = Generator()
    return generator.generate(
        query=query,
        context_docs=context_docs,
        history=history,
        template=template,
        memory_summary=memory_summary,
        case_profile=case_profile,
        synthesized_context=synthesized_context,
    )
