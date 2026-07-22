"""
答案生成模块

使用 DashScope Generation API 基于检索到的参考资料生成药品知识回答。
支持多种回答场景：默认问答、药品对比、用法用量追问。

使用方式:
    from app.online.generator import Generator, GeneratedAnswer

    generator = Generator()
    result = generator.generate(
        query="阿司匹林一天吃几次？",
        context_docs=[...],
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
    template_used: str = "default"  # 使用的提示词模板
    token_count: Optional[int] = None  # 实际消耗 token（API 返回时填充）


# ============================================================
# Generator
# ============================================================
class Generator:
    """
    基于检索结果生成药品知识回答。

    使用 DashScope Generation API (qwen3-max) + 场景化提示词模板。

    使用方式:
        generator = Generator()
        result = generator.generate(
            query="阿司匹林和布洛芬有什么区别？",
            context_docs=[
                {"drug_name": "阿司匹林", "section": "适应症", "chunk_text": "..."},
                {"drug_name": "布洛芬", "section": "适应症", "chunk_text": "..."},
            ],
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
    ) -> GeneratedAnswer:
        """
        生成药品知识回答。

        Args:
            query: 用户问题
            context_docs: 检索到的参考文档列表，每项为 dict，需包含:
                - drug_name: 药品名
                - section: 章节
                - chunk_text: 文本内容
                - score (可选): 相关性得分
            history: 对话历史 [{"role": "user/assistant", "content": "..."}]
            template: 指定提示词模板（"default" / "comparison" / "dosage_followup"）
                      不传则自动检测
            memory_summary: 早期对话的累积摘要（Phase 1 短期记忆）
            user_memories: 跨会话用户中期记忆文本（Phase 2）
            user_profile: 用户画像文本（Phase 3 长期记忆）

        Returns:
            GeneratedAnswer — answer 为生成的回答文本
        """
        request_id = uuid.uuid4().hex[:8]

        if not query or not query.strip():
            logger.warning("空查询，无法生成回答")
            return GeneratedAnswer(
                answer="请提出一个具体的药品相关问题。",
                template_used=template or "default",
            )

        # 自动检测合适的模板
        if template is None:
            template = self._detect_template(query, history)

        # 格式化上下文文本
        context_text = self._format_context(context_docs)

        # 构建 system + user messages
        system_prompt = self._get_system_prompt(template)
        user_prompt = self._get_user_prompt(
            template, context_text, query, history, memory_summary, user_memories, user_profile
        )

        logger.info(
            f"[{request_id}] 开始生成回答: query={query[:60]}..., "
            f"template={template}, docs={len(context_docs)}, "
            f"context_len={len(context_text)}"
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
    def _detect_template(query: str, history: list[dict] | None = None) -> str:
        """
        自动检测最合适的提示词模板。

        检测规则:
        - 如果存在对话历史且 user 消息 ≥ 2 条 → dosage_followup
        - 如果查询包含对比/比较/区别关键词 → comparison
        - 否则 → default
        """
        # 追问检测
        if history and len([h for h in history if h.get("role") == "user"]) >= 1:
            return "dosage_followup"

        # 对比检测
        comparison_patterns = [
            r"(对比|比较|区别|差别|差异|哪个|哪种|vs|VS)",
            r"(有什么不同|有何不同|有什么不一样|有什么区别)",
            r"(可以一起吃|能一起吃|可以同时|能同时)",
            r"(还是哪个|还是哪种|哪个更好|哪种更好)",
        ]
        for pattern in comparison_patterns:
            if re.search(pattern, query):
                return "comparison"

        return "default"

    # ----------------------------------------------------------
    # 上下文格式化
    # ----------------------------------------------------------
    @staticmethod
    def _format_context(docs: list[dict]) -> str:
        """
        将检索到的文档列表格式化为 prompt 中的参考资料文本。

        格式:
            [来源1]
            药品名称: 阿司匹林
            章节: 用法用量
            内容: 口服。成人常用量：一次0.3～0.6g...
        """
        if not docs:
            return "（未检索到相关参考资料）"

        parts: list[str] = []
        for i, doc in enumerate(docs, start=1):
            drug = doc.get("drug_name", "未知药品")
            section = doc.get("section", "")
            text = doc.get("chunk_text", "")

            section_str = f"\n章节: {section}" if section else ""
            parts.append(
                f"[来源{i}]\n"
                f"药品名称: {drug}{section_str}\n"
                f"内容: {text}"
            )

        return "\n\n".join(parts)

    # ----------------------------------------------------------
    # 提示词构建
    # ----------------------------------------------------------
    @staticmethod
    def _get_system_prompt(template: str) -> str:
        """获取指定模板的 system prompt"""
        template_map = {
            "default": "default",
            "comparison": "comparison",
            "dosage_followup": "dosage_followup",
            "general": "general",
        }
        key = template_map.get(template, "default")
        prompt_config = _CHAT_PROMPTS.get(key, _CHAT_PROMPTS["default"])
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
    ) -> str:
        """获取指定模板的 user prompt（填充变量）"""
        template_map = {
            "default": "default",
            "comparison": "comparison",
            "dosage_followup": "dosage_followup",
            "general": "general",
        }
        key = template_map.get(template, "default")
        prompt_config = _CHAT_PROMPTS.get(key, _CHAT_PROMPTS["default"])
        template_text = prompt_config["user"]

        # 格式化近期对话历史（所有模板都注入）
        history_text = ""
        if history:
            history_parts: list[str] = []
            for turn in history[-6:]:  # 最多保留最近 6 条
                role = "用户" if turn.get("role") == "user" else "助手"
                content = turn.get("content", "")
                history_parts.append(f"{role}: {content}")
            history_text = "近期对话：\n" + "\n".join(history_parts)

        # 格式化记忆摘要（短期记忆）
        memory_text = ""
        if memory_summary:
            memory_text = f"前序对话摘要：\n{memory_summary}\n"

        # 中期记忆文本直接使用（UserMemoryManager 已格式化）
        user_memories_text = user_memories or ""

        # 用户画像文本直接使用（UserProfileManager 已格式化）
        user_profile_text = user_profile or ""

        # 填充模板变量
        return template_text.format(
            context=context,
            question=query,
            history=history_text,
            memory_summary=memory_text,
            user_memories=user_memories_text,
            user_profile=user_profile_text,
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
            result_format="message",  # 使用 chat 格式返回 choices
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

        # DashScope 新版本可能返回 choices 或 text 两种格式
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
    ):
        """
        流式版本：逐 token yield 生成结果（用于 SSE）。

        调用 DashScope Generation API with stream=True + incremental_output=True，
        逐个产出文本 token。

        Args:
            query: 用户问题
            context_docs: 检索到的参考文档列表
            history: 对话历史
            template: 提示词模板（不传则自动检测）
            memory_summary: 早期对话的累积摘要（Phase 1）
            user_memories: 跨会话用户中期记忆文本（Phase 2）
            user_profile: 用户画像文本（Phase 3 长期记忆）

        Yields:
            str — 每次产出一个增量 token 文本
        """
        from dashscope import Generation

        if template is None:
            template = self._detect_template(query, history)

        context_text = self._format_context(context_docs)
        system_prompt = self._get_system_prompt(template)
        user_prompt = self._get_user_prompt(
            template, context_text, query, history, memory_summary, user_memories, user_profile
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.info(
            f"开始流式生成: query={query[:60]}..., template={template}, "
            f"docs={len(context_docs)}"
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
            result_format="message",  # 使用 chat 格式返回 choices
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

            # DashScope 新版本可能返回 choices 或 text 两种格式
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
) -> GeneratedAnswer:
    """
    便捷函数：一行调用生成回答。

    Args:
        query: 用户问题
        context_docs: 检索到的参考文档
        history: 对话历史（可选）
        template: 提示词模板（可选，默认自动检测）
        memory_summary: 早期对话的累积摘要（可选）

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
    )
