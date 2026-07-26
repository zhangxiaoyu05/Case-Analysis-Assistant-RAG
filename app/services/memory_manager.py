"""
短期记忆管理器（基于 DashScope 摘要模型）

实现 token-threshold 触发的对话短期记忆：
- 每次请求估算 (history + summary + query) 的总 token 数
- 超过上下文窗口 × 阈值比例时触发摘要压缩
- 压缩后确保总 token < 上下文窗口 × 0.5
- 支持 enable_memory 开关（false 时不加载/不保存历史）

使用方式:
    from app.services.memory_manager import MemoryManager

    manager = MemoryManager()
    summary, recent = await manager.summarize(
        history, existing_summary, query, enable_memory=True
    )
"""

import re
from loguru import logger

from app.config import config

# ============================================================
# Token 估算
# ============================================================
# 尝试加载 tiktoken，不可用时回退到字符估算
_TOKENIZER = None
_TOKENIZER_LOADED = False


def _get_tokenizer():
    """懒加载 tiktoken tokenizer（cl100k_base 对中英文兼容性好）。"""
    global _TOKENIZER, _TOKENIZER_LOADED
    if _TOKENIZER_LOADED:
        return _TOKENIZER
    _TOKENIZER_LOADED = True
    try:
        import tiktoken
        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
        logger.info("Token 估算: 使用 tiktoken cl100k_base")
    except Exception:
        logger.info("Token 估算: tiktoken 不可用，使用字符估算（~1.5 chars/token）")
    return _TOKENIZER


def estimate_tokens(text: str) -> int:
    """
    估算文本的 token 数量。

    优先使用 tiktoken，不可用时回退到启发式估算：
    - CJK 字符：~1.5 chars/token
    - ASCII 字符：~4 chars/token
    """
    if not text:
        return 0
    tokenizer = _get_tokenizer()
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    # 回退：字符估算
    cjk_chars = len(re.findall(r'[一-鿿　-〿＀-￯]', text))
    ascii_chars = len(re.findall(r'[a-zA-Z0-9\s]', text))
    other_chars = len(text) - cjk_chars - ascii_chars
    return int(cjk_chars / 1.5 + ascii_chars / 4 + other_chars / 2)


def estimate_tokens_for_messages(messages: list[dict]) -> int:
    """估算消息列表的总 token 数。"""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        total += estimate_tokens(content)
        # 每条消息的 role + 格式开销约 4 tokens
        total += 4
    return total


# ============================================================
# 摘要提示词
# ============================================================
_SUMMARIZE_SYSTEM_PROMPT = """你是一个对话摘要助手。你的任务是将用户与临床病例分析助手之间的历史对话压缩为简洁的摘要。

摘要要求：
1. **关键信息保留**：记录用户讨论的疾病、症状、鉴别诊断、检查结果、用药方案、指南推荐，以及助手给出的关键分析
2. **时序关系**：如果用户有追问，保留追问的逻辑链条（先问了什么→又追问了什么）
3. **精简**：去除寒暄、感谢等非信息性内容，只保留与临床病例分析相关的实质信息
4. **格式**：使用中文，2-5句话即可，不要超过300字
5. **如果已有前序摘要**：将新对话信息合并到前序摘要中，避免重复，形成连贯的对话历史摘要"""

_SUMMARIZE_USER_PROMPT = """{previous_summary}需要摘要的对话轮次：
{turns_to_summarize}

请生成简洁的对话摘要。"""


# ============================================================
# MemoryManager
# ============================================================
class MemoryManager:
    """
    短期记忆管理器。

    Phase 1: 改为按 token 阈值触发摘要（不再按对话轮数），
    支持 enable_memory 开关。

    使用方式:
        manager = MemoryManager()
        summary, recent = await manager.summarize(
            history, existing_summary, query, enable_memory=True
        )
    """

    def __init__(
        self,
        model: str | None = None,
        recent_turns: int | None = None,
        max_summary_chars: int | None = None,
        api_key: str | None = None,
        threshold_ratio: float | None = None,
        context_window_tokens: int | None = None,
    ) -> None:
        """
        Args:
            model: 摘要模型（默认 qwen-flash）
            recent_turns: 保留最近 N 轮完整对话（默认从 config 读取）
            max_summary_chars: 摘要最大字符数（默认从 config 读取）
            api_key: DashScope API Key（默认从 config 读取）
            threshold_ratio: token 阈值触发比例（默认从 config 读取）
            context_window_tokens: 上下文窗口 token 数（默认从 config 读取）
        """
        self._model = model or "qwen-flash"
        self._recent_turns = recent_turns if recent_turns is not None else config.memory_recent_turns
        self._max_summary_chars = max_summary_chars or config.memory_max_summary_chars
        self._api_key = api_key or config.DASHSCOPE_API_KEY
        self._threshold_ratio = threshold_ratio if threshold_ratio is not None else config.memory_token_threshold_ratio
        self._context_window = context_window_tokens if context_window_tokens is not None else config.memory_context_window_tokens

    # ----------------------------------------------------------
    # 公共 API
    # ----------------------------------------------------------
    async def summarize(
        self,
        history: list[dict],
        existing_summary: str = "",
        query: str = "",
        enable_memory: bool = True,
    ) -> tuple[str, list[dict]]:
        """
        对对话历史进行摘要压缩（token 阈值触发）。

        Args:
            history: 完整对话历史 [{"role": "user"/"assistant", "content": "..."}]
            existing_summary: 已有的前序摘要（用于累积合并）
            query: 当前用户问题（用于估算总 token 数）
            enable_memory: 用户是否启用短期记忆

        Returns:
            (summary_text, recent_history) —
            summary_text: 累积摘要文本（包含前序摘要 + 新压缩的旧轮次）
            recent_history: 保留的最近 N 轮完整消息
        """
        # ---- 全局/用户级关闭 ----
        if not config.memory_enabled or not enable_memory:
            logger.debug("短期记忆已禁用（全局或用户级），返回空历史")
            return "", []

        total_entries = len(history)

        # 空历史直接返回
        if total_entries == 0:
            return existing_summary, history

        # ---- Token 阈值判断 ----
        threshold = int(self._context_window * self._threshold_ratio)
        current_tokens = (
            estimate_tokens_for_messages(history)
            + estimate_tokens(existing_summary)
            + estimate_tokens(query)
        )

        logger.debug(
            f"Token 估算: history={estimate_tokens_for_messages(history)}, "
            f"summary={estimate_tokens(existing_summary)}, "
            f"query={estimate_tokens(query)}, "
            f"total={current_tokens}, threshold={threshold}"
        )

        if current_tokens < threshold:
            logger.debug(
                f"总 token {current_tokens} < 阈值 {threshold}，不触发摘要"
            )
            return existing_summary, history

        # ---- 触发摘要压缩 ----
        # 目标：压缩后总 token < 上下文窗口 × 0.5
        target_tokens = int(self._context_window * 0.5)

        # 计算需要压缩多少轮：从最旧的轮次开始逐轮移除，
        # 直到 estimated(remaining_history) + estimated(summary) + estimated(query) < target
        entries_to_keep = total_entries
        for i in range(2, total_entries + 1, 2):  # 每次移除一轮（2条）
            remaining = history[i:]
            estimated = (
                estimate_tokens_for_messages(remaining)
                + estimate_tokens(existing_summary)
                + estimate_tokens(query)
                # 预留摘要空间
                + 300  # 新摘要的估算 token 数
            )
            if estimated < target_tokens:
                entries_to_keep = len(remaining)
                break
            entries_to_keep = len(remaining)
        else:
            # 即使只剩最后几轮也超过 target → 保留最后 2 轮
            entries_to_keep = min(4, total_entries)

        # 确保至少保留 2 轮（4 条）
        entries_to_keep = max(entries_to_keep, min(4, total_entries))

        old_entries = history[:-entries_to_keep] if entries_to_keep < total_entries else []
        recent_entries = history[-entries_to_keep:]

        if not old_entries:
            logger.debug("无可压缩的旧轮次，保留全部历史")
            return existing_summary, history

        logger.info(
            f"触发摘要: 总 {total_entries} 条 ({current_tokens} tokens) → "
            f"压缩 {len(old_entries)} 条旧轮次, 保留 {len(recent_entries)} 条 "
            f"(目标 < {target_tokens} tokens)"
        )

        # 格式化旧轮次为可读文本
        turns_text = self._format_turns(old_entries)

        # 构建前序摘要前缀
        prev_summary_text = ""
        if existing_summary:
            prev_summary_text = f"前序对话摘要：\n{existing_summary}\n\n"

        # 调用摘要模型
        try:
            summary = await self._call_summarization(
                previous_summary=prev_summary_text,
                turns_text=turns_text,
            )
            # 截断到最大长度
            if len(summary) > self._max_summary_chars:
                summary = summary[:self._max_summary_chars] + "…"
            logger.info(f"摘要完成: {len(summary)} 字符 ({estimate_tokens(summary)} tokens)")
        except Exception as e:
            logger.error(f"摘要失败: {e}，回退到截断旧轮次")
            summary = self._fallback_summary(old_entries, existing_summary)

        return summary, recent_entries

    # ----------------------------------------------------------
    # 格式化
    # ----------------------------------------------------------
    @staticmethod
    def _format_turns(entries: list[dict]) -> str:
        """将消息列表格式化为可读的对话文本。"""
        lines: list[str] = []
        for entry in entries:
            role = entry.get("role", "unknown")
            content = entry.get("content", "")
            # 截断过长的单条消息
            if len(content) > 500:
                content = content[:500] + "…"
            role_label = "用户" if role == "user" else "助手"
            lines.append(f"[{role_label}]: {content}")
        return "\n".join(lines)

    # ----------------------------------------------------------
    # 摘要生成
    # ----------------------------------------------------------
    async def _call_summarization(
        self,
        previous_summary: str,
        turns_text: str,
    ) -> str:
        """调用 DashScope qwen-flash 生成摘要。"""
        from dashscope import Generation

        user_prompt = _SUMMARIZE_USER_PROMPT.format(
            previous_summary=previous_summary,
            turns_to_summarize=turns_text,
        )

        messages = [
            {"role": "system", "content": _SUMMARIZE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # 在线程池中运行同步 DashScope API
        import asyncio
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: Generation.call(
                model=self._model,
                messages=messages,
                temperature=0.2,
                max_tokens=600,
                api_key=self._api_key,
                result_format="message",
            ),
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"摘要 API 错误: status={response.status_code}, "
                f"message={response.message}"
            )

        output = response.output
        if output is None:
            raise RuntimeError("摘要 API 返回了空的 output")

        if output.choices:
            return output.choices[0].message.content
        elif output.text:
            return output.text
        else:
            raise RuntimeError("摘要 API 返回了空的 choices 和 text")

    # ----------------------------------------------------------
    # 回退摘要
    # ----------------------------------------------------------
    def _fallback_summary(
        self,
        old_entries: list[dict],
        existing_summary: str = "",
    ) -> str:
        """
        当 LLM 摘要失败时的简单回退：
        提取旧轮次中用户的问题作为简要摘要。
        """
        user_questions: list[str] = []
        for entry in old_entries:
            if entry.get("role") == "user":
                content = entry.get("content", "")
                if len(content) > 100:
                    content = content[:100] + "…"
                user_questions.append(content)

        if not user_questions:
            return existing_summary

        fallback = "用户曾询问：" + "；".join(user_questions[-5:])

        if existing_summary:
            return existing_summary + "\n" + fallback

        return fallback


# ============================================================
# 便捷函数
# ============================================================
async def summarize_history(
    history: list[dict],
    existing_summary: str = "",
    query: str = "",
    enable_memory: bool = True,
    recent_turns: int = 4,
) -> tuple[str, list[dict]]:
    """
    便捷函数：一行调用对话摘要。

    Args:
        history: 完整对话历史
        existing_summary: 前序摘要（可选）
        query: 当前用户问题（用于 token 估算）
        enable_memory: 是否启用记忆
        recent_turns: 保留最近 N 轮

    Returns:
        (summary, recent_history)
    """
    manager = MemoryManager(recent_turns=recent_turns)
    return await manager.summarize(history, existing_summary, query, enable_memory)
