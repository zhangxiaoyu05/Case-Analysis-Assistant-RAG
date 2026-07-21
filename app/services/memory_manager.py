"""
短期记忆管理器（基于 DashScope 摘要模型）

实现"滑动窗口 + 累积摘要"模式的对话短期记忆：
- 保留最近 N 轮完整对话（recent_turns）
- 旧轮次通过 qwen-flash 压缩为摘要
- 摘要累积更新，跟随对话推进持续合并

使用方式:
    from app.services.memory_manager import MemoryManager

    manager = MemoryManager()
    summary, recent = await manager.summarize(history, existing_summary)
"""

from loguru import logger

from app.config import config

# ============================================================
# 摘要提示词
# ============================================================
_SUMMARIZE_SYSTEM_PROMPT = """你是一个对话摘要助手。你的任务是将用户与药品知识问答助手之间的历史对话压缩为简洁的摘要。

摘要要求：
1. **关键信息保留**：记录用户问过的药品名、症状、关注点，以及助手给出的关键回答
2. **时序关系**：如果用户有追问，保留追问的逻辑链条（先问了什么→又追问了什么）
3. **精简**：去除寒暄、感谢等非信息性内容，只保留与药品知识相关的实质信息
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

    使用 DashScope qwen-flash（快速、便宜）将旧对话轮次压缩为累积摘要，
    保留最近 N 轮完整内容，最大化上下文窗口利用率。

    使用方式:
        manager = MemoryManager()
        summary, recent = await manager.summarize(history, existing_summary)
    """

    def __init__(
        self,
        model: str | None = None,
        recent_turns: int | None = None,
        max_summary_chars: int | None = None,
        api_key: str | None = None,
    ) -> None:
        """
        Args:
            model: 摘要模型（默认 qwen-flash）
            recent_turns: 保留最近 N 轮完整对话（默认 4）
            max_summary_chars: 摘要最大字符数（默认 800）
            api_key: DashScope API Key（默认从 config 读取）
        """
        self._model = model or "qwen-flash"
        self._recent_turns = recent_turns if recent_turns is not None else 4
        self._max_summary_chars = max_summary_chars or 800
        self._api_key = api_key or config.DASHSCOPE_API_KEY

    # ----------------------------------------------------------
    # 公共 API
    # ----------------------------------------------------------
    async def summarize(
        self,
        history: list[dict],
        existing_summary: str = "",
    ) -> tuple[str, list[dict]]:
        """
        对对话历史进行摘要压缩。

        如果历史轮数 ≤ recent_turns，直接返回（不触发摘要）。
        否则，将超出部分压缩为摘要，保留最近 recent_turns 轮。

        Args:
            history: 完整对话历史 [{"role": "user"/"assistant", "content": "..."}]
            existing_summary: 已有的前序摘要（用于累积合并）

        Returns:
            (summary_text, recent_history) —
            summary_text: 累积摘要文本（包含前序摘要 + 新压缩的旧轮次）
            recent_history: 保留的最近 N 轮完整消息
        """
        total_entries = len(history)
        max_recent_entries = self._recent_turns * 2  # 每轮 2 条（user + assistant）

        # 不触发摘要：历史不够长
        if total_entries <= max_recent_entries:
            logger.debug(
                f"历史 {total_entries} 条 ≤ {max_recent_entries} 条，不触发摘要"
            )
            return existing_summary, history

        # 分割：旧轮次（需摘要） + 最近轮次（保留完整）
        old_entries = history[:-max_recent_entries]
        recent_entries = history[-max_recent_entries:]

        logger.info(
            f"触发摘要: 总 {total_entries} 条 → 摘要 {len(old_entries)} 条旧轮次, "
            f"保留 {len(recent_entries)} 条最近轮次"
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
            logger.info(f"摘要完成: {len(summary)} 字符")
        except Exception as e:
            logger.error(f"摘要失败: {e}，回退到截断旧轮次")
            # 回退：简单拼接前 N 条旧轮次的关键内容
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
                # 截断长问题
                if len(content) > 100:
                    content = content[:100] + "…"
                user_questions.append(content)

        if not user_questions:
            return existing_summary

        fallback = "用户曾询问：" + "；".join(user_questions[-5:])  # 最多保留5个问题

        if existing_summary:
            return existing_summary + "\n" + fallback

        return fallback


# ============================================================
# 便捷函数
# ============================================================
async def summarize_history(
    history: list[dict],
    existing_summary: str = "",
    recent_turns: int = 4,
) -> tuple[str, list[dict]]:
    """
    便捷函数：一行调用对话摘要。

    Args:
        history: 完整对话历史
        existing_summary: 前序摘要（可选）
        recent_turns: 保留最近 N 轮

    Returns:
        (summary, recent_history)
    """
    manager = MemoryManager(recent_turns=recent_turns)
    return await manager.summarize(history, existing_summary)
