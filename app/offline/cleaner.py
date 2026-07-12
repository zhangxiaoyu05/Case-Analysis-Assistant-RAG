"""
文本清洗器

对药品说明书原文进行规范化处理：
1. 基础清洗（始终执行）：空白规范化、PDF 伪影去除、Unicode 规范化
2. 可选脱敏（按需开启）：调用 LLM 去除个人隐私信息

使用方式:
    from app.offline.cleaner import clean_text

    cleaned = clean_text(raw_text)
    cleaned = clean_text(raw_text, desensitize=True)  # 含脱敏
"""

import re
from typing import Optional

from loguru import logger
from yaml import safe_load

from app.config import config

from app.config import config, PROJECT_ROOT


# ============================================================
# 基础清洗 — 空白规范化
# ============================================================
def _normalize_whitespace(text: str) -> str:
    """规范化空白字符"""
    # 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 删除零宽字符
    text = re.sub(r"[​‌‍﻿]", "", text)

    # 3+ 连续换行 → 2 换行（保留段落分隔）
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 行内多余空格 → 单一空格
    lines = []
    for line in text.split("\n"):
        # 合并行内连续空格/tab
        line = re.sub(r"[ \t]+", " ", line)
        line = line.strip()
        lines.append(line)

    return "\n".join(lines)


# ============================================================
# 基础清洗 — PDF 伪影去除
# ============================================================
def _remove_pdf_artifacts(text: str) -> str:
    """去除 PDF 常见的页码、页眉页脚等伪影"""
    # 太短的文本不处理（避免误删）
    if len(text) < 30:
        return text

    lines = text.split("\n")
    cleaned_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # 纯页码行: "1", "  123  ", "第 3 页"
        if re.match(r"^\s*\d{1,4}\s*$", stripped):
            continue

        # 中文页码: "第X页", "第 X 页", "第X页/共Y页"
        if re.match(r"^\s*第\s*\d+\s*页\s*(/?\s*共?\s*\d*\s*页?\s*)?\s*$", stripped):
            continue

        # 英文页码: "Page X", "Page X of Y"
        if re.match(r"^\s*[Pp]age\s*\d+\s*(of\s*\d+)?\s*$", stripped):
            continue

        # 常见页眉/页脚: 机构名称 + 页码
        if re.match(r"^\s*第\s*\d+\s*页\s*/\s*共\s*\d+\s*页\s*", stripped):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


# ============================================================
# 基础清洗 — Unicode 规范化
# ============================================================
def _normalize_unicode(text: str) -> str:
    """
    规范化 Unicode 字符。

    将全角 ASCII 字符（数字、字母）转为半角，
    保留中文全角标点符号。
    """
    result: list[str] = []
    for ch in text:
        code = ord(ch)
        # 全角数字 ０-９ (0xFF10-0xFF19) → 半角 0-9 (0x30-0x39)
        if 0xFF10 <= code <= 0xFF19:
            result.append(chr(code - 0xFF10 + 0x30))
        # 全角大写字母 Ａ-Ｚ (0xFF21-0xFF3A) → 半角 A-Z (0x41-0x5A)
        elif 0xFF21 <= code <= 0xFF3A:
            result.append(chr(code - 0xFF21 + 0x41))
        # 全角小写字母 ａ-ｚ (0xFF41-0xFF5A) → 半角 a-z (0x61-0x7A)
        elif 0xFF41 <= code <= 0xFF5A:
            result.append(chr(code - 0xFF41 + 0x61))
        else:
            result.append(ch)

    return "".join(result)


# ============================================================
# 可选脱敏
# ============================================================
def _load_desensitization_prompt() -> tuple[str, str]:
    """
    从 prompts.yaml 加载脱敏提示词。

    Returns:
        (system_prompt, user_prompt_template)
    """
    prompts_path = PROJECT_ROOT / "config" / "prompts.yaml"
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = safe_load(f)

    desensitize_config = prompts.get("desensitization", {})
    system_prompt = desensitize_config.get("system", "")
    user_template = desensitize_config.get("user", "{text}")

    if not system_prompt:
        raise ValueError("prompts.yaml 中缺少 desensitization.system 提示词")

    return system_prompt, user_template


def _desensitize_chunk(text: str, api_key: str, system_prompt: str, user_template: str) -> str:
    """
    调用 DashScope Generation API 对单段文本进行脱敏。

    Args:
        text: 待脱敏文本
        api_key: DashScope API Key
        system_prompt: 系统提示词
        user_template: 用户提示词模板（{text} 占位）

    Returns:
        脱敏后的文本
    """
    from dashscope import Generation

    response = Generation.call(
        model=config.intent_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_template.format(text=text)},
        ],
        temperature=0.1,
        max_tokens=2000,
        result_format="text",
        api_key=api_key,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"脱敏 API 调用失败: status={response.status_code}, "
            f"message={response.message}"
        )

    return response.output.text.strip()


# ============================================================
# 公共 API
# ============================================================
def clean_text(
    text: str,
    desensitize: bool = False,
    api_key: Optional[str] = None,
    desensitize_chunk_size: int = 2000,
) -> str:
    """
    清洗文本。

    Args:
        text: 原始文本
        desensitize: 是否启用 LLM 脱敏（默认 False）
        api_key: DashScope API Key（默认从 config 读取）
        desensitize_chunk_size: 脱敏时每段的最大字符数

    Returns:
        清洗后的文本
    """
    if not text.strip():
        logger.warning("输入文本为空，跳过清洗")
        return text

    # ---- 阶段 1: 基础清洗（始终执行） ----
    logger.info("开始基础清洗...")
    text = _normalize_whitespace(text)
    text = _remove_pdf_artifacts(text)
    text = _normalize_unicode(text)

    # 清理后检查
    if not text.strip():
        logger.warning("基础清洗后文本为空")
        return text

    logger.info(f"基础清洗完成: {len(text)} 字符")

    # ---- 阶段 2: 可选脱敏 ----
    if desensitize:
        logger.info("开始 LLM 脱敏处理...")
        if api_key is None:
            api_key = config.DASHSCOPE_API_KEY

        if not api_key:
            logger.error("DASHSCOPE_API_KEY 未配置，跳过脱敏")
            return text

        try:
            system_prompt, user_template = _load_desensitization_prompt()
        except Exception as e:
            logger.error(f"加载脱敏提示词失败: {e}，跳过脱敏")
            return text

        # 分段脱敏（避免超出 token 限制）
        chunks: list[str] = []
        for i in range(0, len(text), desensitize_chunk_size):
            chunk = text[i : i + desensitize_chunk_size]
            try:
                cleaned_chunk = _desensitize_chunk(chunk, api_key, system_prompt, user_template)
                chunks.append(cleaned_chunk)
            except Exception as e:
                logger.warning(f"脱敏失败 (位置 {i}-{i + desensitize_chunk_size}): {e}，保留原文")
                chunks.append(chunk)

        text = "".join(chunks)
        logger.info(f"脱敏完成: {len(text)} 字符")

    return text
