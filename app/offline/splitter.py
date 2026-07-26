"""
章节感知文本切分器

针对中国药品说明书格式设计：
- 识别 【章节名】 格式的章节标记
- 章节内使用中文友好分隔符进行二次切分
- 跨章节不重叠，同一章节内应用 chunk_overlap

使用方式:
    from app.offline.splitter import split_document, Chunk

    chunks = split_document(text)
    for c in chunks:
        print(f"[{c.section}] {c.chunk_text[:50]}...")
"""

import re
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from app.config import config
from app.offline.splitter_disease import _find_universal_headings


# ============================================================
# 数据类
# ============================================================
@dataclass
class Chunk:
    """切分后的文本块"""

    section: str  # 章节名（不含【】），如 "用法用量"、"禁忌"
    chunk_text: str  # 切分的文本内容
    chunk_index: int  # 文档内全局序号（从 0 开始）
    char_count: int  # 字符数


# ============================================================
# 章节检测
# ============================================================
# 匹配中国药品说明书的章节标记: 【药品名称】、【适应症】等
_SECTION_PATTERN = re.compile(r"【(.+?)】")


def _find_sections(text: str) -> list[tuple[int, str]]:
    """
    查找所有章节标记位置。

    Args:
        text: 原始文本

    Returns:
        [(起始位置, 章节名), ...] 按位置排序，章节名不含【】且已清理空白
    """
    sections: list[tuple[int, str]] = []
    for match in _SECTION_PATTERN.finditer(text):
        name = match.group(1).strip()
        # 清理章节名内的换行/多余空白
        name = re.sub(r"\s+", "", name)
        if name:
            sections.append((match.start(), name))
    return sections


# ============================================================
# 章节拆分
# ============================================================
def _split_by_sections(text: str) -> list[dict]:
    """
    按 【】 章节标记将文本拆分为章节列表。

    Returns:
        [{"section": str, "content": str}, ...]
        文本中第一个章节标记之前的内容使用 section="__preamble__"
    """
    markers = _find_sections(text)

    if not markers:
        # Fallback: 通用章节检测
        universal = _find_universal_headings(text)
        if universal:
            logger.info(f"药品文档未检测到【】标记，通用章节检测发现 {len(universal)} 个标记")
            # 构建与 _find_sections 兼容的格式
            markers = [(pos, name) for pos, name in universal]

    if not markers:
        # 完全无章节标记，整篇作为一个章节
        logger.debug("未检测到任何章节标记，整篇作为单一段落处理")
        return [{"section": "__preamble__", "content": text.strip()}]

    sections: list[dict] = []

    # 第一个标记之前的文本
    first_pos = markers[0][0]
    if first_pos > 0:
        preamble = text[:first_pos].strip()
        if preamble:
            sections.append({"section": "__preamble__", "content": preamble})

    # 各章节内容
    for i, (pos, name) in enumerate(markers):
        start = pos  # 从【开始（包含章节标题）
        # 计算内容结束位置
        if i + 1 < len(markers):
            end = markers[i + 1][0]
        else:
            end = len(text)

        content = text[start:end].strip()
        if content:
            sections.append({"section": name, "content": content})

    logger.info(f"章节检测完成: {len(sections)} 个章节")
    for s in sections:
        logger.debug(f"  [{s['section']}]: {len(s['content'])} 字符")

    return sections


# ============================================================
# 短章节合并
# ============================================================
def _merge_short_sections(
    sections: list[dict],
    min_size: int,
) -> list[dict]:
    """
    将内容过短的章节合并到相邻章节。

    Args:
        sections: 章节列表
        min_size: 最小字符数（低于此值的章节将被合并）

    Returns:
        合并后的章节列表
    """
    if not sections:
        return sections

    merged: list[dict] = []
    i = 0

    while i < len(sections):
        section = sections[i]
        content_len = len(section["content"])

        if content_len >= min_size or len(merged) == 0:
            merged.append(section.copy())
            i += 1
        else:
            # 内容太短，合并到上一个章节
            prev = merged[-1]
            prev["content"] = prev["content"] + "\n" + section["content"]
            logger.debug(
                f"合并短章节: [{section['section']}]({content_len}字符) → [{prev['section']}]"
            )
            i += 1

    # 清理可能产生的纯空白合并（最后一段如果是短 preamble）
    if len(merged) >= 2 and len(merged[-1]["content"]) < min_size:
        last = merged.pop()
        merged[-1]["content"] = merged[-1]["content"] + "\n" + last["content"]
        logger.debug(f"尾部短章节合并: [{last['section']}] → [{merged[-1]['section']}]")

    # 丢弃 __preamble__ 仅当它是唯一章节且内容也短时
    if len(merged) == 1 and merged[0]["section"] == "__preamble__":
        pass  # 保留：至少有一个章节

    return merged


# ============================================================
# 长章节二次切分
# ============================================================
# 中文友好的分隔符优先级（从粗到细）
_SPLIT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，"]


def _find_best_break_drug(text: str, search_from: int, search_to: int) -> int:
    """
    在 [search_from, search_to] 范围内从后往前找最佳断点（药品说明书版本）。

    优先级与 _split_by_chars 一致，但额外支持 config 配置的分隔符。

    Returns:
        断点位置（新 chunk 从此处开始），找不到则返回 0。
    """
    # 优先使用配置的分隔符
    configured_sep = config.splitter_separator
    if configured_sep:
        pos = text.rfind(configured_sep, search_from, search_to)
        if pos >= 0:
            return pos + len(configured_sep)

    # 优先级 1: 段落边界
    pos = text.rfind("\n\n", search_from, search_to)
    if pos >= 0:
        return pos + 2

    # 优先级 2: 句子结尾
    for sep in ("。\n", "。", "！\n", "！", "？\n", "？",
                ".\n", "!\n", "?\n"):
        pos = text.rfind(sep, search_from, search_to)
        if pos >= 0:
            return pos + len(sep)

    # 优先级 3: 换行
    pos = text.rfind("\n", search_from, search_to)
    if pos >= 0:
        return pos + 1

    # 优先级 4: 子句分隔
    for sep in ("；", "，", ";", ","):
        pos = text.rfind(sep, search_from, search_to)
        if pos >= 0:
            return pos + len(sep)

    # 优先级 5: 空格
    pos = text.rfind(" ", search_from, search_to)
    if pos >= 0:
        return pos + 1

    return 0


def _split_long_section(
    section_name: str,
    content: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
) -> list[Chunk]:
    """
    将过长的章节内容按句子边界切分成多个 chunk。

    策略：
    1. 在 chunk 后半段寻找最自然的断点（段落 > 句子 > 换行 > 子句 > 硬切）
    2. 滑动窗口控制 chunk 间重叠
    3. 尾块如果太短则合并到前一块

    Args:
        section_name: 章节名
        content: 章节内容
        chunk_size: 目标 chunk 大小
        chunk_overlap: chunk 间重叠字符数
        min_chunk_size: 最小 chunk 大小（尾块低于此值则合并）

    Returns:
        该章节的 Chunk 列表（chunk_index 在此处暂不分配全局序号）
    """
    if len(content) <= chunk_size:
        return [Chunk(
            section=section_name,
            chunk_text=content,
            chunk_index=-1,  # 调用方填充
            char_count=len(content),
        )]

    chunks: list[Chunk] = []
    start = 0

    while start < len(content):
        end = min(start + chunk_size, len(content))

        if end >= len(content):
            # 最后一段
            chunk_text = content[start:].strip()
            if chunk_text:
                chunks.append(Chunk(
                    section=section_name,
                    chunk_text=chunk_text,
                    chunk_index=-1,
                    char_count=len(chunk_text),
                ))
            break

        # 只在后半段找断点: [start + chunk_size//2, end]
        search_from = start + chunk_size // 2
        best = _find_best_break_drug(content, search_from, end)

        if best > start:
            chunk_text = content[start:best].strip()
            start = best
        else:
            # 没有自然分隔点，硬切
            chunk_text = content[start:end].strip()
            start = end

        if chunk_text:
            chunks.append(Chunk(
                section=section_name,
                chunk_text=chunk_text,
                chunk_index=-1,
                char_count=len(chunk_text),
            ))

        # 应用重叠
        start = max(0, start - chunk_overlap)

        # 防止死循环：确保前进
        if len(chunks) >= 1 and start <= chunks[-1].char_count:
            start = min(start + chunk_size, len(content))

    # 尾块合并：如果最后一块太小，合并到前一块
    if len(chunks) >= 2 and chunks[-1].char_count < min_chunk_size:
        last = chunks.pop()
        prev = chunks[-1]
        merged_text = prev.chunk_text + last.chunk_text
        chunks[-1] = Chunk(
            section=section_name,
            chunk_text=merged_text,
            chunk_index=-1,
            char_count=len(merged_text),
        )
        logger.debug(f"尾块合并: {last.char_count}字符 → 前一块 (合并后{len(merged_text)}字符)")

    return chunks


# ============================================================
# 公共 API
# ============================================================
def split_document(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    min_chunk_size: Optional[int] = None,
) -> list[Chunk]:
    """
    对药品说明书文本进行章节感知切分。

    Args:
        text: 清洗后的文本
        chunk_size: 目标 chunk 大小（默认 config.splitter_chunk_size = 500）
        chunk_overlap: chunk 间重叠字符数（默认 config.splitter_chunk_overlap = 50）
        min_chunk_size: 最小 chunk 大小，尾块低于此值则合并（默认 config.splitter_min_chunk_size = 100）

    Returns:
        Chunk 列表（chunk_index 为文档内全局序号 0..N-1）
    """
    if chunk_size is None:
        chunk_size = config.splitter_chunk_size
    if chunk_overlap is None:
        chunk_overlap = config.splitter_chunk_overlap
    if min_chunk_size is None:
        min_chunk_size = config.splitter_min_chunk_size

    logger.info(
        f"开始切分: chunk_size={chunk_size}, overlap={chunk_overlap}, "
        f"min={min_chunk_size}, 文本长度={len(text)}"
    )

    # 1. 按章节拆分
    sections = _split_by_sections(text)

    # 2. 合并过短章节
    sections = _merge_short_sections(sections, min_chunk_size)

    # 3. 对每个章节进行二次切分
    all_chunks: list[Chunk] = []
    for section in sections:
        section_chunks = _split_long_section(
            section_name=section["section"],
            content=section["content"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap if section["section"] != "__preamble__" else 0,
            min_chunk_size=min_chunk_size,
        )
        all_chunks.extend(section_chunks)

    # 4. 分配全局 chunk_index
    for i, chunk in enumerate(all_chunks):
        chunk.chunk_index = i

    # 5. 检查是否有 chunk 超过 Milvus VARCHAR(5000) 限制
    max_allowed = 5000
    for chunk in all_chunks:
        if chunk.char_count > max_allowed:
            logger.warning(
                f"Chunk [{chunk.section}] 超过 5000 字符限制 "
                f"({chunk.char_count} 字符)，将被截断"
            )
            chunk.chunk_text = chunk.chunk_text[:max_allowed]
            chunk.char_count = max_allowed

    logger.info(
        f"切分完成: {len(all_chunks)} 个 chunk, "
        f"平均 {sum(c.char_count for c in all_chunks) // max(1, len(all_chunks))} 字符/chunk"
    )

    return all_chunks
