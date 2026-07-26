"""
疾病知识文本切分器 (v1.0.0)

针对疾病知识文档格式设计：
- 识别 Markdown 标题（## / ###）
- 识别编号列表（1. 概述 / 2. 临床表现）
- 识别关键词行（定义/概述/病因/病理/流行病学/临床表现/诊断...）
- 回退到通用分隔符切分

chunk_size 默认 800（疾病知识密度高，需更大上下文）。

使用方式:
    from app.offline.splitter_disease import split_disease_document, Chunk

    chunks = split_disease_document(disease_text)
"""

import re
from dataclasses import dataclass

from loguru import logger

from app.config import config


# 复用 splitter.py 的 Chunk dataclass
@dataclass
class Chunk:
    """切分后的文本块"""
    section: str
    chunk_text: str
    chunk_index: int
    char_count: int


# ============================================================
# 章节检测模式（按优先级排列）
# ============================================================

# 模式 1: Markdown 标题
_MD_HEADING_PATTERN = re.compile(r"^#{2,4}\s+(.+)$", re.MULTILINE)

# 模式 2: 编号列表（1. / 2.1 / 一、/ （一））
_NUMBERED_PATTERN = re.compile(
    r"^((?:\d+[\.\)]\s*)|(?:[一二三四五六七八九十]+[、）\)]\s*)|(?:（[一二三四五六七八九十]+）))(.+)$",
    re.MULTILINE,
)

# 模式 3: 关键词行（疾病教科书常用章节名）
_KEYWORD_PATTERNS = [
    r"^(定义|概述|简介|引言|前言)\b",
    r"^(病因|病原学|病理|病理生理|发病机制|病理生理学)\b",
    r"^(流行病学|流行特征|发病率|患病率|危险因素)\b",
    r"^(临床表现|症状|体征|临床特征|自然史)\b",
    r"^(诊断|诊断标准|诊断依据|影像学|影像学检查|实验室检查|辅助检查)\b",
    r"^(鉴别诊断|鉴别)\b",
    r"^(治疗|治疗原则|治疗目标|药物治疗|非药物治疗|手术治疗|治疗方案)\b",
    r"^(预后|转归|并发症|合并症)\b",
    r"^(预防|筛查|监测|随访|管理)\b",
    r"^(分类|分型|分度|分级|分期)\b",
]

_KEYWORD_REGEX = re.compile("|".join(_KEYWORD_PATTERNS), re.IGNORECASE)


# ============================================================
# 切分参数
# ============================================================
def _get_chunk_params():
    """获取切分参数，疾病知识使用稍大的 chunk_size。"""
    return {
        "chunk_size": getattr(config, 'splitter_chunk_size', 500) + 300,  # 800
        "chunk_overlap": getattr(config, 'splitter_chunk_overlap', 50) + 50,  # 100
        "min_chunk_size": getattr(config, 'splitter_min_chunk_size', 100),
        "separator": "\n\n",
    }


# ============================================================
# 章节检测
# ============================================================
def _find_sections(text: str) -> list[tuple[int, str]]:
    """
    查找所有章节标记位置，返回 [(字符位置, 章节名), ...]。

    优先级: Markdown 标题 > 编号列表 > 关键词行
    """
    sections: list[tuple[int, str]] = []

    # 模式 1: Markdown 标题
    for m in _MD_HEADING_PATTERN.finditer(text):
        sections.append((m.start(), m.group(1).strip()))

    # 模式 2: 编号列表（仅在没有 Markdown 标题时生效）
    if not sections:
        for m in _NUMBERED_PATTERN.finditer(text):
            sections.append((m.start(), m.group(2).strip()))

    # 模式 3: 关键词行（兜底）
    if not sections:
        for m in _KEYWORD_REGEX.finditer(text):
            sections.append((m.start(), m.group(0).strip()))

    return sorted(sections, key=lambda x: x[0])


# ============================================================
# 字符级切分（复用 splitter.py 逻辑）
# ============================================================
def _split_by_chars(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    section: str = "",
) -> list[Chunk]:
    """
    句子边界感知的文本切分。

    在 chunk_size 附近寻找最自然的断点，优先级:
    1. 段落边界 (\\n\\n) — 最强语义边界
    2. 句子结尾 (。！？) — 自然阅读单元
    3. 换行 (\\n) — 行级边界
    4. 子句分隔 (；，) — 弱但可用
    5. 硬切 — 在 chunk_size 处裁断（最后手段）

    只在 chunk 后半段搜索断点，避免切出过短的 chunk。
    """
    if len(text) <= chunk_size:
        return [Chunk(
            section=section,
            chunk_text=text,
            chunk_index=0,
            char_count=len(text),
        )]

    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # 只在后半段找断点: [start + chunk_size//2, end]
            search_from = start + chunk_size // 2
            best = _find_best_break(text, search_from, end)

            if best > start:
                end = best

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                section=section,
                chunk_text=chunk_text,
                chunk_index=chunk_index,
                char_count=len(chunk_text),
            ))
            chunk_index += 1

        if end >= len(text):
            break

        prev_start = start
        start = end - chunk_overlap
        # 防止死循环：确保指针至少前进 1 个字符
        if start <= prev_start:
            start = prev_start + 1

    return chunks


def _find_best_break(text: str, search_from: int, search_to: int) -> int:
    """
    在 [search_from, search_to] 范围内从后往前找最佳断点。

    Returns:
        断点位置（新 chunk 从此处开始），找不到则返回 0。
    """
    # 优先级 1: 段落边界（双换行）
    pos = text.rfind("\n\n", search_from, search_to)
    if pos >= 0:
        return pos + 2

    # 优先级 2: 句子结尾（中英文句号、问号、感叹号后跟换行或空格）
    for sep in ("。\n", "。", "！\n", "！", "？\n", "？",
                ".\n", ".\n", "!\n", "?\n"):
        pos = text.rfind(sep, search_from, search_to)
        if pos >= 0:
            return pos + len(sep)

    # 优先级 3: 换行
    pos = text.rfind("\n", search_from, search_to)
    if pos >= 0:
        return pos + 1

    # 优先级 4: 子句分隔（分号、逗号）
    for sep in ("；", "，", ";", ","):
        pos = text.rfind(sep, search_from, search_to)
        if pos >= 0:
            return pos + len(sep)

    # 优先级 5: 空格（英文词边界）
    pos = text.rfind(" ", search_from, search_to)
    if pos >= 0:
        return pos + 1

    # 找不到任何自然断点 → 返回 0，由调用方硬切
    return 0


# ============================================================
# 通用章节检测（fallback）
# ============================================================
_UNIVERSAL_HEADING_PATTERNS = [
    # 编号章节: "1. xxx", "1、xxx", "1) xxx", "1.1. xxx"
    re.compile(r"^\s*(\d+(?:[\.\)、]\d*)*\s+.+)$", re.MULTILINE),
    # 中文编号: "一、xxx", "（一）xxx", "(一) xxx"
    re.compile(r"^\s*[（\(]?[一二三四五六七八九十]+[）\)、]\s*.+$", re.MULTILINE),
    # 章/节标记: "第X章 xxx", "第X节 xxx"
    re.compile(r"^\s*(第[一二三四五六七八九十\d]+[章节部分篇]\s*.+)$", re.MULTILINE),
    # 全大写英文标题（≥4个字符）: "INTRODUCTION", "METHODS AND MATERIALS"
    re.compile(r"^\s*([A-Z][A-Z\s&]{3,40})$", re.MULTILINE),
    # 分隔线（常作为段落标记）
    re.compile(r"^\s*([-=*_]{4,})\s*$", re.MULTILINE),
]


def _find_universal_headings(text: str) -> list[tuple[int, str]]:
    """
    通用章节检测 — 当文档类型特定的结构识别失败时回退使用。

    按优先级依次匹配:
    1. 数字编号 (1. / 1、/ 1) / 1.1.)
    2. 中文编号 (一、/（一）/(一))
    3. 章节标记 (第X章 / 第X节)
    4. 全大写英文标题
    5. 分隔线 (==== / ----)

    Returns:
        [(位置, 章节名), ...] 去重且按位置排序
    """
    sections: list[tuple[int, str]] = []
    for pattern in _UNIVERSAL_HEADING_PATTERNS:
        for m in pattern.finditer(text):
            name = (m.group(1) or m.group(0)).strip()
            if len(name) >= 2:
                sections.append((m.start(), name))

    # 按位置去重（同一位置只保留第一次匹配）
    sections.sort(key=lambda x: x[0])
    seen: set[int] = set()
    unique: list[tuple[int, str]] = []
    for pos, name in sections:
        # 允许 ±5 字符的容差（不同模式可能匹配到同一行的不同位置）
        near = [p for p in seen if abs(p - pos) <= 5]
        if not near:
            seen.add(pos)
            unique.append((pos, name))

    return unique


# ============================================================
# 主切分函数
# ============================================================
def split_disease_document(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """
    切分疾病知识文档。

    流程:
    1. 检测章节标记
    2. 无章节→直接切分全文
    3. 有章节→逐章节内部切分
    4. 合并短章节 + 尾块

    Args:
        text: 疾病知识全文
        chunk_size: 每块目标字符数（默认 800）
        chunk_overlap: 块间重叠字符数（默认 100）

    Returns:
        Chunk 列表（全局 chunk_index）
    """
    params = _get_chunk_params()
    chunk_size = chunk_size or params["chunk_size"]
    chunk_overlap = chunk_overlap or params["chunk_overlap"]
    min_chunk_size = params["min_chunk_size"]

    if not text or not text.strip():
        logger.warning("空文本，返回空列表")
        return []

    logger.info(f"开始切分疾病知识文档: {len(text)} 字符")

    # 1. 检测章节
    sections = _find_sections(text)

    if not sections:
        # Fallback: 通用章节检测
        sections = _find_universal_headings(text)
        if sections:
            logger.info(f"通用章节检测发现 {len(sections)} 个章节标记")

    if not sections:
        # 完全无结构 → 全文切分
        logger.info("未检测到任何章节标记，全文切分")
        return _split_by_chars(text, chunk_size, chunk_overlap, section="全文")

    # 2. 逐章节切分
    all_chunks: list[Chunk] = []

    # 处理第一个章节前的前言
    if sections[0][0] > 0:
        preamble = text[:sections[0][0]].strip()
        if len(preamble) > min_chunk_size:
            all_chunks.extend(_split_by_chars(
                preamble, chunk_size, chunk_overlap,
                section="前言",
            ))

    # 逐章节处理
    for i, (pos, section_name) in enumerate(sections):
        section_start = pos
        section_end = sections[i + 1][0] if i + 1 < len(sections) else len(text)
        section_text = text[section_start:section_end].strip()

        if len(section_text) <= min_chunk_size:
            continue  # 跳过太短的章节

        # 章节内二次切分
        chunks = _split_by_chars(
            section_text, chunk_size, chunk_overlap,
            section=section_name,
        )
        all_chunks.extend(chunks)

    # 3. 回退：如果所有章节都因太短被跳过，全文切分
    if not all_chunks:
        logger.info("所有章节均过短被跳过，回退到全文切分")
        return _split_by_chars(text, chunk_size, chunk_overlap, section="全文")

    # 4. 合并短块
    merged = _merge_short_chunks(all_chunks, min_chunk_size)

    # 5. 重新编号
    for i, chunk in enumerate(merged):
        chunk.chunk_index = i

    logger.info(f"疾病知识切分完成: {len(merged)} 个 chunk")
    return merged


def _merge_short_chunks(
    chunks: list[Chunk],
    min_chunk_size: int,
) -> list[Chunk]:
    """合并过短的 chunk 到前一个块"""
    if not chunks:
        return []

    result = [chunks[0]]
    for chunk in chunks[1:]:
        if chunk.char_count < min_chunk_size and result:
            # 合并到前一个块
            prev = result[-1]
            prev.chunk_text = prev.chunk_text + "\n" + chunk.chunk_text
            prev.char_count = len(prev.chunk_text)
        else:
            result.append(chunk)
    return result


# ============================================================
# 命令行测试
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            text = f.read()

        chunks = split_disease_document(text)
        for c in chunks:
            print(f"[{c.chunk_index}] [{c.section}] ({c.char_count}chars): {c.chunk_text[:100]}...")
        print(f"\nTotal: {len(chunks)} chunks")
