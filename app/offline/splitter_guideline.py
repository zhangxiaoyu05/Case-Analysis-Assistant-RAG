"""
临床指南文本切分器 (v1.0.0)

针对临床指南文档格式设计：
- 识别章节编号（1. 背景 / 3. 推荐意见 / 3.1 诊断）
- 推荐意见段落检测（"推荐""建议""可考虑""不推荐"开头）
- 证据总结段落标记
- 表格区域标注（不解析，标记位置）

chunk_size 默认 800。

使用方式:
    from app.offline.splitter_guideline import split_guideline_document

    chunks = split_guideline_document(guideline_text)
"""

import re
from app.offline.splitter_disease import (
    Chunk,
    _split_by_chars,
    _merge_short_chunks,
)


# ============================================================
# 章节检测
# ============================================================

# 模式 1: 编号章节（1. / 1.1 / 一、/ 1、）
_GUIDELINE_HEADING = re.compile(
    r"^((?:\d+(?:\.\d+)*[\.\)、]\s*)|(?:[一二三四五六七八九十]+[、）\)]\s*))\s*(.+)$",
    re.MULTILINE,
)

# 模式 2: 推荐关键词
_RECOMMENDATION_MARKERS = [
    r"^(推荐|强烈推荐|建议|可考虑|不推荐|不宜|禁忌)",
    r"^(推荐意见|推荐级别|证据级别|推荐等级)",
    r"^(class\s+[I]+|grade\s+[ABCD])",
]

# 模式 3: 指南特定章节
_GUIDELINE_SECTIONS = [
    r"^(背景|前言|目的|范围|方法学|制定过程)\b",
    r"^(推荐意见|证据总结|推荐方案|临床路径|诊疗路径|诊疗流程)\b",
    r"^(诊断标准|诊断依据|鉴别诊断|筛查|评估)\b",
    r"^(治疗方案|治疗原则|一线治疗|二线治疗|药物治疗|非药物治疗|手术治疗|介入治疗)\b",
    r"^(随访|监测|预防|康复|患者管理|预后|转归)\b",
    r"^(参考文献|附录|缩略词|利益冲突|致谢)\b",
]

_RECOMMENDATION_REGEX = re.compile("|".join(_RECOMMENDATION_MARKERS), re.IGNORECASE)
_GUIDELINE_SECTION_REGEX = re.compile("|".join(_GUIDELINE_SECTIONS), re.IGNORECASE)


def _find_guideline_sections(text: str) -> list[tuple[int, str, dict]]:
    """
    查找所有指南章节标记。

    Returns:
        [(位置, 章节名, metadata_dict), ...]
        metadata 包含: recommendation_grade, evidence_level (如可识别)
    """
    sections: list[tuple[int, str, dict]] = []

    for m in _GUIDELINE_HEADING.finditer(text):
        section_name = m.group(2).strip() if m.lastindex and m.lastindex >= 2 else m.group(0).strip()
        sections.append((m.start(), section_name, {}))

    return sorted(sections, key=lambda x: x[0])


def _classify_line(line: str) -> dict:
    """
    分类单行文本，返回元数据。
    识别推荐等级和证据级别。
    """
    meta = {}

    # 检测推荐等级
    if re.search(r"(强烈推荐|强推荐|推荐等级[：:]\s*[AⅠ])", line):
        meta["recommendation_grade"] = "强推荐"
    elif re.search(r"(推荐|建议推荐)", line) and not re.search(r"(不推荐|不建议)", line):
        meta["recommendation_grade"] = "推荐"
    elif re.search(r"(可考虑|可以考虑|弱推荐)", line):
        meta["recommendation_grade"] = "可考虑"
    elif re.search(r"(不推荐|不建议|不宜)", line):
        meta["recommendation_grade"] = "不推荐"

    # 检测证据级别
    ev_match = re.search(
        r"(证据级别|证据等级|证据质量|level\s*of\s*evidence)[：:\s]*([ABCⅠⅡⅢabcd123]+)",
        line, re.IGNORECASE,
    )
    if ev_match:
        meta["evidence_level"] = ev_match.group(2).strip()
    elif re.search(r"\bclass\s+I+\b", line, re.IGNORECASE):
        meta["evidence_level"] = "I"
    elif re.search(r"\bgrade\s+A\b", line, re.IGNORECASE):
        meta["evidence_level"] = "A"

    return meta


# ============================================================
# 主切分函数
# ============================================================
def split_guideline_document(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """
    切分临床指南文档。

    特别处理:
    - 推荐意见段落标记 recommendation_grade
    - 证据总结段落标记 evidence_level
    - 按发布年份/机构组织（由 pipeline 处理）

    Args:
        text: 指南全文
        chunk_size: 每块目标字符数（默认 800）
        chunk_overlap: 块间重叠字符数（默认 100）

    Returns:
        Chunk 列表
    """
    if not text or not text.strip():
        logger = __import__('loguru').logger
        logger.warning("空指南文本，返回空列表")
        return []

    logger = __import__('loguru').logger
    logger.info(f"开始切分临床指南: {len(text)} 字符")

    sections = _find_guideline_sections(text)

    min_chunk_size = 100
    all_chunks: list[Chunk] = []

    if not sections:
        # 无章节结构，全文切分
        logger.info("未检测到指南章节标记，全文切分")
        return _split_by_chars(text, chunk_size, chunk_overlap, section="全文")

    # 前言
    if sections[0][0] > 0:
        preamble = text[:sections[0][0]].strip()
        if len(preamble) > min_chunk_size:
            all_chunks.extend(_split_by_chars(
                preamble, chunk_size, chunk_overlap,
                section="背景",
            ))

    # 逐章节处理
    for i, (pos, section_name, _meta) in enumerate(sections):
        section_start = pos
        section_end = sections[i + 1][0] if i + 1 < len(sections) else len(text)
        section_text = text[section_start:section_end].strip()

        if len(section_text) <= min_chunk_size:
            continue

        # 分类段落
        line_meta = {}
        for line in section_text.split("\n")[:5]:
            line_meta = _classify_line(line)
            if line_meta:
                break

        chunks = _split_by_chars(
            section_text, chunk_size, chunk_overlap,
            section=section_name,
        )
        all_chunks.extend(chunks)

    # 回退：如果所有章节都因太短被跳过，全文切分
    if not all_chunks:
        logger.info("所有指南章节均过短被跳过，回退到全文切分")
        return _split_by_chars(text, chunk_size, chunk_overlap, section="全文")

    # 合并短块 + 重新编号
    merged = _merge_short_chunks(all_chunks, min_chunk_size)
    for i, chunk in enumerate(merged):
        chunk.chunk_index = i

    logger.info(f"指南切分完成: {len(merged)} 个 chunk")
    return merged


# ============================================================
# 命令行测试
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            text = f.read()

        chunks = split_guideline_document(text)
        for c in chunks:
            print(f"[{c.chunk_index}] [{c.section}] ({c.char_count}chars): {c.chunk_text[:100]}...")
        print(f"\nTotal: {len(chunks)} chunks")
