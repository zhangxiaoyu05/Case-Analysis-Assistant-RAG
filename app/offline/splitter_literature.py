"""
学术文献文本切分器 (v1.0.0)

针对学术文献的 IMRaD 结构设计：
- 识别 IMRaD 章节（Introduction / Methods / Results / Discussion / Conclusion）
- 中文文献（引言/背景 / 方法/资料与方法 / 结果 / 讨论 / 结论）
- 无结构识别时按摘要/正文/参考文献三大段切分

chunk_size 默认 800。

使用方式:
    from app.offline.splitter_literature import split_literature_document

    chunks = split_literature_document(literature_text)
"""

import re
from app.offline.splitter_disease import (
    Chunk,
    _split_by_chars,
    _merge_short_chunks,
    _find_universal_headings,
)


# ============================================================
# IMRaD 章节模式
# ============================================================

# 英文 IMRaD
_ENGLISH_IMRAD = [
    (r"^(Introduction|Background)\b", "Introduction"),
    (r"^(Methods?|Materials?\s*(and|&)\s*Methods?|Methodology|Experimental\s*Section|Study\s*Design)\b", "Methods"),
    (r"^(Results?|Findings)\b", "Results"),
    (r"^(Discussion|General\s*Discussion)\b", "Discussion"),
    (r"^(Conclusions?|Concluding\s*Remarks|Summary)\b", "Conclusion"),
    (r"^(Abstract|Summary)\b", "Abstract"),
    (r"^(References?|Bibliography|Literature\s*Cited)\b", "References"),
    (r"^(Acknowledgments?|Funding|Disclosure|Conflict\s*of\s*Interest)\b", "Appendix"),
    (r"^(Supplementary|Supporting\s*Information|Appendix|Appendices)\b", "Appendix"),
]

# 中文 IMRaD
_CHINESE_IMRAD = [
    (r"^(引言|前言|背景|研究背景)\b", "Introduction"),
    (r"^(方法|资料与方法|材料与方法|研究方法|实验方法|研究设计|对象与方法|病例与方法)\b", "Methods"),
    (r"^(结果|研究结果|实验结果|临床结果)\b", "Results"),
    (r"^(讨论|分析与讨论|总结与讨论|结果与讨论)\b", "Discussion"),
    (r"^(结论|小结|总结|研究结论)\b", "Conclusion"),
    (r"^(摘要|概要|内容提要)\b", "Abstract"),
    (r"^(参考文献|文献|参考资料|文献列表)\b", "References"),
    (r"^(致谢|基金|资助|利益冲突|作者贡献|声明)\b", "Appendix"),
    (r"^(附录|补充材料|附加文件)\b", "Appendix"),
]

# 二级子章节
_SUB_SECTIONS = [
    (r"^(Patients?|Participants?|Subjects?|Study\s*Population|研究对象|纳入标准|排除标准)\b", "Methods-Population"),
    (r"^(Statistical\s*Analysis|数据分析|统计学方法|统计方法)\b", "Methods-Statistics"),
    (r"^(Outcomes?|Endpoints?|结局指标|观察指标|终点指标)\b", "Methods-Outcomes"),
    (r"^(Baseline\s*Characteristics?|基线特征|基线资料|一般资料)\b", "Results-Baseline"),
    (r"^(Primary\s*Outcome|主要结局|主要终点)\b", "Results-Primary"),
    (r"^(Secondary\s*Outcome|次要结局|次要终点|亚组分析|Subgroup)\b", "Results-Secondary"),
    (r"^(Adverse\s*Events?|Safety|不良反应|安全性|不良事件)\b", "Results-Safety"),
    (r"^(Limitations?|局限性|不足|缺陷)\b", "Discussion-Limitations"),
    (r"^(Strengths?|优势|优势与不足)\b", "Discussion-Strengths"),
    (r"^(Clinical\s*Implications?|临床意义|临床应用)\b", "Discussion-Implications"),
]


def _find_literature_sections(text: str) -> list[tuple[int, str]]:
    """
    查找文献章节标记。

    优先级: 英文 IMRaD > 中文 IMRaD > 二级子章节
    """
    sections: list[tuple[int, str]] = []

    # 合并所有模式
    all_patterns = [(p, name) for p, name in _ENGLISH_IMRAD]
    all_patterns += [(p, name) for p, name in _CHINESE_IMRAD]
    all_patterns += [(p, name) for p, name in _SUB_SECTIONS]

    # 按行扫描，匹配第一个命中的模式
    lines = text.split("\n")
    pos = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            pos += len(line) + 1
            continue

        best_name = None
        for pattern, name in all_patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                best_name = name
                break

        if best_name:
            # 确保匹配位置正确
            line_start = text.find(stripped, pos)
            if line_start < 0:
                line_start = pos
            sections.append((line_start, best_name))

        pos += len(line) + 1

    return sorted(sections, key=lambda x: x[0])


# ============================================================
# 证据级别推断（牛津循证医学中心分级）
# ============================================================
def _infer_evidence_level(study_type: str) -> str:
    """
    根据研究类型推断牛津证据级别。

    Args:
        study_type: RCT / meta-analysis / systematic_review / cohort / etc.

    Returns:
        证据级别字符串（1a/1b/2a/2b/3a/3b/4/5）
    """
    mapping = {
        "meta-analysis": "1a",
        "systematic_review": "1a",
        "rct": "1b",
        "cohort": "2b",
        "case_control": "3b",
        "case_series": "4",
        "case_report": "4",
        "expert_opinion": "5",
    }
    return mapping.get(study_type.lower(), "5")


# ============================================================
# 主切分函数
# ============================================================
def split_literature_document(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """
    切分学术文献文档。

    Args:
        text: 文献全文
        chunk_size: 每块目标字符数（默认 800）
        chunk_overlap: 块间重叠字符数（默认 100）

    Returns:
        Chunk 列表
    """
    if not text or not text.strip():
        logger = __import__('loguru').logger
        logger.warning("空文献文本，返回空列表")
        return []

    logger = __import__('loguru').logger
    logger.info(f"开始切分学术文献: {len(text)} 字符")

    sections = _find_literature_sections(text)

    min_chunk_size = 100
    all_chunks: list[Chunk] = []

    if not sections:
        # Fallback: 通用章节检测
        universal = _find_universal_headings(text)
        if universal:
            logger = __import__('loguru').logger
            logger.info(f"通用章节检测发现 {len(universal)} 个标记")
            sections = universal

    if not sections:
        # 无 IMRaD 结构：按摘要/正文/参考文献三段切
        logger = __import__('loguru').logger
        logger.info("未检测到 IMRaD 结构，按三段式切分")
        return _split_by_fallback(text, chunk_size, chunk_overlap)

    # 前言
    if sections[0][0] > 0:
        preamble = text[:sections[0][0]].strip()
        if len(preamble) > min_chunk_size:
            all_chunks.extend(_split_by_chars(
                preamble, chunk_size, chunk_overlap,
                section="Preamble",
            ))

    # 逐章节处理
    for i, (pos, section_name) in enumerate(sections):
        section_start = pos
        section_end = sections[i + 1][0] if i + 1 < len(sections) else len(text)
        section_text = text[section_start:section_end].strip()

        if len(section_text) <= min_chunk_size:
            continue

        chunks = _split_by_chars(
            section_text, chunk_size, chunk_overlap,
            section=section_name,
        )
        all_chunks.extend(chunks)

    # 回退：如果所有章节都因太短被跳过，全文切分
    if not all_chunks:
        logger.info("所有文献章节均过短被跳过，回退到全文切分")
        return _split_by_chars(text, chunk_size, chunk_overlap, section="FullText")

    # 合并短块 + 重新编号
    merged = _merge_short_chunks(all_chunks, min_chunk_size)
    for i, chunk in enumerate(merged):
        chunk.chunk_index = i

    logger.info(f"文献切分完成: {len(merged)} 个 chunk")
    return merged


def _split_by_fallback(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """
    无 IMRaD 结构时的三段式回退切分：
    摘要段 → 正文段 → 参考文献段
    """
    import re

    chunks: list[Chunk] = []
    idx = 0

    # 尝试找 References/参考文献 分段
    ref_match = re.search(
        r"\n(References?|Bibliography|参考文献|文献列表)\s*\n",
        text, re.IGNORECASE,
    )
    if ref_match:
        ref_start = ref_match.start()

        # 正文部分 (含摘要)
        body_text = text[:ref_start].strip()
        ref_text = text[ref_start:].strip()

        # 正文按章节切
        body_chunks = _split_by_chars(body_text, chunk_size, chunk_overlap, section="Body")
        for c in body_chunks:
            c.chunk_index = idx
            idx += 1
        chunks.extend(body_chunks)

        # 参考文献
        ref_chunks = _split_by_chars(ref_text[:chunk_size * 3], chunk_size, chunk_overlap, section="References")
        for c in ref_chunks:
            c.chunk_index = idx
            idx += 1
        chunks.extend(ref_chunks)
    else:
        # 无法识别分段，全文切分
        chunks = _split_by_chars(text, chunk_size, chunk_overlap, section="FullText")

    return chunks


# ============================================================
# 命令行测试
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            text = f.read()

        chunks = split_literature_document(text)
        for c in chunks:
            print(f"[{c.chunk_index}] [{c.section}] ({c.char_count}chars): {c.chunk_text[:100]}...")
        print(f"\nTotal: {len(chunks)} chunks")
