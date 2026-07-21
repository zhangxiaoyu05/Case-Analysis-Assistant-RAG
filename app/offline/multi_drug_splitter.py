"""
多药品文档智能拆分器

检测并拆分包含多种药品说明书的合集文档，
使每种药品能够独立进入后续的清洗→切分→入库流程。

检测策略:
    1. 【药品名称】章节标记出现 ≥ 2 次
    2. "通用名称：" 模式出现 ≥ 2 次

拆分策略 (按优先级):
    1. ===== 分隔符拆分（常见于合集文档格式）
    2. 【药品名称】标记位置切分
    3. 兜底：作为单一文档返回

使用方式:
    from app.offline.multi_drug_splitter import detect_multi_drug, split_multi_drug

    if detect_multi_drug(raw_text):
        sub_docs = split_multi_drug(raw_text)
        for sub in sub_docs:
            print(f"{sub.drug_name}: {len(sub.text)} 字符")
"""

import re
from dataclasses import dataclass

from loguru import logger


# ============================================================
# 数据类
# ============================================================
@dataclass
class SubDocument:
    """拆分后的单个药品文档"""

    drug_name: str       # 提取的药品通用名称
    text: str            # 该药品的原始文本
    index: int           # 在原合集文档中的序号（从 0 开始）


# ============================================================
# 检测逻辑
# ============================================================
# 匹配章节标题 "【药品名称】"（允许前面有空白）
_DRUG_NAME_SECTION = re.compile(r"【药品名称】")

# 匹配 "通用名称：XXX" 或 "通用名称: XXX"
_GENERIC_NAME_PATTERN = re.compile(r"通用名称[：:]\s*(.+)")


def detect_multi_drug(text: str) -> bool:
    """
    检测文本是否包含多种药品说明书。

    使用双重规则（OR 关系）：
    1. 【药品名称】章节标记出现 ≥ 2 次 → 判定为合集
    2. "通用名称：" 模式出现 ≥ 2 次 → 判定为合集

    Args:
        text: 待检测的原始文本

    Returns:
        True 表示检测到多种药品，应进行拆分
    """
    if not text or not text.strip():
        return False

    # 规则 1: 【药品名称】计数
    drug_name_markers = _DRUG_NAME_SECTION.findall(text)
    if len(drug_name_markers) >= 2:
        logger.info(
            f"检测到多药品文档: 【药品名称】出现 {len(drug_name_markers)} 次"
        )
        return True

    # 规则 2: "通用名称：" 计数
    generic_names = _GENERIC_NAME_PATTERN.findall(text)
    if len(generic_names) >= 2:
        logger.info(
            f"检测到多药品文档: '通用名称：' 出现 {len(generic_names)} 次"
        )
        return True

    return False


# ============================================================
# 药名提取
# ============================================================
# 匹配标题行 "XXX说明书"（用于回退提取药名）
_TITLE_LINE_PATTERN = re.compile(r"^(.+?)说明书\s*$")


def extract_drug_name(text: str) -> str:
    """
    从单个药品的文本片段中提取通用名称。

    提取策略（按优先级）：
    1. 正则匹配 "通用名称：XXX" 取捕获组
    2. 取第一行，去掉末尾"说明书"后缀
    3. 返回空字符串（调用方负责兜底命名）

    Args:
        text: 单个药品的文本片段

    Returns:
        提取的药品名称，失败时返回 ""
    """
    if not text or not text.strip():
        return ""

    # 策略 1: 匹配 "通用名称：XXX"
    match = _GENERIC_NAME_PATTERN.search(text)
    if match:
        name = match.group(1).strip()
        # 清理药名中的多余字符（保留中文、字母、数字、括号）
        name = re.sub(r"\s+", "", name)
        if name:
            logger.debug(f"从'通用名称'字段提取药名: '{name}'")
            return name

    # 策略 2: 取第一行，去掉"说明书"后缀
    lines = text.strip().split("\n")
    if lines:
        first_line = lines[0].strip()
        # 尝试匹配 "XXX说明书"
        title_match = _TITLE_LINE_PATTERN.match(first_line)
        if title_match:
            name = title_match.group(1).strip()
            if name:
                logger.debug(f"从标题行提取药名: '{name}'")
                return name
        # 如果第一行有意义（不是分隔符、不是空行），直接用
        if first_line and not re.match(r"^=+$", first_line):
            # 去掉"说明书"后缀（如果紧跟在药名后）
            name = re.sub(r"说明书$", "", first_line).strip()
            if name and len(name) >= 2:
                logger.debug(f"从首行提取药名: '{name}'")
                return name

    return ""


# ============================================================
# 拆分逻辑
# ============================================================
# 匹配分隔符行（一整行等号）
_SEPARATOR_PATTERN = re.compile(r"^=+\s*$", re.MULTILINE)

# 匹配章节标记位置（用于回退拆分）
_SECTION_MARKER_PATTERN = re.compile(r"【药品名称】")


def _split_by_separator(text: str) -> list[str]:
    """
    策略 1: 按 ===== 分隔符行拆分。

    Returns:
        拆分后的片段列表（已去空白、去空），若只有 1 个片段则返回空列表
    """
    parts = _SEPARATOR_PATTERN.split(text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 2:
        logger.info(f"按分隔符 '=====' 拆分: {len(parts)} 个片段")
        return parts
    return []


def _split_by_section_marker(text: str) -> list[str]:
    """
    策略 2: 按 【药品名称】 章节标记位置切分。

    每个 【药品名称】 标记开始一个新的药品文档。

    Returns:
        拆分后的片段列表（已去空白、去空），若只有 1 个片段则返回空列表
    """
    # 找到所有 【药品名称】 的位置
    positions: list[int] = []
    for match in _SECTION_MARKER_PATTERN.finditer(text):
        positions.append(match.start())

    if len(positions) < 2:
        return []

    # 按位置切分
    parts: list[str] = []
    for i, pos in enumerate(positions):
        if i + 1 < len(positions):
            end = positions[i + 1]
        else:
            end = len(text)
        part = text[pos:end].strip()
        if part:
            parts.append(part)

    # 处理第一个标记之前的文本（如果有有意义的内容）
    if positions[0] > 0:
        preamble = text[:positions[0]].strip()
        if preamble and len(preamble) > 20:
            # 可能是缺少标记的药品，也加入
            parts.insert(0, preamble)

    if len(parts) >= 2:
        logger.info(f"按【药品名称】标记拆分: {len(parts)} 个片段")
        return parts

    return []


def split_multi_drug(text: str) -> list[SubDocument]:
    """
    将多药品合集文档拆分为独立的药品文档列表。

    拆分策略（按优先级自动选择）：
    1. 按 ===== 分隔符行拆分
    2. 按 【药品名称】 标记位置切分
    3. 兜底：返回整个文档作为单一 SubDocument

    Args:
        text: 待拆分的合集文档原始文本

    Returns:
        SubDocument 列表，每个包含 drug_name、text、index
    """
    if not text or not text.strip():
        logger.warning("输入文本为空，返回空列表")
        return []

    # 策略 1: 按分隔符拆分
    parts = _split_by_separator(text)

    # 策略 2: 按章节标记拆分
    if not parts:
        parts = _split_by_section_marker(text)

    # 策略 3: 兜底 — 整个文档作为一个
    if not parts:
        logger.debug("未能拆分，作为单一文档处理")
        parts = [text.strip()]

    # 为每个片段提取药名并构建 SubDocument
    sub_docs: list[SubDocument] = []
    for i, part in enumerate(parts):
        drug_name = extract_drug_name(part)
        if not drug_name:
            drug_name = f"未知药品_{i + 1}"
            logger.warning(f"片段 {i} 无法提取药名，使用兜底名称: {drug_name}")

        # 清理药名中不适合文件名/数据库的字符
        drug_name = drug_name.replace("/", "-").replace("\\", "-")

        sub_docs.append(SubDocument(
            drug_name=drug_name,
            text=part,
            index=i,
        ))

    logger.info(
        f"多药品拆分完成: {len(sub_docs)} 种药品 → "
        + ", ".join(s.drug_name for s in sub_docs)
    )
    return sub_docs
