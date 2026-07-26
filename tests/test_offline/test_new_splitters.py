"""
测试 v1.0.0 新切分器 (splitter_disease / splitter_guideline / splitter_literature)
"""

import pytest

from app.offline.splitter_disease import (
    Chunk,
    split_disease_document,
    _find_sections as _find_disease_sections,
)
from app.offline.splitter_guideline import split_guideline_document
from app.offline.splitter_literature import split_literature_document


# ============================================================
# splitter_disease
# ============================================================
class TestSplitDiseaseDocument:
    """测试疾病知识切分器。"""

    def test_empty_text(self):
        """空文本返回空列表。"""
        chunks = split_disease_document("")
        assert chunks == []

    def test_single_short_text(self):
        """短文本单 chunk。"""
        chunks = split_disease_document("原发性高血压是一种常见的心血管疾病。")
        assert len(chunks) == 1
        assert "原发性高血压" in chunks[0].chunk_text

    def test_markdown_headings_detected(self):
        """Markdown 标题检测。"""
        text = "## 病因\n高血压的病因包括遗传因素和环境因素。\n\n## 临床表现\n早期通常无症状。"
        chunks = split_disease_document(text)
        assert len(chunks) > 0

    def test_numbered_sections_fallback(self):
        """无 Markdown 标题时回退到编号检测。"""
        text = "1. 概述\n高血压定义\n\n2. 诊断标准\n诊室血压≥140/90mmHg"
        chunks = split_disease_document(text)
        assert len(chunks) > 0

    def test_keyword_sections_detected(self):
        """关键词行检测。"""
        text = "概述\n\n高血压是一种常见疾病。\n\n诊断标准\n血压持续升高。\n\n治疗原则\n生活方式干预和药物治疗。"
        chunks = split_disease_document(text)
        assert len(chunks) > 0

    def test_custom_chunk_size(self):
        """自定义 chunk_size。"""
        text = "高血压防治知识。" * 200  # 较长文本
        chunks = split_disease_document(text, chunk_size=300, chunk_overlap=50)
        assert len(chunks) > 2

    def test_chunks_have_required_fields(self):
        """每个 chunk 有必需字段。"""
        text = "原发性高血压\n\n## 定义\n高血压是指..."
        chunks = split_disease_document(text)
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.section
            assert c.chunk_text
            assert c.chunk_index >= 0
            assert c.char_count > 0


# ============================================================
# splitter_guideline
# ============================================================
class TestSplitGuidelineDocument:
    """测试临床指南切分器。"""

    def test_empty_text(self):
        """空文本返回空列表。"""
        chunks = split_guideline_document("")
        assert chunks == []

    def test_short_guideline(self):
        """短指南单 chunk。"""
        text = "推荐意见：ACEI/ARB作为心衰患者的一线治疗药物。\n\n证据级别：IA"
        chunks = split_guideline_document(text)
        assert len(chunks) >= 1

    def test_numbered_sections(self):
        """编号章节切分。"""
        text = (
            "1. 背景\n心力衰竭是常见的心血管疾病终末阶段。\n\n"
            "2. 推荐意见\n"
            "2.1 药物治疗\n推荐使用ACEI、β受体阻滞剂等。\n"
            "2.2 非药物治疗\nCRT适用于符合适应症的患者。"
        )
        chunks = split_guideline_document(text, chunk_size=300)
        assert len(chunks) > 0

    def test_recommendation_markers(self):
        """推荐意见标记。"""
        text = (
            "推荐使用SGLT2i治疗心衰（推荐等级：强推荐，证据级别：IA）\n"
            "可考虑使用维利西呱（推荐等级：可考虑）\n"
            "不推荐常规使用硝酸酯类药物"
        )
        chunks = split_guideline_document(text)
        assert len(chunks) > 0

    def test_chunks_have_required_fields(self):
        """每个 chunk 有必需字段。"""
        text = "1. 诊断\n心力衰竭的诊断依据：\n2. 治疗\nGDMT方案。"
        chunks = split_guideline_document(text)
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.section
            assert c.chunk_text
            assert c.chunk_index >= 0
            assert c.char_count > 0


# ============================================================
# splitter_literature
# ============================================================
class TestSplitLiteratureDocument:
    """测试学术文献切分器。"""

    def test_empty_text(self):
        """空文本返回空列表。"""
        chunks = split_literature_document("")
        assert chunks == []

    def test_imrad_structure(self):
        """IMRaD 结构检测。"""
        text = (
            "Introduction\nHeart failure (HF) affects millions worldwide.\n\n"
            "Methods\nWe conducted a randomized trial of 200 patients.\n\n"
            "Results\nSGLT2i reduced mortality by 25%.\n\n"
            "Discussion\nThese results confirm the efficacy of SGLT2i.\n\n"
            "Conclusion\nSGLT2i should be considered for all HF patients.\n\n"
            "References\n1. Smith et al. 2023."
        )
        chunks = split_literature_document(text, chunk_size=300)
        assert len(chunks) > 0

    def test_chinese_imrad(self):
        """中文 IMRaD 检测。"""
        text = (
            "引言\n心力衰竭是常见的心血管疾病。\n\n"
            "方法\n纳入200例心衰患者。\n\n"
            "结果\n治疗组死亡率显著降低。\n\n"
            "讨论\n本研究证实了SGLT2i的疗效。\n\n"
            "结论\n推荐心衰患者使用SGLT2i。\n\n"
            "参考文献\n[1] 张三. 2023."
        )
        chunks = split_literature_document(text, chunk_size=300)
        assert len(chunks) > 0

    def test_fallback_three_section(self):
        """无 IMRaD 结构时三段式回退。"""
        text = (
            "摘要\n本研究评估了SGLT2i在心衰患者中的疗效。\n\n"
            "正文内容\n纳入了200例患者，随机分组。\n\n"
            "详细的实验方法和统计分析方法。\n\n"
            "参考文献\n[1] Smith et al. 2023.\n[2] Zhang et al. 2024."
        )
        chunks = split_literature_document(text, chunk_size=500)
        assert len(chunks) > 0

    def test_chunks_have_required_fields(self):
        """每个 chunk 有必需字段。"""
        text = "Methods\nStudy design.\n\nResults\nPrimary endpoint met."
        chunks = split_literature_document(text)
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.section
            assert c.chunk_text
            assert c.chunk_index >= 0
            assert c.char_count > 0
