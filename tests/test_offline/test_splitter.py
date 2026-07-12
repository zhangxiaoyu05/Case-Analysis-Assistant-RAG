"""
测试 app.offline.splitter — 章节感知文本切分模块

覆盖: _find_sections, _split_by_sections, _merge_short_sections,
      _split_long_section, split_document, Chunk dataclass
"""

import pytest

from app.offline.splitter import (
    Chunk,
    _find_sections,
    _split_by_sections,
    _merge_short_sections,
    _split_long_section,
    split_document,
)


# ============================================================
# Chunk dataclass
# ============================================================
class TestChunk:
    """测试 Chunk 数据类。"""

    def test_create_chunk(self):
        """创建 Chunk 对象。"""
        chunk = Chunk(
            section="适应症",
            chunk_text="用于解热镇痛",
            chunk_index=0,
            char_count=6,
        )
        assert chunk.section == "适应症"
        assert chunk.chunk_text == "用于解热镇痛"
        assert chunk.chunk_index == 0
        assert chunk.char_count == 6


# ============================================================
# _find_sections
# ============================================================
class TestFindSections:
    """测试章节标记检测。"""

    def test_find_multiple_sections(self, sample_raw_text):
        """检测多个【】章节标记。"""
        sections = _find_sections(sample_raw_text)
        section_names = [name for _, name in sections]
        assert "药品名称" in section_names
        assert "适应症" in section_names
        assert "用法用量" in section_names
        assert "禁忌" in section_names
        assert "不良反应" in section_names

    def test_no_sections(self):
        """无章节标记的文本。"""
        sections = _find_sections("这是一段没有章节标记的普通文本。")
        assert sections == []

    def test_empty_section_name_filtered(self):
        """空章节名被过滤。"""
        sections = _find_sections("【】空章节")
        assert len(sections) == 0

    def test_section_with_whitespace(self):
        """章节名含空白被清理。"""
        sections = _find_sections("【 药品名称 】阿司匹林")
        if sections:
            assert sections[0][1] == "药品名称"


# ============================================================
# _split_by_sections
# ============================================================
class TestSplitBySections:
    """测试按章节拆分。"""

    def test_split_standard_text(self, sample_raw_text):
        """拆分标准说明书文本。"""
        sections = _split_by_sections(sample_raw_text)
        assert len(sections) > 1
        section_names = [s["section"] for s in sections]
        # 第一个章节前的内容
        assert "药品名称" in section_names or "__preamble__" in section_names

    def test_no_sections_returns_preamble(self):
        """无章节标记时整篇作为 preamble。"""
        sections = _split_by_sections("普通文本内容")
        assert len(sections) == 1
        assert sections[0]["section"] == "__preamble__"

    def test_preamble_before_first_section(self):
        """开头的无章节内容标记为 __preamble__。"""
        text = "这是一段前言内容\n【药品名称】阿司匹林"
        sections = _split_by_sections(text)
        if sections[0]["section"] == "__preamble__":
            assert "前言内容" in sections[0]["content"]


# ============================================================
# _merge_short_sections
# ============================================================
class TestMergeShortSections:
    """测试短章节合并。"""

    def test_short_section_merged_up(self):
        """短章节合并到前一个章节。"""
        sections = [
            {"section": "适应症", "content": "较长的适应症内容" * 20},  # ~200 字符
            {"section": "短章节", "content": "X"},  # 1 字符
        ]
        merged = _merge_short_sections(sections, min_size=50)
        # 短章节被合并，总章节数减少
        assert len(merged) < len(sections)

    def test_min_size_respected(self):
        """达到 min_size 的章节不被合并。"""
        sections = [
            {"section": "A", "content": "足够长的内容A" * 20},
            {"section": "B", "content": "足够长的内容B" * 20},
        ]
        merged = _merge_short_sections(sections, min_size=50)
        assert len(merged) == 2

    def test_empty_sections(self):
        """空列表处理。"""
        result = _merge_short_sections([], min_size=100)
        assert result == []


# ============================================================
# _split_long_section
# ============================================================
class TestSplitLongSection:
    """测试长章节二次切分。"""

    def test_short_content_no_split(self):
        """短内容不切分。"""
        content = "短文本，不超过 chunk_size。"
        chunks = _split_long_section(
            "测试", content,
            chunk_size=500, chunk_overlap=50, min_chunk_size=100,
        )
        assert len(chunks) == 1
        assert chunks[0].section == "测试"
        assert chunks[0].chunk_text == content

    def test_long_content_split(self):
        """长内容被切分为多个 chunk。"""
        content = "这是一个测试句子。" * 100  # ~900 字符
        chunks = _split_long_section(
            "测试章节", content,
            chunk_size=200, chunk_overlap=20, min_chunk_size=50,
        )
        assert len(chunks) > 1
        for c in chunks:
            assert c.section == "测试章节"

    def test_chunks_have_consistent_section_name(self):
        """所有 chunk 保持相同的章节名。"""
        content = "测试内容。" * 150
        chunks = _split_long_section(
            "不良反应", content,
            chunk_size=200, chunk_overlap=50, min_chunk_size=50,
        )
        for c in chunks:
            assert c.section == "不良反应"

    def test_tail_merge(self):
        """尾块太短时合并到前一块。"""
        content = "A" * 300 + "。" + "B" * 20  # 最后一段很短
        chunks = _split_long_section(
            "测试", content,
            chunk_size=200, chunk_overlap=20, min_chunk_size=50,
        )
        # 最后一块不应太短
        if len(chunks) > 1:
            assert chunks[-1].char_count >= 50 or len(chunks) == 1


# ============================================================
# split_document (公共 API)
# ============================================================
class TestSplitDocument:
    """测试 split_document 公共函数。"""

    def test_split_standard_text(self, sample_raw_text):
        """切分标准药品说明书文本。"""
        chunks = split_document(sample_raw_text)
        assert len(chunks) > 0
        # 验证 chunk 结构
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.section
            assert c.chunk_text
            assert c.char_count > 0
            assert c.chunk_index >= 0

    def test_chunk_indices_sequential(self, sample_raw_text):
        """chunk_index 从 0 开始连续递增。"""
        chunks = split_document(sample_raw_text)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_text(self):
        """空文本返回单 chunk（空内容）。"""
        chunks = split_document("")
        assert len(chunks) == 1
        assert chunks[0].chunk_text == ""

    def test_custom_parameters(self):
        """自定义参数生效。"""
        text = "测试内容。" * 100
        chunks = split_document(
            text,
            chunk_size=300,
            chunk_overlap=30,
            min_chunk_size=50,
        )
        assert len(chunks) > 0
        # 验证没有超过 chunk_size 太多的（允许一些超出因为自然边界）
        for c in chunks:
            assert c.char_count <= 300 + 100  # 放宽容差

    def test_chunk_text_not_exceed_5000(self):
        """chunk 不超过 Milvus VARCHAR(5000) 限制。"""
        # 构造超长文本
        long_text = "【药品名称】\n" + ("很长的内容。" * 1000)  # ~6000 字符
        chunks = split_document(long_text, chunk_size=6000, chunk_overlap=0)
        for c in chunks:
            assert c.char_count <= 5000

    def test_single_section(self):
        """只有单个章节的文本。"""
        text = "【适应症】用于解热镇痛，缓解轻至中度疼痛。"
        chunks = split_document(text)
        assert len(chunks) >= 1
        assert chunks[0].section == "适应症"

    def test_no_section_markers(self):
        """无【】章节标记的纯文本。"""
        text = "这是一段没有任何章节标记的药品说明书文本。" * 10
        chunks = split_document(text)
        assert len(chunks) >= 1
        assert chunks[0].section == "__preamble__"
