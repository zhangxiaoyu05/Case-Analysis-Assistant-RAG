"""
测试 app.offline.cleaner — 文本清洗模块

覆盖: _normalize_whitespace, _remove_pdf_artifacts, _normalize_unicode, clean_text
"""

from unittest.mock import patch, MagicMock

import pytest

from app.offline.cleaner import (
    _normalize_whitespace,
    _remove_pdf_artifacts,
    _normalize_unicode,
    clean_text,
)


# ============================================================
# _normalize_whitespace
# ============================================================
class TestNormalizeWhitespace:
    """测试空白规范化。"""

    def test_crlf_to_lf(self):
        """CRLF → LF。"""
        text = "line1\r\nline2\r\nline3"
        result = _normalize_whitespace(text)
        assert "\r\n" not in result
        assert "\r" not in result

    def test_multiple_newlines_collapsed(self):
        """3+ 连续换行 → 2 换行。"""
        text = "para1\n\n\n\npara2"
        result = _normalize_whitespace(text)
        assert result == "para1\n\npara2"

    def test_inline_spaces_collapsed(self):
        """行内多余空格合并。"""
        text = "药品   名称：   阿司匹林"
        result = _normalize_whitespace(text)
        assert result == "药品 名称： 阿司匹林"

    def test_zero_width_chars_removed(self):
        """零宽字符被移除。"""
        # 零宽空格 U+200B
        text = "药​品​名​称"
        result = _normalize_whitespace(text)
        assert "​" not in result

    def test_tabs_to_space(self):
        """Tab 转空格。"""
        text = "药品\t名称：\t阿司匹林"
        result = _normalize_whitespace(text)
        assert "\t" not in result
        assert "药品 名称： 阿司匹林" == result


# ============================================================
# _remove_pdf_artifacts
# ============================================================
class TestRemovePdfArtifacts:
    """测试 PDF 伪影去除。"""

    def test_remove_page_numbers(self):
        """去除纯页码行（需要 >30 字符避免短路）。"""
        text = "这是第一行内容足够长" * 2 + "\n123\n" + "这是后续内容也足够长" * 2
        result = _remove_pdf_artifacts(text)
        assert "123" not in result.split("\n")

    def test_remove_chinese_page_numbers(self):
        """去除中文页码（需要 >30 字符避免短路）。"""
        text = "这是第一行内容足够长用于测试" + "\n第 3 页\n" + "这是更多内容行也够长"
        result = _remove_pdf_artifacts(text)
        assert "第 3 页" not in result.split("\n")

    def test_remove_english_page_numbers(self):
        """去除英文页码。"""
        text = "content\nPage 5 of 10\nmore content"
        result = _remove_pdf_artifacts(text)
        assert "Page 5 of 10" not in result

    def test_short_text_preserved(self):
        """短文本（< 30 字符）不处理。"""
        text = "12"
        result = _remove_pdf_artifacts(text)
        assert result == "12"

    def test_normal_content_preserved(self):
        """正常内容行保留。"""
        text = "【药品名称】阿司匹林肠溶片\n【适应症】解热镇痛"
        result = _remove_pdf_artifacts(text)
        assert "【药品名称】" in result
        assert "【适应症】" in result


# ============================================================
# _normalize_unicode
# ============================================================
class TestNormalizeUnicode:
    """测试 Unicode 规范化。"""

    def test_fullwidth_digits_to_halfwidth(self):
        """全角数字 → 半角。"""
        text = "一次０．３～０．６ｇ"  # 全角数字
        result = _normalize_unicode(text)
        assert "0" in result  # 半角数字
        assert "０" not in result

    def test_fullwidth_letters_to_halfwidth(self):
        """全角字母 → 半角。"""
        text = "Ａｓｐｉｒｉｎ"  # 全角大写
        result = _normalize_unicode(text)
        assert "A" in result
        assert "Ａ" not in result

    def test_chinese_punctuation_preserved(self):
        """中文标点保留。"""
        text = "【药品名称】阿司匹林（肠溶片）"
        result = _normalize_unicode(text)
        assert "【" in result
        assert "】" in result
        assert "（" in result
        assert "）" in result


# ============================================================
# clean_text
# ============================================================
class TestCleanText:
    """测试 clean_text 公共函数。"""

    def test_basic_cleaning(self, sample_raw_text):
        """基础清洗（不脱敏）。"""
        result = clean_text(sample_raw_text)
        assert len(result) > 0
        assert "【药品名称】" in result
        assert "【适应症】" in result
        # 不应有 CRLF
        assert "\r\n" not in result

    def test_empty_input(self):
        """空输入原样返回。"""
        result = clean_text("")
        assert result == ""

    def test_whitespace_only(self):
        """仅空白输入。"""
        result = clean_text("   \n  \t  ")
        assert result.strip() == ""

    def test_desensitize_without_api_key(self):
        """无 API Key 时脱敏被跳过。"""
        result = clean_text("患者张三，身份证110101199001011234", desensitize=True, api_key=None)
        assert len(result) > 0

    def test_desensitize_with_api_key(self):
        """有 API Key 时走脱敏流程（mock）。"""
        with patch("app.offline.cleaner._load_desensitization_prompt") as mock_prompt, \
             patch("app.offline.cleaner._desensitize_chunk") as mock_desensitize:
            mock_prompt.return_value = ("system prompt", "user: {text}")
            mock_desensitize.return_value = "脱敏后的文本"

            result = clean_text(
                "患者张三", desensitize=True, api_key="test-key"
            )
            assert result == "脱敏后的文本"

    def test_desensitize_chunk_failure_graceful(self):
        """脱敏失败时保留原文。"""
        with patch("app.offline.cleaner._load_desensitization_prompt") as mock_prompt, \
             patch("app.offline.cleaner._desensitize_chunk") as mock_desensitize:
            mock_prompt.return_value = ("system prompt", "user: {text}")
            mock_desensitize.side_effect = RuntimeError("API error")

            result = clean_text(
                "患者张三", desensitize=True, api_key="test-key"
            )
            assert result == "患者张三"
