"""
测试 app.offline.loader — 文档加载模块

覆盖: load_pdf, load_docx, load_txt, infer_drug_name, load_document, load_documents_from_dir
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.offline.loader import (
    LoadedDocument,
    LoaderError,
    infer_drug_name,
    load_document,
    load_documents_from_dir,
    load_pdf,
    load_docx,
    load_txt,
)


# ============================================================
# infer_drug_name
# ============================================================
class TestInferDrugName:
    """测试从文件名推断药品名称。"""

    def test_simple_drug_name(self):
        """文件名 => 推断药名。"""
        assert infer_drug_name(Path("阿司匹林肠溶片说明书.pdf")) == "阿司匹林肠溶片"

    def test_with_manufacturer_suffix(self):
        """含厂家后缀的文件名。"""
        name = infer_drug_name(Path("布洛芬缓释胶囊说明书_中美史克.pdf"))
        assert "布洛芬缓释胶囊" in name

    def test_txt_file(self):
        """TXT 文件。"""
        assert infer_drug_name(Path("对乙酰氨基酚片说明书_test.txt")) == "对乙酰氨基酚片"

    def test_docx_file(self):
        """DOCX 文件。"""
        name = infer_drug_name(Path("头孢克洛胶囊说明书.docx"))
        assert "头孢克洛胶囊" in name

    def test_with_year_suffix(self):
        """含年份后缀。"""
        name = infer_drug_name(Path("阿莫西林胶囊说明书_2024版.pdf"))
        assert "阿莫西林胶囊" in name

    def test_no_recognizable_name(self):
        """无法推断药名的文件名。"""
        assert infer_drug_name(Path("说明书.pdf")) is None


# ============================================================
# load_txt
# ============================================================
class TestLoadTxt:
    """测试 TXT 文件加载。"""

    def test_load_utf8(self):
        """加载 UTF-8 编码的 TXT 文件。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        ) as f:
            f.write("【药品名称】阿司匹林肠溶片\n【适应症】解热镇痛")
            tmp_path = Path(f.name)

        try:
            text = load_txt(tmp_path)
            assert "阿司匹林肠溶片" in text
            assert "解热镇痛" in text
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_load_empty_file(self):
        """加载空文件。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        ) as f:
            f.write("")
            tmp_path = Path(f.name)

        try:
            text = load_txt(tmp_path)
            assert text == ""
        finally:
            tmp_path.unlink(missing_ok=True)


# ============================================================
# load_document
# ============================================================
class TestLoadDocument:
    """测试单文档加载主函数。"""

    def test_load_txt_document(self):
        """加载 TXT 文档并返回 LoadedDocument。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        ) as f:
            f.write("【药品名称】测试药品\n【适应症】测试适应症")
            tmp_path = Path(f.name)

        try:
            doc = load_document(tmp_path)
            assert isinstance(doc, LoadedDocument)
            assert "测试药品" in doc.raw_text
            assert doc.file_type == "txt"
            assert doc.source_file == str(tmp_path)
            assert doc.inferred_drug_name is not None
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_file_not_found(self):
        """文件不存在时抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            load_document(Path("/nonexistent/file.pdf"))

    def test_unsupported_format(self):
        """不支持的文件格式抛出 LoaderError。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xyz", encoding="utf-8", delete=False
        ) as f:
            f.write("dummy")
            tmp_path = Path(f.name)

        try:
            with pytest.raises(LoaderError, match="不支持的文件格式"):
                load_document(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_empty_document_warning(self, caplog):
        """空文档加载后会有 warning 日志。"""
        import logging
        caplog.set_level(logging.WARNING)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        ) as f:
            f.write("   ")
            tmp_path = Path(f.name)

        try:
            doc = load_document(tmp_path)
            assert doc.raw_text.strip() == ""
        finally:
            tmp_path.unlink(missing_ok=True)


# ============================================================
# load_documents_from_dir
# ============================================================
class TestLoadDocumentsFromDir:
    """测试目录批量加载。"""

    def test_load_directory(self):
        """加载目录中所有支持的文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # 创建测试文件
            (tmpdir / "药1.txt").write_text("药品1内容", encoding="utf-8")
            (tmpdir / "药2.txt").write_text("药品2内容", encoding="utf-8")
            (tmpdir / "not_a_drug.png").write_text("image")

            docs = load_documents_from_dir(tmpdir)
            assert len(docs) == 2
            drug_names = {d.inferred_drug_name for d in docs}
            assert "药1" in drug_names
            assert "药2" in drug_names

    def test_load_empty_directory(self):
        """加载空目录返回空列表。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = load_documents_from_dir(Path(tmpdir))
            assert docs == []

    def test_nonexistent_directory(self):
        """不存在的目录抛出异常。"""
        with pytest.raises(NotADirectoryError):
            load_documents_from_dir(Path("/nonexistent_dir"))


# ============================================================
# load_pdf — mock
# ============================================================
class TestLoadPdf:
    """测试 PDF 加载（mock pypdf）。"""

    def test_load_pdf_success(self):
        """成功加载 PDF。"""
        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "【药品名称】测试"
        mock_reader.pages = [mock_page, mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            text = load_pdf(Path("test.pdf"))
            assert "【药品名称】测试" in text

    def test_load_pdf_empty_pages(self):
        """PDF 含空页面。"""
        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = None
        mock_reader.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            text = load_pdf(Path("test.pdf"))
            assert text == ""


# ============================================================
# load_docx — mock
# ============================================================
class TestLoadDocx:
    """测试 DOCX 加载（mock python-docx）。"""

    def test_load_docx_success(self):
        """成功加载 DOCX。"""
        mock_doc = MagicMock()
        mock_para = MagicMock()
        mock_para.text = "【药品名称】测试药品"
        mock_doc.paragraphs = [mock_para]

        with patch("docx.Document", return_value=mock_doc):
            text = load_docx(Path("test.docx"))
            assert "测试药品" in text


# ============================================================
# LoadedDocument dataclass
# ============================================================
class TestLoadedDocument:
    """测试 LoadedDocument 数据类。"""

    def test_default_values(self):
        """默认字段值。"""
        doc = LoadedDocument(raw_text="test", source_file="/tmp/test.txt")
        assert doc.inferred_drug_name is None
        assert doc.file_type is None
        assert doc.metadata == {}

    def test_full_fields(self):
        """完整字段。"""
        doc = LoadedDocument(
            raw_text="测试内容",
            source_file="/tmp/药.pdf",
            inferred_drug_name="测试药",
            file_type="pdf",
            metadata={"pages": 3},
        )
        assert doc.raw_text == "测试内容"
        assert doc.inferred_drug_name == "测试药"
        assert doc.file_type == "pdf"
        assert doc.metadata["pages"] == 3
