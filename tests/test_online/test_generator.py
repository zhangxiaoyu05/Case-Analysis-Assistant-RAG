"""
测试 app.online.generator — 答案生成模块

覆盖: GeneratedAnswer, Generator 类, generate_answer 便捷函数
"""

from unittest.mock import MagicMock, patch

import pytest

from app.online.generator import (
    GeneratedAnswer,
    Generator,
    generate_answer,
)


# ============================================================
# GeneratedAnswer
# ============================================================
class TestGeneratedAnswer:
    """测试 GeneratedAnswer 数据类。"""

    def test_create(self):
        """创建 GeneratedAnswer。"""
        answer = GeneratedAnswer(
            answer="根据药品说明书，阿司匹林的不良反应包括...",
            sources=[{"drug_name": "阿司匹林肠溶片", "section": "不良反应"}],
            template_used="default",
            token_count=150,
        )
        assert answer.template_used == "default"
        assert answer.token_count == 150
        assert len(answer.sources) == 1


# ============================================================
# Generator.__init__
# ============================================================
class TestGeneratorInit:
    """测试初始化。"""

    def test_init_with_defaults(self):
        """默认参数初始化。"""
        gen = Generator(api_key="test-key")
        assert gen._api_key == "test-key"
        assert gen._model is not None
        assert gen._temperature > 0

    def test_init_without_api_key_raises(self):
        """无 API Key 抛异常。"""
        with patch("app.online.generator.config") as mock_config:
            mock_config.DASHSCOPE_API_KEY = ""
            mock_config.chat_model = "test-model"
            mock_config.chat_temperature = 0.3
            mock_config.chat_max_tokens = 2000
            mock_config.chat_top_p = 0.95
            with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
                Generator()


# ============================================================
# Generator._detect_template
# ============================================================
class TestDetectTemplate:
    """测试模板检测。"""

    def test_default_template(self):
        """普通查询返回 default 模板。"""
        template = Generator._detect_template("阿司匹林有什么不良反应？")
        assert template == "default"

    def test_comparison_template(self):
        """对比类查询返回 comparison 模板。"""
        test_cases = [
            "阿司匹林和布洛芬有什么区别？",
            "阿司匹林与对乙酰氨基酚哪个更好？",
            "比较一下阿司匹林和布洛芬",
            "阿司匹林 vs 布洛芬",
        ]
        for query in test_cases:
            template = Generator._detect_template(query)
            assert template == "comparison", f"'{query}' should be comparison"

    def test_dosage_followup_template(self):
        """对比类问题返回 comparison 模板。"""
        template = Generator._detect_template("阿司匹林和布洛芬有什么区别？")
        assert template == "comparison"

    def test_default_template_with_history(self):
        """即使有历史，普通追问仍返回 default。"""
        history = [{"role": "assistant", "content": "成人一次0.3～0.6g"}]
        template = Generator._detect_template("那儿童呢？", history=history)
        assert template == "default"


# ============================================================
# Generator._format_context
# ============================================================
class TestFormatContext:
    """测试上下文格式化。"""

    def test_format_single_doc(self):
        """单个文档格式化。"""
        docs = [{"drug_name": "阿司匹林", "section": "用法用量",
                 "chunk_text": "成人一次0.3～0.6g，一日3次。"}]
        context = Generator._format_context(docs)
        assert "阿司匹林" in context
        assert "用法用量" in context

    def test_format_multiple_docs(self):
        """多个文档格式化。"""
        docs = [
            {"drug_name": "阿司匹林", "section": "适应症", "chunk_text": "解热镇痛。"},
            {"drug_name": "阿司匹林", "section": "禁忌", "chunk_text": "过敏者禁用。"},
        ]
        context = Generator._format_context(docs)
        assert "适应症" in context
        assert "禁忌" in context


# ============================================================
# Generator.generate
# ============================================================
class TestGeneratorGenerate:
    """测试生成方法。"""

    @pytest.fixture
    def sample_context_docs(self):
        """测试用上下文文档。"""
        return [
            {"drug_name": "阿司匹林肠溶片", "section": "适应症",
             "chunk_text": "用于解热镇痛，缓解轻至中度疼痛。", "score": 0.95,
             "doc_id": 1, "chunk_index": 0},
            {"drug_name": "阿司匹林肠溶片", "section": "用法用量",
             "chunk_text": "成人一次0.3～0.6g，一日3次。", "score": 0.92,
             "doc_id": 1, "chunk_index": 1},
        ]

    def test_generate_success(self, sample_context_docs, mock_dashscope_response):
        """成功生成回答。"""
        mock_resp = mock_dashscope_response(
            choices_content="根据说明书，阿司匹林用于解热镇痛，成人一次0.3～0.6g，一日3次。"
        )

        with patch("dashscope.Generation") as mock_gen:
            mock_gen.call.return_value = mock_resp
            gen = Generator(api_key="test-key")
            result = gen.generate("阿司匹林怎么吃？", sample_context_docs)

        assert isinstance(result, GeneratedAnswer)
        assert len(result.answer) > 0
        assert result.template_used in ("default", "comparison", "dosage_followup")

    def test_generate_with_history(self, sample_context_docs, mock_dashscope_response):
        """带对话历史生成。"""
        mock_resp = mock_dashscope_response(
            choices_content="儿童用量需咨询医师或药师。"
        )

        history = [
            {"role": "user", "content": "阿司匹林怎么吃？"},
            {"role": "assistant", "content": "成人一次0.3～0.6g，一日3次。"},
        ]

        with patch("dashscope.Generation") as mock_gen:
            mock_gen.call.return_value = mock_resp
            gen = Generator(api_key="test-key")
            result = gen.generate("那儿童呢？", sample_context_docs, history=history)

        assert isinstance(result, GeneratedAnswer)

    def test_generate_api_failure(self, sample_context_docs, mock_dashscope_response):
        """API 失败时返回兜底回答。"""
        mock_resp = mock_dashscope_response(status_code=500)

        with patch("dashscope.Generation") as mock_gen:
            mock_gen.call.return_value = mock_resp
            gen = Generator(api_key="test-key")
            result = gen.generate("阿司匹林怎么吃？", sample_context_docs)

        assert isinstance(result, GeneratedAnswer)
        assert len(result.answer) > 0  # 兜底回答

    def test_generate_with_explicit_template(self, sample_context_docs, mock_dashscope_response):
        """显式指定模板。"""
        mock_resp = mock_dashscope_response(
            choices_content="阿司匹林和布洛芬的区别..."
        )

        with patch("dashscope.Generation") as mock_gen:
            mock_gen.call.return_value = mock_resp
            gen = Generator(api_key="test-key")
            result = gen.generate(
                "阿司匹林和布洛芬的区别？",
                sample_context_docs,
                template="comparison",
            )
        assert result.template_used == "comparison"


# ============================================================
# Generator.generate_stream
# ============================================================
class TestGeneratorStream:
    """测试流式生成。"""

    @pytest.fixture
    def sample_context_docs(self):
        """测试用上下文文档。"""
        return [
            {"drug_name": "阿司匹林肠溶片", "section": "用法用量",
             "chunk_text": "成人一次0.3～0.6g。", "score": 0.95,
             "doc_id": 1, "chunk_index": 0},
        ]

    def test_generate_stream_yields_tokens(self, sample_context_docs):
        """流式生成产出 token。"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        # 模拟流式输出
        mock_output = MagicMock()
        mock_output.text = "token"
        mock_output.choices = [MagicMock()]
        mock_output.choices[0].message.content = "阿司匹林"
        mock_resp.output = mock_output

        with patch("dashscope.Generation") as mock_gen:
            mock_gen.call.return_value = [mock_resp]  # 流式返回迭代器
            gen = Generator(api_key="test-key")
            tokens = list(gen.generate_stream("查询", sample_context_docs))
            # 应有 token 产出
            assert len(tokens) > 0


# ============================================================
# generate_answer 便捷函数
# ============================================================
class TestGenerateAnswer:
    """测试 generate_answer 便捷函数。"""

    def test_returns_generated_answer(self, sample_chunks):
        """返回 GeneratedAnswer。"""
        with patch("app.online.generator.Generator.generate") as mock_generate:
            mock_generate.return_value = GeneratedAnswer(
                answer="测试回答", sources=sample_chunks,
                template_used="default", token_count=10,
            )
            result = generate_answer("测试", sample_chunks)
            assert isinstance(result, GeneratedAnswer)
            assert result.answer == "测试回答"
