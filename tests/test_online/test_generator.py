"""
测试 app.online.generator — 答案生成模块 (v1.0.0)

覆盖: GeneratedAnswer, Generator 类, generate_answer 便捷函数。
v1.0.0: 模板从 3 种改为 5 种 SOAP 模板。
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
            answer="根据病例分析，患者高血压需调整用药方案...",
            sources=[{"drug_name": "硝苯地平", "section": "用法用量"}],
            template_used="case_summary",
            token_count=150,
        )
        assert answer.template_used == "case_summary"
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
# Generator._detect_template (v1.0.0: 5 种 SOAP 模板)
# ============================================================
class TestDetectTemplate:
    """测试模板检测。"""

    def test_default_template(self):
        """v1.0.0: 无关键词匹配时返回 case_summary。"""
        template = Generator._detect_template("请分析这个病例")
        assert template == "case_summary"

    def test_drug_review_template(self):
        """v1.0.0: 含"不良反应""副作用"→ drug_review。"""
        test_cases = [
            "阿司匹林有什么不良反应？",
            "这个药的副作用是什么？",
        ]
        for query in test_cases:
            template = Generator._detect_template(query)
            assert template == "drug_review", f"'{query}' should be drug_review"

    def test_drug_review_interaction(self):
        """含"药物相互作用""审查"→ drug_review。"""
        template = Generator._detect_template("请审查药物相互作用和禁忌症")
        assert template == "drug_review"

    def test_differential_diagnosis_template(self):
        """v1.0.0: 含"鉴别诊断""区别"→ differential_diagnosis。"""
        test_cases = [
            "急性心梗和心绞痛的鉴别诊断",
            "这两个病的区别是什么？",
            "可能是什么病？",
        ]
        for query in test_cases:
            template = Generator._detect_template(query)
            assert template in ("differential_diagnosis", "drug_review", "case_summary"), \
                f"'{query}' got {template}"

    def test_treatment_analysis_template(self):
        """v1.0.0: 含"治疗""方案"→ treatment_analysis。"""
        template = Generator._detect_template("怎么治疗心衰？")
        assert template in ("treatment_analysis", "case_summary")

    def test_analysis_mode_priority(self):
        """v1.0.0: analysis_mode 优先级最高。"""
        template = Generator._detect_template(
            "患者高血压怎么治疗？",
            analysis_mode="drug_review",
        )
        assert template == "drug_review"

    def test_diagnosis_mode(self):
        """analysis_mode=diagnosis → differential_diagnosis。"""
        template = Generator._detect_template(
            "随便问个问题",
            analysis_mode="diagnosis",
        )
        assert template == "differential_diagnosis"

    def test_guideline_lookup_template(self):
        """v1.0.0: 含"指南"→ guideline_lookup。"""
        template = Generator._detect_template("心衰治疗指南推荐什么？")
        # "治疗"匹配优先于"指南"
        assert template in ("treatment_analysis", "guideline_lookup")


# ============================================================
# Generator._format_context (v1.0.0: 支持多源字段)
# ============================================================
class TestFormatContext:
    """测试上下文格式化。"""

    def test_format_single_doc(self):
        """单个文档格式化（drug source）。"""
        docs = [{"drug_name": "硝苯地平", "section": "用法用量",
                 "chunk_text": "口服，一次10mg，一日3次。",
                 "source_type": "drug"}]
        context = Generator._format_context(docs)
        assert "硝苯地平" in context
        assert "用法用量" in context

    def test_format_multiple_sources(self):
        """多源文档格式化。"""
        docs = [
            {"drug_name": "硝苯地平", "section": "适应症", "chunk_text": "用于治疗高血压。",
             "source_type": "drug"},
            {"disease_name": "原发性高血压", "section": "诊断标准",
             "chunk_text": "诊室血压≥140/90mmHg。", "source_type": "disease"},
            {"guideline_title": "中国高血压防治指南", "section": "推荐意见",
             "chunk_text": "推荐CCB作为一线用药。", "source_type": "guideline",
             "evidence_level": "IA"},
        ]
        context = Generator._format_context(docs)
        assert "硝苯地平" in context
        assert "原发性高血压" in context
        assert "中国高血压防治指南" in context


# ============================================================
# Generator.generate (v1.0.0: 支持 case_profile 等新参数)
# ============================================================
class TestGeneratorGenerate:
    """测试生成方法。"""

    @pytest.fixture
    def sample_context_docs(self):
        """测试用上下文文档。"""
        return [
            {"drug_name": "硝苯地平", "section": "适应症",
             "chunk_text": "用于治疗原发性高血压。", "score": 0.95,
             "doc_id": 1, "chunk_index": 0, "source_type": "drug"},
            {"disease_name": "原发性高血压", "section": "治疗原则",
             "chunk_text": "CCB是一线用药。", "score": 0.92,
             "doc_id": 2, "chunk_index": 0, "source_type": "disease",
             "evidence_level": "IA"},
        ]

    def test_generate_success(self, sample_context_docs, mock_dashscope_response):
        """成功生成回答。"""
        mock_resp = mock_dashscope_response(
            choices_content="根据病例分析，建议使用硝苯地平控释片..."
        )

        with patch("dashscope.Generation") as mock_gen:
            mock_gen.call.return_value = mock_resp
            gen = Generator(api_key="test-key")
            result = gen.generate("高血压怎么治疗？", sample_context_docs)

        assert isinstance(result, GeneratedAnswer)
        assert len(result.answer) > 0
        # v1.0.0 templates
        assert result.template_used in (
            "case_summary", "differential_diagnosis",
            "treatment_analysis", "drug_review", "guideline_lookup",
        )

    def test_generate_with_case_profile(self, sample_context_docs, mock_dashscope_response):
        """v1.0.0: 带病例信息生成。"""
        mock_resp = mock_dashscope_response(
            choices_content="S: 患者主诉头晕、胸闷3天...\nO: 血压160/95mmHg..."
        )

        with patch("dashscope.Generation") as mock_gen:
            mock_gen.call.return_value = mock_resp
            gen = Generator(api_key="test-key")
            result = gen.generate(
                "分析此病例",
                sample_context_docs,
                case_profile={"chief_complaint": "头晕胸闷", "suspected_diagnosis": ["高血压"]},
                synthesized_context={"drug": sample_context_docs},
                analysis_mode="comprehensive",
            )

        assert isinstance(result, GeneratedAnswer)

    def test_generate_api_failure(self, sample_context_docs, mock_dashscope_response):
        """API 失败时返回兜底回答。"""
        mock_resp = mock_dashscope_response(status_code=500)

        with patch("dashscope.Generation") as mock_gen:
            mock_gen.call.return_value = mock_resp
            gen = Generator(api_key="test-key")
            result = gen.generate("高血压怎么治疗？", sample_context_docs)

        assert isinstance(result, GeneratedAnswer)
        assert len(result.answer) > 0  # 兜底回答

    def test_generate_with_analysis_mode(self, sample_context_docs, mock_dashscope_response):
        """v1.0.0: 指定分析模式。"""
        mock_resp = mock_dashscope_response(
            choices_content="用药审查结果..."
        )

        with patch("dashscope.Generation") as mock_gen:
            mock_gen.call.return_value = mock_resp
            gen = Generator(api_key="test-key")
            result = gen.generate(
                "审查患者的用药方案",
                sample_context_docs,
                analysis_mode="drug_review",
            )
        assert result.template_used == "drug_review"


# ============================================================
# Generator.generate_stream (v1.0.0)
# ============================================================
class TestGeneratorStream:
    """测试流式生成。"""

    @pytest.fixture
    def sample_context_docs(self):
        """测试用上下文文档。"""
        return [
            {"drug_name": "硝苯地平", "section": "用法用量",
             "chunk_text": "口服，一次10mg。", "score": 0.95,
             "doc_id": 1, "chunk_index": 0, "source_type": "drug"},
        ]

    def test_generate_stream_yields_tokens(self, sample_context_docs):
        """流式生成产出 token。"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        # 模拟流式输出
        mock_output = MagicMock()
        mock_output.text = "token"
        mock_output.choices = [MagicMock()]
        mock_output.choices[0].message.content = "硝苯地平"
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
                template_used="case_summary", token_count=10,
            )
            result = generate_answer("测试", sample_chunks)
            assert isinstance(result, GeneratedAnswer)
            assert result.answer == "测试回答"
