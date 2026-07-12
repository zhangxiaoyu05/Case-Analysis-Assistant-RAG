"""
测试 app.online.intent — 意图识别模块

覆盖: IntentResult, IntentClassifier, classify_intent 便捷函数
"""

from unittest.mock import MagicMock, patch

import pytest

from app.online.intent import (
    IntentClassifier,
    IntentResult,
    classify_intent,
)


# ============================================================
# IntentResult
# ============================================================
class TestIntentResult:
    """测试 IntentResult 数据类。"""

    def test_drug_inquiry_result(self):
        """药品查询结果。"""
        result = IntentResult(intent="drug_inquiry", confidence=0.95)
        assert result.intent == "drug_inquiry"
        assert result.confidence == 0.95

    def test_other_result(self):
        """非药品查询结果。"""
        result = IntentResult(intent="other", confidence=0.9)
        assert result.intent == "other"


# ============================================================
# IntentClassifier.__init__
# ============================================================
class TestIntentClassifierInit:
    """测试初始化。"""

    def test_init_with_defaults(self):
        """默认参数初始化。"""
        classifier = IntentClassifier(api_key="test-key")
        assert classifier._api_key == "test-key"
        assert classifier._model is not None
        assert classifier._temperature > 0

    def test_init_without_api_key_raises(self):
        """无 API Key 抛异常。"""
        with patch("app.online.intent.config") as mock_config:
            mock_config.DASHSCOPE_API_KEY = ""
            mock_config.intent_model = "test-model"
            mock_config.intent_temperature = 0.1
            mock_config.intent_max_tokens = 200
            with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
                IntentClassifier()


# ============================================================
# IntentClassifier._quick_classify
# ============================================================
class TestQuickClassify:
    """测试快速预判。"""

    def test_non_drug_keywords(self):
        """非药品关键词快速判定为 other。"""
        classifier = IntentClassifier(api_key="test-key")

        test_cases = [
            "今天天气怎么样？",
            "推荐一只股票给我",
            "有什么好看的电影？",
            "你会什么编程语言？",
            "你是谁？你能做什么？",
            "帮我翻译一段英文",
        ]
        for query in test_cases:
            result = classifier._quick_classify(query)
            if result is not None:
                assert result.intent == "other"

    def test_drug_signals_return_none(self):
        """药品信号词返回 None（需 LLM 精确分类）。"""
        classifier = IntentClassifier(api_key="test-key")

        test_cases = [
            "阿司匹林一天吃几次？",
            "布洛芬有什么副作用？",
            "头孢类抗生素的禁忌",
            "对乙酰氨基酚的用法用量",
        ]
        for query in test_cases:
            result = classifier._quick_classify(query)
            assert result is None, f"'{query}' should return None for LLM classification"


# ============================================================
# IntentClassifier.classify
# ============================================================
class TestIntentClassifierClassify:
    """测试 classify 主方法。"""

    def test_empty_query(self):
        """空查询返回默认 drug_inquiry。"""
        classifier = IntentClassifier(api_key="test-key")
        result = classifier.classify("")
        assert result.intent == "drug_inquiry"

    def test_whitespace_query(self):
        """全空白查询返回默认 drug_inquiry。"""
        classifier = IntentClassifier(api_key="test-key")
        result = classifier.classify("   \n  ")
        assert result.intent == "drug_inquiry"

    def test_quick_classify_non_drug(self):
        """快速预判为非药品问题。"""
        classifier = IntentClassifier(api_key="test-key")
        result = classifier.classify("今天天气怎么样？")
        assert result.intent == "other"
        assert result.confidence > 0.9

    def test_llm_classify_drug(self, mock_dashscope_response):
        """LLM 分类为药品问题。"""
        mock_resp = mock_dashscope_response(
            choices_content='{"intent": "drug_inquiry", "confidence": 0.95}'
        )

        with patch("dashscope.Generation") as mock_gen:
            mock_gen.call.return_value = mock_resp
            classifier = IntentClassifier(api_key="test-key")
            result = classifier.classify("阿司匹林有什么副作用？")
            assert result.intent == "drug_inquiry"
            assert result.confidence == 0.95

    def test_llm_classify_other(self, mock_dashscope_response):
        """LLM 分类为非药品问题。"""
        mock_resp = mock_dashscope_response(
            choices_content='{"intent": "other", "confidence": 0.88}'
        )

        with patch("dashscope.Generation") as mock_gen:
            mock_gen.call.return_value = mock_resp
            classifier = IntentClassifier(api_key="test-key")
            result = classifier.classify("推荐一本书给我")
            assert result.intent == "other"

    def test_api_failure_graceful(self, mock_dashscope_response):
        """API 失败时降级为 drug_inquiry（宽容策略）。"""
        mock_resp = mock_dashscope_response(status_code=500)

        with patch("dashscope.Generation") as mock_gen:
            mock_gen.call.return_value = mock_resp
            classifier = IntentClassifier(api_key="test-key")
            result = classifier.classify("阿司匹林怎么吃？")
            assert result.intent == "drug_inquiry"
            assert result.confidence == 0.5  # 默认置信度


# ============================================================
# IntentClassifier._parse_response
# ============================================================
class TestParseResponse:
    """测试响应解析。"""

    def test_parse_valid_json(self):
        """解析合法的 JSON 响应。"""
        classifier = IntentClassifier(api_key="test-key")
        result = classifier._parse_response('{"intent": "drug_inquiry", "confidence": 0.92}')
        assert result.intent == "drug_inquiry"
        assert result.confidence == 0.92

    def test_parse_json_with_markdown_wrapper(self):
        """解析被 markdown 代码块包裹的 JSON。"""
        classifier = IntentClassifier(api_key="test-key")
        result = classifier._parse_response('```json\n{"intent": "other", "confidence": 0.85}\n```')
        assert result.intent == "other"
        assert result.confidence == 0.85

    def test_parse_invalid_json_fallback(self):
        """无效 JSON 回退到默认值。"""
        classifier = IntentClassifier(api_key="test-key")
        result = classifier._parse_response("not valid json")
        assert result.intent == "drug_inquiry"  # 宽容回退
        assert result.confidence == 0.5

    def test_parse_empty_string(self):
        """空字符串回退。"""
        classifier = IntentClassifier(api_key="test-key")
        result = classifier._parse_response("")
        assert result.intent == "drug_inquiry"
        assert result.confidence == 0.5

    def test_confidence_clamped(self):
        """置信度钳制在 [0, 1]。"""
        classifier = IntentClassifier(api_key="test-key")
        result = classifier._parse_response('{"intent": "drug_inquiry", "confidence": 1.5}')
        assert result.confidence == 1.0

    def test_invalid_intent_value(self):
        """无效 intent 值回退为 other。"""
        classifier = IntentClassifier(api_key="test-key")
        result = classifier._parse_response('{"intent": "invalid", "confidence": 0.5}')
        assert result.intent == "other"


# ============================================================
# classify_intent 便捷函数
# ============================================================
class TestClassifyIntent:
    """测试 classify_intent 便捷函数。"""

    def test_returns_intent_result(self):
        """返回 IntentResult。"""
        with patch("app.online.intent.IntentClassifier.classify") as mock_classify:
            mock_classify.return_value = IntentResult(intent="drug_inquiry", confidence=0.9)
            result = classify_intent("测试查询", api_key="test-key")
            assert isinstance(result, IntentResult)
            assert result.intent == "drug_inquiry"
