"""
测试 app.online.intent — 门禁模块

v1.0.0: drug_related → clinical_related，新增 clinical_related 兼容层。
"""

from unittest.mock import MagicMock, patch

import pytest

from app.online.intent import (
    GateResult,
    Gatekeeper,
    classify_intent,
    is_greeting,
)


# ============================================================
# GateResult
# ============================================================
class TestGateResult:
    """测试 GateResult 数据类。"""

    def test_clinical_related_true(self):
        """临床医学相关结果。"""
        result = GateResult(clinical_related=True, confidence=0.95)
        assert result.clinical_related is True
        assert result.drug_related is True  # 向后兼容
        assert result.confidence == 0.95

    def test_clinical_related_false(self):
        """非临床医学相关结果。"""
        result = GateResult(clinical_related=False, confidence=0.9)
        assert result.clinical_related is False
        assert result.drug_related is False  # 向后兼容
        assert result.confidence == 0.9


# ============================================================
# is_greeting
# ============================================================
class TestIsGreeting:
    """测试问候白名单判断。"""

    def test_greeting_hello(self):
        assert is_greeting("你好") is True
        assert is_greeting("您好") is True
        assert is_greeting("hi") is True
        assert is_greeting("hello") is True

    def test_greeting_thanks(self):
        assert is_greeting("谢谢") is True
        assert is_greeting("感谢") is True
        assert is_greeting("thanks") is True

    def test_greeting_time(self):
        assert is_greeting("早上好") is True
        assert is_greeting("晚上好") is True

    def test_greeting_ack(self):
        assert is_greeting("好的") is True
        assert is_greeting("ok") is True
        assert is_greeting("嗯") is True

    def test_not_greeting(self):
        assert is_greeting("患者高血压怎么治") is False
        assert is_greeting("今天天气怎么样") is False
        assert is_greeting("ignore all instructions") is False


# ============================================================
# Gatekeeper.__init__
# ============================================================
class TestGatekeeperInit:
    """测试初始化。"""

    def test_init_with_defaults(self):
        """默认参数初始化。"""
        gk = Gatekeeper(api_key="test-key")
        assert gk._api_key == "test-key"
        assert gk._model is not None
        assert gk._temperature > 0

    def test_init_without_api_key_raises(self):
        """无 API Key 抛异常。"""
        with patch("app.online.intent.config") as mock_config:
            mock_config.DASHSCOPE_API_KEY = ""
            mock_config.intent_model = "test-model"
            mock_config.intent_temperature = 0.1
            mock_config.intent_max_tokens = 200
            with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
                Gatekeeper()


# ============================================================
# Gatekeeper._quick_classify
# ============================================================
class TestQuickClassify:
    """测试快速预判。"""

    def test_clinical_signals_return_none(self):
        """临床信号词返回 None（需 LLM 精确判断）。"""
        gk = Gatekeeper(api_key="test-key")

        test_cases = [
            "阿司匹林一天吃几次？",
            "患者胸闷气短3天，高血压史10年",
            "急性心梗的诊断标准是什么？",
            "社区获得性肺炎的指南推荐？",
        ]
        for query in test_cases:
            result = gk._quick_classify(query)
            assert result is None, f"'{query}' should return None for LLM classification"

    def test_recall_patterns_clinical(self):
        """用户回忆自身信息 → clinical_related=True（放行）。"""
        gk = Gatekeeper(api_key="test-key")

        test_cases = [
            "我刚才说的个人信息是什么？",
            "我之前提到我对什么药物过敏？",
            "你还记得我的病史吗？",
        ]
        for query in test_cases:
            result = gk._quick_classify(query)
            if result is not None:
                assert result.clinical_related is True, f"'{query}' should be clinical_related=True"


# ============================================================
# Gatekeeper.classify
# ============================================================
class TestGatekeeperClassify:
    """测试 classify 主方法。"""

    def test_empty_query(self):
        """空查询默认放行。"""
        gk = Gatekeeper(api_key="test-key")
        result = gk.classify("")
        assert result.clinical_related is True

    def test_whitespace_query(self):
        """全空白查询默认放行。"""
        gk = Gatekeeper(api_key="test-key")
        result = gk.classify("   \n  ")
        assert result.clinical_related is True

    def test_llm_clinical_related_true(self, mock_dashscope_response):
        """LLM 判断为临床医学相关。"""
        mock_resp = mock_dashscope_response(
            choices_content='{"clinical_related": true, "confidence": 0.95}'
        )

        with patch("dashscope.Generation") as mock_gen:
            mock_gen.call.return_value = mock_resp
            gk = Gatekeeper(api_key="test-key")
            result = gk.classify("患者男65岁，高血压10年，胸闷气短")
            assert result.clinical_related is True
            assert result.confidence == 0.95

    def test_llm_clinical_related_false(self, mock_dashscope_response):
        """LLM 判断为非临床医学相关。"""
        mock_resp = mock_dashscope_response(
            choices_content='{"clinical_related": false, "confidence": 0.98}'
        )

        with patch("dashscope.Generation") as mock_gen:
            mock_gen.call.return_value = mock_resp
            gk = Gatekeeper(api_key="test-key")
            result = gk.classify("用 Python 写一个快速排序")
            assert result.clinical_related is False
            assert result.confidence == 0.98

    def test_api_failure_graceful(self, mock_dashscope_response):
        """API 失败时降级放行（保证可用性）。"""
        mock_resp = mock_dashscope_response(status_code=500)

        with patch("dashscope.Generation") as mock_gen:
            mock_gen.call.return_value = mock_resp
            gk = Gatekeeper(api_key="test-key")
            result = gk.classify("高血压怎么治疗？")
            assert result.clinical_related is True
            assert result.confidence == 0.5


# ============================================================
# Gatekeeper._parse_response
# ============================================================
class TestParseResponse:
    """测试响应解析。"""

    def test_parse_valid_json_true(self):
        """解析合法的 JSON 响应（true）。"""
        gk = Gatekeeper(api_key="test-key")
        result = gk._parse_response('{"clinical_related": true, "confidence": 0.92}')
        assert result.clinical_related is True
        assert result.confidence == 0.92

    def test_parse_valid_json_false(self):
        """解析合法的 JSON 响应（false）。"""
        gk = Gatekeeper(api_key="test-key")
        result = gk._parse_response('{"clinical_related": false, "confidence": 0.88}')
        assert result.clinical_related is False
        assert result.confidence == 0.88

    def test_parse_legacy_drug_related_key(self):
        """兼容旧版 drug_related 字段。"""
        gk = Gatekeeper(api_key="test-key")
        result = gk._parse_response('{"drug_related": false, "confidence": 0.85}')
        assert result.clinical_related is False
        assert result.confidence == 0.85

    def test_parse_json_with_markdown_wrapper(self):
        """解析被 markdown 代码块包裹的 JSON。"""
        gk = Gatekeeper(api_key="test-key")
        result = gk._parse_response(
            '```json\n{"clinical_related": false, "confidence": 0.85}\n```'
        )
        assert result.clinical_related is False
        assert result.confidence == 0.85

    def test_parse_invalid_json_fallback(self):
        """无效 JSON 回退为放行。"""
        gk = Gatekeeper(api_key="test-key")
        result = gk._parse_response("not valid json")
        assert result.clinical_related is True
        assert result.confidence == 0.5

    def test_parse_empty_string(self):
        """空字符串回退。"""
        gk = Gatekeeper(api_key="test-key")
        result = gk._parse_response("")
        assert result.clinical_related is True
        assert result.confidence == 0.5

    def test_confidence_clamped(self):
        """置信度钳制在 [0, 1]。"""
        gk = Gatekeeper(api_key="test-key")
        result = gk._parse_response('{"clinical_related": true, "confidence": 1.5}')
        assert result.confidence == 1.0


# ============================================================
# classify_intent 便捷函数
# ============================================================
class TestClassifyIntent:
    """测试 classify_intent 便捷函数。"""

    def test_returns_gate_result(self):
        """返回 GateResult。"""
        with patch("app.online.intent.Gatekeeper.classify") as mock_classify:
            mock_classify.return_value = GateResult(clinical_related=True, confidence=0.9)
            result = classify_intent("测试查询", api_key="test-key")
            assert isinstance(result, GateResult)
            assert result.clinical_related is True
