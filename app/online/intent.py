"""
意图识别模块

判断用户问题是否属于药品知识领域，用于过滤无关问题和路由决策。

使用方式:
    from app.online.intent import IntentClassifier, IntentResult

    classifier = IntentClassifier()
    result = classifier.classify("阿司匹林一天吃几次？")
    if result.intent == "drug_inquiry":
        print(f"药品问题，置信度: {result.confidence}")
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger

from app.config import config

# 加载 prompts.yaml 中的意图识别模板
_PROMPTS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "prompts.yaml"
with open(_PROMPTS_PATH, "r", encoding="utf-8") as _f:
    _PROMPTS = yaml.safe_load(_f)

_INTENT_SYSTEM = _PROMPTS["intent"]["system"]
_INTENT_FEW_SHOT = _PROMPTS["intent"]["few_shot_examples"]


# ============================================================
# 数据类
# ============================================================
@dataclass
class IntentResult:
    """意图识别结果"""

    intent: str  # "drug_inquiry" | "chitchat" | "other"
    confidence: float  # 0.0 ~ 1.0


# ============================================================
# IntentClassifier
# ============================================================
class IntentClassifier:
    """
    药品问题意图分类器。

    使用 DashScope Generation API + intent 提示词模板
    判断用户问题是否属于药品知识领域。

    使用方式:
        classifier = IntentClassifier()
        result = classifier.classify("布洛芬可以和酒精一起用吗？")
        print(result.intent)  # "drug_inquiry"
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        api_key: str | None = None,
    ) -> None:
        """
        初始化意图分类器。

        Args:
            model: 模型名（默认 config.intent_model = qwen-flash）
            temperature: 温度（默认 config.intent_temperature = 0.1）
            max_tokens: 最大 token（默认 config.intent_max_tokens = 200）
            api_key: DashScope API Key（默认从 config 读取）
        """
        self._model = model or config.intent_model
        self._temperature = temperature if temperature is not None else config.intent_temperature
        self._max_tokens = max_tokens or config.intent_max_tokens
        self._api_key = api_key or config.DASHSCOPE_API_KEY

        if not self._model:
            raise ValueError(
                "intent_model 未配置。请在 config/config.yaml 的 models.intent.model 中设置。"
            )
        if not self._api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY 未配置。请设置环境变量或在初始化 IntentClassifier 时传入 api_key。"
            )

    def classify(self, query: str) -> IntentResult:
        """
        对用户问题执行意图分类。

        Args:
            query: 用户问题文本

        Returns:
            IntentResult — intent 为 "drug_inquiry" 或 "other"，confidence 为置信度
        """
        if not query or not query.strip():
            logger.warning("收到空查询，返回 drug_inquiry 默认意图")
            return IntentResult(intent="drug_inquiry", confidence=0.3)

        # 快速预判：明显的非药品问题直接返回
        quick_check = self._quick_classify(query)
        if quick_check is not None:
            return quick_check

        # 构造消息
        messages = self._build_messages(query)

        try:
            response_text = self._call_generation(messages)
            return self._parse_response(response_text)

        except Exception as e:
            logger.warning(f"意图识别 API 调用失败: {e}，默认视为药品问题")
            return IntentResult(intent="drug_inquiry", confidence=0.5)

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------
    def _quick_classify(self, query: str) -> IntentResult | None:
        """
        快速预判：基于关键词/模式的启发式分类。
        返回 None 表示无法快速判断，需走 LLM 流程。
        """
        query_stripped = query.strip()

        # 明显的问候/闲聊（无需 LLM，直接返回）
        chitchat_patterns = [
            r"^(你好|您好|hi|hello|嗨|喂|在吗|在不在|有人在吗)[\s!！。.,，]*$",
            r"^(谢谢|感谢|多谢|thanks|thank you|3q)[\s!！。.,，]*$",
            r"^(早上好|下午好|晚上好|早安|晚安|中午好)[\s!！。.,，]*$",
            r"^(好的|ok|OK|嗯|哦|知道了|明白了)[\s!！。.,，]*$",
        ]
        for pattern in chitchat_patterns:
            if re.search(pattern, query_stripped, re.IGNORECASE):
                logger.info(f"快速预判 — 闲聊（匹配问候模式）")
                return IntentResult(intent="chitchat", confidence=0.95)

        # 明显的非药品问题关键词
        non_drug_patterns = [
            r"天气",
            r"股票",
            r"汇率",
            r"新闻",
            r"电影",
            r"游戏",
            r"足球|篮球|比赛",
            r"旅游|攻略",
            r"菜谱|做饭|怎么做",
            r"编程|代码|python|java",
            r"今天是几号|现在几点",
            r"翻译|translate",
        ]
        query_lower = query.lower()
        for pattern in non_drug_patterns:
            if re.search(pattern, query):
                logger.info(f"快速预判 — 非药品问题（匹配: {pattern}）")
                return IntentResult(intent="other", confidence=0.95)

        # 明显的药品问题关键词
        drug_signals = [
            "药", "片", "胶囊", "丸", "注射液", "口服液",
            "剂量", "用法", "用量", "禁忌", "不良反应",
            "副作用", "适应症", "说明书", "抗生素",
            "mg", "毫克", "一天", "每日", "饭前", "饭后",
            "布洛芬", "阿司匹林", "对乙酰", "头孢", "阿莫西林",
            "服用", "吃药", "停药", "忌口", "过敏",
        ]
        has_drug_signal = any(s in query for s in drug_signals)

        if has_drug_signal:
            return None  # 有药品信号词，走 LLM 精确分类

        # 有问号但没有药品信号词 → 可能模糊，走 LLM
        if "?" in query or "？" in query or "吗" in query:
            return None

        # 其他情况：极短文本、无明显信息 → 默认药品问题（宽容策略）
        return None

    def _build_messages(self, query: str) -> list[dict]:
        """构造意图识别的 messages"""
        messages = [{"role": "system", "content": _INTENT_SYSTEM}]

        # 加入 few-shot 示例
        for example in _INTENT_FEW_SHOT:
            messages.append({"role": "user", "content": example["question"]})
            messages.append({"role": "assistant", "content": example["answer"]})

        messages.append({"role": "user", "content": query})
        return messages

    def _call_generation(self, messages: list[dict]) -> str:
        """调用 DashScope Generation API"""
        from dashscope import Generation

        response = Generation.call(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            api_key=self._api_key,
            result_format="message",  # 使用 chat 格式返回 choices
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"DashScope API 错误: status={response.status_code}, "
                f"message={response.message}"
            )

        output = response.output
        # DashScope 新版本可能返回 choices 或 text 两种格式
        if output.choices:
            return output.choices[0].message.content
        elif output.text:
            return output.text
        else:
            raise RuntimeError("DashScope API 返回了空的 choices 和 text")

    def _parse_response(self, text: str) -> IntentResult:
        """解析 LLM 返回的 JSON"""
        if not text:
            return IntentResult(intent="drug_inquiry", confidence=0.5)

        text = text.strip()

        # 尝试提取 JSON（去除可能的 markdown 代码块包裹）
        json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

        try:
            data = json.loads(text)
            intent = data.get("intent", "other")
            confidence = float(data.get("confidence", 0.5))

            # 校验
            if intent not in ("drug_inquiry", "chitchat", "other"):
                intent = "drug_inquiry"  # 兜底宽容

            # 钳制置信度到 [0, 1]
            confidence = max(0.0, min(1.0, confidence))

            return IntentResult(intent=intent, confidence=confidence)

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"意图识别 JSON 解析失败: {e}，原始文本: {text[:200]}")
            # 启发式回退：如果文本中包含 drug_inquiry 字样
            if "drug_inquiry" in text.lower():
                return IntentResult(intent="drug_inquiry", confidence=0.7)
            return IntentResult(intent="drug_inquiry", confidence=0.5)


# ============================================================
# 便捷函数
# ============================================================
def classify_intent(
    query: str,
    api_key: str | None = None,
) -> IntentResult:
    """
    便捷函数：一行调用完成意图分类。

    Args:
        query: 用户问题
        api_key: API Key（可选）

    Returns:
        IntentResult
    """
    classifier = IntentClassifier(api_key=api_key)
    return classifier.classify(query)
