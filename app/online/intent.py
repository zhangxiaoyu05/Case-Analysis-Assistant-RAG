"""
门禁模块（Gatekeeper）

二元判断用户问题是否与临床医学/病例分析相关。与临床无关的问题全部拦截，
不再区分 chitchat / general / attack 子类。

问候白名单由 intent_node 在调用 Gatekeeper 之前单独处理。

v1.0.0: drug_related → clinical_related，关键词从药品扩展为临床医学。

使用方式:
    from app.online.intent import Gatekeeper, GateResult, is_greeting

    if is_greeting(query):
        # 友好回应
        ...
    else:
        gk = Gatekeeper()
        result = gk.classify("患者男65岁，高血压10年，胸闷气短...")
        if result.clinical_related:
            print(f"临床问题，置信度: {result.confidence}")
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger

from app.config import config

# 加载 prompts.yaml 中的门禁模板
_PROMPTS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "prompts.yaml"
with open(_PROMPTS_PATH, "r", encoding="utf-8") as _f:
    _PROMPTS = yaml.safe_load(_f)

_GATEKEEPER_SYSTEM = _PROMPTS["gatekeeper"]["system"]
_GATEKEEPER_FEW_SHOT = _PROMPTS["gatekeeper"]["few_shot_examples"]

# ============================================================
# 问候白名单（供 intent_node 调用，不走门禁 LLM）
# ============================================================
_GREETING_PATTERNS = [
    r"^(你好|您好|hi|hello|嗨|喂|在吗|在不在|有人在吗)[\s!！。.,，]*$",
    r"^(谢谢|感谢|多谢|thanks|thank you|3q)[\s!！。.,，]*$",
    r"^(早上好|下午好|晚上好|早安|晚安|中午好)[\s!！。.,，]*$",
    r"^(好的|ok|OK|嗯|哦|知道了|明白了)[\s!！。.,，]*$",
]


def is_greeting(query: str) -> bool:
    """
    判断用户输入是否为日常问候/闲聊。

    由 intent_node 在调用 Gatekeeper 之前使用，
    命中则直接路由到 chitchat_node 返回友好回应。
    """
    query_stripped = query.strip()
    for pattern in _GREETING_PATTERNS:
        if re.search(pattern, query_stripped, re.IGNORECASE):
            return True
    return False


# ============================================================
# 数据类
# ============================================================
@dataclass
class GateResult:
    """门禁判断结果"""

    clinical_related: bool  # True = 临床医学相关，放行；False = 拦截
    confidence: float  # 0.0 ~ 1.0

    # 向后兼容别名
    @property
    def drug_related(self) -> bool:
        return self.clinical_related


# ============================================================
# Gatekeeper
# ============================================================
class Gatekeeper:
    """
    临床医学门禁。

    使用 DashScope Generation API + gatekeeper 提示词模板
    二元判断用户问题是否与临床医学/病例分析相关。

    使用方式:
        gk = Gatekeeper()
        result = gk.classify("患者胸闷气短3天，既往高血压史，如何诊治？")
        print(result.clinical_related)  # True
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        api_key: str | None = None,
    ) -> None:
        """
        初始化门禁。

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
                "DASHSCOPE_API_KEY 未配置。请设置环境变量或在初始化 Gatekeeper 时传入 api_key。"
            )

    def classify(self, query: str) -> GateResult:
        """
        判断用户问题是否与临床医学相关。

        Args:
            query: 用户问题文本

        Returns:
            GateResult — clinical_related=True 放行，False 拦截
        """
        if not query or not query.strip():
            logger.warning("收到空查询，默认放行")
            return GateResult(clinical_related=True, confidence=0.3)

        # 快速预判：明显的临床医学信号词直接放行
        quick = self._quick_classify(query)
        if quick is not None:
            return quick

        # 构造消息
        messages = self._build_messages(query)

        try:
            response_text = self._call_generation(messages)
            return self._parse_response(response_text)

        except Exception as e:
            logger.warning(f"门禁 API 调用失败: {e}，默认放行（保证可用性）")
            return GateResult(clinical_related=True, confidence=0.5)

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------
    def _quick_classify(self, query: str) -> GateResult | None:
        """
        快速预判：基于关键词的启发式分类。
        返回 None 表示无法快速判断，需走 LLM 流程。

        注意：问候白名单由 intent_node 在调用 Gatekeeper 之前处理，
        此处不再重复判断。
        """
        query_stripped = query.strip()

        # 对话回忆类问题 — 用户回忆自己之前分享的个人/医疗信息
        recall_patterns = [
            r"我(刚才|之前|前面|上次)(说|提到|问|讲)(的|过)?.*(信息|过敏|病史|情况|内容|话|药|病例)",
            r"(我的|我个人)(过敏|病史|用药|病历|健康|身体|信息|病例).*(是什么|有什么|多少|哪些|还记得|吗)",
            r"复述.*(我的|我刚才|我之前|我的病例)",
            r"(还记得|记不记得|记得吗).*(我的|我说|我提到|我的过敏|我的病史|我的病例)",
            r"(告诉我|说说|讲一下).*(我的|我说过|我提到过).*(信息|过敏|病史|情况|病例)",
        ]
        for pattern in recall_patterns:
            if re.search(pattern, query):
                logger.info("快速预判 — 对话回忆（用户询问自身信息），放行")
                return GateResult(clinical_related=True, confidence=0.85)

        # 明显的临床医学信号词 → 直接放行
        clinical_signals = [
            # 药品相关
            "药", "片", "胶囊", "丸", "注射液", "口服液",
            "剂量", "用法", "用量", "禁忌", "不良反应",
            "副作用", "适应症", "说明书", "抗生素",
            "mg", "毫克", "服用", "吃药", "停药", "忌口",
            # 临床医学
            "患者", "病例", "诊断", "治疗", "手术",
            "检查", "CT", "MRI", "X线", "超声", "心电图",
            "血压", "血糖", "体温", "心率", "呼吸",
            "主诉", "现病史", "既往史", "体格检查",
            "化验", "实验室", "影像学", "病理",
            "症状", "体征", "综合征", "并发症",
            "指南", "循证", "临床路径", "诊疗",
            "出院", "入院", "转科", "会诊",
            "高血压", "糖尿病", "冠心病", "心衰",
            "肺炎", "肝炎", "肾炎", "贫血",
            "布洛芬", "阿司匹林", "对乙酰", "头孢", "阿莫西林",
        ]
        if any(s in query for s in clinical_signals):
            return None  # 有信号词，走 LLM 精确判断

        # 有问号但没有信号词 → 可能模糊，走 LLM
        if "?" in query or "？" in query or "吗" in query:
            return None

        # 其他情况：极短文本、无明显信息 → 走 LLM
        return None

    def _build_messages(self, query: str) -> list[dict]:
        """构造门禁判断的 messages"""
        messages = [{"role": "system", "content": _GATEKEEPER_SYSTEM}]

        # 加入 few-shot 示例
        for example in _GATEKEEPER_FEW_SHOT:
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
            result_format="message",
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"DashScope API 错误: status={response.status_code}, "
                f"message={response.message}"
            )

        output = response.output
        if output.choices:
            return output.choices[0].message.content
        elif output.text:
            return output.text
        else:
            raise RuntimeError("DashScope API 返回了空的 choices 和 text")

    def _parse_response(self, text: str) -> GateResult:
        """解析 LLM 返回的 JSON"""
        if not text:
            return GateResult(clinical_related=True, confidence=0.5)

        text = text.strip()

        # 尝试提取 JSON（去除可能的 markdown 代码块包裹）
        json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

        try:
            data = json.loads(text)
            # 兼容新旧两种字段名
            clinical_related = data.get(
                "clinical_related",
                data.get("drug_related", True)
            )
            confidence = float(data.get("confidence", 0.5))

            # 确保 clinical_related 是 bool
            if not isinstance(clinical_related, bool):
                clinical_related = True  # 兜底放行

            # 钳制置信度到 [0, 1]
            confidence = max(0.0, min(1.0, confidence))

            return GateResult(clinical_related=clinical_related, confidence=confidence)

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"门禁 JSON 解析失败: {e}，原始文本: {text[:200]}")
            # 启发式回退
            if "true" in text.lower():
                return GateResult(clinical_related=True, confidence=0.7)
            return GateResult(clinical_related=True, confidence=0.5)


# ============================================================
# 便捷函数
# ============================================================
def classify_intent(
    query: str,
    api_key: str | None = None,
) -> GateResult:
    """
    便捷函数：一行调用完成门禁判断。

    Args:
        query: 用户问题
        api_key: API Key（可选）

    Returns:
        GateResult
    """
    gk = Gatekeeper(api_key=api_key)
    return gk.classify(query)
