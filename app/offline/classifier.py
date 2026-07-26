"""
文档自动分类器 (v1.1.0)

使用 qwen-flash 自动识别文档类型（drug/disease/guideline/literature），
并根据类型提取元数据（疾病名、指南标题、研究类型、证据级别等）。

参照 app/online/intent.py:Gatekeeper 的 LLM 调用模式设计。
包含规则 fallback，在 LLM 调用失败时兜底。

使用方式:
    from app.offline.classifier import classify_document, ClassifyResult

    result = classify_document(raw_text, filename="doc.txt")
    print(result.source_type)  # "disease"
    print(result.inferred_name)  # "原发性高血压"
    print(result.extra_fields)  # {"department": "心内科", ...}
"""

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

from app.config import config

# ============================================================
# 加载 prompt 模板
# ============================================================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PROMPTS_PATH = _PROJECT_ROOT / "config" / "prompts.yaml"

with open(_PROMPTS_PATH, "r", encoding="utf-8") as _f:
    _PROMPTS = yaml.safe_load(_f)

_CLASSIFIER_SYSTEM = _PROMPTS["document_classifier"]["system"]
_CLASSIFIER_USER = _PROMPTS["document_classifier"]["user"]


# ============================================================
# 数据类
# ============================================================
@dataclass
class ClassifyResult:
    """文档分类结果"""

    source_type: str  # "drug" | "disease" | "guideline" | "literature"
    inferred_name: str  # 文档名称
    confidence: str = "medium"  # "high" | "medium" | "low"
    extra_fields: dict = field(default_factory=dict)
    classification_method: str = "llm"  # "llm" | "rule"
    elapsed_ms: float = 0.0


# ============================================================
# 规则 fallback 分类
# ============================================================
def _rule_based_classify(text: str, filename: str = "") -> ClassifyResult:
    """
    基于关键词规则的文档分类（LLM 失败时的 fallback）。

    检测顺序按特异性从高到低：
    1. drug: 【药品名称】/【适应症】等章节标记
    2. guideline: 推荐意见 + 推荐等级/证据级别
    3. literature: IMRaD 结构 / DOI / 研究类型声明
    4. disease: 病因/流行病学/诊断/临床表现/并发症/治疗综合模式
    5. 兜底: drug
    """
    text_sample = text[:3000]  # 只检查前 3000 字符
    inferred_name = filename.rsplit(".", 1)[0] if filename else "unknown"

    # 1. Drug detection: 【章节名】模式
    drug_markers = ["【药品名称】", "【适应症】", "【用法用量】", "【不良反应】",
                    "【禁忌】", "【注意事项】", "【药理作用】"]
    drug_score = sum(1 for m in drug_markers if m in text_sample)
    if drug_score >= 2:
        # 尝试提取药品名称
        name_match = re.search(r"通用名称[：:]\s*(.+)", text_sample)
        if name_match:
            inferred_name = name_match.group(1).strip()
        return ClassifyResult(
            source_type="drug",
            inferred_name=inferred_name,
            confidence="high",
            classification_method="rule",
        )

    # 2. Guideline detection: 推荐等级 + 证据级别
    guideline_keywords = ["推荐意见", "推荐等级", "证据级别", "GRADE",
                          "强推荐", "Ⅱa", "Ⅱb", "Ⅲ类"]
    guideline_score = sum(1 for k in guideline_keywords if k in text_sample)
    if guideline_score >= 3:
        # 尝试提取指南标题
        title_match = re.search(r"#\s*(.+?)(?:指南|摘要)", text_sample)
        if title_match:
            inferred_name = title_match.group(1).strip() + "指南"
        return ClassifyResult(
            source_type="guideline",
            inferred_name=inferred_name,
            confidence="high" if guideline_score >= 5 else "medium",
            classification_method="rule",
        )

    # 3. Literature detection: IMRaD / DOI / study type
    imrad_markers = ["Abstract", "Introduction", "Methods", "Results",
                     "Discussion", "Conclusions"]
    imrad_score = sum(1 for m in imrad_markers if m in text_sample)
    lit_markers = ["doi:", "DOI:", "RCT", "meta-analysis", "randomized controlled trial",
                   "systematic review", "cohort study", "observational study",
                   "PubMed", "ClinicalTrials.gov"]
    lit_score = sum(1 for m in lit_markers if m.lower() in text_sample.lower())
    if imrad_score >= 3 or lit_score >= 3:
        # 尝试提取文献标题
        title_match = re.search(r"#\s*(.+?)(?:\n|$)", text_sample)
        if title_match:
            inferred_name = title_match.group(1).strip()
        study_type = ""
        for st in ["meta-analysis", "RCT", "cohort", "observational", "systematic review"]:
            if st.lower() in text_sample.lower():
                study_type = st
                break
        extra = {"study_type": study_type}
        ev_match = re.search(r"Evidence Level[:\s]*(\S+)", text_sample)
        if ev_match:
            extra["evidence_level"] = ev_match.group(1)
        return ClassifyResult(
            source_type="literature",
            inferred_name=inferred_name,
            confidence="high" if imrad_score >= 4 else "medium",
            extra_fields=extra,
            classification_method="rule",
        )

    # 4. Disease detection: 综合性医学知识模式
    disease_keywords = ["病因", "流行病学", "诊断标准", "临床表现",
                        "并发症", "治疗原则", "随访", "监测",
                        "危险因素", "靶器官", "分期", "预后"]
    disease_score = sum(1 for k in disease_keywords if k in text_sample)
    if disease_score >= 4:
        # 尝试提取疾病名称
        title_match = re.search(r"#\s*(.+?)(?:\n|$)", text_sample)
        if title_match:
            inferred_name = title_match.group(1).strip()
        return ClassifyResult(
            source_type="disease",
            inferred_name=inferred_name,
            confidence="high" if disease_score >= 6 else "medium",
            classification_method="rule",
        )

    # 5. Fallback: drug（保守默认）
    return ClassifyResult(
        source_type="drug",
        inferred_name=inferred_name,
        confidence="low",
        classification_method="rule",
    )


# ============================================================
# LLM 分类器
# ============================================================
class DocumentClassifier:
    """
    基于 LLM 的文档分类器。

    使用 qwen-flash 对文档前 ~2000 字符进行分析：
    1. 判断 source_type（drug/disease/guideline/literature）
    2. 提取文档名称
    3. 提取类型专属元数据

    LLM 调用失败时自动降级到规则 fallback。

    使用方式:
        classifier = DocumentClassifier()
        result = classifier.classify(raw_text, filename="高血压.txt")
    """

    def __init__(self) -> None:
        self._model: str = config.classifier_model
        self._temperature: float = config.classifier_temperature
        self._max_tokens: int = config.classifier_max_tokens
        self._api_key: str = config.DASHSCOPE_API_KEY

    def classify(self, text: str, filename: str = "") -> ClassifyResult:
        """
        对文档进行分类。

        Args:
            text: 文档原始文本
            filename: 文件名（用于辅助推断和日志）

        Returns:
            ClassifyResult — 含 source_type, inferred_name, confidence, extra_fields
        """
        t_start = time.time()

        # 截取前 2000 字符（足够分类，节省 token）
        content_sample = text[:2000]

        try:
            result = self._llm_classify(content_sample, filename)
            result.elapsed_ms = (time.time() - t_start) * 1000
            logger.info(
                f"🤖 LLM 分类完成: {result.source_type} "
                f"→ \"{result.inferred_name}\" "
                f"(confidence={result.confidence}, {result.elapsed_ms:.0f}ms)"
            )
            return result
        except Exception as e:
            logger.warning(f"LLM 分类失败，降级到规则分类: {e}")
            result = _rule_based_classify(text, filename)
            result.elapsed_ms = (time.time() - t_start) * 1000
            logger.info(
                f"📏 规则分类完成: {result.source_type} "
                f"→ \"{result.inferred_name}\" "
                f"(confidence={result.confidence}, {result.elapsed_ms:.0f}ms)"
            )
            return result

    def _llm_classify(self, content: str, filename: str) -> ClassifyResult:
        """调用 DashScope LLM 进行分类"""
        from dashscope import Generation

        user_prompt = _CLASSIFIER_USER.format(
            filename=filename or "未知",
            content=content,
        )

        messages = [
            {"role": "system", "content": _CLASSIFIER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

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
        if output is None:
            raise RuntimeError("DashScope 返回空 output")

        raw_content: str = ""
        if output.choices and output.choices[0].message.content:
            raw_content = output.choices[0].message.content
        elif output.text:
            raw_content = output.text
        else:
            raise RuntimeError("DashScope 返回空 choices 和 text")

        return self._parse_response(raw_content, content, filename)

    def _parse_response(
        self, raw: str, content: str, filename: str
    ) -> ClassifyResult:
        """解析 LLM 的 JSON 输出"""
        # 1. 尝试直接 JSON 解析
        try:
            data = json.loads(raw.strip())
            return self._validate_and_build(data, filename)
        except json.JSONDecodeError:
            pass

        # 2. 尝试从 markdown 代码块中提取 JSON
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1).strip())
                return self._validate_and_build(data, filename)
            except json.JSONDecodeError:
                pass

        # 3. 尝试用正则提取关键字段
        source_type_match = re.search(
            r'"source_type"\s*:\s*"(drug|disease|guideline|literature)"', raw
        )
        name_match = re.search(r'"inferred_name"\s*:\s*"([^"]+)"', raw)
        conf_match = re.search(r'"confidence"\s*:\s*"(high|medium|low)"', raw)

        if source_type_match:
            source_type = source_type_match.group(1)
            inferred_name = name_match.group(1) if name_match else (
                filename.rsplit(".", 1)[0] if filename else "unknown"
            )
            confidence = conf_match.group(1) if conf_match else "low"
            logger.warning(f"JSON 解析失败，regex 提取部分字段: {raw[:100]}")
            return ClassifyResult(
                source_type=source_type,
                inferred_name=inferred_name,
                confidence=confidence,
                extra_fields={},
                classification_method="llm",
            )

        # 4. 完全无法解析，降级到规则
        logger.warning(f"LLM 输出无法解析，降级到规则分类: {raw[:100]}")
        return _rule_based_classify(content, filename)

    def _validate_and_build(
        self, data: dict, filename: str
    ) -> ClassifyResult:
        """验证 JSON 数据并构建 ClassifyResult"""
        source_type = data.get("source_type", "drug")
        if source_type not in ("drug", "disease", "guideline", "literature"):
            source_type = "drug"

        inferred_name = data.get("inferred_name", "")
        if not inferred_name and filename:
            inferred_name = filename.rsplit(".", 1)[0]

        confidence = data.get("confidence", "medium")
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"

        extra_fields = data.get("extra_fields", {})
        if not isinstance(extra_fields, dict):
            extra_fields = {}

        return ClassifyResult(
            source_type=source_type,
            inferred_name=inferred_name,
            confidence=confidence,
            extra_fields=extra_fields,
            classification_method="llm",
        )


# ============================================================
# 模块级便捷函数
# ============================================================
# 模块级单例（避免重复创建）
_classifier_instance: Optional[DocumentClassifier] = None


def classify_document(text: str, filename: str = "") -> ClassifyResult:
    """
    对文档进行自动分类的便捷函数。

    Args:
        text: 文档原始文本
        filename: 文件名（用于辅助推断）

    Returns:
        ClassifyResult — 含 source_type, inferred_name, confidence, extra_fields
    """
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = DocumentClassifier()
    return _classifier_instance.classify(text, filename)
