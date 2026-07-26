"""
长期记忆管理器 —— 医生画像（执业信息）

实现：
- LLM 从每轮对话中异步提取医生明确陈述的执业信息
- 新信息覆盖旧信息（医生最新陈述为准）
- 无衰减（长期记忆永不过期）
- 低置信度字段在 prompt 中标注"医生自称"

使用方式:
    from app.services.user_profile_manager import UserProfileManager

    manager = UserProfileManager()

    # 对话完成后异步提取
    await manager.extract_and_save(user_id, session_id, user_msg, assistant_msg)

    # 每次请求加载医生画像
    profile_text = manager.format_profile_for_prompt(user_id)
"""

import asyncio
import json
import re
from typing import Optional

from loguru import logger

from app.config import config

# ============================================================
# 画像提取提示词
# ============================================================
_EXTRACT_SYSTEM_PROMPT = """你是一个医生画像提取助手。分析医生与临床病例分析助手的对话，提取医生**明确陈述**的个人执业信息。

可提取的字段（只提取医生明确说出的信息，绝对不要推测）：
1. name — 姓名
2. title — 职称（主任医师/副主任医师/主治医师/住院医师/临床药师等）
3. department — 科室（心内科/内分泌科/呼吸科/ICU等）
4. hospital — 所在医院/机构
5. specialty — 专业领域（心力衰竭/介入心脏病学/糖尿病等）
6. license_years — 执业年限（数字）
7. guideline_preference — 指南使用偏好（中国指南/NICE/ESC/ACC-AHA等）
8. patient_population — 主要患者群体（老年/儿童/孕产妇/重症等）
9. common_diseases — 日常常见病种

提取规则：
- **只提取医生明确陈述的内容**，不要推测、不要推断
- 每条记录带 confidence 评分（0.0~1.0），表示医生陈述的明确程度
- 直接陈述 → confidence 0.8~1.0（如"我是心内科的"→0.95）
- 模糊陈述 → confidence 0.4~0.6（如"我好像提到过我在心内科"→0.5）
- 间接提及但不是明确陈述 → 不要提取
- 如果对话中没有可提取的个人信息，返回空数组 []

返回格式：纯 JSON 数组，不要有其他文字。
如果没有可提取的信息，返回 []。
[
  {"field_name": "department", "field_value": "心血管内科", "confidence": 0.95},
  {"field_name": "specialty", "field_value": "心力衰竭", "confidence": 0.9}
]"""

# 有效的画像字段（医生维度）
_VALID_FIELDS = frozenset({
    "name",              # 姓名
    "title",             # 职称（主任医师/副主任医师/主治医师/住院医师/临床药师...）
    "department",        # 科室（心内科/内分泌科/呼吸科/ICU...）
    "hospital",          # 所在医院/机构
    "specialty",         # 专业领域（心力衰竭/介入心脏病学/糖尿病...）
    "license_years",     # 执业年限
    "guideline_preference", # 指南使用偏好（中国指南/NICE/ESC/ACC-AHA...）
    "patient_population",   # 主要患者群体（老年/儿童/孕产妇/重症...）
    "common_diseases",   # 日常常见病种
})

# 字段中文标签
_FIELD_LABELS = {
    "name": "姓名",
    "title": "职称",
    "department": "科室",
    "hospital": "所在医院",
    "specialty": "专业领域",
    "license_years": "执业年限",
    "guideline_preference": "指南偏好",
    "patient_population": "患者群体",
    "common_diseases": "常见病种",
}


# ============================================================
# UserProfileManager
# ============================================================
class UserProfileManager:
    """
    长期记忆管理器 —— 用户画像。

    使用方式:
        manager = UserProfileManager()
        await manager.extract_and_save(1, "sess_abc", "我有高血压...", "了解...")
        profile_text = manager.format_profile_for_prompt(1)
    """

    def __init__(self, mysql_client=None):
        """
        Args:
            mysql_client: MySQLClient 实例（不传则自动创建）
        """
        self._mysql = mysql_client
        self._own_mysql = mysql_client is None
        self._extract_model = config.user_profile_extract_model
        self._min_confidence = config.user_profile_min_confidence

    @property
    def mysql(self):
        if self._mysql is None:
            from app.db.mysql_client import MySQLClient
            self._mysql = MySQLClient()
            self._mysql.connect()
        return self._mysql

    # ----------------------------------------------------------
    # 公开 API
    # ----------------------------------------------------------
    async def extract_and_save(
        self,
        user_id: int,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
    ) -> list[dict]:
        """
        从一轮对话中异步提取用户画像字段并持久化。

        Args:
            user_id: 用户 ID
            session_id: 来源会话 ID（仅用于日志）
            user_msg: 用户消息
            assistant_msg: 助手回答

        Returns:
            提取到的画像字段列表
        """
        if not user_msg or not user_msg.strip():
            return []

        try:
            loop = asyncio.get_running_loop()
            fields = await loop.run_in_executor(
                None,
                self._extract_via_llm,
                user_msg,
                assistant_msg,
            )
        except Exception as e:
            logger.warning(f"画像提取 LLM 调用失败 [user={user_id}]: {e}")
            return []

        if not fields:
            logger.debug(f"本轮无画像字段可提取 [user={user_id}]")
            return []

        saved = []
        for f in fields:
            field_name = (f.get("field_name") or "").strip().lower()
            field_value = (f.get("field_value") or "").strip()[:500]
            confidence = float(f.get("confidence", 0.5))

            if not field_name or not field_value:
                continue
            if field_name not in _VALID_FIELDS:
                logger.debug(f"忽略无效画像字段: {field_name}")
                continue
            if confidence < self._min_confidence:
                logger.debug(
                    f"画像字段置信度过低 [user={user_id}]: "
                    f"{field_name}={field_value}, confidence={confidence:.2f}"
                )
                continue

            result = self._upsert_field(user_id, field_name, field_value, confidence)
            if result:
                saved.append(result)

        if saved:
            logger.info(
                f"画像提取完成 [user={user_id}]: "
                f"提取 {len(fields)} 字段, 保存 {len(saved)} 字段"
            )

        return saved

    def get_profile(self, user_id: int) -> dict:
        """
        获取用户完整画像。

        Returns:
            {field_name: {"field_value": str, "confidence": float}, ...}
        """
        conn = self.mysql.conn
        with conn.cursor() as cursor:
            sql = (
                "SELECT field_name, field_value, confidence "
                "FROM user_profiles "
                "WHERE user_id = %s"
            )
            cursor.execute(sql, (user_id,))
            rows = cursor.fetchall()

        return {
            row["field_name"]: {
                "field_value": row["field_value"],
                "confidence": row["confidence"],
            }
            for row in rows
        }

    def format_profile_for_prompt(self, user_id: int) -> str:
        """
        获取用户画像并格式化为 prompt 友好的文本。

        Returns:
            格式化的画像文本，如：
            "医生执业信息（基于对话中明确陈述的内容）：
            - 科室：心血管内科
            - 专业领域：心力衰竭（医生自称）"
            无画像时返回空字符串。
        """
        profile = self.get_profile(user_id)
        if not profile:
            return ""
        return self._format_profile_text(profile)

    def get_field(self, user_id: int, field_name: str) -> dict | None:
        """获取单个画像字段。"""
        conn = self.mysql.conn
        with conn.cursor() as cursor:
            sql = (
                "SELECT field_name, field_value, confidence "
                "FROM user_profiles "
                "WHERE user_id = %s AND field_name = %s"
            )
            cursor.execute(sql, (user_id, field_name))
            row = cursor.fetchone()
            if row:
                return {
                    "field_name": row["field_name"],
                    "field_value": row["field_value"],
                    "confidence": row["confidence"],
                }
            return None

    def delete_field(self, user_id: int, field_name: str) -> bool:
        """删除单个画像字段。返回是否实际删除了记录。"""
        conn = self.mysql.conn
        with conn.cursor() as cursor:
            sql = "DELETE FROM user_profiles WHERE user_id = %s AND field_name = %s"
            cursor.execute(sql, (user_id, field_name))
            conn.commit()
            return cursor.rowcount > 0

    def upsert_field(
        self, user_id: int, field_name: str, field_value: str, confidence: float = 1.0
    ) -> dict | None:
        """
        公开的画像字段写入接口（供 API 手动编辑使用）。

        用户手动编辑的字段默认 confidence=1.0（最高置信度）。
        field_value 为空时自动删除该字段。

        Raises:
            ValueError: field_name 不在有效字段列表中
        """
        field_name = field_name.strip().lower()
        if field_name not in _VALID_FIELDS:
            raise ValueError(f"无效的画像字段: {field_name}")
        field_value = (field_value or "").strip()[:500]
        if not field_value:
            self.delete_field(user_id, field_name)
            return None
        return self._upsert_field(user_id, field_name, field_value, confidence)

    @classmethod
    def get_valid_fields(cls) -> list[dict]:
        """返回所有有效画像字段及其中文标签。"""
        return [
            {"field_name": fn, "label": _FIELD_LABELS.get(fn, fn)}
            for fn in sorted(_VALID_FIELDS)
        ]

    def update_profile_batch(self, user_id: int, fields: list[dict]) -> list[dict]:
        """
        批量更新画像字段。

        Args:
            fields: [{"field_name": str, "field_value": str, "confidence": float}, ...]
                field_value 为空字符串表示删除该字段

        Returns:
            实际保存/删除的字段列表
        """
        results = []
        for f in fields:
            field_name = (f.get("field_name") or "").strip().lower()
            field_value = (f.get("field_value") or "").strip()[:500]
            confidence = float(f.get("confidence", 1.0))
            if not field_name or field_name not in _VALID_FIELDS:
                continue
            if not field_value:
                self.delete_field(user_id, field_name)
                results.append({"field_name": field_name, "deleted": True})
            else:
                r = self._upsert_field(user_id, field_name, field_value, confidence)
                if r:
                    results.append(r)
        return results

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------
    def _upsert_field(
        self,
        user_id: int,
        field_name: str,
        field_value: str,
        confidence: float,
    ) -> dict | None:
        """
        插入或更新画像字段。

        规则：新信息覆盖旧信息（用户最新陈述为准）。
        使用 INSERT ... ON DUPLICATE KEY UPDATE。
        """
        conn = self.mysql.conn
        with conn.cursor() as cursor:
            sql = (
                "INSERT INTO user_profiles "
                "(user_id, field_name, field_value, confidence) "
                "VALUES (%s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE "
                "field_value = VALUES(field_value), "
                "confidence = VALUES(confidence)"
            )
            cursor.execute(sql, (user_id, field_name, field_value, confidence))
            conn.commit()

            # 判断是插入还是更新
            if cursor.rowcount == 1:
                action = "新增"
            else:
                action = "更新"

        logger.debug(
            f"画像字段{action}: user={user_id}, "
            f"{field_name}={field_value}, confidence={confidence:.2f}"
        )
        return {
            "field_name": field_name,
            "field_value": field_value,
            "confidence": confidence,
            "action": action,
        }

    def _extract_via_llm(
        self,
        user_msg: str,
        assistant_msg: str,
    ) -> list[dict]:
        """调用 DashScope LLM 从对话中提取画像字段（同步方法，在线程池中执行）。"""
        from dashscope import Generation

        user_prompt = (
            f"用户消息：{user_msg[:500]}\n\n"
            f"助手回答：{assistant_msg[:300]}\n\n"
            f"请从以上对话中提取用户明确陈述的个人信息。"
        )

        response = Generation.call(
            model=self._extract_model,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=400,
            api_key=config.DASHSCOPE_API_KEY,
            result_format="message",
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"画像提取 API 错误: status={response.status_code}, "
                f"message={response.message}"
            )

        output = response.output
        if output is None:
            raise RuntimeError("画像提取 API 返回了空的 output")

        text = ""
        if output.choices:
            text = output.choices[0].message.content
        elif output.text:
            text = output.text

        if not text:
            return []

        # 解析 JSON（可能被 markdown 代码块包裹）
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
            return []
        except json.JSONDecodeError:
            logger.warning(f"画像提取 JSON 解析失败: {text[:200]}")
            return []

    @staticmethod
    def _format_profile_text(profile: dict) -> str:
        """
        将画像字典格式化为 prompt 友好文本。

        格式：
            医生执业信息（基于对话中明确陈述的内容）：
            - 科室：心血管内科
            - 专业领域：心力衰竭
            - 职称：主任医师（医生自称）

        低置信度（< 0.7）的字段标注"（医生自称）"。
        """
        if not profile:
            return ""

        lines = ["\n医生执业信息（基于对话中明确陈述的内容）："]
        for field_name, info in profile.items():
            label = _FIELD_LABELS.get(field_name, field_name)
            value = info["field_value"]
            confidence = info.get("confidence", 1.0)

            if confidence < 0.7:
                lines.append(f"- {label}：{value}（医生自称）")
            else:
                lines.append(f"- {label}：{value}")

        return "\n".join(lines) + "\n"

    def close(self) -> None:
        if self._own_mysql and self._mysql is not None:
            self._mysql.disconnect()
            self._mysql = None
