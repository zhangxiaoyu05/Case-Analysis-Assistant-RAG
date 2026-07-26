"""
中期记忆管理器 —— 跨会话医生临床动态记忆

实现：
- LLM 从每轮对话中异步提取医生的临床关注点/疑难点/诊疗偏好/计划/执业特征
- 关键词重叠度 Upsert 合并去重
- 每日衰减遗忘机制（×decay_factor）
- Top-K 召回 + 重要性累加

使用方式:
    from app.services.user_memory_manager import UserMemoryManager

    manager = UserMemoryManager()

    # 对话完成后异步提取
    await manager.extract_and_save(user_id, session_id, user_msg, assistant_msg)

    # 每次请求加载医生记忆
    memories_text = manager.get_top_memories(user_id)
"""

import asyncio
import json
import re
import threading
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from app.config import config

# ============================================================
# 记忆提取提示词
# ============================================================
_EXTRACT_SYSTEM_PROMPT = """你是一个医生画像提取助手。分析医生与临床病例分析助手的对话，提取医生在对话中表现出的关键特征。

提取记忆类型（只提取确定的信息，不要推测）：
1. **clinical_interest** — 医生频繁关注的疾病领域或诊疗问题（如"心衰""糖尿病""抗菌药物"）
2. **clinical_concern** — 医生表达的临床疑难点或不确定（如"对他汀不耐受的处理不确定"）
3. **clinical_preference** — 医生表达的诊疗偏好或处方习惯（如"倾向使用ACEI而非ARB"）
4. **clinical_plan** — 医生提到的学习研究计划（如"准备系统学习心衰指南"）
5. **clinical_fact** — 医生的执业信息/科室特征（如"我们心内科CCU""常见糖尿病合并CKD患者"）

提取规则：
- 只提取医生明确陈述的内容，不要推测
- 每条记忆 content 字段控制在 100 字以内
- keywords 字段用逗号分隔 1-5 个关键词
- 如果对话中没有有意义的信息，返回空数组 []
- 不要提取助手的回答内容，只提取医生表达的信息

返回格式：纯 JSON 数组，不要有其他文字。
如果没有可提取的记忆，返回 []。
[
  {"memory_type": "clinical_interest", "content": "医生关注慢性心力衰竭的GDMT方案", "keywords": "心衰,GDMT"},
  {"memory_type": "clinical_preference", "content": "偏好使用SGLT2i作为心衰基础治疗", "keywords": "SGLT2i,心衰"}
]"""


# ============================================================
# UserMemoryManager
# ============================================================
class UserMemoryManager:
    """
    中期记忆管理器。

    使用方式:
        manager = UserMemoryManager()
        await manager.extract_and_save(1, "sess_abc", "阿司匹林怎么吃？", "成人一次...")
        memories_text = manager.get_top_memories(1)
        manager.apply_decay(1)
    """

    def __init__(self, mysql_client=None):
        """
        Args:
            mysql_client: MySQLClient 实例（不传则自动创建）
        """
        self._mysql = mysql_client
        self._own_mysql = mysql_client is None
        self._extract_model = config.user_memory_extract_model
        self._max_per_user = config.user_memory_max_per_user
        self._decay_factor = config.user_memory_decay_factor
        self._min_importance = config.user_memory_min_importance
        self._recall_top_k = config.user_memory_recall_top_k
        self._merge_threshold = config.user_memory_merge_threshold

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
        从一轮对话中异步提取记忆并持久化。

        Args:
            user_id: 用户 ID
            session_id: 来源会话 ID
            user_msg: 用户消息
            assistant_msg: 助手回答

        Returns:
            提取到的记忆列表
        """
        if not user_msg or not user_msg.strip():
            return []

        try:
            # 在线程池中运行同步 LLM 调用
            loop = asyncio.get_running_loop()
            memories = await loop.run_in_executor(
                None,
                self._extract_via_llm,
                user_msg,
                assistant_msg,
            )
        except Exception as e:
            logger.warning(f"记忆提取 LLM 调用失败 [user={user_id}]: {e}")
            return []

        if not memories:
            logger.debug(f"本轮无有效记忆可提取 [user={user_id}]")
            return []

        # Upsert 每条记忆
        saved = []
        for mem in memories:
            memory_type = mem.get("memory_type", "clinical_fact")
            content = (mem.get("content") or "").strip()[:500]
            keywords = (mem.get("keywords") or "").strip()[:300]

            if not content or memory_type not in (
                "clinical_interest", "clinical_concern", "clinical_preference",
                "clinical_plan", "clinical_fact",
            ):
                continue

            result = self._upsert_memory(
                user_id, memory_type, content, keywords, session_id
            )
            if result:
                saved.append(result)

        if saved:
            logger.info(
                f"记忆提取完成 [user={user_id}]: "
                f"提取 {len(memories)} 条, 保存 {len(saved)} 条"
            )

        # 强制执行数量上限（删除 importance 最低的）
        self._enforce_limit(user_id)

        return saved

    def get_top_memories(self, user_id: int, limit: int | None = None) -> list[dict]:
        """
        获取用户最重要的 K 条记忆（按 importance_score 降序）。

        同时更新 access_count 和 last_accessed_at，小幅恢复 importance。

        Args:
            user_id: 用户 ID
            limit: 返回条数（默认从 config 读取）

        Returns:
            [{"memory_type": str, "content": str, "keywords": str, "importance_score": float}, ...]
        """
        limit = limit if limit is not None else self._recall_top_k

        conn = self.mysql.conn
        with conn.cursor() as cursor:
            sql = (
                "SELECT id, memory_type, content, keywords, importance_score, "
                "source_session_id, access_count "
                "FROM user_memories "
                "WHERE user_id = %s "
                "ORDER BY importance_score DESC "
                "LIMIT %s"
            )
            cursor.execute(sql, (user_id, limit))
            rows = cursor.fetchall()

        # 更新访问计数 + 重要性小幅恢复
        for row in rows:
            new_importance = min(row["importance_score"] + 0.1, 1.0)
            with conn.cursor() as cursor:
                sql = (
                    "UPDATE user_memories "
                    "SET access_count = access_count + 1, "
                    "    last_accessed_at = NOW(), "
                    "    importance_score = %s "
                    "WHERE id = %s"
                )
                cursor.execute(sql, (new_importance, row["id"]))
            conn.commit()

        return [
            {
                "memory_type": row["memory_type"],
                "content": row["content"],
                "keywords": row["keywords"],
                "importance_score": row["importance_score"],
                "source_session_id": row["source_session_id"],
                "access_count": row["access_count"],
            }
            for row in rows
        ]

    def format_memories_for_prompt(
        self, user_id: int, limit: int | None = None, max_tokens: int | None = None
    ) -> str:
        """
        获取用户记忆并格式化为 prompt 友好的文本，支持 token 预算截断。

        Args:
            user_id: 用户 ID
            limit: 召回条数（默认从 config 读取）
            max_tokens: token 预算上限（默认从 config 读取），超出则逐条截断

        Returns:
            格式化的记忆文本。无记忆时返回空字符串。
        """
        if max_tokens is None:
            max_tokens = config.user_memory_max_tokens_in_prompt

        memories = self.get_top_memories(user_id, limit)
        if not memories:
            return ""

        from app.services.memory_manager import estimate_tokens

        header = "\n医生临床特征（基于历史对话）：\n"
        current_tokens = estimate_tokens(header)
        lines = []

        type_labels = {
            "clinical_interest": "关注领域",
            "clinical_concern": "临床疑难点",
            "clinical_preference": "诊疗偏好",
            "clinical_plan": "学习/研究计划",
            "clinical_fact": "执业特征",
        }

        truncated = False
        for mem in memories:
            mtype = mem.get("memory_type", "")
            content = mem.get("content", "")
            label = type_labels.get(mtype, mtype)
            line = f"- [{label}] {content}\n"
            line_tokens = estimate_tokens(line)

            if current_tokens + line_tokens > max_tokens:
                truncated = True
                break

            lines.append(line)
            current_tokens += line_tokens

        if not lines:
            return ""

        result = header + "".join(lines)
        if truncated:
            result += "…(记忆已截断)\n"

        return result

    def apply_decay(self, user_id: int) -> dict:
        """
        对指定用户的所有记忆执行一次衰减。

        Returns:
            {"decayed": int, "deleted": int}
        """
        conn = self.mysql.conn

        # 衰减
        with conn.cursor() as cursor:
            sql = (
                "UPDATE user_memories "
                "SET importance_score = importance_score * %s "
                "WHERE user_id = %s"
            )
            cursor.execute(sql, (self._decay_factor, user_id))
            conn.commit()
            decayed = cursor.rowcount

        # 清理低于阈值的记忆
        with conn.cursor() as cursor:
            sql = (
                "DELETE FROM user_memories "
                "WHERE user_id = %s AND importance_score < %s"
            )
            cursor.execute(sql, (user_id, self._min_importance))
            conn.commit()
            deleted = cursor.rowcount

        if decayed or deleted:
            logger.info(
                f"记忆衰减完成 [user={user_id}]: "
                f"衰减 {decayed} 条, 清理 {deleted} 条"
            )

        return {"decayed": decayed, "deleted": deleted}

    def apply_decay_all_users(self) -> dict:
        """
        对所有用户的记忆执行一次衰减（后台定时任务调用）。

        Returns:
            {"users": int, "decayed": int, "deleted": int}
        """
        conn = self.mysql.conn

        # 获取所有有记忆的用户
        with conn.cursor() as cursor:
            cursor.execute("SELECT DISTINCT user_id FROM user_memories")
            user_ids = [row["user_id"] for row in cursor.fetchall()]

        total_decayed, total_deleted = 0, 0
        for uid in user_ids:
            result = self.apply_decay(uid)
            total_decayed += result["decayed"]
            total_deleted += result["deleted"]

        logger.info(
            f"全量记忆衰减完成: {len(user_ids)} 用户, "
            f"衰减 {total_decayed} 条, 清理 {total_deleted} 条"
        )

        return {
            "users": len(user_ids),
            "decayed": total_decayed,
            "deleted": total_deleted,
        }

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------
    def _upsert_memory(
        self,
        user_id: int,
        memory_type: str,
        content: str,
        keywords: str,
        source_session_id: str,
    ) -> dict | None:
        """
        插入或合并记忆。

        合并条件：关键词重叠度 > merge_threshold。
        合并时 importance_score 累加（上限 1.0）。
        """
        conn = self.mysql.conn

        # 查找该用户同类型的已有记忆
        with conn.cursor() as cursor:
            sql = (
                "SELECT id, content, keywords, importance_score "
                "FROM user_memories "
                "WHERE user_id = %s AND memory_type = %s"
            )
            cursor.execute(sql, (user_id, memory_type))
            existing = cursor.fetchall()

        # 计算关键词重叠度，找到最佳匹配
        new_kw_set = set(k.strip() for k in (keywords or "").split(",") if k.strip())
        best_match = None
        best_overlap = 0.0

        for row in existing:
            old_kw_set = set(
                k.strip() for k in (row["keywords"] or "").split(",") if k.strip()
            )
            if not new_kw_set or not old_kw_set:
                continue
            overlap = len(new_kw_set & old_kw_set) / len(new_kw_set | old_kw_set)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = row

        if best_match and best_overlap >= self._merge_threshold:
            # 合并：更新内容 + 累加重要性
            new_importance = min(best_match["importance_score"] + 1.0, 1.0)
            with conn.cursor() as cursor:
                sql = (
                    "UPDATE user_memories "
                    "SET content = %s, keywords = %s, importance_score = %s, "
                    "    source_session_id = %s "
                    "WHERE id = %s"
                )
                cursor.execute(
                    sql,
                    (content, keywords, new_importance, source_session_id, best_match["id"]),
                )
                conn.commit()
            logger.debug(
                f"记忆合并: type={memory_type}, overlap={best_overlap:.2f}, "
                f"importance={best_match['importance_score']:.2f}→{new_importance:.2f}"
            )
            return {"id": best_match["id"], "memory_type": memory_type,
                    "content": content, "merged": True}
        else:
            # 新增
            with conn.cursor() as cursor:
                sql = (
                    "INSERT INTO user_memories "
                    "(user_id, memory_type, content, keywords, source_session_id) "
                    "VALUES (%s, %s, %s, %s, %s)"
                )
                cursor.execute(
                    sql,
                    (user_id, memory_type, content, keywords, source_session_id),
                )
                conn.commit()
                mem_id = cursor.lastrowid
            logger.debug(
                f"记忆新增: id={mem_id}, type={memory_type}, "
                f"best_overlap={best_overlap:.2f}"
            )
            return {"id": mem_id, "memory_type": memory_type,
                    "content": content, "merged": False}

    def _enforce_limit(self, user_id: int) -> None:
        """强制执行每个用户的记忆数量上限（删除 importance 最低的）。"""
        conn = self.mysql.conn
        with conn.cursor() as cursor:
            sql = (
                "SELECT COUNT(*) AS cnt FROM user_memories WHERE user_id = %s"
            )
            cursor.execute(sql, (user_id,))
            row = cursor.fetchone()
            count = row["cnt"] if row else 0

        if count > self._max_per_user:
            excess = count - self._max_per_user
            with conn.cursor() as cursor:
                sql = (
                    "DELETE FROM user_memories "
                    "WHERE user_id = %s "
                    "ORDER BY importance_score ASC "
                    "LIMIT %s"
                )
                cursor.execute(sql, (user_id, excess))
                conn.commit()
            logger.info(
                f"记忆数量超限 [user={user_id}]: "
                f"{count} → {self._max_per_user}（删除 {excess} 条）"
            )

    def _extract_via_llm(
        self,
        user_msg: str,
        assistant_msg: str,
    ) -> list[dict]:
        """调用 DashScope LLM 从对话中提取记忆（同步方法，在线程池中执行）。"""
        from dashscope import Generation

        user_prompt = (
            f"用户消息：{user_msg[:500]}\n\n"
            f"助手回答：{assistant_msg[:300]}\n\n"
            f"请从以上对话中提取用户的关键信息。"
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
                f"记忆提取 API 错误: status={response.status_code}, "
                f"message={response.message}"
            )

        output = response.output
        if output is None:
            raise RuntimeError("记忆提取 API 返回了空的 output")

        text = ""
        if output.choices:
            text = output.choices[0].message.content
        elif output.text:
            text = output.text

        if not text:
            return []

        # 解析 JSON（可能被 markdown 代码块包裹）
        text = text.strip()
        # 去掉 ```json ... ``` 包裹
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
            return []
        except json.JSONDecodeError:
            logger.warning(f"记忆提取 JSON 解析失败: {text[:200]}")
            return []

    @staticmethod
    def _format_memories_text(memories: list[dict]) -> str:
        """
        将记忆列表格式化为 prompt 友好文本。

        格式：
            医生临床特征（基于历史对话）：
            - 关注疾病：心衰、糖尿病
            - 倾向使用SGLT2i作为心衰基础治疗
        """
        if not memories:
            return ""

        lines = ["\n医生临床特征（基于历史对话）："]
        for mem in memories:
            mtype = mem.get("memory_type", "")
            content = mem.get("content", "")

            type_labels = {
                "clinical_interest": "关注领域",
                "clinical_concern": "临床疑难点",
                "clinical_preference": "诊疗偏好",
                "clinical_plan": "学习/研究计划",
                "clinical_fact": "执业特征",
            }
            label = type_labels.get(mtype, mtype)
            lines.append(f"- [{label}] {content}")

        return "\n".join(lines) + "\n"

    def close(self) -> None:
        if self._own_mysql and self._mysql is not None:
            self._mysql.disconnect()
            self._mysql = None
