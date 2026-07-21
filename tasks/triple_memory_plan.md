# 三段记忆体系 — 分步实施任务清单

> 状态：规划完成，待按 Phase 依次执行。
> 每个 Phase 独立可交付，包含完整的修改清单、测试要求、验证方案。

---

## Phase 1: 短期记忆完善 + 用户身份基础设施

**目标：** 补全短期记忆的 3 个缺陷 + 建立 user_id 机制，为中/长期记忆铺路。

### 修改清单

| 文件 | 改动 |
|------|------|
| `config/config.yaml` | 新增 `memory.token_threshold_ratio: 0.7`（token 阈值触发比例） |
| `app/config.py` | 新增 `memory_token_threshold_ratio` property |
| `app/services/memory_manager.py` | ① 新增 `_estimate_tokens()` 方法（tiktoken 估算）② `summarize()` 支持 token 阈值触发 ③ 读取 `memory_enabled` 配置 |
| `app/schemas/chat.py` | `ChatRequest` 增加 2 个可选字段：`user_id: Optional[str]`、`enable_memory: bool = True` |
| `app/api/routers/chat.py` | ① 检查 `enable_memory` + `memory_enabled` 开关 ② 不启用时跳过摘要直接传全部历史 ③ 接收并传递 `user_id` |
| `frontend/streamlit_app.py` | ① localStorage 自动生成/加载持久化 `user_id` ② 每次 API 请求传入 `user_id` ③ 增加短期记忆开关 UI 控件 |

### 新增文件
无（全部是修改）

### 测试要求
- 运行现有 288 个测试，确保无回归
- 手动验证：关闭 `enable_memory` 后，AI 不再具备短期记忆

### 验证方案
1. `python -m pytest tests/ -x -v` 全量通过
2. Chrome DevTools：`enable_memory=false` 时问"我叫X"，下一轮问"我叫什么？" → 应回答不知道
3. Chrome DevTools：`enable_memory=true`（默认）→ 记忆正常

---

## Phase 2: 中期记忆 —— 跨会话用户动态记忆

**目标：** 实现跨会话的用户偏好/关注点/兴趣记忆 + 衰减遗忘机制。

### 新增文件

| 文件 | 说明 |
|------|------|
| `scripts/migration_memory.sql` | DDL：`user_memories` 表 |
| `app/services/user_memory_manager.py` | 核心服务：LLM 提取 + CRUD + 衰减 + Top-K 召回 |
| `tests/test_services/test_user_memory_manager.py` | 单元测试（~15 cases） |

### 修改清单

| 文件 | 改动 |
|------|------|
| `config/config.yaml` | 新增 `user_memory` 配置节（extract_model, max_memories_per_user, decay_days, decay_factor, min_importance） |
| `app/config.py` | 新增 5+ 个 user_memory property |
| `config/prompts.yaml` | 所有 4 个 chat 模板 user prompt 增加 `{user_memories}` 占位符 |
| `app/online/generator.py` | ① `generate()` + `generate_stream()` 增加 `user_memories: str = ""` 参数 ② `_get_user_prompt()` 注入 user_memories |
| `app/graph/nodes.py` | `generate_node` + `general_node` 传递 `user_memories` |
| `app/graph/state.py` | 增加 `user_memories: str` 字段 |
| `app/api/routers/chat.py` | ① 有 user_id 时加载中期记忆并注入 ② 回答完成后 `asyncio.create_task` 异步提取新记忆 |
| `app/main.py` 或 `app/lifespan.py` | 注册每日衰减定时任务（asyncio 后台循环） |

### 中期记忆数据结构

```sql
CREATE TABLE user_memories (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    memory_type VARCHAR(32) NOT NULL COMMENT 'preference/concern/drug_interest/plan/fact',
    content VARCHAR(500) NOT NULL,
    keywords VARCHAR(300),
    importance_score FLOAT DEFAULT 1.0,
    source_session_id VARCHAR(32),
    access_count INT DEFAULT 0,
    last_accessed_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_score (user_id, importance_score DESC),
    INDEX idx_user_type (user_id, memory_type)
);
```

### 测试要求
- `test_user_memory_manager.py` ≥ 15 个测试
- 运行全量测试无回归

### 验证方案
1. 单元测试全部通过
2. Chrome DevTools 跨会话测试：
   - 场景 1：在一个会话中反复问"阿司匹林"相关问题 → 换新会话（新 session_id，同 user_id）→ 问"我之前关注什么药品？" → AI 应回答阿司匹林
   - 场景 2：多轮对话后等 10 分钟 → 检查 Redis TTL 过期后中期记忆仍存在（MySQL 持久化）

---

## Phase 3: 长期记忆 Part B —— 用户不可变人口属性

**目标：** 存储用户明确声明的姓名、生日、病史等事实，永不过期。

### 新增文件

| 文件 | 说明 |
|------|------|
| `app/services/user_profile_manager.py` | 核心服务：LLM 提取 + CRUD |
| `tests/test_services/test_user_profile_manager.py` | 单元测试（~10 cases） |

### 修改清单

| 文件 | 改动 |
|------|------|
| `scripts/migration_memory.sql` | 追加 `user_profiles` 表 DDL（或新建独立 SQL） |
| `config/prompts.yaml` | 所有 4 个 chat 模板 user prompt 增加 `{user_profile}` 占位符 |
| `app/online/generator.py` | ① `generate()` + `generate_stream()` 增加 `user_profile: str = ""` 参数 ② `_get_user_prompt()` 注入 user_profile |
| `app/graph/nodes.py` | `generate_node` + `general_node` 传递 `user_profile` |
| `app/graph/state.py` | 增加 `user_profile: str` 字段 |
| `app/api/routers/chat.py` | ① 有 user_id 时加载用户画像并注入 ② 回答完成后异步提取新画像字段 |
| `app/services/system_prompt_builder.py`（可选） | 统一构建最终 System Prompt，确保组装顺序正确 |

### 用户画像数据结构

```sql
CREATE TABLE user_profiles (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    field_name VARCHAR(64) NOT NULL COMMENT 'name/birthday/gender/company/medical_history',
    field_value VARCHAR(500) NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_user_field (user_id, field_name)
);
```

### 最终 Prompt 组装顺序（Phase 3 完成后）

```
System Prompt（角色 + 安全规则）
  ↓
RAG 检索结果（context）            ← 长期记忆 Part A
  ↓
用户画像（user_profile）           ← 长期记忆 Part B [Phase 3 新增]
  ↓
记忆摘要（memory_summary）         ← 短期记忆压缩
  ↓
近期对话（history）               ← 短期记忆原始
  ↓
用户背景（user_memories Top-5）    ← 中期记忆 [Phase 2 新增]
  ↓
当前用户问题
```

### 测试要求
- `test_user_profile_manager.py` ≥ 10 个测试
- 运行全量测试无回归

### 验证方案
1. 单元测试全部通过
2. Chrome DevTools 端到端测试：
   - 场景 1：告诉 AI "我叫张潇予，有高血压" → 刷新/换 session → 问"我的名字是什么？有什么病？" → 应正确回答
   - 场景 2：告诉 AI "我的公司是XX药企" → 新会话问"我在哪工作？" → 应正确回答
3. `python -m pytest tests/ -x -v` 全量通过

---

## 附录：决策记录

| 决策 | 结论 |
|------|------|
| 用户身份 | localStorage 自动生成持久化 UUID，用户无感 |
| 异步方式 | `asyncio.create_task`，不引入 Celery |
| 中期记忆更新 | Upsert 合并去重 + importance_score 累加 |
| 用户画像提取 | 自动提取，Prompt 中声明透明化 |
| 衰减策略 | 每日定时任务，7 天未访问 ×0.95，<0.1 自动删除 |
