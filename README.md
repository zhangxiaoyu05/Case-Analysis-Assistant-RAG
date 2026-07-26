# 🏥 RAG 临床病例分析助手

基于 **RAG（检索增强生成）** 的临床病例智能分析系统。支持上传病例文档（PDF / DOCX / TXT），自动提取关键信息，生成 **SOAP 格式**结构化分析报告，并标注循证引用来源和证据级别。

**面向用户**：执业医师 / 临床药师

## 🏗️ 技术架构

```
┌──────────────────────────────────────────────────────────┐
│                    前端（四界面）                          │
│   Web 原生界面（index.html / login.html / manage.html / profile.html）│
│   文件拖拽上传 + 分析模式选择 + AI 病例提取面板 + SOAP 结果渲染   │
└───────────────────────┬──────────────────────────────────┘
                        │ HTTP / SSE (multipart/form-data)
┌───────────────────────▼──────────────────────────────────┐
│               FastAPI 后端（app/api/）                     │
│   /api/v1/chat  /auth  /conversations  /user  /knowledge  │
│   JWT 鉴权 + API Key + 速率限制 + 安全响应头               │
└───────────────────────┬──────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────┐
│           LangGraph 流程编排（app/graph/） 8 节点           │
│                                                          │
│   START → intent ──┬─ clinical → case_preprocess         │
│                    │       → multi_retrieve → rank       │
│                    │       → synthesize → generate → END │
│                    ├─ chitchat → END（问候直达）            │
│                    └─ not_clinical → reject → END（拦截）  │
│                                                          │
│   各节点调用 app/online/ 的组件：                           │
│   门禁判断 / 病例预处理 / 多路检索 / 重排序 / 上下文合成 / 生成│
└───────────────────────┬──────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────┐
│                      数据存储层                            │
│   Milvus（向量检索）+ MySQL（文档/元数据/BM25全文索引）      │
│   + Redis（会话历史 + 记忆摘要缓存）                        │
└──────────────────────────────────────────────────────────┘
```

## ✨ 核心功能

### 临床病例分析

| 模块 | 功能 |
|------|------|
| 病例上传 | 支持 PDF / DOCX / TXT，拖拽或点击上传（≤20MB） |
| 病例预处理 | LLM 结构化提取（主诉、现病史、既往史、检查结果、用药、诊断等） |
| 分析模式 | 综合分析 / 鉴别诊断 / 诊疗评估 / 用药审查 |
| SOAP 输出 | S（主观）/ O（客观）/ A（评估）/ P（计划）标准临床格式 |
| 循证引用 | 每条建议标注来源 + 证据级别（IA/IB/IIA/IIB/III/IV） |
| 多源知识库 | 药品说明书 + 疾病知识 + 临床指南 + 学术文献，4 路并行检索 |

### 离线知识库构建

| 模块 | 功能 |
|------|------|
| 文档加载 | 支持 PDF / DOCX / TXT 格式，自动识别 UTF-8 / GBK 编码 |
| 多源识别 | 自动检测多药品合集并拆分为独立文档 |
| 文本清洗 | 空白规范化、PDF 伪影去除、Unicode 规范化、可选 LLM 脱敏 |
| 4 种切分器 | 药品（`【章节】`）+ 疾病（Markdown/编号/关键词）+ 指南（推荐等级/证据级别）+ 文献（IMRaD/牛津等级） |
| 向量化 | DashScope text-embedding-v4，1024 维，支持批量 + 自动重试 |
| 入库 | MySQL（9 张表 + BM25 全文索引）+ Milvus（4 个 Collection） |

### 在线智能问答

| 模块 | 功能 |
|------|------|
| 门禁判断 | qwen-flash，二元分类（clinical / not_clinical），极简 prompt Few-shot |
| 病例提取 | qwen-flash，结构化提取 10 个临床字段（主诉/现病史/既往史/检查/用药/诊断/异常等） |
| 混合检索 | Milvus 向量检索 + MySQL BM25 全文检索 → RRF 融合 |
| 重排序 | qwen3-rerank，对检索结果二次排序，失败自动回退到原始排序 |
| 上下文合成 | 按疾病/指南/药品/文献四个维度组织检索结果 |
| 答案生成 | qwen3-max，5 种 SOAP 模板（case_summary/differential_diagnosis/treatment_analysis/drug_review/guideline_lookup） |
| 流式输出 | SSE（Server-Sent Events），逐 token 实时返回 |

### 用户系统与记忆体系

| 模块 | 功能 |
|------|------|
| 用户认证 | JWT 登录/注册，bcrypt 密码哈希，7 天 token 有效期 |
| 多会话管理 | 多对话窗口（创建/切换/删除），LLM 自动生成对话标题 |
| 短期记忆 | Redis 滑动窗口 + qwen-flash 累积摘要，旧对话压缩注入 Prompt |
| 中期记忆 | MySQL 持久化，5 种类型（关注领域/临床疑难点/诊疗偏好/学习计划/执业特征），每日衰减 ×0.95 |
| 长期记忆 | EAV 模式的医生执业画像（9 个字段：姓名/职称/科室/医院/专业领域/执业年限/指南偏好/患者群体/常见病种），LLM 自动提取 + 手动编辑，永不过期 |

### 安全防护

- **认证层**：JWT 登录鉴权 + bcrypt 密码哈希 + 7 天 token 有效期
- **接口层**：API Key 鉴权（知识库管理）+ 基于 IP 的速率限制
- **AI 层**：二元门禁 + 路由隔离（clinical → RAG 检索 / not_clinical → 统一拦截），问候白名单零 token 回应
- **HTTP 层**：安全响应头（X-Content-Type-Options / X-Frame-Options / XSS Protection）
- **攻击检测**：提示词注入 / 越狱 / 间接注入，由门禁 LLM 统一拦截

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Docker & Docker Compose
- 阿里云 DashScope API Key（[申请地址](https://dashscope.console.aliyun.com/)）

### 1. 克隆项目

```bash
git clone git@github.com:zhangxiaoyu05/Case-Analysis-Assistant-RAG.git
cd Case-Analysis-Assistant-RAG
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 DASHSCOPE_API_KEY
```

### 3. 启动依赖服务

```bash
docker compose up -d mysql milvus redis
```

### 4. 初始化存储层

```bash
python scripts/init_collection.py
```

### 5. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 6. 离线入库（构建知识库）

```bash
# 药品说明书入库
python scripts/run_offline.py --file data/raw/阿莫西林胶囊.txt

# 临床指南入库
python scripts/run_offline.py --file guideline.pdf --source-type guideline \
    --guideline-title "中国心力衰竭诊疗指南2024"

# 疾病知识入库
python scripts/run_offline.py --file disease.txt --source-type disease \
    --disease-name "2型糖尿病"

# 学术文献入库
python scripts/run_offline.py --file paper.pdf --source-type literature

# 目录批量入库
python scripts/run_offline.py --dir data/raw/

# 干跑预览（不入库）
python scripts/run_offline.py --file doc.pdf --dry-run
```

### 7. 启动服务

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

访问 http://localhost:8000 使用 Web 界面，或 http://localhost:8000/docs 查看 Swagger API 文档。

### Docker 一键部署

```bash
docker compose up -d
```

## 🤖 模型配置

所有模型使用阿里云 DashScope 平台：

| 环节 | 模型 | 说明 |
|------|------|------|
| 嵌入 | `text-embedding-v4` | 文本转向量，1024 维 |
| 门禁判断 | `qwen-flash` | 轻量模型，二元分类（clinical/not_clinical） |
| 病例提取 | `qwen-flash` | 结构化提取 10 个临床字段 |
| 重排序 | `qwen3-rerank` | 检索结果二次排序 |
| 答案生成 | `qwen3-max` | 高质量 SOAP 格式回答 |
| 短期记忆 | `qwen-flash` | 对话摘要压缩 |
| 中期记忆 | `qwen-flash` | 多轮对话提取 5 类记忆 |
| 长期记忆 | `qwen-flash` | 用户画像字段提取 |

## 📡 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | → 重定向到 `/app` |
| `GET` | `/login` | 登录/注册页面 |
| `GET` | `/app` | 主应用界面（index.html） |
| `GET` | `/manage` | 知识库管理界面 |
| `GET` | `/profile` | 用户个人资料页面 |
| `GET` | `/docs` | Swagger API 文档 |
| `GET` | `/health` | 基础健康检查 |
| `GET` | `/health/ready` | 就绪检查（含依赖服务状态） |
| `POST` | `/api/v1/auth/register` | 用户注册 |
| `POST` | `/api/v1/auth/login` | 用户登录 |
| `GET` | `/api/v1/auth/me` | 获取当前用户信息 |
| `POST` | `/api/v1/chat` | 单轮问答（multipart/form-data，支持文件上传） |
| `POST` | `/api/v1/chat/stream` | 流式问答（SSE，支持文件上传） |
| `GET` | `/api/v1/chat/history/{id}` | 获取对话历史 |
| `DELETE` | `/api/v1/chat/history/{id}` | 清除会话 |
| `GET` | `/api/v1/conversations` | 获取对话列表 |
| `POST` | `/api/v1/conversations` | 创建新对话 |
| `PATCH` | `/api/v1/conversations/{id}` | 更新对话 |
| `DELETE` | `/api/v1/conversations/{id}` | 删除对话 |
| `GET` | `/api/v1/user/settings` | 获取用户设置 |
| `PUT` | `/api/v1/user/settings` | 更新昵称 |
| `GET` | `/api/v1/user/profile` | 获取个人画像 |
| `PUT` | `/api/v1/user/profile` | 批量更新画像字段 |
| `POST` | `/api/v1/knowledge/upload` | 上传文档构建知识库（支持 drug/disease/guideline/literature） |
| `GET` | `/api/v1/knowledge/sources` | 列出所有 source_type 的知识条目（含统计） |
| `GET` | `/api/v1/knowledge/drugs` | 列出已入库药品（向后兼容） |
| `DELETE` | `/api/v1/knowledge/source/{type}/{id}` | 删除指定来源类型文档（MySQL + Milvus 同步） |
| `DELETE` | `/api/v1/knowledge/drug/{id}` | 删除指定药品（向后兼容） |

## 🧪 运行测试

```bash
pip install -r requirements-dev.txt
pytest tests/ -v          # 407 个测试用例，全部通过
pytest tests/ -q          # 简洁输出
pytest tests/ --cov=app   # 含覆盖率报告
```

## 📂 项目结构

```
├── app/
│   ├── api/                    # FastAPI 接口层
│   │   ├── main.py             # 应用入口 + lifespan
│   │   ├── auth.py             # API Key 鉴权
│   │   ├── middleware.py        # 速率限制 + 安全头
│   │   ├── dependencies.py     # 依赖注入
│   │   └── routers/            # 路由：chat / auth / conversations / user / health / knowledge
│   ├── config.py               # 统一配置（.env + config.yaml）
│   ├── db/                     # 数据库客户端
│   │   ├── milvus_client.py    # Milvus 向量数据库（多 Collection 统一 schema）
│   │   └── mysql_client.py     # MySQL + BM25（9 表 + 通用路由方法）
│   ├── graph/                  # LangGraph 流程编排（8 节点）
│   │   ├── state.py            # RagState + GraphResult
│   │   ├── nodes.py            # intent / case_preprocess / multi_retrieve / rank / synthesize / generate / chitchat / reject
│   │   ├── edges.py            # 条件路由
│   │   └── graph.py            # 图构建 + 编译
│   ├── offline/                # 离线入库（4 种知识源切分器）
│   │   ├── loader.py           # 文档加载（PDF/DOCX/TXT）
│   │   ├── cleaner.py          # 文本清洗
│   │   ├── splitter.py         # 药品说明书切分器
│   │   ├── splitter_disease.py # 疾病知识切分器
│   │   ├── splitter_guideline.py # 临床指南切分器（含推荐等级/证据级别检测）
│   │   ├── splitter_literature.py # 学术文献切分器（IMRaD + 牛津证据等级）
│   │   ├── multi_drug_splitter.py # 多药品合集智能检测与拆分
│   │   ├── embedder.py         # 向量化
│   │   └── pipeline.py         # 流程编排（source_type 路由）
│   ├── online/                 # 在线问答
│   │   ├── intent.py           # 门禁判断（clinical/not_clinical）
│   │   ├── retriever.py        # 混合检索 + 多源并行检索（4 collection RRF 融合）
│   │   ├── ranker.py           # 重排序
│   │   └── generator.py        # 答案生成（5 种 SOAP 模板）
│   ├── schemas/                # Pydantic 模型
│   └── services/               # 业务服务（记忆/会话/用户）
├── config/
│   ├── config.yaml             # 业务参数
│   └── prompts.yaml            # 提示词模板（gatekeeper + case_extraction + 5 SOAP 模板）
├── frontend/                   # Web 前端（4 页面：病例分析/登录/知识库管理/执业画像）
├── scripts/                    # 工具脚本（含 migration_v3.sql 数据库增量迁移）
└── tests/                      # 测试（407 tests，全部通过）
```

## 📄 许可

MIT License
