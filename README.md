# 💊 RAG 药品说明书智能问答系统

基于 **RAG（检索增强生成）** 的药品知识问答系统。支持上传药品说明书文档（PDF / DOCX / TXT），自动构建知识库，提供基于语义检索 + 大模型生成的智能问答服务。

## 🏗️ 技术架构

```
┌──────────────────────────────────────────────────────┐
│                   前端（双界面）                       │
│   Streamlit（streamlit_app.py） + 原生 HTML/CSS/JS   │
└─────────────────────┬────────────────────────────────┘
                      │ HTTP / SSE
┌─────────────────────▼────────────────────────────────┐
│              FastAPI 后端（app/api/）                  │
│   /api/v1/chat  /chat/stream  /knowledge/*  /health  │
└─────────────────────┬────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────┐
│           LangGraph 流程编排（app/graph/）              │
│   START → intent → retrieve → rank → generate → END  │
│            ├─ chitchat → END（闲聊直达）               │
│            ├─ general → END（非药品问题 LLM 直接回答）   │
│            └─ attack → END（攻击/越狱安全拒绝）         │
└─────────────────────┬────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────┐
│               在线流程（app/online/）                   │
│   意图识别 → 混合检索（向量 + BM25）→ 重排序 → 答案生成  │
└─────────────────────┬────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────┐
│                     数据存储层                         │
│  Milvus（向量） + MySQL（结构化 + BM25） + Redis（会话） │
└──────────────────────────────────────────────────────┘
```

## ✨ 核心功能

### 离线知识库构建

| 模块 | 功能 |
|------|------|
| 文档加载 | 支持 PDF / DOCX / TXT 格式，自动识别 UTF-8 / GBK 编码 |
| 文本清洗 | 空白规范化、PDF 伪影去除、Unicode 规范化、可选 LLM 脱敏 |
| 章节感知切分 | 识别 `【章节名】` 标记，智能合并短章节和尾块 |
| 向量化 | DashScope text-embedding-v4，1024 维，支持批量 + 自动重试 |
| 入库 | MySQL（原始文档 + 文本块 BM25 全文索引）+ Milvus（向量） |

### 在线智能问答

| 模块 | 功能 |
|------|------|
| 意图识别 | qwen-flash，快速预判 + LLM 精确分类（drug_inquiry / chitchat / general / attack）四层防御 |
| 混合检索 | Milvus 向量检索 + MySQL BM25 全文检索 → RRF 融合 |
| 重排序 | qwen3-rerank，对检索结果二次排序，失败自动回退到原始排序 |
| 答案生成 | qwen3-max，支持 4 种场景模板（默认问答 / 药品对比 / 用法用量追问 / 通用问答） |
| 短期记忆 | qwen-flash 滑动窗口 + 累积摘要，旧对话压缩为摘要注入 Prompt，最大化上下文利用率 |
| 流式输出 | SSE（Server-Sent Events），逐 token 实时返回 |

### 会话与知识库管理

- Redis 多轮对话历史（自动 TTL 过期 + 轮数裁剪）+ 记忆摘要持久化
- 短期记忆：滑动窗口 + qwen-flash 累积摘要，旧对话压缩注入 Prompt
- 知识库药品 CRUD（上传入库 / 列表查询 / 删除）
- 健康检查（Milvus / MySQL / Redis 连接状态）

### 安全防护

- 四层防御架构：输入检测 → 提示词加固 → 路由隔离 → 输出过滤
- API Key 鉴权 + 基于 IP 的速率限制
- HTTP 安全响应头（X-Content-Type-Options / X-Frame-Options / XSS Protection）
- 攻击检测：提示词注入 / 越狱 / 间接注入 / 语义诱导

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Docker & Docker Compose
- 阿里云 DashScope API Key（[申请地址](https://dashscope.console.aliyun.com/)）

### 1. 克隆项目

```bash
git clone https://github.com/zhangxiaoyu05/Medication-Instructions-RAG.git
cd Medication-Instructions-RAG
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
# 单个文件入库
python scripts/run_offline.py --file data/raw/阿司匹林肠溶片说明书_test.txt

# 目录批量入库
python scripts/run_offline.py --dir data/raw/

# 干跑预览（不入库，仅查看切分效果）
python scripts/run_offline.py --file data/raw/阿司匹林肠溶片说明书_test.txt --dry-run

# 指定药品名称和厂家
python scripts/run_offline.py --file doc.pdf --drug-name "布洛芬" --manufacturer "拜耳医药"
```

### 7. 启动问答服务

```bash
# FastAPI 后端（端口 8000）
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

# 或 Streamlit 前端（端口 8501）
streamlit run frontend/streamlit_app.py
```

访问 http://localhost:8000 使用 Web 界面，或 http://localhost:8000/docs 查看 Swagger API 文档。

### Docker 一键部署

```bash
docker compose up -d
```

自动启动 Milvus + MySQL + Redis + FastAPI 全套服务。

## 📂 项目结构

```
├── app/
│   ├── api/                    # FastAPI 接口层
│   │   ├── main.py             # 应用入口 + lifespan 生命周期
│   │   ├── auth.py             # API Key 鉴权模块
│   │   ├── middleware.py        # 速率限制 + 安全响应头中间件
│   │   ├── dependencies.py     # 依赖注入（单例获取）
│   │   └── routers/            # 路由模块
│   │       ├── chat.py         # 问答接口（单轮 + 流式 + 短期记忆集成）
│   │       ├── health.py       # 健康检查 / 就绪检查
│   │       └── knowledge.py    # 知识库管理（上传 / 查询 / 删除）
│   ├── config.py               # 统一配置层（.env + config.yaml 合并）
│   ├── db/                     # 数据库客户端
│   │   ├── milvus_client.py    # Milvus 向量数据库封装
│   │   └── mysql_client.py     # MySQL 封装（含 BM25 全文检索）
│   ├── graph/                  # LangGraph 流程编排
│   │   ├── state.py            # RAG 图状态定义（TypedDict）
│   │   ├── nodes.py            # 节点函数（intent / retrieve / rank / generate 等）
│   │   ├── edges.py            # 条件路由（意图路由 / 检索后路由）
│   │   └── graph.py            # 图构建 + 编译（模块级单例）
│   ├── offline/                # 离线入库流程
│   │   ├── __init__.py          # Pipeline 接口
│   │   ├── loader.py           # 文档加载器（PDF / DOCX / TXT）
│   │   ├── cleaner.py          # 文本清洗器（规范化 + 可选脱敏）
│   │   ├── splitter.py         # 章节感知切分器
│   │   ├── multi_drug_splitter.py  # 多药品合集文件拆分器
│   │   ├── embedder.py         # 向量化器（DashScope TextEmbedding）
│   │   └── pipeline.py         # 流程编排器（运行完整入库流水线）
│   ├── online/                 # 在线问答流程
│   │   ├── intent.py           # 意图分类器
│   │   ├── retriever.py        # 混合检索器（向量 + BM25 → RRF 融合）
│   │   ├── ranker.py           # 重排序器（qwen3-rerank）
│   │   └── generator.py        # 答案生成器（qwen3-max + 流式输出）
│   ├── schemas/                # Pydantic 数据模型
│   │   ├── chat.py             # 问答请求 / 响应 / 历史模型
│   │   └── common.py           # 通用模型（健康检查 / 错误响应）
│   └── services/               # 业务服务层
│       ├── history_manager.py  # Redis 异步会话历史管理 + 记忆摘要存储
│       └── memory_manager.py   # 短期记忆管理器（滑动窗口 + qwen-flash 累积摘要）
├── config/
│   ├── config.yaml             # 业务参数（模型 / 检索 / 数据库 / 日志）
│   └── prompts.yaml            # 提示词模板（意图 / 生成 / 脱敏 / 质量评估）
├── data/
│   ├── raw/                    # 20 种药品说明书原始文件
│   └── uploads/                # Web 上传文件暂存目录
├── frontend/
│   ├── chat.html               # 原生 Web 问答界面
│   ├── manage.html             # 知识库管理界面
│   └── streamlit_app.py        # Streamlit 前端（含短期记忆支持）
├── scripts/
│   ├── init_collection.py      # 一键初始化所有存储层
│   ├── init_milvus.py          # Milvus Collection 创建 + 索引构建
│   ├── mysql_init.sql          # MySQL 建库建表脚本（Docker 自动执行）
│   ├── run_offline.py          # 离线入库 CLI 工具
│   └── split_drug_file.py      # 药品合集文件拆分工具
├── tasks/                      # 开发任务规划文档
│   └── triple_memory_plan.md   # 三段记忆体系分步实施计划
├── tests/                      # 单元测试（288 tests）
│   ├── conftest.py             # 共享 fixtures + mocks
│   ├── test_offline/           # 离线流程测试
│   ├── test_online/            # 在线流程测试
│   ├── test_graph/             # LangGraph 图测试
│   ├── test_api/               # API 接口测试
│   └── test_services/          # 业务服务测试（含记忆管理）
├── docker-compose.yml          # Docker 服务编排（6 个容器）
├── Dockerfile                  # API 服务容器镜像
├── pyproject.toml              # 项目元数据 + 工具配置
├── requirements.txt            # 运行时依赖
└── requirements-dev.txt        # 开发依赖（测试 / 代码检查）
```

## 🤖 模型配置

所有模型使用阿里云 DashScope 平台，通过 `config/config.yaml` 配置：

| 环节 | 模型 | 说明 |
|------|------|------|
| 嵌入 | `text-embedding-v4` | 文本转向量，1024 维 |
| 意图识别 | `qwen-flash` | 轻量模型，四分类（drug_inquiry/chitchat/general/attack） |
| 重排序 | `qwen3-rerank` | 检索结果二次排序 |
| 答案生成 | `qwen3-max` | 高质量生成回答 |
| 短期记忆 | `qwen-flash` | 对话摘要压缩 |

## 📡 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | Web 问答界面 |
| `GET` | `/docs` | Swagger API 文档 |
| `GET` | `/health` | 基础健康检查 |
| `GET` | `/health/ready` | 就绪检查（含依赖服务状态） |
| `POST` | `/api/v1/chat` | 单轮问答 |
| `POST` | `/api/v1/chat/stream` | 流式问答（SSE） |
| `GET` | `/api/v1/chat/history/{id}` | 获取多轮对话历史 |
| `DELETE` | `/api/v1/chat/history/{id}` | 清除会话 |
| `POST` | `/api/v1/knowledge/upload` | 上传文档构建知识库 |
| `GET` | `/api/v1/knowledge/status/{id}` | 查询入库批次状态 |
| `GET` | `/api/v1/knowledge/drugs` | 列出已入库药品 |
| `DELETE` | `/api/v1/knowledge/drug/{id}` | 删除指定药品（MySQL + Milvus） |

## 🧪 运行测试

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

## 📄 许可

MIT License
