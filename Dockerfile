# ============================================================
# RAG 药品问答系统 - Python 应用容器镜像
# 构建命令: docker build -t rag-pharma-api:latest .
# 运行命令: docker run -d -p 8000:8000 --env-file .env rag-pharma-api:latest
# ============================================================

# ---- 基础镜像（多阶段构建第一阶段）----
FROM python:3.11-slim AS builder

WORKDIR /app

# 安装编译依赖（psycopg2 等需要编译）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装（缓存层）
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---- 正式镜像 ----
FROM python:3.11-slim

WORKDIR /app

# 安装运行时依赖（纯 Python / 已编译好的 .whl）
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# 安装运行时系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制应用代码
COPY app/ ./app/
COPY config/ ./config/
COPY scripts/ ./scripts/

# 创建非 root 用户（安全加固）
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# 环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV APP_ENV=production

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
