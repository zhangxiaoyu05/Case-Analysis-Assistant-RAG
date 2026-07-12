"""
RAG 药品问答系统 - FastAPI 入口

提供 HTTP 问答接口，对外暴露 /api/v1/chat 等端点。
使用 lifespan 管理 LangGraph 图编译和 Redis 连接生命周期。
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.api.dependencies import set_history_manager
from app.api.routers import chat, health, knowledge
from app.graph.graph import get_graph
from app.services.history_manager import AsyncRedisHistoryManager

# 前端静态文件目录
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


# ============================================================
# 生命周期
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期:
    - startup: 预编译 LangGraph 图 + 初始化 Redis 连接池
    - shutdown: 清理 Redis 连接
    """
    logger.info("=" * 50)
    logger.info("RAG 药品问答系统启动中...")
    logger.info("=" * 50)

    # --- Startup ---
    # 预编译 LangGraph 图（首次调用触发编译）
    get_graph()
    logger.info("LangGraph RAG 图已编译")

    # 初始化 Redis 会话管理器
    history_manager = AsyncRedisHistoryManager()
    set_history_manager(history_manager)
    logger.info("Redis 会话管理器已初始化")

    logger.info("应用启动完成，准备接收请求")

    yield  # 应用运行中...

    # --- Shutdown ---
    logger.info("应用正在关闭...")
    await history_manager.close()
    logger.info("应用已关闭")


# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(
    title="RAG 药品问答系统 API",
    description="基于 LangChain + LangGraph + Milvus 的药品说明书智能问答服务",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 配置（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat.router, prefix="/api/v1", tags=["问答"])
app.include_router(knowledge.router, prefix="/api/v1/knowledge", tags=["知识库"])
app.include_router(health.router, tags=["健康检查"])

# 挂载前端静态文件（CSS/JS 等）
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend_static")


@app.get("/", tags=["根"])
def root():
    """返回前端 Web 界面"""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html; charset=utf-8")
    return {"message": "RAG 药品问答系统 API", "docs": "/docs"}


def main() -> None:
    import uvicorn

    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)
