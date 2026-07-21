"""
RAG 药品问答系统 - FastAPI 入口

提供 HTTP 问答接口，对外暴露 /api/v1/chat 等端点。
使用 lifespan 管理 LangGraph 图编译和 Redis 连接生命周期。
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.api.auth import verify_api_key
from app.api.dependencies import set_history_manager
from app.api.middleware import RateLimitMiddleware, SecurityHeadersMiddleware
from app.api.routers import chat, health, knowledge
from app.config import config
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

# 安全响应头中间件（注入安全相关的 HTTP 头）
app.add_middleware(SecurityHeadersMiddleware)

# 速率限制中间件（基于 IP 的滑动窗口限流）
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=config.rate_limit_requests_per_minute,
    enabled=config.rate_limit_enabled,
)

# 注册路由
# 注意：chat / knowledge 路由添加了 API Key 鉴权依赖；
# health 路由和根路径 (/) 不加依赖，保持公开访问。
app.include_router(
    chat.router, prefix="/api/v1", tags=["问答"],
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    knowledge.router, prefix="/api/v1/knowledge", tags=["知识库"],
    dependencies=[Depends(verify_api_key)],
)
app.include_router(health.router, tags=["健康检查"])

# 挂载前端静态文件（CSS/JS 等）
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend_static")


def _serve_html(filename: str, fallback_message: str = ""):
    """读取前端 HTML 文件并注入 API Key 后返回。"""
    from fastapi.responses import HTMLResponse

    file_path = FRONTEND_DIR / filename
    if file_path.exists():
        html = file_path.read_text(encoding="utf-8")
        html = html.replace("__API_KEY_PLACEHOLDER__", config.APP_API_KEY)
        return HTMLResponse(html, media_type="text/html; charset=utf-8")
    return {"message": fallback_message or f"{filename} 未找到"}


@app.get("/", tags=["根"])
def root():
    """返回药品知识问答页面（自动注入 API Key）"""
    return _serve_html("chat.html", "RAG 药品问答系统 API")


@app.get("/manage", tags=["根"])
def manage_page():
    """返回知识库管理页面（自动注入 API Key）"""
    return _serve_html("manage.html", "知识库管理页面未找到")


def main() -> None:
    import uvicorn

    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)
