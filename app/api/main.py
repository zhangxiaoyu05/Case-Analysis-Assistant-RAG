"""
RAG 药品问答系统 - FastAPI 入口

提供 HTTP 问答接口，对外暴露 /api/v1/chat 等端点。
使用 lifespan 管理 LangGraph 图编译和 Redis 连接生命周期。

Phase 0: 新增用户系统（登录/注册）+ 多会话管理（左侧边栏 + 对话区）。
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.api.auth import verify_api_key
from app.api.dependencies import set_history_manager
from app.api.middleware import RateLimitMiddleware, SecurityHeadersMiddleware
from app.api.routers import auth, chat, conversations, health, knowledge, user
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
    - startup: 预编译 LangGraph 图 + 初始化 Redis 连接池 + 启动衰减定时任务
    - shutdown: 清理 Redis 连接 + 取消定时任务
    """
    logger.info("=" * 50)
    logger.info("RAG 药品问答系统启动中...")
    logger.info("=" * 50)

    # --- Startup ---
    get_graph()
    logger.info("LangGraph RAG 图已编译")

    history_manager = AsyncRedisHistoryManager()
    set_history_manager(history_manager)
    logger.info("Redis 会话管理器已初始化")

    # Phase 2: 启动中期记忆每日衰减后台任务
    decay_task = asyncio.create_task(_background_decay_loop())
    logger.info("中期记忆衰减后台任务已启动")

    logger.info("应用启动完成，准备接收请求")

    yield

    # --- Shutdown ---
    logger.info("应用正在关闭...")
    decay_task.cancel()
    try:
        await decay_task
    except asyncio.CancelledError:
        pass
    await history_manager.close()
    logger.info("应用已关闭")


# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(
    title="RAG 药品问答系统 API",
    description="基于 LangGraph + Milvus 的药品说明书智能问答服务",
    version="0.4.0",
    lifespan=lifespan,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 安全响应头中间件
app.add_middleware(SecurityHeadersMiddleware)

# 速率限制中间件
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=config.rate_limit_requests_per_minute,
    enabled=config.rate_limit_enabled,
)

# ============================================================
# 注册路由
# ============================================================
# 认证路由（无需鉴权）
app.include_router(auth.router, prefix="/api/v1", tags=["认证"])

# 会话管理路由（需要 JWT 鉴权，鉴权在路由内部通过 Depends 处理）
app.include_router(conversations.router, prefix="/api/v1", tags=["会话"])

# 问答路由（Phase 0: 使用 JWT 鉴权代替 API Key）
app.include_router(chat.router, prefix="/api/v1", tags=["问答"])

# 用户路由（需要 JWT 鉴权，鉴权在路由内部通过 Depends 处理）
app.include_router(user.router, prefix="/api/v1", tags=["用户"])

# 知识库路由（保持 API Key 鉴权）
app.include_router(
    knowledge.router, prefix="/api/v1/knowledge", tags=["知识库"],
    dependencies=[Depends(verify_api_key)],
)

# 健康检查路由（无需鉴权）
app.include_router(health.router, tags=["健康检查"])

# 挂载前端静态文件（CSS/JS 等）
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend_static")


# ============================================================
# 前端页面路由
# ============================================================
def _read_html(filename: str) -> str:
    """读取前端 HTML 文件内容。"""
    file_path = FRONTEND_DIR / filename
    if file_path.exists():
        html = file_path.read_text(encoding="utf-8")
        html = html.replace("__API_KEY_PLACEHOLDER__", config.APP_API_KEY)
        return html
    return ""


@app.get("/login", tags=["前端"])
def login_page():
    """登录/注册页面"""
    content = _read_html("login.html")
    if content:
        return HTMLResponse(content, media_type="text/html; charset=utf-8")
    return {"message": "login.html 未找到"}


@app.get("/app", tags=["前端"])
def app_page():
    """主应用页面（侧边栏 + 对话区）"""
    content = _read_html("index.html")
    if content:
        return HTMLResponse(content, media_type="text/html; charset=utf-8")
    return {"message": "index.html 未找到"}


@app.get("/", tags=["根"])
def root():
    """根路径 → 重定向到主应用页面"""
    return RedirectResponse(url="/app")


@app.get("/manage", tags=["根"])
def manage_page():
    """知识库管理页面（保持原有功能）"""
    content = _read_html("manage.html")
    if content:
        return HTMLResponse(content, media_type="text/html; charset=utf-8")
    return {"message": "manage.html 未找到"}


@app.get("/profile", tags=["根"])
def profile_page():
    """用户个人资料页面"""
    content = _read_html("profile.html")
    if content:
        return HTMLResponse(content, media_type="text/html; charset=utf-8")
    return {"message": "profile.html 未找到"}


# ============================================================
# Phase 2: 中期记忆每日衰减后台任务
# ============================================================
async def _background_decay_loop():
    """
    每日执行一次记忆衰减（每 86400 秒检查一次）。

    优雅处理启动瞬间的首次衰减：
    - 启动后等待 60 秒再执行首次衰减（给 MySQL 连接池就绪时间）
    - 之后每 24 小时执行一次
    """
    # 首次等待 60 秒让服务完全就绪
    await asyncio.sleep(60)

    while True:
        try:
            from app.services.user_memory_manager import UserMemoryManager
            umm = UserMemoryManager()
            try:
                result = umm.apply_decay_all_users()
                logger.info(f"每日记忆衰减完成: {result}")
            finally:
                umm.close()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"记忆衰减异常: {e}")

        # 每 24 小时执行一次
        await asyncio.sleep(86400)


def main() -> None:
    import uvicorn

    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)
