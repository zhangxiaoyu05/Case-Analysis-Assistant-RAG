"""
安全中间件

包含:
  - SecurityHeadersMiddleware : 注入安全相关 HTTP 响应头
  - RateLimitMiddleware      : 基于 IP 的滑动窗口速率限制
"""

import time
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


# ============================================================
# 安全响应头中间件
# ============================================================
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    向所有 HTTP 响应注入安全头。

    注入的头:
      X-Content-Type-Options: nosniff
      X-Frame-Options: DENY
      X-XSS-Protection: 1; mode=block
      Referrer-Policy: strict-origin-when-cross-origin
      Cache-Control: no-store, no-cache, must-revalidate, max-age=0
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        headers_to_set = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        }

        for header_name, header_value in headers_to_set.items():
            if header_name not in response.headers:
                response.headers[header_name] = header_value

        return response


# ============================================================
# 速率限制中间件
# ============================================================
class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    基于客户端 IP 的滑动窗口速率限制。

    - 按 IP 追踪请求时间戳
    - 窗口大小固定为 60 秒
    - 超过限制返回 429 Too Many Requests
    - 每 5 分钟自动清理过期 IP 记录

    公共路径（不计入限流）:
      /health*, /docs*, /openapi.json*, /redoc*, /static*, /
    """

    # 不受限流限制的路径前缀
    PUBLIC_PATH_PREFIXES = (
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/static",
    )

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        enabled: bool = True,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.enabled = enabled
        self.window_size: float = 60.0  # 滑动窗口大小（秒）
        self._clients: dict[str, list[float]] = {}
        self._last_cleanup: float = time.time()

    async def dispatch(self, request: Request, call_next):
        # 限流开关
        if not self.enabled:
            return await call_next(request)

        # 公共路径跳过限流
        if self._is_public_path(request.url.path):
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        allowed, retry_after = self._check_rate(client_ip)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "请求过于频繁，请稍后重试。",
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)

    # ---- 内部方法 ----

    @staticmethod
    def _is_public_path(path: str) -> bool:
        """判断路径是否为公共路径（不受限流）。"""
        # 根路径
        if path == "/":
            return True
        # 前缀匹配
        for prefix in RateLimitMiddleware.PUBLIC_PATH_PREFIXES:
            if path.startswith(prefix):
                return True
        return False

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """获取客户端真实 IP（优先检查反向代理头）。"""
        forwarded: Optional[str] = request.headers.get("X-Forwarded-For")
        if forwarded:
            # X-Forwarded-For 格式: client, proxy1, proxy2 ...
            return forwarded.split(",")[0].strip()
        real_ip: Optional[str] = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        if request.client:
            return request.client.host
        return "unknown"

    def _check_rate(self, client_ip: str) -> tuple[bool, int]:
        """
        检查客户端是否超过速率限制。

        Returns:
            (allowed, retry_after_seconds): 是否放行 + 需等待秒数
        """
        now = time.time()
        window_start = now - self.window_size

        # 获取或初始化该 IP 的时间戳列表
        if client_ip not in self._clients:
            self._clients[client_ip] = []

        # 移除窗口外的时间戳
        self._clients[client_ip] = [
            ts for ts in self._clients[client_ip] if ts > window_start
        ]

        # 判断是否超限
        if len(self._clients[client_ip]) >= self.requests_per_minute:
            oldest = self._clients[client_ip][0]
            retry_after = int(self.window_size - (now - oldest))
            return False, max(1, retry_after)

        # 记录本次请求
        self._clients[client_ip].append(now)

        # 定期清理过期 IP（每 5 分钟）
        if now - self._last_cleanup > 300:
            self._cleanup(now)
            self._last_cleanup = now

        return True, 0

    def _cleanup(self, now: float) -> None:
        """移除所有已过期 IP 的记录。"""
        window_start = now - self.window_size
        expired_ips: list[str] = []
        for ip, timestamps in self._clients.items():
            # 过滤窗口内的时间戳
            active = [ts for ts in timestamps if ts > window_start]
            if active:
                self._clients[ip] = active
            else:
                expired_ips.append(ip)
        for ip in expired_ips:
            del self._clients[ip]
