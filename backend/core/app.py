# backend/core/app.py

import logging
import sys

import dashscope
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 从创建的模块中导入依赖
from ..api.endpoints import router as api_router
from ..config.settings import settings

# --- 日志基础配置 ---
# 在应用的核心模块配置一次全局日志格式，这样所有模块都能沿用
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,  # 确保日志输出到控制台
)

# 获取一个日志记录器实例
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    应用工厂函数：负责创建和配置 FastAPI 应用实例。
    """
    logger.info("--- 正在创建 FastAPI 应用 ---")

    # --- 1. 创建 FastAPI 实例 ---
    app = FastAPI(
        title="Realtime Voice Assistant API",
        version="2.0.0",
        description="一个基于 FastAPI 的后端，通过 WebSocket 提供异步、流式的 STT/RAG/TTS 语音对话功能。",
    )

    # --- 2. 挂载 CORS 中间件 ---
    # 它允许来自任何源 (*) 的前端页面访问我们的 API。
    # 在生产环境中，为了安全，可能需要将 "*" 替换为前端域名列表。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 允许所有来源
        allow_credentials=True,
        allow_methods=["*"],  # 允许所有 HTTP 方法
        allow_headers=["*"],  # 允许所有 HTTP 请求头
    )
    logger.info("CORS 中间件已配置，允许所有来源。")

    # --- 3. 注册 API 路由 ---
    # 将 api/endpoints.py 中定义的路由组包含到主应用中。
    # 所有在 api_router 中定义的路径都会被添加到 app 中。
    app.include_router(api_router)
    logger.info("API 路由已成功挂载 (前缀: /ws)。")

    # --- 4. 定义应用生命周期事件 ---
    @app.on_event("startup")
    async def on_startup():
        """
        在应用启动时执行的异步函数。
        """
        logger.info("应用启动事件触发: 正在初始化全局依赖...")

        # 初始化 DashScope API Key
        # 必须在使用 dashscope 的任何功能之前设置
        if settings.DASHSCOPE_API_KEY:
            dashscope.api_key = settings.DASHSCOPE_API_KEY
            logger.info("DashScope API Key 初始化成功。")
        else:
            # 这个检查理论上在 settings.py 中已经做过，但在这里再确认一次更稳健
            logger.critical("FATAL: DASHSCOPE_API_KEY 未设置！应用无法启动。")
            sys.exit(1)

        logger.info("应用启动完成，已准备好接受连接。")

    @app.on_event("shutdown")
    async def on_shutdown():
        """
        在应用关闭时执行的异步函数。
        """
        logger.info("应用关闭事件触发: 正在执行清理操作...")
        # 目前没有需要特别清理的全局资源，为未来预留。
        logger.info("清理完成。")

    return app

# --- 创建一个全局可用的 app 实例 ---
# 在主入口文件 main.py 就可以直接导入它
app = create_app()