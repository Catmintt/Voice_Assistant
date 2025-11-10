# main.py

import uvicorn
import logging

# 直接导入已经创建并配置好的 app 实例。
from .core.app import app
from .config.settings import settings

# 获取一个日志记录器实例
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("--- 服务启动 ---")
    logger.info(f"访问文档地址: http://127.0.0.1:8000/docs")

    # 使用 uvicorn.run() 来启动服务。
    uvicorn.run(
        "backend.core.app:app",
        host="0.0.0.0",  # 监听所有网络接口，允许局域网内的其他设备访问
        port=8000,       # 监听 8000 端口
        reload=True,     # 开启热重载，当你修改代码并保存后，服务会自动重启，非常适合开发环境
    )