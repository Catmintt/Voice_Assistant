# backend/config/settings.py

import os
import sys
import logging
from functools import lru_cache
from dotenv import load_dotenv

# 获取一个日志记录器实例
logger = logging.getLogger(__name__)

# 在模块加载时，就执行一次 .env 文件的加载
# 确保了在任何地方导入 settings 对象之前，环境变量都已准备就绪
load_dotenv()

class Settings:
    """
    一个用于管理应用所有配置的类。
    它会从环境变量中读取配置，并提供类型提示和默认值。
    """

    def __init__(self):
        # --- 日志配置 ---
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

        # --- 代理配置 (可选) ---
        # 检查并设置 HTTP/HTTPS 代理
        for proxy_key in ("HTTP_PROXY", "HTTPS_PROXY"):
            proxy_value = os.getenv(proxy_key)
            if proxy_value:
                os.environ[proxy_key] = proxy_value
                logger.info(f"成功设置代理: {proxy_key}={proxy_value}")

        # --- 核心服务 API Keys 和 Endpoints ---
        # DashScope (通义千问) API Key
        self.DASHSCOPE_API_KEY: str | None = os.getenv("DASHSCOPE_API_KEY")

        # Zilliz Cloud (Milvus) 连接凭证
        self.ZILLIZ_URI: str | None = os.getenv("ZILLIZ_CLOUD_URI")
        self.ZILLIZ_TOKEN: str | None = os.getenv("ZILLIZ_CLOUD_TOKEN")

        # SiliconFlow (Reranker) API Key
        self.SILICONFLOW_API_KEY: str | None = os.getenv("SILICONFLOW_API_KEY")

        # Ollama (Embedding Model) 服务地址
        # 提供了一个默认的本地地址，但仍然推荐在 .env 中显式设置
        # self.OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.OLLAMA_BASE_URL: str | None = os.getenv("OLLAMA_BASE_URL")

        # --- LangChain 模型配置 ---
        # 使用的大模型名称
        self.LLM_MODEL_NAME: str = "qwen-flash"
        # 嵌入模型的名称
        self.EMBEDDING_MODEL_NAME: str = "Qwen3-Embedding-8B-Q8_0:latest"

        # 执行关键配置的检查
        self._validate_critical_settings()

    def _validate_critical_settings(self):
        """检查所有必需的环境变量是否已设置，如果缺少则直接退出程序。"""
        critical_vars = {
            "DASHSCOPE_API_KEY": self.DASHSCOPE_API_KEY,
            "ZILLIZ_URI": self.ZILLIZ_URI,
            "ZILLIZ_TOKEN": self.ZILLIZ_TOKEN,
            "SILICONFLOW_API_KEY": self.SILICONFLOW_API_KEY,
        }
        missing_vars = [key for key, value in critical_vars.items() if not value]
        if missing_vars:
            logger.error(f"FATAL: 缺少必要的环境变量: {', '.join(missing_vars)}")
            logger.error("请检查你的 .env 文件或系统环境变量后重试。")
            sys.exit(1) # 立即终止程序

# 使用 @lru_cache(maxsize=1) 装饰器
# 能确保 Settings 类只被实例化一次 (单例模式)。
# 无论在代码中导入 get_settings 多少次，返回的都是同一个配置对象实例。
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回一个全局唯一的 Settings 实例。"""
    logger.info("正在初始化应用配置...")
    return Settings()

# 创建一个全局可用的 settings 对象，方便在其他模块中直接导入使用。
# 例如: from backend.config.settings import settings
settings = get_settings()