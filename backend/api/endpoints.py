# backend/api/endpoints.py

import asyncio
import logging
from types import SimpleNamespace

from dashscope.audio.qwen_omni import OmniRealtimeConversation
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from langchain_core.runnables import Runnable

# 从模块中导入依赖
from ..services.websocket_callbacks import WebSocketSttCallback
from ..rag.chain_factory import create_rag_chain, create_summarize_chain
from ..rag.retriever_factory import create_hybrid_retriever
from ..config.settings import settings
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import OllamaEmbeddings


# 获取一个日志记录器实例
logger = logging.getLogger(__name__)

# 创建一个 APIRouter 实例。
# 我们可以给它加个前缀，这样所有在这个 router 中定义的路由都会自动带上 /ws 前缀。
router = APIRouter(prefix="/ws")

# --- 依赖注入系统 ---
# FastAPI/LangChain
# 不希望每次 WebSocket 连接都重新创建整个 RAG 链，
# 在这里定义一些函数，它们会在应用启动时被调用一次，
# 然后将结果（例如 RAG 链实例）缓存起来。
# FastAPI 的 Depends() 系统会在每次请求时，高效地提供这些缓存好的实例。

# 缓存llm和embeddings
_llm = ChatTongyi(model=settings.LLM_MODEL_NAME, temperature=0.7)
_embeddings = OllamaEmbeddings(
    base_url=settings.OLLAMA_NGROK_URL, model=settings.EMBEDDING_MODEL_NAME
)

# 缓存 retriever
_retriever = create_hybrid_retriever(
    embeddings=_embeddings,
    zilliz_uri=settings.ZILLIZ_URI,
    zilliz_token=settings.ZILLIZ_TOKEN,
    siliconflow_api_key=settings.SILICONFLOW_API_KEY,
)

# 缓存 rag_chain
_rag_chain = create_rag_chain(retriever=_retriever, llm=_llm)

# 缓存 summarize_chain
_summarize_chain = create_summarize_chain(llm=_llm)


def get_rag_chain() -> Runnable:
    """依赖项函数，用于提供全局唯一的 RAG 链实例。"""
    return _rag_chain

def get_summarize_chain() -> Runnable:
    """依赖项函数，用于提供全局唯一的总结链实例。"""
    return _summarize_chain


@router.websocket("/chat")
async def websocket_endpoint(
    websocket: WebSocket,
    rag_chain: Runnable = Depends(get_rag_chain),
    summarize_chain: Runnable = Depends(get_summarize_chain),
):
    """
    处理实时语音聊天 WebSocket 连接的核心端点。

    参数:
        websocket (WebSocket): 由 FastAPI 自动传入的 WebSocket 连接对象。
        rag_chain (Runnable): 通过 Depends 系统注入的 RAG 链单例。
        summarize_chain (Runnable): 通过 Depends 系统注入的总结链单例。
    """
    await websocket.accept()
    logger.info(f"WebSocket 连接已接受: {websocket.client}")

    # 获取当前正在运行的事件循环，我们需要把它传递给回调类
    loop = asyncio.get_running_loop()

    # --- 实例化与连接 ---
    # 1. 为每个 WebSocket 连接创建一个 STT 回调实例。
    #    将注入的 rag_chain 和 summarize_chain 传递给了回调类的构造函数。
    stt_callback = WebSocketSttCallback(
        websocket=websocket,
        loop=loop,
        rag_chain=rag_chain,
        summarize_chain=summarize_chain,
    )

    # 2. 创建通义千问的实时会话实例，并将回调实例注册进去。
    conversation = OmniRealtimeConversation(
        model="qwen-omni-turbo-realtime-latest", callback=stt_callback
    )
    conversation.connect()

    # 3. 配置会话参数，例如指定输入音频格式、启用语音转文字等。
    conversation.update_session(
        output_modalities=[],
        voice="aixiaoxin",
        input_audio_format=SimpleNamespace(format_str="pcm_16000hz_mono_16bit"),
        enable_input_audio_transcription=True,
        input_audio_transcription_model="gummy-realtime-v1",
        enable_turn_detection=True,
        turn_detection_type="server_vad",
    )
    logger.info("DashScope STT 会话已成功配置并连接。")

    try:
        # 进入一个无限循环，持续接收来自客户端的音频数据
        while True:
            # 等待并接收前端发送过来的文本数据（这里是 base64 编码的音频块）
            audio_b64 = await websocket.receive_text()
            # 将接收到的音频数据块立即送入 STT 服务进行处理
            conversation.append_audio(audio_b64)
    except WebSocketDisconnect:
        logger.info(f"客户端断开 WebSocket 连接: {websocket.client}")
    except Exception as e:
        logger.error(f"WebSocket 端点发生未知错误: {e}", exc_info=True)
    finally:
        # --- 清理资源 ---
        # 无论连接是正常关闭还是异常中断，都要确保关闭 STT 会话。
        # (回调类中已经设计好，关闭 STT 会话会自动关闭 TTS 会话)
        logger.info("正在清理 WebSocket 连接资源...")
        conversation.close()
        logger.info("资源清理完毕。")