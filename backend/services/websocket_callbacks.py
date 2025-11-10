# backend/services/websocket_callbacks.py

from __future__ import annotations

import asyncio
import logging
from asyncio import AbstractEventLoop
from typing import List

from dashscope.audio.qwen_omni import OmniRealtimeCallback
from dashscope.audio.qwen_tts_realtime import (
    QwenTtsRealtime,
    QwenTtsRealtimeCallback,
    AudioFormat,
)
from fastapi import WebSocket
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from starlette.websockets import WebSocketState

# 获取一个日志记录器实例
logger = logging.getLogger(__name__)

class WebSocketTtsCallback(QwenTtsRealtimeCallback):
    """
    这个回调类负责处理来自通义千问TTS服务的事件。
    它将接收到的音频数据块通过WebSocket异步地发送到前端。
    """

    def __init__(self, websocket: WebSocket, loop: AbstractEventLoop):
        """
        初始化TTS回调。

        参数:
            websocket (WebSocket): 当前的FastAPI WebSocket连接对象，用于向客户端发送数据。
            loop (AbstractEventLoop): FastAPI运行的事件循环。需要它来安全地在同步回调中调度异步任务。
        """
        self.websocket = websocket
        self.loop = loop
        # 创建一个异步队列，作为同步回调和异步发送任务之间的缓冲区
        self.task_queue = asyncio.Queue()
        # 创建一个后台任务，专门从队列中取出数据并发送
        self.worker_task = asyncio.create_task(self._worker())

    async def _worker(self):
        """
        一个独立的异步任务，作为消费者，不断从队列中获取数据并发送。
        这样可以避免在同步的 on_event 回调中直接调用 await，从而防止阻塞。
        """
        while True:
            try:
                # 从队列中异步等待数据，如果得到 None，则表示结束信号
                audio_chunk_b64 = await self.task_queue.get()
                if audio_chunk_b64 is None:
                    break

                # 发送前再次检查WebSocket连接状态
                if self.websocket.client_state == WebSocketState.CONNECTED:
                    # 将音频数据块包装成JSON格式发送给前端
                    await self.websocket.send_json(
                        {"type": "tts_chunk", "data": audio_chunk_b64}
                    )
                else:
                    logger.warning("WebSocket连接已断开，丢弃TTS音频块。")
                    break  # 连接已关闭，退出worker

                self.task_queue.task_done()
            except Exception as e:
                logger.error(f"TTS WebSocket worker 发生错误: {e}")
                break

    def on_open(self):
        """当TTS连接成功建立时被调用。"""
        logger.info("TTS 服务连接成功。")

    def on_close(self, code, msg):
        """当TTS连接关闭时被调用。"""
        logger.info(f"TTS 服务连接关闭: {msg}")
        # 向队列发送一个None作为结束信号，停止_worker任务
        self.task_queue.put_nowait(None)

    def on_event(self, response):
        """
        当收到TTS服务的事件时被调用。这是一个同步方法。
        """
        try:
            # 音频增量数据
            if response["type"] == "response.audio.delta":
                audio_chunk_b64 = response["delta"]
                # 将数据放入队列，让 worker 去处理
                self.task_queue.put_nowait(audio_chunk_b64)
            # 会话结束事件
            elif response["type"] == "session.finished":
                logger.info("TTS 会话完成。")
                # 使用 run_coroutine_threadsafe 从同步线程安全地调度异步任务
                asyncio.run_coroutine_threadsafe(
                    self.websocket.send_json({"type": "tts_end"}), self.loop
                )
        except Exception as e:
            logger.error(f"[TTS 回调错误] {e}")


class WebSocketSttCallback(OmniRealtimeCallback):
    """
    这个回调类是整个流程的“总指挥”。
    它处理来自STT服务的事件，调用RAG链，并驱动TTS服务。
    """
    SUMMARIZATION_THRESHOLD = 50  # 定义需要进行文本摘要的长度阈值
    FALLBACK_TRIGGER_PHRASES = [
        "我暂时还没有学到相关的知识",  # 标准回答
        "不太清楚您的意思",         # 捕获 LLM 可能的“自由发挥”
        "可能输入有误",
        "无法回答您的问题",
    ]

    def __init__(
        self,
        websocket: WebSocket,
        loop: AbstractEventLoop,
        rag_chain: Runnable,
        summarize_chain: Runnable,
    ):
        """
        初始化STT回调，采用“依赖注入”的设计模式。

        参数:
            websocket (WebSocket): FastAPI的WebSocket连接实例。
            loop (AbstractEventLoop): 异步事件循环。
            rag_chain (Runnable): 已经实例化的RAG链。
            summarize_chain (Runnable): 已经实例化的总结链。
        """
        self.websocket = websocket
        self.loop = loop
        self.rag_chain = rag_chain
        self.summarize_chain = summarize_chain
        self.chat_history: List[BaseMessage] = []

        # 初始化 TTS 服务和其回调
        self.tts_callback = WebSocketTtsCallback(self.websocket, self.loop)
        self.tts_realtime = QwenTtsRealtime(
            model="qwen3-tts-flash-realtime", callback=self.tts_callback
        )
        self.tts_realtime.connect()
        # 设置TTS的音色和音频格式
        self.tts_realtime.update_session(
            voice="Cherry",
            response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
        )

    def on_open(self):
        """当STT连接成功建立时被调用。"""
        logger.info("STT 服务连接成功。")

    def on_close(self, code, msg):
        """当STT连接关闭时被调用。"""
        logger.info(f"STT 服务连接关闭: {msg}")
        # 确保STT关闭时，TTS连接也一并关闭
        if self.tts_realtime:
            self.tts_realtime.close()

    def on_event(self, response):
        """
        当收到STT服务的事件时被调用。这是一个同步方法。
        """
        # 一句话识别完成的事件
        if response["type"] == "conversation.item.input_audio_transcription.completed":
            user_question = response.get("transcript")
            if not user_question:
                return

            logger.info(f'STT 识别到用户问题: "{user_question}"')

            # 立即通知前端停止录音
            # 说完话麦克风图标就停止了，而不是等AI思考完。
            asyncio.run_coroutine_threadsafe(
                self.websocket.send_json({"type": "stt_end"}), self.loop
            )

            # 将耗时的RAG处理任务调度到后台事件循环中执行，避免阻塞当前回调线程。
            asyncio.run_coroutine_threadsafe(
                self.process_question(user_question), self.loop
            )

    async def process_question(self, user_question: str):
        """
        异步处理整个“提问->RAG->总结->TTS”的流程。
        """
        try:
            logger.info(">>> 正在调用 RAG 链获取详细答案...")
            result = await self.rag_chain.ainvoke(
                {"input": user_question, "chat_history": self.chat_history}
            )
            detailed_answer = result.get("answer", "抱歉，我暂时无法回答这个问题。")
            logger.info(f'>>> RAG 生成的详细答案: "{detailed_answer}"')

            spoken_answer = ""

            # 智能处理“无法回答”的情况
            if any(phrase in detailed_answer for phrase in self.FALLBACK_TRIGGER_PHRASES):
                logger.info(">>> 检测到“无法回答”的回复，正在重新格式化...")
                standard_fallback = "我暂时还没有学到相关的知识"
                # 重新构建回复，引用用户的问题，使对话更自然
                spoken_answer = f'关于“{user_question}”这个问题，{standard_fallback}呢，建议您关注我们的官方通知获取最新信息。'
                detailed_answer = spoken_answer # 同时更新详细答案，确保历史记录一致
            else:
                # 如果RAG能回答，则根据长度判断是否需要总结
                if len(detailed_answer) > self.SUMMARIZATION_THRESHOLD:
                    logger.info(f">>> 答案长度 ({len(detailed_answer)}) 超出阈值 ({self.SUMMARIZATION_THRESHOLD})，正在进行总结...")
                    summarization_result = await self.summarize_chain.ainvoke(
                        {"detailed_answer": detailed_answer}
                    )
                    spoken_answer = summarization_result.content
                    logger.info(f'>>> 总结后的口播答案: "{spoken_answer}"')
                else:
                    logger.info(">>> 答案长度在阈值内，直接使用。")
                    spoken_answer = detailed_answer

            # 更新对话历史。用 detailed_answer 来保持历史记录的完整性
            self.chat_history.extend(
                [HumanMessage(content=user_question), AIMessage(content=detailed_answer)]
            )

            # 向前端发送最终的【文本】答案，用于在界面上显示
            await self.websocket.send_json(
                {"type": "final_answer", "data": spoken_answer}
            )

            # 将最终要【播报】的答案送入TTS服务进行语音合成
            self.tts_realtime.append_text(spoken_answer)
            self.tts_realtime.finish()

        except Exception as e:
            logger.error(f"处理 RAG/TTS 流程时出错: {e}", exc_info=True)
            # 如果发生任何错误，都要通知前端
            await self.websocket.send_json({"type": "error", "message": str(e)})