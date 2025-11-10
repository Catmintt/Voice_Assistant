# test_rag.py

import asyncio
import logging
import sys
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

# --- 关键导入 ---
# 导入 RAG 链的“提供者”函数。
# 导入这个函数会自动触发 endpoints.py 文件顶层的代码执行，
# 从而完成所有重量级对象（LLM, Retriever, RAG Chain 等）的加载和初始化。
from backend.api.endpoints import get_rag_chain

# 配置日志，以便能看到后端模块的详细输出
logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


async def main():
    """
    一个异步的主函数，用于运行交互式 RAG 测试客户端。
    """
    print("--- RAG 链交互式测试客户端 ---")
    print("正在加载 RAG 链，这可能需要几秒钟...")

    # 1. 获取已初始化的 RAG 链实例
    # 这和 FastAPI 在处理真实请求时通过 Depends(get_rag_chain) 所做的事情完全一样。
    try:
        rag_chain = get_rag_chain()
        print("✅ RAG 链加载成功！")
    except Exception as e:
        logger.error(f"加载 RAG 链失败: {e}", exc_info=True)
        return

    # 2. 初始化一个空的聊天历史列表
    # RAG 链需要这个列表来理解对话的上下文。
    chat_history: List[BaseMessage] = []

    print('\n请输入你的问题。输入 "exit" 或 "quit" 退出程序。')

    # 3. 进入主循环，接收用户输入
    while True:
        try:
            # 获取用户在命令行中的输入
            user_question = input("\n👤 你: ")

            # 检查退出命令
            if user_question.lower() in ["exit", "quit"]:
                print("👋 感谢使用，再见！")
                break

            print("\n🤖 助手: ...正在思考中...")

            # 4. 【核心】异步调用 RAG 链
            # 使用 .ainvoke() 方法，因为它是一个异步链。
            # 传入的字典结构必须和链的期望输入完全一致。
            result = await rag_chain.ainvoke({
                "input": user_question,
                "chat_history": chat_history
            })
            
            # 从返回结果中提取答案
            answer = result.get("answer", "抱歉，我遇到了一个错误，无法回答。")

            print(f"🤖 助手: {answer}")

            # 5. 【重要】更新聊天历史
            # 将当前的用户问题和模型的回答追加到历史记录中，
            # 以便下一次提问时，模型能够“记住”之前聊了什么。
            chat_history.extend([
                HumanMessage(content=user_question),
                AIMessage(content=answer)
            ])

        except KeyboardInterrupt:
            # 允许用户通过 Ctrl+C 优雅地退出
            print("\n👋 检测到中断，程序退出。")
            break
        except Exception as e:
            logger.error(f"在处理请求时发生错误: {e}", exc_info=True)


if __name__ == "__main__":
    # 使用 asyncio.run() 来启动异步 main 函数
    asyncio.run(main())