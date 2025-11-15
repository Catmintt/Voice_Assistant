# test_rag.py

import asyncio
import logging
import sys
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from backend.api.endpoints import get_rag_chain, _llm

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

    contextualize_q_system_prompt = (
        "给定一段聊天历史和用户最新的一个问题，"
        "该问题可能引用了聊天历史中的上下文。"
        "你的任务是将这个问题改写成一个独立的、无需聊天历史就能被完全理解的新问题。"
        "【重要规则】如果用户的问题本身已经是一个独立的、完整的句子，并且不需要参考聊天历史就能理解，那么请【直接原样返回】该问题，不要做任何修改或添加任何额外内容。"
        "请注意，你的唯一任务是改写或确认问题，绝对不要回答问题。"
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    rewrite_chain = contextualize_q_prompt | _llm

    # 3. 进入主循环，接收用户输入
    while True:
        try:
            user_question = input("\n👤 你: ")
            if user_question.lower() in ["exit", "quit"]:
                print("👋 感谢使用，再见！")
                break

            # --- 在调用 RAG 链前，先执行并显示问题改写步骤 ---
            print("\n--- 步骤 1: 问题改写 (History-Aware) ---")
            rewritten_question = ""
            if chat_history:
                # 如果有历史记录，则调用创建的演示链
                print("检测到对话历史，正在执行问题改写...")
                rewritten_result = await rewrite_chain.ainvoke({
                    "input": user_question,
                    "chat_history": chat_history
                })
                rewritten_question = rewritten_result.content
                print(f"🤖 改写后的独立问题: {rewritten_question}")
            else:
                # 如果没有历史记录，则模拟 RAG 链的行为，直接跳过
                print("对话历史为空，跳过问题改写步骤。")
                rewritten_question = user_question
                print(f"🤖 用于检索的问题: {rewritten_question}")
            # --- 修改结束 ---

            print("\n🤖 助手: ...正在思考中...")

            # 异步调用 RAG 链
            # 注意：这里的调用保持不变，它会在内部独立地、再次执行上面的改写逻辑
            result = await rag_chain.ainvoke({
                "input": user_question,
                "chat_history": chat_history
            })
            
            answer = result.get("answer", "抱歉，我遇到了一个错误，无法回答。")
            print(f"🤖 助手: {answer}")

            # 更新聊天历史
            chat_history.extend([
                HumanMessage(content=user_question),
                AIMessage(content=answer)
            ])

        except KeyboardInterrupt:
            print("\n👋 检测到中断，程序退出。")
            break
        except Exception as e:
            logger.error(f"在处理请求时发生错误: {e}", exc_info=True)


if __name__ == "__main__":
    # 使用 asyncio.run() 来启动异步 main 函数
    asyncio.run(main())