# debug_rag_flow.py (Corrected Version)

import asyncio
import logging
from re import U
import sys
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- 关键导入 ---
# 我们需要直接访问底层的构建模块，而不仅仅是最终的链
from backend.api.endpoints import _llm, _embeddings, settings
from backend.rag.retriever_factory import create_hybrid_retriever

# 配置日志
logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


async def main(user_question: str
):
    """
    一个用于分解和调试 RAG 检索流程的异步函数。
    """
    print("--- RAG 流程分解与调试客户端 ---")
    print("正在加载 RAG 组件...")

    try:
        # --- 1. 初始化所有需要的组件 ---
        
        # 直接获取 LLM 实例
        llm = _llm
        
        # 创建完整的、包含重排器的高级检索器
        compression_retriever = create_hybrid_retriever(
            embeddings=_embeddings,
            zilliz_uri=settings.ZILLIZ_URI,
            zilliz_token=settings.ZILLIZ_TOKEN,
            siliconflow_api_key=settings.SILICONFLOW_API_KEY,
        )
        
        # 从高级检索器中“解构”出基础检索器
        # base_retriever 就是 EnsembleRetriever
        ensemble_retriever = compression_retriever.base_retriever
        # 再从 EnsembleRetriever 中解构出两个更基础的检索器
        bm25_retriever = ensemble_retriever.retrievers[0]
        vector_retriever = ensemble_retriever.retrievers[1]

        print("✅ 所有 RAG 组件加载成功！\n")

    except Exception as e:
        logger.error(f"加载 RAG 组件失败: {e}", exc_info=True)
        return

    # --- 2. 模拟用户输入 ---
    chat_history = []  # 假设是新对话

    print(f"👤 原始问题: {user_question}\n")

    # --- 3. 【第1步】执行问题改写 ---
    print("--- 步骤 1: 问题改写 (History-Aware) ---")
    
    # 在这里直接定义问题改写所需的 Prompt
    contextualize_q_system_prompt = (
        "给定一段聊天历史和用户最新的一个问题，"
        "该问题可能引用了聊天历史中的上下文。"
        "你的任务是将这个问题改写成一个独立的、无需聊天历史就能被完全理解的新问题。"
        "请注意，你不需要回答这个问题，只需要完成改写任务。"
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # 构建改写链
    rewrite_chain = contextualize_q_prompt | llm
    if chat_history:
        print("检测到对话历史，正在执行问题改写...")
        # 调用链来获取改写后的问题
        rewritten_result = await rewrite_chain.ainvoke({
            "input": user_question,
            "chat_history": chat_history
        })
        rewritten_question = rewritten_result.content
    else:
        print("对话历史为空，跳过问题改写步骤。")
        rewritten_question = user_question

    print(f"🤖 LLM 改写后的独立问题: {rewritten_question}\n")
    
    # --- 4. 【第2步】执行基础检索 (并行) ---
    print("--- 步骤 2: 基础检索 (Vector vs BM25) ---")

    # 调用向量检索器
    vector_docs = await vector_retriever.ainvoke(rewritten_question)
    print(f"✅ 向量检索器 (Vector Retriever) 返回了 {len(vector_docs)} 个文档:")
    for i, doc in enumerate(vector_docs):
        print(f"   - [Vec {i+1}] 内容: '{doc.page_content}'")
        print(f"     元数据: {doc.metadata}\n")

    # 调用 BM25 检索器
    bm25_docs = await bm25_retriever.ainvoke(rewritten_question)
    print(f"✅ BM25 检索器 (BM25 Retriever) 返回了 {len(bm25_docs)} 个文档:")
    for i, doc in enumerate(bm25_docs):
        print(f"   - [BM25 {i+1}] 内容: '{doc.page_content}'")
        print(f"     元数据: {doc.metadata}\n")
        
    # --- 5. 【第3步】执行混合检索 ---
    print("--- 步骤 3: 混合检索 (Ensemble) ---")
    
    ensemble_docs = await ensemble_retriever.ainvoke(rewritten_question)
    print(f"✅ 混合检索器 (Ensemble Retriever) 融合并排序后，返回了 {len(ensemble_docs)} 个文档:")
    for i, doc in enumerate(ensemble_docs):
        print(f"   - [Ens {i+1}] 内容: '{doc.page_content}'")
        print(f"     元数据: {doc.metadata}\n")

    # --- 6. 【第4步】执行最终的重排/压缩 ---
    print("--- 步骤 4: 重排压缩 (Reranker) ---")
    
    final_docs = await compression_retriever.ainvoke(rewritten_question)
    print(f"✅ 重排器 (Reranker) 最终筛选出 top_{compression_retriever.base_compressor.top_n} 个最相关的文档，将它们传入LLM生成最终答案:")
    for i, doc in enumerate(final_docs):
        print(f"   - [Final {i+1}] 内容: '{doc.page_content}'")
        # Reranker 会把自己的分数也加入元数据
        print(f"     元数据: {doc.metadata}\n")
    print(f"✅ 最终答案: {final_docs[0].page_content}")


if __name__ == "__main__":
    asyncio.run(main(user_question="新兴领域赛道是什么"))