# rag/chain_factory.py

import logging
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


def create_rag_chain(retriever: BaseRetriever, llm: BaseChatModel) -> Runnable:
    """
    创建一个完整的、带有对话历史感知的 RAG (检索增强生成) 链。

    这个函数负责将 RAG 的各个部分组装起来：
    1.  **历史感知检索器 (History-Aware Retriever)**:
        -   接收对话历史和用户新问题。
        -   使用 LLM 将新问题改写成一个独立的、无需上下文就能理解的问题。
        -   用改写后的问题去调用我们之前创建的 retriever。
    2.  **问答链 (Question-Answer Chain)**:
        -   接收改写后的问题和检索到的相关文档。
        -   将这些信息填入一个精心设计的 Prompt 中。
        -   调用 LLM 生成最终的、基于文档内容的回答。

    参数:
        retriever: 一个配置好的 LangChain 检索器实例 (由 retriever_factory 创建)。
        llm: 一个 LangChain 聊天模型实例 (例如 ChatTongyi)。

    返回:
        一个可执行的 LangChain 链 (Runnable)。
    """
    logger.info("--- 正在配置 RAG 链 ---")

    # --- 第一部分: 构建“历史感知检索器” ---
    # 这个Prompt的目的是让LLM扮演一个“问题改写员”
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
    
    # create_history_aware_retriever 是 LangChain 的一个便捷函数。
    # 它会自动创建一个链，该链接收输入，通过 LLM 和上面的 Prompt 生成一个新的查询，
    # 然后用这个新查询去调用传入的 retriever。
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    logger.info("✅ 历史感知检索器 (History-Aware Retriever) 创建成功。")

    # --- 第二部分: 构建最终的“问答链” ---
    # 这个Prompt定义了AI助手的最终角色和行为准则
    qa_system_prompt_template = (
        "你是一位热情、专业的“人工智能＋社会工作”创新应用大赛的官方赛事助手。"
        "你的任务是根据提供的背景知识，亲切并准确地回答参赛人员的各种问题，不允许自由发挥，严格基于知识库回答问题。"
        "请遵循以下沟通指南：\n"
        "1. **语气友好亲切**：总是使用鼓励和欢迎的语气。\n"
        "2. **回答精准**：严格基于下面提供的“上下文”信息进行回答。\n"
        "3. **【处理未知问题的铁律】**：如果上下文中没有提到相关信息，或者根据上下文无法回答用户的问题，你的回答【必须且只能】是“关于这个问题，我暂时还没有学到相关的知识呢，建议您关注我们的官方通知获取最新信息。”，【禁止】做任何形式的修改或自由发挥。\n"
        "4. **【格式铁律】**：你的最终回复【绝对禁止】包含任何表情符号或Markdown格式。\n"
        "\n上下文:\n"
        "----------------\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # create_stuff_documents_chain 是另一个便捷函数。
    # “stuff”方法是把所有检索到的文档内容，直接“塞”进上面Prompt模板中的 {context} 变量里。
    # 这个链的作用就是接收文档和问题，然后生成最终答案。
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    logger.info("✅ 问答链 (Question-Answer Chain) 创建成功。")

    # --- 第三部分: 组合成完整的检索链 ---
    # create_retrieval_chain 将上面两个部分连接起来。
    # 它的内部工作流是：
    # 1. 接收原始输入 (input, chat_history)。
    # 2. 将输入传递给 history_aware_retriever，得到改写后的问题和检索到的文档。
    # 3. 将这些结果再传递给 question_answer_chain 来生成最终答案。
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    logger.info("✅ 完整的 RAG 链已成功组装！")

    return rag_chain


def create_summarize_chain(llm: BaseChatModel) -> Runnable:
    """
    创建一个用于将长文本概括为口语化摘要的链。

    参数:
        llm: 一个 LangChain 聊天模型实例。

    返回:
        一个可执行的 LangChain 链 (Runnable)。
    """
    summarization_prompt_template = (
        "你是一个专业的口语表达专家。"
        "你的任务是将以下提供的详细文本，概括成一段简短、自然、易于口头表达的核心摘要。"
        "请遵循以下规则：\n"
        "1. **极其简练**：只保留最重要的核心要点。\n"
        "2. **口语化**：使用像日常对话一样的语言，避免书面语。\n"
        "3. **直接回答**：不要说“好的，这是概括...”或类似的前缀，直接输出概括后的内容。\n"
        "4. **保持原意**：确保概括后的内容与原文意思一致，不添加任何额外信息。\n"
        "\n需要概括的详细文本：\n"
        "----------------\n"
        "{detailed_answer}"
    )
    summarization_prompt = ChatPromptTemplate.from_template(summarization_prompt_template)

    # 这是 LangChain 表达式语言 (LCEL) 的用法。
    # 管道符号 | 意味着“将前一步的输出作为下一步的输入”。
    # 整个流程是：输入字典 -> 填入Prompt模板 -> 传递给LLM -> 输出结果
    summarize_chain = summarization_prompt | llm
    logger.info("✅ 口语化摘要链 (Summarize Chain) 创建成功。")

    return summarize_chain