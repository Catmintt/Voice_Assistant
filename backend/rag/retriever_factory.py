# rag/retriever_factory.py

import logging
from langchain_community.vectorstores import Milvus
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pymilvus import connections

# 从模块中导入 Reranker 组件
from .components import SiliconFlowReranker

logger = logging.getLogger(__name__)

def create_hybrid_retriever(
    embeddings: Embeddings,
    zilliz_uri: str,
    zilliz_token: str,
    siliconflow_api_key: str,
) -> BaseRetriever:
    """
    创建一个集成了“混合搜索”和“重排”功能的高级检索器。

    这个函数封装了以下复杂的步骤：
    1. 连接到 Zilliz Cloud (Milvus) 向量数据库。
    2. 初始化一个基于向量的相似性搜索检索器 (Vector Retriever)。
    3. 初始化一个基于关键词的 BM25 检索器。
    4. 使用 EnsembleRetriever 将上述两种检索器融合成“混合搜索”。
    5. 使用自定义的 SiliconFlowReranker 对混合搜索的结果进行重排。
    6. 将重排器包装在 ContextualCompressionRetriever 中，形成最终的检索器。

    参数:
        embeddings: 用于文档向量化的嵌入模型实例。
        zilliz_uri: Zilliz Cloud 的连接 URI。
        zilliz_token: Zilliz Cloud 的连接令牌。
        siliconflow_api_key: SiliconFlow 的 API 密钥，用于重排。

    返回:
        一个配置完成的、可直接在 LangChain 中使用的 BaseRetriever 实例。
    """
    logger.info("--- 正在配置高级检索器 (混合搜索 + 重排) ---")
    
    # 步骤 1: 连接到 Milvus/Zilliz 数据库
    connections.connect("default", uri=zilliz_uri, token=zilliz_token)
    logger.info("✅ 已成功连接到 Zilliz Cloud。")

    # 步骤 2: 初始化向量检索器
    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name="knowledge_base_collection",
        connection_args={"uri": zilliz_uri, "token": zilliz_token},
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    logger.info(f"✅ 向量检索器 (Vector Retriever) 初始化完成，每次将检索 {vector_retriever.search_kwargs['k']} 个结果。")

    # 步骤 3 & 4: 尝试构建混合检索器，如果失败则降级
    try:
        # BM25 需要先获取所有文档来构建其内部索引
        logger.info("正在获取所有文档以初始化 BM25 检索器...")
        all_docs = vectorstore.similarity_search(query=" ", k=1000) # 不设定关键词，用空查询获取大量文档
        
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = 5
        logger.info(f"✅ BM25 检索器初始化完成，每次将检索 {bm25_retriever.k} 个结果。")

        # 使用 EnsembleRetriever 将 BM25 和向量检索器结合起来
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
        )
        logger.info("✅ 混合检索器 (Ensemble Retriever) 创建成功。")
        base_retriever = ensemble_retriever

    except Exception as e:
        logger.warning(
            f"初始化 BM25 检索器失败，将降级为仅使用向量搜索。错误: {e}"
        )
        # 如果 BM25 失败，就只用向量检索器作为基础
        base_retriever = vector_retriever

    # 步骤 5 & 6: 创建重排器和压缩检索器
    # 实例化组件
    compressor = SiliconFlowReranker(api_key=siliconflow_api_key, top_n=3)
    
    # ContextualCompressionRetriever 是一个包装器
    # 它的工作模式是：
    # 1. 调用 base_retriever (混合检索器) 获取一批初始文档。
    # 2. 将这批文档和原始查询交给 base_compressor。
    # 3. 返回压缩器处理后的、更少但更相关的文档。
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    logger.info("✅ 重排压缩检索器 (Compression Retriever) 配置完成。")

    return compression_retriever