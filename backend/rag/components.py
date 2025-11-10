# rag/components.py

from __future__ import annotations

import logging
from typing import Sequence

import requests
from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.schema import Document

# 获取一个日志记录器实例，这样我们就可以在这个模块中记录信息
logger = logging.getLogger(__name__)


class SiliconFlowReranker(BaseDocumentCompressor):
    """
    一个 LangChain 文档压缩器，它使用 SiliconFlow 的 Rerank API 
    来对检索到的文档进行重新排序，以提高相关性。
    """
    api_key: str
    model: str = "BAAI/bge-reranker-v2-m3"
    top_n: int = 3

    # 重写父类的 compress_documents 方法，实现文档压缩
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """
        同步地对文档进行重新排序和筛选。
        """
        if not documents:
            return []

        url = "https://api.siliconflow.cn/v1/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # API 需要的是纯文本列表，而不是 LangChain 的 Document 对象
        doc_texts = [doc.page_content for doc in documents]
        
        payload = {
            "model": self.model,
            "query": query,
            "documents": doc_texts,
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()  # 如果请求失败 (例如 4xx 或 5xx 错误)，则会抛出异常
            results = response.json().get("results", [])
        except requests.exceptions.RequestException as e:
            logger.warning(f"SiliconFlow Rerank API 请求失败: {e}. 将返回原始文档的前 top_n 个。")
            return documents[: self.top_n]

        # --- 结果处理 ---
        reranked_docs = []
        # 创建一个从索引到原始文档的映射，方便快速查找
        docs_dict = {i: doc for i, doc in enumerate(documents)}
        
        for result in results:
            if len(reranked_docs) >= self.top_n:
                break  # 如果已经收集够了 top_n 个文档，就停止

            doc_index = result.get("index")
            if doc_index is not None:
                original_doc = docs_dict.get(doc_index)
                if original_doc:
                    # 将重排分数添加到元数据中，这对于调试和分析很有用
                    original_doc.metadata["rerank_score"] = result.get("relevance_score")
                    reranked_docs.append(original_doc)
                    
        return reranked_docs

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """
        异步地对文档进行重新排序和筛选。
        
        注意：这是一个简单的实现，它在内部调用了同步方法。
        对于 I/O 密集型任务，这通常是可接受的，因为它满足了 LangChain 异步流程的接口要求。
        """
        return self.compress_documents(documents, query, callbacks)