# update_knowledge_base.py

import os
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pymilvus import utility, connections

def update_knowledge_base():
    """采用更精细的分块策略，并使用本地Ollama Embedding模型来更新知识库。"""
    print("\n--- 正在更新 Zilliz Cloud 知识库 (优化分块策略) ---")
    
    # --- 1. 加载配置 ---
    load_dotenv()
    ZILLIZ_URI = os.getenv("ZILLIZ_CLOUD_URI")
    ZILLIZ_TOKEN = os.getenv("ZILLIZ_CLOUD_TOKEN")
    # 8b表示使用Qwen3-Embedding-0.6B:Q8_0模型来生成嵌入向量
    COLLECTION_NAME = "knowledge_base_collection"
    KNOWLEDGE_FILE = "knowledge_re.md"

    # --- 2. 连接到 Milvus 并删除旧集合 ---
    print("1. 正在连接到 Zilliz Cloud...")
    connections.connect("default", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
    if utility.has_collection(COLLECTION_NAME):
        print(f"2. 发现旧集合 '{COLLECTION_NAME}'，正在删除...")
        utility.drop_collection(COLLECTION_NAME)
        print("   ✅ 旧集合已删除。")

    # --- 3. 文档读取与【优化后】的分块 ---
    print(f"3. 正在从 '{KNOWLEDGE_FILE}' 读取新文档...")
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        markdown_document = f.read()
    
    # 步骤 3.1: 首先按标题分割，目的是为了保留标题作为元数据
    headers_to_split_on = [("#", "header_1"), ("##", "header_2")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_document)
    print(f"   - 步骤1: Markdown标题分割完成，得到 {len(md_header_splits)} 个大块。")

    # 步骤 3.2: 接着，对每个大块进行更细粒度的递归字符分割
    # chunk_size 决定了每个块的大小，这是关键参数，需要根据文档内容微调
    # chunk_overlap 确保了文本的连续性
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # 每个块的目标大小（字符数）
        chunk_overlap=30, # 块之间的重叠字符数
    )
    
    documents = text_splitter.split_documents(md_header_splits)
    print(f"   - 步骤2: 递归字符分割完成，最终得到 {len(documents)} 个语义集中的文本块。")

    # --- 4. 初始化 Embedding 模型 (优先Ollama，失败则切换到SiliconFlow) ---
    print("4. 正在初始化 Embedding 模型...")
    
    # 从配置中获取模型名称和服务地址
    from config.settings import settings

    embeddings = None
    try:
        print("   - 尝试连接本地 Ollama Embedding 服务...")
        OLLAMA_NGROK_URL = settings.OLLAMA_NGROK_URL
        embeddings = OllamaEmbeddings(
            base_url=OLLAMA_NGROK_URL,
            model="dengcao/Qwen3-Embedding-0.6B:Q8_0",
        )
        # 发送一个虚拟请求来真实地测试连接
        embeddings.embed_query("test connection")
        print("   ✅ Ollama Embedding 模型初始化并连接成功。")
    except Exception as e:
        print(f"   ⚠️ 警告: 连接 Ollama 服务失败: {e}")
        return None
                

    # --- 5. 创建新集合并插入数据  ---
    print("5. 正在创建新集合并存入数据...")

    try:
        # 步骤1: 创建一个空的 Milvus 实例，强制它使用已有的 "default" 连接
        vectorstore = Milvus(
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
            connection_args={"uri": ZILLIZ_URI, "token": ZILLIZ_TOKEN},
            auto_id=True
        )
        
        # 步骤2: 调用实例的 .add_documents() 方法来添加数据
        print(f"   - 正在将 {len(documents)} 个文本块存入 Milvus...")
        vectorstore.add_documents(documents)
        
        print(f"6. ✅ 成功创建新集合并存入 {len(documents)} 个文本块。")

    except Exception as e:
        print(f"❌ 错误: 操作 Milvus 时失败。")
        print(f"详细错误: {e}")
        
    print("\n--- 知识库更新完成 ---")


if __name__ == "__main__":
    update_knowledge_base()