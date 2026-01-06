"""
Semantic Search Engine with LangChain
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings  # Only needed if using OpenAI

# ============================================
# 1. DOCUMENTS AND DOCUMENT LOADERS
# ============================================
print("=" * 60)
print("1. Loading Documents")
print("=" * 60)

# Load PDF document
file_path = r"C:\Users\xiaoh\AI\VN_RAG\VN_knowledgeBase.pdf"

loader = PyPDFLoader(file_path)
docs = loader.load()

print(f"Loaded {len(docs)} pages from PDF")
print(f"\nFirst page preview:\n{docs[0].page_content[:200]}...\n")
print(f"Metadata: {docs[0].metadata}\n")

# ============================================
# 2. TEXT SPLITTING
# ============================================
print("=" * 60)
print("2. Splitting Documents")
print("=" * 60)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 中文文档使用更小的chunk_size
    chunk_overlap=100,   # 相应减小重叠部分
    add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(f"Split {len(docs)} pages into {len(all_splits)} chunks")
print(f"\nFirst chunk preview:\n{all_splits[0].page_content[:200]}...\n")
print(f"Chunk metadata: {all_splits[0].metadata}\n")

# ============================================
# 3. EMBEDDINGS
# ============================================
print("=" * 60)
print("3. Setting up Embeddings")
print("=" * 60)

from langchain_huggingface import HuggingFaceEmbeddings
# 使用中文嵌入模型 - BAAI/bge-base-zh-v1.5 专门针对中文优化
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("✓ Using Chinese embeddings model (BAAI/bge-base-zh-v1.5)")

# # Check if OpenAI API key is set
# if not os.environ.get("OPENAI_API_KEY"):
#     print("⚠️  OPENAI_API_KEY not found in environment variables")
    
#     from langchain_huggingface import HuggingFaceEmbeddings
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     print("✓ Using HuggingFace embeddings (all-MiniLM-L6-v2)")
# else:
#     from langchain_openai import OpenAIEmbeddings
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
#     print("✓ Using OpenAI embeddings (text-embedding-3-large)")



# ============================================
# 4. VECTOR STORE
# ============================================
print("\n" + "=" * 60)
print("4. Creating Vector Store")
print("=" * 60)

# Create Chroma vector store from documents with persistence
persist_directory = "./chroma_db_VN"
vector_store = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings,
    collection_name="VN_knowledgeBase",
    persist_directory=persist_directory
)
print(f"✓ Vector store created with Chroma (persisted to: {persist_directory})")

# ============================================
# 5. SEMANTIC SEARCH - SIMILARITY SEARCH
# ============================================
print("\n" + "=" * 60)
print("5. Testing Semantic Search")
print("=" * 60)

# Test query 1
query1 = "整个项目的系统结构是怎样的?"
print(f"\nQuery: '{query1}'")
results1 = vector_store.similarity_search(query1, k=2)
print(f"\nTop result:")
print(f"Content: {results1[0].page_content[:300]}...")
print(f"Metadata: {results1[0].metadata}")

# Test query 2
query2 = "车辆的外观检查内容有哪些?"
print(f"\n\nQuery: '{query2}'")
results2 = vector_store.similarity_search(query2, k=1)
print(f"\nTop result:")
print(f"Content: {results2[0].page_content[:300]}...")
print(f"Metadata: {results2[0].metadata}")

# Test query 3
query3 = "在给AGV建图的时候有哪些注意事项?"
print(f"\n\nQuery: '{query3}'")
results3 = vector_store.similarity_search(query3, k=1)
print(f"\nTop result:")
print(f"Content: {results3[0].page_content[:300]}...")
print(f"Metadata: {results3[0].metadata}")

# ============================================
# 6. RETRIEVER
# ============================================
print("\n" + "=" * 60)
print("6. Using Retriever")
print("=" * 60)

# Create a retriever from the vector store
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # 增加检索数量以获得更多相关上下文
)

print("✓ Retriever created")

# Test batch retrieval
queries = [
    "放货方法主要有哪些?",
    "什么是重复性和一致性?"
]

print(f"\nBatch retrieval for {len(queries)} queries:")
results = retriever.batch(queries)
for i, (query, docs) in enumerate(zip(queries, results)):
    print(f"\n{i+1}. Query: '{query}'")
    print(f"   Answer found in: Page {docs[0].metadata.get('page', 'N/A')}")
    print(f"   Content preview: {docs[0].page_content[:150]}...")

# ============================================
# 7. ADVANCED: SIMILARITY SEARCH WITH SCORE
# ============================================
print("\n" + "=" * 60)
print("7. Similarity Search with Scores")
print("=" * 60)

query = "墩孔检测法的基本逻辑是什么?"
docs_with_scores = vector_store.similarity_search_with_score(query, k=3)

print(f"\nQuery: '{query}'")
print(f"\nTop 3 results with similarity scores:")
for i, (doc, score) in enumerate(docs_with_scores, 1):
    print(f"\n{i}. Score: {score:.4f}")
    print(f"   Page: {doc.metadata.get('page', 'N/A')}")
    print(f"   Content: {doc.page_content[:200]}...")

print("\n" + "=" * 60)
print("Semantic Search Engine Ready!")
print("=" * 60)
print("\nYou can now use:")
print("  - vector_store.similarity_search(query, k=n)")
print("  - retriever.invoke(query)")
print("  - retriever.batch([query1, query2, ...])")
print("\nNext steps: Run a RAG application on top of this!")