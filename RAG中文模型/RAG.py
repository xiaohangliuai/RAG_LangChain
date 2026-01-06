"""
Load the existing vector store and perform custom searches with AI-powered answers
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

print("=" * 60)
print("AI小助手 - 知识库检索")
print("=" * 60)
print("\nLoading vector store...")

# Initialize the same embeddings model used during indexing
# 使用中文嵌入模型 - 必须与vector_store_retrieval.py中的模型一致
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Load the existing vector store from persistent directory
persist_directory = "./chroma_db_VN"
try:
    vector_store = Chroma(
        collection_name="VN_knowledgeBase",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    
    # Check if the vector store has documents
    collection = vector_store._collection
    count = collection.count()
    
    if count == 0:
        print("❌ Vector store is empty!")
        print("\nPlease run 'python vector_store_retrieval.py' first to create and populate the vector store.")
        sys.exit(1)
    
    print(f"✓ Vector store loaded successfully! ({count} document chunks)")
except Exception as e:
    print(f"❌ Error loading vector store: {e}")
    print("\nPlease run 'python vector_store_retrieval.py' first to create the vector store.")
    sys.exit(1)

# Create a retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}  # 增加检索数量以获得更多相关上下文
)

# ============================================
# SET UP RAG WITH OPENAI
# ============================================
use_rag = False
llm = None

if os.environ.get("OPENAI_API_KEY"):
    try:
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        template = """您是一名 AI 助手，帮助回答有关VN项目的知识问题。请使用从文档中检索到的以下上下文来回答问题。
        如果您无法根据上下文回答问题，请说“我无法在提供的上下文中找到该信息。”请简明而具体地回答.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Helper function to format documents
        def format_docs(docs):
            return "\n\n".join(f"[Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}" for doc in docs)
        
        # Create RAG chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        use_rag = True
        print("\n✅ OpenAI RAG mode enabled! AI will answer your questions using retrieved context.")
    except Exception as e:
        print(f"\n⚠️  OpenAI setup failed: {e}")
        print("Falling back to search-only mode.")
else:
    print("\n💡 Tip: Set OPENAI_API_KEY environment variable to enable AI-powered answers!")
    print("   For now, showing retrieved passages only.")

print("\n" + "=" * 60)
if use_rag:
    print("🚀 Ready! Ask questions - AI will answer using VN project knowledge base.")
else:
    print("Ready! Ask questions about VN project knowledge base.")
print("Type 'quit' or 'exit' to stop.")
print("=" * 60)

print("\n💡 Example questions:")
print("  - 整个项目的系统结构是怎样的?")
print("  - 车辆的外观检查内容有哪些?")
print("  - 在给AGV建图的时候有哪些注意事项?")


while True:
    try:
        print("\n" + "-" * 60)
        user_query = input("\n❓ Your question: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! 👋")
            break
        
        if not user_query:
            continue
        
        print(f"\n🔍 Searching for: '{user_query}'")
        
        # Get relevant documents
        results = retriever.invoke(user_query)
        
        # If RAG is enabled, get AI answer
        if use_rag:
            print("\n🤖 AI Answer:")
            print("=" * 60)
            try:
                answer = rag_chain.invoke(user_query)
                print(answer)
            except Exception as e:
                print(f"Error generating answer: {e}")
            print("=" * 60)
        
        # Show retrieved passages
        print("\n📄 Retrieved Passages:\n")
        for i, doc in enumerate(results, 1):
            print(f"{i}. Page {doc.metadata.get('page', 'N/A')}")
            print("-" * 40)
            # Show first 400 chars for readability
            content = doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else "")
            print(content)
            print()
        
        # Show similarity scores
        results_with_scores = vector_store.similarity_search_with_score(user_query, k=3)
        print("📊 Similarity Scores (lower is better):")
        for i, (doc, score) in enumerate(results_with_scores, 1):
            print(f"  {i}. Page {doc.metadata.get('page', 'N/A')}: {score:.4f}")
        
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")
        break   
    except Exception as e:
        print(f"\n❌ Error: {e}")

