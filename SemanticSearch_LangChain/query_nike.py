"""
Interactive Query Script for Nike 10-K Semantic Search with RAG
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
print("Nike 10-K Semantic Search - Interactive Mode")
print("=" * 60)
print("\nLoading vector store...")

# Initialize the same embeddings model used during indexing
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the existing vector store from persistent directory
persist_directory = "./chroma_db"
try:
    vector_store = Chroma(
        collection_name="nike_10k_2023",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    
    # Check if the vector store has documents
    collection = vector_store._collection
    count = collection.count()
    
    if count == 0:
        print("❌ Vector store is empty!")
        print("\nPlease run 'python semanticSearch.py' first to create and populate the vector store.")
        sys.exit(1)
    
    print(f"✓ Vector store loaded successfully! ({count} document chunks)")
except Exception as e:
    print(f"❌ Error loading vector store: {e}")
    print("\nPlease run 'python semanticSearch.py' first to create the vector store.")
    sys.exit(1)

# Create a retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}  
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
        
        template = """You are an AI assistant helping to answer questions about Nike's 2023 10-K filing.
Use the following context retrieved from the document to answer the question.
If you cannot answer the question based on the context, say "I cannot find that information in the provided context."
Be concise and specific in your answer.

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
    print("🚀 Ready! Ask questions - AI will answer using Nike's 2023 10-K filing.")
else:
    print("Ready! Ask questions about Nike's 2023 10-K filing.")
print("Type 'quit' or 'exit' to stop.")
print("=" * 60)

print("\n💡 Example questions:")
print("  - How were Nike's margins impacted in 2023?")
print("  - How many distribution centers does Nike have in the US?")
print("  - When was Nike incorporated?")
# print("  - What were Nike's total revenues in fiscal 2023?")
# print("  - Tell me about Nike's retail stores")                               
# print("  - How much did Converse revenues increase?")

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

