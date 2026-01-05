"""
RAG (Retrieval-Augmented Generation) Example
Builds on top of the semantic search engine to answer questions using an LLM
"""

import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ============================================
# 1. LOAD EXISTING VECTOR STORE
# ============================================
print("=" * 60)
print("Loading Vector Store")
print("=" * 60)

# Check if OpenAI API key is set
if not os.environ.get("OPENAI_API_KEY"):
    print("⚠️  OPENAI_API_KEY not found!")
    print("Please set it: os.environ['OPENAI_API_KEY'] = 'your-api-key'")
    print("\nYou need to run semanticSearch.py first to create the vector store.")
    exit(1)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

try:
    vector_store = Chroma(
        collection_name="nike_10k_2023",
        embedding_function=embeddings
    )
    print("✓ Vector store loaded successfully")
except Exception as e:
    print(f"❌ Error loading vector store: {e}")
    print("\nPlease run semanticSearch.py first to create the vector store.")
    exit(1)

# Create retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
)

# ============================================
# 2. CREATE RAG CHAIN
# ============================================
print("\n" + "=" * 60)
print("Creating RAG Chain")
print("=" * 60)

# Create prompt template
template = """You are an assistant for question-answering tasks about Nike's 2023 10-K filing.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer based on the context, just say that you don't know.
Keep the answer concise and informative.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Create LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Helper function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("✓ RAG chain created")

# ============================================
# 3. ASK QUESTIONS
# ============================================
print("\n" + "=" * 60)
print("Question Answering with RAG")
print("=" * 60)

questions = [
    "How were Nike's margins impacted in 2023?",
    "How many distribution centers does Nike have in the US?",
    "When was Nike incorporated?",
    "What were Nike's revenue trends in 2023?",
    "What are the main business activities of Nike?"
]

for i, question in enumerate(questions, 1):
    print(f"\n{'='*60}")
    print(f"Question {i}: {question}")
    print('='*60)
    
    # Get relevant documents (for reference)
    relevant_docs = retriever.invoke(question)
    print(f"\n📚 Retrieved {len(relevant_docs)} relevant chunks:")
    for j, doc in enumerate(relevant_docs, 1):
        print(f"  {j}. Page {doc.metadata.get('page', 'N/A')}")
    
    # Get answer from RAG chain
    print("\n💡 Answer:")
    answer = rag_chain.invoke(question)
    print(answer)

# ============================================
# 4. INTERACTIVE MODE
# ============================================
print("\n" + "=" * 60)
print("Interactive Question Answering")
print("=" * 60)
print("\nYou can now ask questions about Nike's 2023 10-K filing.")
print("Type 'quit' or 'exit' to stop.\n")

while True:
    try:
        user_question = input("\n❓ Your question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! 👋")
            break
        
        if not user_question:
            continue
        
        # Get relevant documents
        relevant_docs = retriever.invoke(user_question)
        print(f"\n📚 Found {len(relevant_docs)} relevant sections (Pages: {', '.join(str(d.metadata.get('page', 'N/A')) for d in relevant_docs)})")
        
        # Get answer
        print("\n💡 Answer:")
        answer = rag_chain.invoke(user_question)
        print(answer)
        
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}")

