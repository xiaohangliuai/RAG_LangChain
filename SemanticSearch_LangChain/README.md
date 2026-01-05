# A semantic search engine built with LangChain

## Features

- 📄 PDF document loading
- ✂️ Intelligent text splitting with overlap
- 🔢 Document embeddings (OpenAI, HuggingFace, or Ollama)
- 🗄️ Vector storage with Chroma
- 🔍 Semantic similarity search
- 🎯 Retriever interface for easy integration

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up API Keys (if using OpenAI)

Create a `.env` file:

```bash
OPENAI_API_KEY=your-api-key-here
```

Or set it in your code:

```python
import os
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
```

### 3. Alternative: Use Free Embeddings

If you don't have an OpenAI API key, you can use free alternatives:

**Option A: HuggingFace (runs locally)**
```python
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

**Option B: Ollama (requires Ollama installed)**
```python
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="llama2")
```

## Usage

### Run the Semantic Search Engine

```bash
python semanticSearch.py
```

### Use in Your Own Code

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Load existing vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma(
    collection_name="nike_10k_2023",
    embedding_function=embeddings
)

# Search
results = vector_store.similarity_search("your query here", k=3)
for doc in results:
    print(doc.page_content)
    print(doc.metadata)
```

### Use as a Retriever

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Single query
docs = retriever.invoke("What are Nike's revenues?")

# Batch queries
results = retriever.batch([
    "What are Nike's revenues?",
    "How many stores does Nike have?"
])
```

## Key Components

### 1. Document Loaders
- Loads PDF files page by page
- Preserves metadata (page number, source)

### 2. Text Splitters
- Splits documents into chunks (default: 1000 chars)
- Uses overlap (default: 200 chars) to preserve context
- Tracks original position via `start_index`

### 3. Embeddings
- Converts text to numerical vectors
- Supports multiple providers (OpenAI, HuggingFace, Ollama, etc.)

### 4. Vector Stores
- Stores document embeddings
- Enables fast similarity search
- Uses Chroma (lightweight, persistent)

### 5. Retrievers
- Runnable interface for document retrieval
- Supports batch operations
- Can be chained with other components

## Next Steps

### Build a RAG Application

Combine this search engine with an LLM to answer questions:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Create RAG chain
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-3.5-turbo")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Ask questions
answer = rag_chain.invoke("How were Nike's margins impacted in 2023?")
print(answer)
```

## Resources

- [LangChain Documentation](https://docs.langchain.com/)
- [Tutorial: Build a semantic search engine](https://docs.langchain.com/oss/python/langchain/knowledge-base)
- [Tutorial: Build a RAG application](https://python.langchain.com/docs/tutorials/rag/)

## License

MIT

