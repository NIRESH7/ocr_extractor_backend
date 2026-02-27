import os
import time
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http import models
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from database import get_qdrant_client

# Configuration
COLLECTION_NAME = "local_documents"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Global Model Singletons (Initializes only ONCE on startup)
print(f"--- [RAG] Initializing Local Embeddings & LLM (Ollama: {OLLAMA_MODEL}) ---")

# Purely Local HuggingFace Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("--- [RAG] Using Local HuggingFace Embeddings (all-MiniLM-L6-v2) ---")

# Purely Local Ollama
print("--- [RAG] Using Local Ollama Engine ---")
llm = Ollama(
    model=OLLAMA_MODEL,
    temperature=0,
    num_predict=64,
    top_p=0.9
)
print("--- [RAG] Models Loaded and Ready ---")

def get_rag_chain(folder_name: str = None):
    # 2. Vector Store - Use Singleton
    client = get_qdrant_client()
    vector_store = QdrantVectorStore(
        client=client, 
        collection_name=COLLECTION_NAME, 
        embedding=embeddings
    )
    
    search_kwargs = {"k": 3}
    if folder_name and folder_name != "All":
        print(f"--- [RAG] Filtering by folder: {folder_name} ---")
        # Use Qdrant's Filter model for better compatibility with langchain-qdrant
        search_kwargs["filter"] = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.folder", 
                    match=models.MatchValue(value=folder_name)
                )
            ]
        )
    
    # Increase k for better context
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    
    # 4. Prompt
    template = """You are a helpful assistant that answers questions based ONLY on the provided context.
    
    Rules:
    - If you don't know the answer, say "Query not found in document."
    - Answer the question directly and as briefly as possible.
    - Do not explain yourself or use any introduction.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:"""
    
    prompt = PromptTemplate.from_template(template)
    
    # 5. Chain with LCEL
    rag_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()} 
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return retriever, rag_chain

import json
import re

def clean_ocr_text(text: str):
    """Clean and structure OCR text for the LLM and logging."""
    # Remove weird characters and normalize spacing
    text = re.sub(r'\[.*?\]', '', text) # Remove [CLIENT NAME] style placeholders
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return "\n".join(lines)

def query_rag(question: str, folder_name: str = None):
    print("\n" + "🚀 " + "="*60)
    print(f"--- [RAG] NEW USER QUERY ---".center(60))
    print(f"--- [QUESTION]: {question} ---".center(60))
    print("="*60)
    
    try:
        retriever, chain = get_rag_chain(folder_name)
        docs = retriever.invoke(question)
        
        if not docs:
            return "Query not found in document."

        # STEP 2: SHOW DATA TO USER IN BACKEND
        print("\n" + "╔" + "═"*58 + "╗")
        print("║" + " SOURCE DATA EXTRACTED FROM DOCUMENT ".center(58) + "║")
        print("╠" + "═"*58 + "╢")
        
        found_data = []
        for i, doc in enumerate(docs):
            cleaned_text = clean_ocr_text(doc.page_content)
            found_data.append({
                "chunk": i + 1,
                "file": os.path.basename(doc.metadata.get("source", "unknown")),
                "text": cleaned_text
            })
            print(f"📄 CHUNK {i+1} ({found_data[-1]['file']}):")
            print(f"--------------------------------------------------")
            print(cleaned_text)
            print(f"--------------------------------------------------\n")

        print("╚" + "═"*58 + "╝")

        context_str = "\n".join([d["text"] for d in found_data])

        # STEP 3: GENERATION
        result = chain.invoke({"context": context_str, "question": question})
        
        clean_ans = result.strip()
        
        # LOG FINAL ANSWER AS JSON FOR USER
        print("\n" + "╔" + "═"*58 + "╗")
        print("║" + " ENGINE FINAL ANSWER ".center(58) + "║")
        print("╠" + "═"*58 + "╢")
        print(json.dumps({"question": question, "answer": clean_ans}, indent=2))
        print("╚" + "═"*58 + "╝\n")

        return clean_ans

    except Exception as e:
        print(f"--- [RAG ERROR]: {e} ---")
        return "Error in extraction node."
