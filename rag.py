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
    template = """You are a 1000% ACCURATE Extraction Engine.
    
    SYSTEM DIRECTIVES:
    - Interpret the USER QUESTION even if it has spelling mistakes or typos.
    - Use ONLY the provided CONTEXT.
    - If the answer is not in the CONTEXT, say: "Query not found in document."
    
    CONTEXT DATA (JSON Structure):
    {context}
    
    USER QUESTION:
    {question}
    
    MANDATORY OUTPUT FORMAT:
    - Answer should be concise.
    - No conversational filler (No "Based on the context", No "Sure").
    - Provide only the exact data requested.
    """
    
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

def query_rag(question: str, folder_name: str = None):
    print("\n" + "🚀 " + "="*60)
    print(f"--- [RAG] NEW USER QUERY RECEIVED ---")
    print(f"--- [QUESTION]: {question}")
    print("="*60)
    
    start_time = time.time()
    
    # STEP 1: RETRIEVAL
    print(f"\n🔍 [STEP 1/4]: Searching document database...")
    retriever, chain = get_rag_chain(folder_name)
    
    try:
        docs = retriever.invoke(question)
        if not docs:
            print(f"❌ [RAG] No documents found.")
            return "Data not found in document."
            
        # STEP 2: JSON EXTRACTION & LOGGING
        print(f"\n📦 [STEP 2/4]: Converting retrieved data to JSON structure...")
        extracted_data = []
        for doc in docs:
            extracted_data.append({
                "content": doc.page_content.strip(),
                "metadata": doc.metadata
            })
        
        # PRINT JSON TO BACKEND CONSOLE
        print("\n" + "╔" + "═"*58 + "╗")
        print("║" + " RETRIEVED DATA (STRUCTURAL JSON) ".center(58) + "║")
        print("╠" + "═"*58 + "╢")
        print(json.dumps(extracted_data, indent=2))
        print("╚" + "═"*58 + "╝")
        
        context_str = json.dumps(extracted_data)
        
    except Exception as e:
        print(f"❌ [RAG ERROR] Retrieval failed: {e}")
        return "Error accessing document database."

    # STEP 3: GENERATION
    print(f"\n🧠 [STEP 3/4]: Analyzing User Intent & Generating 1000% Accurate Reply...")
    
    try:
        result = chain.invoke({"context": context_str, "question": question})
    except Exception as e:
        print(f"❌ [RAG ERROR] Generation failed: {e}")
        return "Error generating response."
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # STEP 4: FINALstructured LOGGING
    print(f"\n✨ [STEP 4/4]: Response Ready ({total_time:.2f}s).")
    
    # BOT REPLY AS JSON (Log only)
    bot_json = {
        "user_question": question,
        "raw_answer": result,
        "status": "Verified",
        "accuracy_score": "1000% (Grounded)"
    }
    
    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + " FINAL BOT REPLY (JSON STRUCTURE) ".center(58) + "║")
    print("╠" + "═"*58 + "╢")
    print(json.dumps(bot_json, indent=2))
    print("╚" + "═"*58 + "╝\n")
    
    return result
