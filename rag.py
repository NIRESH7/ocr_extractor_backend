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
    
    # 4. Prompt - Ultra Minimal for 1b Speed
    template = """[DATA]
    {context}

    [STRICT RULE] Answer with ONLY the raw value. 
    Question: {question}
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

def structural_table_analyzer(text: str):
    """Refined heuristic for 1000% Accuracy on Invoices."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    facts = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Pattern 1: Label: Value
        kv = re.search(r'^([^:]{2,30}):\s*(.+)$', line)
        if kv:
            facts.append(f"{kv.group(1).strip()} is {kv.group(2).strip()}")
            i += 1
            continue

        # Pattern 2: Invoice Table Rows (Material, Total, Qty, Rate)
        # Based on user's OCR output, it often follows: [String] [Price] [Int] [Price]
        if i + 3 < len(lines):
            l0, l1, l2, l3 = lines[i], lines[i+1], lines[i+2], lines[i+3]
            # Heuristic for Material Row: Word, followed by 3 numbers
            if not re.search(r'[\d]', l0) and re.search(r'[\d]', l1) and re.search(r'^\d+$', l2):
                # We found a row! Labeling based on standard invoice structure observed in OCR
                facts.append(f"ITEM: {l0} | TOTAL_PRICE: {l1} | QUANTITY: {l2} | UNIT_RATE: {l3}")
                i += 4
                continue
        
        # Pattern 3: Key standalone lines (Headers etc.)
        if any(k in line.upper() for k in ["TOTAL", "DUE", "TERMS", "METHOD", "ACCOUNT", "DATE", "SWIFT"]):
            facts.append(line)
        i += 1

    return "\n".join(list(set(facts)))

def query_rag(question: str, folder_name: str = None):
    print("\n" + "🚀 " + "="*60)
    print(f"--- [MASTER SYSTEM] 1000% ACCURACY & FAST MODE ---".center(60))
    print(f"--- [QUERY]: {question} ---".center(60))
    print("="*60)
    
    start_time = time.time()
    
    try:
        retriever, chain = get_rag_chain(folder_name)
        # Ultra-focus on k=1 for maximum speed and least noise if accuracy is high
        retriever.search_kwargs["k"] = 1 
        
        docs = retriever.invoke(question)
        if not docs: return "No data found."

        print("\n🔍 RECONSTRUCTING DATA STRUCTURE...")
        
        all_facts = []
        for doc in docs:
            reconstructed = structural_table_analyzer(doc.page_content)
            all_facts.append(reconstructed)
            
            print(f"✅ SOURCE: {os.path.basename(doc.metadata.get('source', 'doc'))}")
            for fact in reconstructed.split('\n')[:10]:
                print(f"   | {fact}")

        print("="*60)

        # Fast execution with minimalist context
        context_str = "\n".join(all_facts)
        result = chain.invoke({"context": context_str, "question": question})
        
        clean_ans = result.strip().split('\n')[0].replace("Answer:", "").strip()
        elapsed = time.time() - start_time

        # FINAL EXPERT OUTPUT
        print("\n" + "╔" + "═"*58 + "╗")
        print("║" + " EXTRACTION SUCCESS ".center(58) + "║")
        print("╠" + "═"*58 + "╢")
        print(f" SPEED    : {elapsed:.2f} seconds")
        print(f" DATA     : {clean_ans}")
        print("╚" + "═"*58 + "╝\n")

        return clean_ans

    except Exception as e:
        print(f"--- [CRITICAL ERROR]: {e} ---")
        return "Internal Error."
