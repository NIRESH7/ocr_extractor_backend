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
    template = """[SYSTEM: EXTRACTION BOT]
    1. EXCLUSIVELY use the JSON CONTEXT provided.
    2. Respond with ONLY the raw value(s) requested.
    3. NO intro ("The answer is..."), NO sentences, NO conversational filler.
    4. If not found, say "Query not found in document."
    5. Handle typos in user question by looking for semantic matches in context.

    CONTEXT:
    {context}

    USER QUESTION:
    {question}

    FINAL OUTPUT (RAW VALUE ONLY):
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
import re

def text_to_json_keys(text: str):
    """Smarter OCR parsing: Pairs labels with values and removes 'unstructured' prefixes."""
    data_dict = {}
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for Key: Value or Key - Value
        match = re.search(r'^(.{1,30}?)([:\-])\s*(.*)$', line)
        if match:
            key = match.group(1).strip()
            val = match.group(3).strip()
            if key and val:
                data_dict[key] = val
                i += 1
                continue
            elif key and not val and (i + 1 < len(lines)):
                # Key on this line, Value on next?
                next_line = lines[i+1]
                data_dict[key] = next_line
                i += 2
                continue

        # Look for typical headers or keys without colons (e.g. "Invoice Number")
        # If the line is short and title-case/uppercase, treat as key for the next line
        if len(line.split()) < 4 and i + 1 < len(lines):
             next_val = lines[i+1]
             # If next line is a number or date, it's definitely a pair
             if re.search(r'[\d]', next_val):
                 data_dict[line] = next_val
                 i += 2
                 continue

        # Fallback: Just put the line as its own key/value if short
        if len(line.split()) < 6:
            data_dict[line] = "---"
        else:
            data_dict[f"info_{i}"] = line
        i += 1
                 
    return data_dict

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

        # STEP 2: JSON EXTRACTION FOR LOGGING
        structured_logs = []
        for i, doc in enumerate(docs):
            kv_pairs = text_to_json_keys(doc.page_content)
            structured_logs.append({
                "chunk": i + 1,
                "file": os.path.basename(doc.metadata.get("source", "unknown")),
                "data": kv_pairs
            })

        # PRINT CLEAN JSON TO CONSOLE
        print("\n" + "╔" + "═"*58 + "╗")
        print("║" + " SOURCE DATA (STRUCTURED JSON) ".center(58) + "║")
        print("╠" + "═"*58 + "╢")
        print(json.dumps(structured_logs, indent=2))
        print("╚" + "═"*58 + "╝")

        context_str = json.dumps([s["data"] for s in structured_logs])

        # STEP 3: GENERATION
        result = chain.invoke({"context": context_str, "question": question})
        
        # CLEANUP: Remove any remaining bot filler (Ollama sometimes adds intro text)
        clean_ans = result.strip()
        # If it looks like a sentence start, cut it (heuristic)
        if clean_ans.lower().startswith("the "):
            clean_ans = clean_ans.split(" is ", 1)[-1] if " is " in clean_ans else clean_ans
        
        # LOG FINAL ANSWER
        print("\n" + "╔" + "═"*58 + "╗")
        print("║" + " ENGINE EXTRACTION (FINAL) ".center(58) + "║")
        print("╠" + "═"*58 + "╢")
        print(json.dumps({"question": question, "answer": clean_ans}, indent=2))
        print("╚" + "═"*58 + "╝\n")

        return clean_ans

    except Exception as e:
        print(f"--- [RAG ERROR]: {e} ---")
        return "Error in extraction node."
