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
print(f"--- [RAG] Initializing Embeddings & LLM ({OLLAMA_MODEL}) ---")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Support for remote or local APIs
if os.getenv("OPENAI_API_KEY"):
    from langchain_openai import ChatOpenAI
    print("--- [RAG] Using OpenAI API ---")
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
elif os.getenv("OLLAMA_API_KEY") or OLLAMA_BASE_URL != "http://localhost:11434":
    print(f"--- [RAG] Using Remote/Cloud Ollama at {OLLAMA_BASE_URL} ---")
    llm = Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_predict=64
    )
else:
    print("--- [RAG] Using Local Ollama ---")
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
    template = """You are a STRICT document-grounded assistant.

Your knowledge is LIMITED to the provided CONTEXT only.
The CONTEXT may contain OCR text and may include noise.

════════════════════════════════════
ABSOLUTE RULES (NON-NEGOTIABLE)
════════════════════════════════════
1. Use ONLY information explicitly present in the CONTEXT.
2. NEVER use prior knowledge, assumptions, or inference.
3. NEVER guess or complete missing information.
4. Use values from ONE clearly identifiable document, page, and company only.
5. If more than one company, invoice, or document appears → respond EXACTLY:
   Query not found in document.
6. If ANY requested value is missing, unclear, or not explicitly written → respond EXACTLY:
   Query not found in document.
8. OCR Merged Text Handling: The text is from an OCR engine and lacks layout. Data values often merge with the NEXT header (e.g. `Due Date Email: info@company.com 02-17-2025` means Due Date=02-17-2025 and Email=info@company.com).
9. Proximity Rule: When multiple similar fields exist (like two different phone numbers or emails), associate values based on chronological proximity to the target entity (e.g. info@topconstruction.com belongs to TopConstructionInc, michael.brown@example.com belongs to Michael Brown).

════════════════════════════════════
QUESTION TYPE HANDLING (MANDATORY)
════════════════════════════════════
A. FACT QUESTIONS
- Extract ONLY the exact value(s) explicitly written in the CONTEXT.

B. MULTI-FIELD QUESTIONS
- Identify EACH requested field.
- Extract ALL values from the SAME document/page.
- If even ONE value is missing → STOP and respond:
  Query not found in document.

C. VALIDATION QUESTIONS (e.g., “Is this correct?”)
- If the statement is fully correct → respond EXACTLY:
  ✅ Correct
- If the statement is incorrect → respond EXACTLY:
  ❌ Not correct
- After ❌ Not correct, output ONLY the correct value(s) from the document.
- Do NOT add explanations.

════════════════════════════════════
ANSWER FORMAT (MANDATORY)
════════════════════════════════════
- Output ONE single line only.
- No explanations, reasoning, labels, or headings.
- Do NOT restate the question.
- Combine multiple values using commas.
- Preserve original formatting (dates, currency symbols).

════════════════════════════════════

CONTEXT:
{context}

USER QUESTION:
{question}
"""
    
    prompt = PromptTemplate.from_template(template)
    
    # 5. Chain with LCEL
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()} 
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return retriever, rag_chain

def query_rag(question: str, folder_name: str = None):
    print("\n" + "🚀 " + "="*50)
    print(f"--- [RAG] NEW USER QUERY RECEIVED ---")
    print(f"--- [QUESTION]: {question}")
    print("="*50)
    
    start_time = time.time()
    
    # STEP 1: RETRIEVAL
    print(f"\n🔍 [STEP 1/4]: Searching document database for relevant sections...")
    
    retriever, chain = get_rag_chain(folder_name)
    
    try:
        docs = retriever.invoke(question)
        print(f"--- [RAG] Retrieved {len(docs)} chunks ---")
        
        if not docs:
            print(f"❌ [RAG] No documents found. Aborting generation.")
            return "Data not found in document."
            
        import re
        # Print the fully retrieved data to the terminal in Key: Value format
        for i, doc in enumerate(docs):
            print(f"\n" + "═"*70)
            print(f"📄 [RAG] CHUNK {i+1} FULL DATA")
            print("═"*70)
            
            # To guarantee NO data is lost, we print the FULL raw text but add line breaks 
            # before known labels and headers so it resembles a key-value layout.
            import re
            text = doc.page_content.strip()
            
            # Identify likely fields generically. Instead of hardcoding "Date" or "Phone",
            # we look for:
            # 1. Any word(s) ending in a colon (e.g. "Name:")
            # 2. Or any sequence of Title Case words (e.g. "Invoice Number")
            # 3. Or any sequence of UPPERCASE words (e.g. "TOTAL AMOUNT")
            # and inject newlines before them to create a dynamic list format for ANY document.
            formatted_text = re.sub(
                r'(\b[A-Za-z\s]+:\s*|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b[A-Z]+(?:\s+[A-Z]+)*\b)', 
                r'\n\1', 
                text
            )
            
            # Clean and print every single piece of data
            lines = [line.strip() for line in formatted_text.split('\n') if line.strip()]
            
            for line in lines:
                # If there's a natural split like 'Phone: 123', make it neat
                if ':' in line:
                    parts = line.split(':', 1)
                    print(f"{parts[0].strip().ljust(25)} : {parts[1].strip()}")
                else:
                    # Otherwise print the whole phrase cleanly
                    print(line)
                
                print() # Keep the empty gap between lines
            
            print("═"*70 + "\n")
        context_str = "\n\n".join(doc.page_content for doc in docs)
        
    except Exception as e:
        print(f"❌ [RAG ERROR] Retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return "Error accessing document database."

    # STEP 2: GENERATION
    print(f"🧠 [STEP 2/4]: Found relevant info. Thinking & formulating response...")
    
    try:
        # Pass context and question explicitly
        result = chain.invoke({"context": context_str, "question": question})
    except Exception as e:
        error_msg = str(e)
        if "model requires more system memory" in error_msg:
             print(f"❌ [RAG ERROR]: OOM Error: {error_msg}")
             return (
                 "I apologize, but I cannot answer your question right now because the AI model "
                 "is too large for the available system memory. \n\n"
                 "Please try switching to a smaller model or close other applications to free up RAM."
             )
        raise e
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # STEP 3 & 4: SUCCESS
    print(f"✨ [STEP 3/4]: Response formulated successfully.")
    print(f"📝 [STEP 4/4]: Final Answer generated in {total_time:.2f}s.")
    
    print("\n" + "🤖 " + "-"*50)
    print(f"--- [RAG] BOT REPLY ---")
    print(f"{result}")
    print("-"*50 + "\n")
    return result
