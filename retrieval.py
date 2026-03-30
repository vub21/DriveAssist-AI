from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from dotenv import load_dotenv
import os
from pathlib import Path 

# Load environment variables (API keys, etc.)
load_dotenv()

# Global Configuration
persist_directory = "./chroma_db"
collection_name = "documind_data"
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_vectorstore():
    if not os.path.exists(persist_directory):
        return None
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function,
        collection_name=collection_name
    )

def get_available_models():
    """
    Returns a unique list of car manuals currently stored in the database.
    """
    vectorstore = get_vectorstore()
    if not vectorstore:
        return []
    
    db_data = vectorstore.get()
    if not db_data or not db_data['metadatas']:
        return []
    
    # Extract unique source names (filenames)
    sources = set()
    for meta in db_data['metadatas']:
        source_path = meta.get("source", "unknown")
        sources.add(Path(source_path).name)
    
    return sorted(list(sources))

def setup_hybrid_retriever(model_filename=None):
    """
    Initializes a hybrid retriever. If model_filename is provided,
    it filters results to ONLY that specific manual.
    """
    vectorstore = get_vectorstore()
    if not vectorstore:
        print("❌ Error: Database not found.")
        return None

    print(f"📚 Preparing search index for: {model_filename if model_filename else 'All Models'}...")
    
    # 1. Fetch chunks (filtered if model_filename is set)
    if model_filename:
        # We fetch all data and filter manually to build the BM25 correctly
        db_data = vectorstore.get()
        chunks = []
        for text, meta in zip(db_data['documents'], db_data['metadatas']):
            if Path(meta.get("source", "")).name == model_filename:
                chunks.append(Document(page_content=text, metadata=meta))
    else:
        db_data = vectorstore.get()
        chunks = [Document(page_content=text, metadata=meta) for text, meta in zip(db_data['documents'], db_data['metadatas'])]

    if not chunks:
        print("⚠️ No documents found for the selected model.")
        return None

    # 2. Build Keyword Retriever (BM25)
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = 5

    # 3. Build Semantic Retriever (Vector)
    search_kwargs = {"k": 5}
    if model_filename:
        # Chroma filter uses the absolute path stored in DB, but the user picked 
        # based on basename. Let's find one mapping.
        # Quickest way: use the filtered chunks' source path
        sample_path = chunks[0].metadata.get("source")
        search_kwargs["filter"] = {"source": sample_path}

    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )

    # 4. Combine into Ensemble
    hybrid_retriever = EnsembleRetriever(
        retrievers=[keyword_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    return hybrid_retriever

def retrieve(query: str, hybrid_retriever):
    """
    Search the hybrid database using a pre-initialized retriever.
    """
    if not hybrid_retriever:
        return []

    print(f"\n🔍 Querying: '{query}'")
    results = hybrid_retriever.invoke(query)
    
    print(f"Done! Found {len(results)} relevant results.")
    for i, res in enumerate(results):
        source = Path(res.metadata.get("source", "unknown")).name
        # PyPDFLoader uses 0-based indexing; +1 gives the PDF viewer page number.
        raw_page = res.metadata.get("page")
        page = (int(raw_page) + 1) if raw_page is not None else "?"
        print(f"   [{i+1}] {source} (page {page}): {res.page_content[:100]}...")
    
    return results

if __name__ == "__main__":
    # --- TEST SEARCH ---
    models = get_available_models()
    if models:
        print("Available models:")
        for i, m in enumerate(models):
            print(f"[{i+1}] {m}")
        
        selection = int(input("\nSelect model number (or 0 for all): "))
        selected_model = models[selection-1] if selection > 0 else None
        
        retriever = setup_hybrid_retriever(selected_model)
        test_query = "How do I check the engine oil level?"
        retrieve(test_query, retriever)




