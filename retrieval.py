# retrieval.py
# PURPOSE: Given a question, find the most relevant chunks from the database

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder


def build_retrievers(chunks):
    """
    Sets up the hybrid retrieval system.
    Call this once at startup and reuse the returned retriever.
    """
    
    # --- BM25 Retriever (keyword-based) ---
    print("📚 Setting up BM25 retriever...")
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 10  # Return top 10 keyword matches
    
    # --- Vector Retriever (semantic/meaning-based) ---
    print("🔢 Loading vector database...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="documind_docs"
    )
    vector = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # --- Hybrid: Combine Both ---
    # weights=[0.4, 0.6] means: 40% keyword influence, 60% semantic influence
    # You can tune these — for technical docs with lots of specific terms,
    # try [0.5, 0.5] or even [0.6, 0.4]
    hybrid = EnsembleRetriever(
        retrievers=[bm25, vector],
        weights=[0.4, 0.6]
    )
    print("✅ Hybrid retriever ready")
    return hybrid, chunks


def rerank(query: str, chunks: list, top_k: int = 5) -> list:
    """
    Re-ranking: A smarter but slower scoring pass.
    
    Regular retrieval scores the query alone.
    CrossEncoder reads BOTH query + chunk together — much more accurate.
    
    We only use it on the final 20 candidates (not all 10,000 chunks)
    because it's too slow to run on everything.
    """
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Create (query, chunk_text) pairs
    pairs = [(query, chunk.page_content) for chunk in chunks]
    
    # Score each pair — higher score = more relevant
    scores = reranker.predict(pairs)
    
    # Sort by score, return top_k
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in ranked[:top_k]]


def retrieve(query: str, hybrid_retriever) -> list:
    """
    Full retrieval pipeline:
    1. Hybrid search → 20 candidates
    2. Re-rank → top 5
    """
    print(f"\n🔍 Query: '{query}'")
    
    # Stage 1: Fast hybrid retrieval — get 20 candidates
    candidates = hybrid_retriever.get_relevant_documents(query)
    print(f"   Stage 1 (hybrid): {len(candidates)} candidates retrieved")
    
    # Stage 2: Accurate re-ranking — pick best 5
    best = rerank(query, candidates, top_k=5)
    print(f"   Stage 2 (rerank): narrowed to {len(best)} best chunks")
    
    # Show what was found
    for i, chunk in enumerate(best):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "?")
        print(f"   [{i+1}] {source} (page {page}): {chunk.page_content[:80]}...")
    
    return best


# Quick test — run this file directly to test retrieval
if __name__ == "__main__":
    from ingest import load_documents, split_into_chunks
    
    print("Loading documents for BM25...")
    docs   = load_documents()
    chunks = split_into_chunks(docs)
    
    hybrid, chunks = build_retrievers(chunks)
    
    # Test with a sample question
    test_query = "What are the main topics covered in these documents?"
    results = retrieve(test_query, hybrid)
    
    print(f"\n✅ Found {len(results)} relevant chunks for your query")