# ingest.py
# PURPOSE: Load PDFs → split into chunks → convert to numbers → save to database

from glob import glob
import os
from dotenv import load_dotenv
from langchain_community.document_loaders  import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv() 

# 1. Load PDF

def load_documents(data_dir="./data"):

    print(f"📂 Loading PDFs from '{data_dir}'...")
    
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",          # This command loads all .pdf files
        loader_cls=PyPDFLoader    # Use PDF-specific loader
    )
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} pages across all PDFs")
    return docs

def split_into_chunks(docs):
    """
    Splits large documents into smaller chunks.
    
    Why chunk_size=700? 
      - Too large (2000+): Less precise retrieval, expensive
      - Too small (100):   Loses context, chunks are meaningless
      - 700 is the sweet spot for most documents
    
    Why chunk_overlap=100?
      - Prevents cutting a sentence right at the boundary
      - Neighboring chunks share 100 characters, so no info is lost
    """
    print("✂️  Splitting documents into chunks...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        # It tries to split at paragraph breaks first (\n\n),
        # then line breaks (\n), then sentences (". "), then words
        separators=["\n\n", "\n", ". ", " "]
    )
    
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Print a sample so you can SEE what a chunk looks like
    print("\n" + "="*50)
    print("SAMPLE CHUNK (so you understand the data):")
    print("="*50)
    print(f"Text: {chunks[0].page_content[:400]}")
    print(f"Metadata: {chunks[0].metadata}")
    print("="*50 + "\n")
    
    return chunks

def embed_and_store(chunks):
    """
    Converts each chunk's text into a list of numbers (embedding).
    Similar text = similar numbers = can search by meaning.
    
    Example:
      "refund policy" → [0.23, -0.11, 0.87, ...]  (768 numbers)
      "money back guarantee" → [0.21, -0.09, 0.85, ...]  (very similar!)
      "chocolate cake recipe" → [-0.45, 0.33, -0.12, ...]  (very different)
    """
    print("🔢 Converting chunks to embeddings...")
    print("   (First time downloads a ~400MB model — be patient)")
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    print("💾 Saving to ChromaDB vector database...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db",    # Saves to disk
        collection_name="documind_docs"
    )
    vectorstore.persist()
    
    count = vectorstore._collection.count()
    print(f"✅ Database ready! Contains {count} chunks")
    return vectorstore


if __name__ == "__main__":
    print("🚀 Starting AI Powered Vehicle Assistant ingestion pipeline...\n")
    docs   = load_documents()
    chunks = split_into_chunks(docs)
    embed_and_store(chunks)
    print("\n🎉 Done! Your documents are ingested and ready to search.")
    print("   Next step: Run retrieval.py to test searching")