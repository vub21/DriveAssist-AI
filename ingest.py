# Step 1: Import necessary libraries
from pathlib import Path             # To handle file paths efficiently
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Split large texts
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import chromadb
import numpy as np

# Load environment variables (API keys, etc.)
load_dotenv()

print("📂 Loading the PDF's...")

# Step 2: Define the path to your owner's manuals
manuals_path = Path("data/owners_manual/")  # Folder containing all manuals (PDF, TXT, DOCX)

# Step 3: Load documents from the directory
if not manuals_path.exists():
    print(f"⚠️  Warning: Directory {manuals_path} not found. Creating it...")
    manuals_path.mkdir(parents=True, exist_ok=True)

loader = PyPDFDirectoryLoader(str(manuals_path))

# Step 4: Load all documents into memory
documents = loader.load()  # Each document is now a LangChain Document object

if not documents:
    print("❌ No PDF documents found in 'data/owners_manual/'. Please add some PDFs and run again.")
    exit()

print(f"✅ Loaded {len(documents)} pages.")

print("✂️  Splitting the documents into chunks...")
# Step 5: Split documents into smaller chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,    # Max 700 characters per chunk
    chunk_overlap=100   # Overlap 100 characters for context continuity
)

chunks = text_splitter.split_documents(documents)

print(f"✅ Created {len(chunks)} chunks.")

print("🔢 Creating embeddings...")
# Step 7: Create embeddings
# This model is small and fast for local use
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [chunk.page_content for chunk in chunks]
embeddings = model.encode(texts, show_progress_bar=True)

print("✅ Embeddings created successfully.")

# -----------------------------
# 3️⃣ Create local Chroma DB
# -----------------------------
print("💾 Saving to ChromaDB vector database...")
# Using PersistentClient for modern ChromaDB (>= 0.4.0)
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get a collection to store your embeddings
collection_name = "documind_data"
collection = client.get_or_create_collection(name=collection_name)

# -----------------------------
# 4️⃣ Add embeddings + metadata
# -----------------------------
# Prepare metadata and IDs
metadatas = [chunk.metadata for chunk in chunks]
ids = [str(i) for i in range(len(chunks))]

collection.add(
    documents=texts,
    embeddings=embeddings.tolist(),  # Chroma expects a list
    metadatas=metadatas,
    ids=ids
)

print(f"✅ Data stored successfully in collection: '{collection_name}'")
print("\n🎉 Done! Your documents are ingested and ready to search.")

