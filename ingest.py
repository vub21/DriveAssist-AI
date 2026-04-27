from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import chromadb

load_dotenv()


def ingest_manual(filepath: str, on_progress=None):
    """
    Ingest a single PDF manual into ChromaDB.
    on_progress(step, message, percent) is called at each stage if provided.
    """
    def report(step, message, percent):
        print(message)
        if on_progress:
            on_progress(step, message, percent)

    report(1, "Loading PDF...", 5)
    loader = PyPDFLoader(filepath)
    documents = loader.load()
    if not documents:
        raise ValueError("No pages found in the PDF.")
    report(1, f"Loaded {len(documents)} pages.", 15)

    report(2, "Splitting into chunks...", 20)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    report(2, f"Created {len(chunks)} chunks.", 35)

    report(3, "Generating embeddings...", 40)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)
    report(3, f"Generated {len(embeddings)} embedding vectors.", 70)

    report(4, "Saving to database...", 75)
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="documind_data")

    # Remove any existing entries for this file before re-adding
    stem = Path(filepath).name
    try:
        existing = collection.get(where={"source": filepath})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"{stem}_{i}" for i in range(len(chunks))]

    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids,
    )
    report(4, f"Saved {len(chunks)} chunks to ChromaDB.", 95)
    report(5, "Done! Manual is ready to use.", 100)


if __name__ == "__main__":
    manuals_path = Path("data/owners_manual/")
    if not manuals_path.exists():
        print(f"⚠️  Directory {manuals_path} not found. Creating it...")
        manuals_path.mkdir(parents=True, exist_ok=True)

    pdfs = list(manuals_path.glob("*.pdf"))
    if not pdfs:
        print("❌ No PDF documents found in 'data/owners_manual/'. Please add some PDFs and run again.")
        exit()

    for pdf in pdfs:
        print(f"\n📄 Ingesting: {pdf.name}")
        ingest_manual(str(pdf))

    print("\n🎉 Done! All documents are ingested and ready to search.")
