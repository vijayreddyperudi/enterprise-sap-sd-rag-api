import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PDF_PATH = os.getenv("PDF_PATH", "data/Sample.pdf")
INDEX_PATH = os.getenv("INDEX_PATH", "data/faiss_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


def resolve_pdf_path(path: str) -> str:
    """Resolve PDF path with case-tolerance for cross-platform compatibility."""
    if os.path.exists(path):
        return path
    # Try alternative casing if default doesn't match directly
    dirname = os.path.dirname(path) or "."
    basename = os.path.basename(path).lower()
    for f in os.listdir(dirname) if os.path.exists(dirname) else []:
        if f.lower() == basename:
            return os.path.join(dirname, f)
    raise FileNotFoundError(f"Source PDF file not found at: {path}")


def create_index(pdf_path: str = PDF_PATH, index_path: str = INDEX_PATH):
    resolved_pdf = resolve_pdf_path(pdf_path)
    print(f"Loading PDF from {resolved_pdf}...")
    loader = PyPDFLoader(resolved_pdf)
    documents = loader.load()
    print(f"Pages loaded: {len(documents)}")

    print("Creating chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Chunks created: {len(chunks)}")

    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Creating FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    print(f"Saving FAISS index to {index_path}...")
    os.makedirs(index_path, exist_ok=True)
    vector_store.save_local(index_path)
    print("FAISS index saved successfully.")

    return {
        "pages": len(documents),
        "chunks": len(chunks),
        "index_path": index_path,
    }


if __name__ == "__main__":
    result = create_index()
    print("\n================================")
    print("INGESTION COMPLETE")
    print("================================")
    print(f"Pages: {result['pages']}")
    print(f"Chunks: {result['chunks']}")
    print(f"Index: {result['index_path']}")

