"""
Task 3 — Retriever module.

Loads the ChromaDB collection built from complaint_embeddings.parquet
and performs semantic similarity search using all-MiniLM-L6-v2.

Prerequisites:
    Run notebooks/03_build_vector_store_from_parquet.ipynb once first.
"""
from typing import List, Dict, Any, Optional
import chromadb
from src.embeddings import embed_texts
from src.config import VECTOR_STORE_DIR, CHROMA_COLLECTION_NAME

TOP_K = 5
_collection = None


def _get_collection() -> chromadb.Collection:
    """
    Load and cache the ChromaDB collection as a module-level singleton.
    Opens the client only once per process to avoid repeated disk reads.

    Raises
    ------
    FileNotFoundError
        If vector_store/ does not exist — run notebook 03 first.
    ValueError
        If the collection name is not found inside the store.
    """
    global _collection
    if _collection is not None:
        return _collection

    if not VECTOR_STORE_DIR.exists():
        raise FileNotFoundError(
            f"Vector store not found at: {VECTOR_STORE_DIR}\n"
            "Run notebooks/03_build_vector_store_from_parquet.ipynb first."
        )

    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))

    try:
        _collection = client.get_collection(CHROMA_COLLECTION_NAME)
    except Exception:
        available = [c.name for c in client.list_collections()]
        raise ValueError(
            f"Collection '{CHROMA_COLLECTION_NAME}' not found.\n"
            f"Collections available in store: {available}\n"
            "Update CHROMA_COLLECTION_NAME in src/config.py to match."
        )

    print(f"✓ Loaded collection '{CHROMA_COLLECTION_NAME}' "
          f"({_collection.count():,} chunks)")
    return _collection


def retrieve_chunks(
    question: str,
    k: int = TOP_K,
    product_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Embed `question` and return the top-k most similar complaint chunks.

    Parameters
    ----------
    question : str
        Plain-English user question.
    k : int
        Number of chunks to retrieve.
    product_filter : str or None
        If set, restricts search to this product_category value.
        E.g. "Credit Card", "Personal Loan"

    Returns
    -------
    List[Dict[str, Any]]
        Each dict has keys:
            text, complaint_id, product_category, distance,
            chunk_index, issue, company, state

    Raises
    ------
    ValueError
        If the question is empty or no results are returned.
    """
    if not question or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    collection = _get_collection()

    # Embed the question using the same model as the parquet embeddings
    query_embedding = embed_texts([question], show_progress=False)

    # Build optional metadata filter
    where_filter = None
    if product_filter and product_filter.strip() != "All Products":
        where_filter = {"product_category": {"$eq": product_filter}}

    query_kwargs: Dict[str, Any] = {
        "query_embeddings": query_embedding.tolist(),
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        query_kwargs["where"] = where_filter

    try:
        results = collection.query(**query_kwargs)
    except Exception as e:
        raise ValueError(f"ChromaDB query failed: {e}")

    if not results["documents"] or not results["documents"][0]:
        raise ValueError(
            f"No chunks retrieved for: '{question}'\n"
            "The vector store may be empty or the product filter too restrictive."
        )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":             doc,
            "complaint_id":     meta.get("complaint_id", "N/A"),
            "product_category": meta.get("product_category", "Unknown"),
            "distance":         round(float(dist), 4),
            "chunk_index":      meta.get("chunk_index", ""),
            "issue":            meta.get("issue", ""),
            "company":          meta.get("company", ""),
            "state":            meta.get("state", ""),
        })

    return chunks