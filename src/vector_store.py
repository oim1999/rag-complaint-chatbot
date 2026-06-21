import pandas as pd
import numpy as np
import chromadb
from typing import Optional
from src.config import VECTOR_STORE_DIR, CHROMA_COLLECTION_NAME


def get_chroma_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client stored at VECTOR_STORE_DIR."""
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))


def build_and_persist_vector_store(
    chunks_df: pd.DataFrame,
    embeddings: np.ndarray,
    collection_name: str = CHROMA_COLLECTION_NAME,
    batch_size: int = 500,
) -> chromadb.Collection:
    """
    Create a ChromaDB collection and index all chunk embeddings.
    Any existing collection with the same name is replaced.

    Parameters
    ----------
    chunks_df : pd.DataFrame
        Output of chunk_dataframe(). Must have columns:
        complaint_id, product_category, chunk_text, chunk_index, total_chunks.
    embeddings : np.ndarray
        Embedding array, shape (len(chunks_df), 384). Rows must align with chunks_df.
    collection_name : str
        ChromaDB collection name.
    batch_size : int
        Rows upserted per batch (reduce if memory is tight).

    Returns
    -------
    chromadb.Collection
        The populated, persisted collection.

    Raises
    ------
    ValueError
        If chunks_df is empty or row counts diverge from embeddings.
    """
    if chunks_df.empty:
        raise ValueError("Cannot build vector store: chunks_df is empty.")
    if len(chunks_df) != len(embeddings):
        raise ValueError(
            f"Row mismatch — chunks_df: {len(chunks_df)} rows, "
            f"embeddings: {len(embeddings)} rows."
        )

    client = get_chroma_client()

    # Clean rebuild: delete collection if it already exists
    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection '{collection_name}'.")
    except Exception:
        pass  # Collection did not exist yet

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    optional_meta_cols = ["issue", "sub_issue", "company", "state", "date_received"]
    total = len(chunks_df)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = chunks_df.iloc[start:end]
        batch_embs = embeddings[start:end]

        ids = [
            f"{row['complaint_id']}_chunk_{row['chunk_index']}"
            for _, row in batch.iterrows()
        ]
        documents = batch["chunk_text"].tolist()

        metadatas = []
        for _, row in batch.iterrows():
            meta = {
                "complaint_id": str(row["complaint_id"]),
                "product_category": str(row["product_category"]),
                "chunk_index": int(row["chunk_index"]),
                "total_chunks": int(row["total_chunks"]),
            }
            for col in optional_meta_cols:
                if col in row.index and pd.notna(row[col]):
                    meta[col] = str(row[col])
            metadatas.append(meta)

        collection.upsert(
            ids=ids,
            embeddings=batch_embs.tolist(),
            documents=documents,
            metadatas=metadatas,
        )
        print(f"  Indexed {end:,}/{total:,} chunks...", end="\r")

    print(f"\n✓ Vector store persisted → {VECTOR_STORE_DIR}")
    print(f"  Collection '{collection_name}' — {collection.count():,} chunks total.")
    return collection


def load_vector_store(
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> chromadb.Collection:
    """
    Load an existing persisted ChromaDB collection.

    Raises
    ------
    ValueError
        If the collection does not exist.
    """
    client = get_chroma_client()
    try:
        return client.get_collection(collection_name)
    except Exception:
        raise ValueError(
            f"Collection '{collection_name}' not found at {VECTOR_STORE_DIR}.\n"
            "Run the Task 2 notebook to build the vector store first."
        )