from typing import List, Dict, Any
import pandas as pd
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split a string into overlapping character-level chunks.

    Parameters
    ----------
    text : str
        Input narrative text.
    chunk_size : int
        Maximum characters per chunk.
    chunk_overlap : int
        Characters of overlap between consecutive chunks.

    Returns
    -------
    List[str]
        List of non-empty text chunks. Returns [] for blank input.
    """
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start = end - chunk_overlap

    return chunks


def chunk_dataframe(
    df: pd.DataFrame,
    text_col: str = "cleaned_narrative",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> pd.DataFrame:
    """
    Explode each complaint row into one row per text chunk.

    Required columns in df:
        complaint_id, product_category, <text_col>

    Optional metadata columns (included if present):
        issue, sub_issue, company, state, date_received

    Returns
    -------
    pd.DataFrame with columns:
        complaint_id, product_category, chunk_text,
        chunk_index, total_chunks, [optional metadata cols]

    Raises
    ------
    ValueError
        If df is empty or required columns are missing, or if
        chunking produces zero records.
    """
    if df.empty:
        raise ValueError("Cannot chunk an empty DataFrame.")

    required = {"complaint_id", "product_category", text_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns for chunking: {missing}\n"
            f"Available: {list(df.columns)}"
        )

    optional_meta = ["issue", "sub_issue", "company", "state", "date_received"]
    available_meta = [c for c in optional_meta if c in df.columns]

    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        text = str(row[text_col]) if pd.notna(row[text_col]) else ""
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        total = len(chunks)

        for idx, chunk in enumerate(chunks):
            record: Dict[str, Any] = {
                "complaint_id": row["complaint_id"],
                "product_category": row["product_category"],
                "chunk_text": chunk,
                "chunk_index": idx,
                "total_chunks": total,
            }
            for col in available_meta:
                record[col] = row.get(col)
            records.append(record)

    if not records:
        raise ValueError(
            "Chunking produced zero records. "
            "Ensure cleaned_narrative column contains non-empty text."
        )

    return pd.DataFrame(records)