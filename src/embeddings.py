from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME

_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    """
    Return the singleton SentenceTransformer, loading it on first call.
    Model: all-MiniLM-L6-v2 (384-dim, ~80 MB).
    """
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Model loaded.")
    return _model


def embed_texts(
    texts: List[str],
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode a list of strings into L2-normalised embedding vectors.

    Parameters
    ----------
    texts : List[str]
        Strings to encode.
    batch_size : int
        Texts per encoding batch (tune down if you run out of RAM).
    show_progress : bool
        Display a tqdm progress bar during encoding.

    Returns
    -------
    np.ndarray
        Shape (len(texts), 384), dtype float32.
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings