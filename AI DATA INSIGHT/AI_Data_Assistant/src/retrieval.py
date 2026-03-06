"""
Retrieval Module for RAG-Based AI Data Insight Assistant.

Stores embeddings in a FAISS index and retrieves relevant text chunks
based on user query (using similarity search).
"""

from typing import List, Optional
import numpy as np


def build_faiss_index(embeddings: np.ndarray):
    """
    Build a FAISS index from embedding vectors for fast similarity search.

    Args:
        embeddings: 2D array of shape (n_chunks, embedding_dim). Must be float32.

    Returns:
        FAISS index object.
    """
    import faiss

    # FAISS expects float32
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance; lower is more similar
    index.add(embeddings)
    return index


def retrieve_chunks(
    query_embedding: np.ndarray,
    index,
    text_chunks: List[str],
    top_k: int = 5,
) -> List[str]:
    """
    Retrieve the top-k most relevant text chunks for a query embedding.

    Uses L2 distance: we return chunks with smallest distance (most similar).

    Args:
        query_embedding: 1D array of shape (embedding_dim,) or 2D (1, dim).
        index: FAISS index built from chunk embeddings.
        text_chunks: List of original text chunks (order must match index).
        top_k: Number of chunks to retrieve.

    Returns:
        List of retrieved text chunk strings.
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    if query_embedding.dtype != np.float32:
        query_embedding = query_embedding.astype(np.float32)

    top_k = min(top_k, index.ntotal)
    if top_k <= 0:
        return []

    distances, indices = index.search(query_embedding, top_k)
    retrieved = []
    for idx in indices[0]:
        if 0 <= idx < len(text_chunks):
            retrieved.append(text_chunks[idx])
    return retrieved
