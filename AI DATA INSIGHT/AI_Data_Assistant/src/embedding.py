"""
Embedding Module for RAG-Based AI Data Insight Assistant.

Converts text chunks into vector embeddings using SentenceTransformer.
These embeddings are used for semantic search in the FAISS vector database.
"""

from typing import List
import numpy as np


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load a SentenceTransformer model for generating text embeddings.

    Args:
        model_name: HuggingFace model name. 'all-MiniLM-L6-v2' is fast and lightweight.

    Returns:
        Loaded SentenceTransformer model.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model


def embed_texts(model, texts: List[str]) -> np.ndarray:
    """
    Convert a list of text strings into embedding vectors.

    Args:
        model: Loaded SentenceTransformer model.
        texts: List of text chunks to embed.

    Returns:
        2D numpy array of shape (len(texts), embedding_dim).
    """
    if not texts:
        return np.array([]).reshape(0, 384)  # MiniLM default dim
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings
