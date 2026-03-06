"""
RAG Pipeline for AI Data Insight Assistant.

Orchestrates: data processing -> text chunks -> embeddings -> FAISS index ->
query embedding -> retrieval -> context for LLM.
"""

from typing import List, Optional, Any
import numpy as np

# Local modules
from .data_processing import process_dataset, insights_to_text_chunks, load_dataset
from .embedding import load_embedding_model, embed_texts
from .retrieval import build_faiss_index, retrieve_chunks


class RAGPipeline:
    """
    End-to-end RAG pipeline: build index from dataset insights and retrieve
    relevant chunks for user questions.
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        target_column: Optional[str] = "price",
    ):
        """
        Initialize the RAG pipeline (load embedding model).

        Args:
            embedding_model_name: SentenceTransformer model name.
            target_column: Target column for correlations in EDA.
        """
        self.embedding_model_name = embedding_model_name
        self.target_column = target_column
        self.model = None
        self.text_chunks = []
        self.index = None
        self._embedding_dim = None

    def load_model(self):
        """Load the SentenceTransformer model (lazy load)."""
        if self.model is None:
            self.model = load_embedding_model(self.embedding_model_name)
            # Get dimension from a dummy encode
            self._embedding_dim = self.model.get_sentence_embedding_dimension()

    def build_index_from_csv(self, csv_path: str) -> List[str]:
        """
        Load CSV, generate insight chunks, embed them, and build FAISS index.

        Args:
            csv_path: Path to the housing (or any) CSV dataset.

        Returns:
            List of text chunks that were indexed.
        """
        self.load_model()
        _, self.text_chunks = process_dataset(csv_path, target_column=self.target_column)
        if not self.text_chunks:
            return []

        # Embed chunks and build FAISS index
        embeddings = embed_texts(self.model, self.text_chunks)
        self.index = build_faiss_index(embeddings)
        return self.text_chunks

    def build_index_from_chunks(self, text_chunks: List[str]) -> None:
        """
        Build FAISS index from precomputed text chunks (e.g., after custom EDA).

        Args:
            text_chunks: List of text strings to index.
        """
        self.load_model()
        self.text_chunks = text_chunks
        if not self.text_chunks:
            self.index = None
            return
        embeddings = embed_texts(self.model, self.text_chunks)
        self.index = build_faiss_index(embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve the top-k most relevant chunks for a natural language query.

        Args:
            query: User question in natural language.
            top_k: Number of chunks to return.

        Returns:
            List of relevant text chunks.
        """
        if self.model is None or self.index is None or not self.text_chunks:
            return []

        query_embedding = embed_texts(self.model, [query])
        return retrieve_chunks(
            query_embedding[0],
            self.index,
            self.text_chunks,
            top_k=top_k,
        )

    def get_context_for_query(self, query: str, top_k: int = 5) -> str:
        """
        Get a single context string (concatenated retrieved chunks) for the LLM.

        Args:
            query: User question.
            top_k: Number of chunks to retrieve.

        Returns:
            Concatenated context string.
        """
        chunks = self.retrieve(query, top_k=top_k)
        return "\n\n".join(chunks) if chunks else ""
