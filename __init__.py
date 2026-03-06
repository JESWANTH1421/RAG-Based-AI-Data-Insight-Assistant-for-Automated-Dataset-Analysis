"""
RAG-Based AI Data Insight Assistant - Source package.

Modules:
- data_processing: EDA, summary stats, correlations, text chunks
- embedding: SentenceTransformer embeddings
- retrieval: FAISS index and retrieval
- rag_pipeline: End-to-end RAG pipeline
- llm_generator: HuggingFace LLM answer generation
"""

from .data_processing import load_dataset, process_dataset, insights_to_text_chunks
from .embedding import load_embedding_model, embed_texts
from .retrieval import build_faiss_index, retrieve_chunks
from .rag_pipeline import RAGPipeline
from .llm_generator import load_llm, generate_answer

__all__ = [
    "load_dataset",
    "process_dataset",
    "insights_to_text_chunks",
    "load_embedding_model",
    "embed_texts",
    "build_faiss_index",
    "retrieve_chunks",
    "RAGPipeline",
    "load_llm",
    "generate_answer",
]
