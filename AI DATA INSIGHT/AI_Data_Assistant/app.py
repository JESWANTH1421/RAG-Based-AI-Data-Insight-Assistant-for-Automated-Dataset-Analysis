"""
Streamlit Application for RAG-Based AI Data Insight Assistant.

Allows users to:
- Upload a CSV dataset (or use default housing dataset)
- Ask natural language questions about the data
- Get AI-generated insights using RAG + LLM
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add project root so we can import from src
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.rag_pipeline import RAGPipeline
from src.llm_generator import load_llm, generate_answer


# Page config
st.set_page_config(
    page_title="RAG-Based AI Data Insight Assistant",
    page_icon="📊",
    layout="centered",
)

# Title and description
st.title("📊 RAG-Based AI Data Insight Assistant")
st.markdown(
    "Ask natural language questions about your dataset. "
    "Upload a CSV or use the default housing dataset."
)

# Default dataset path
DEFAULT_DATASET = project_root / "dataset" / "housing.csv"


def get_rag_pipeline(csv_path: str, target_column: str = "price") -> RAGPipeline:
    """Build or reuse RAG pipeline for the given CSV."""
    cache_key = f"rag_{csv_path}_{target_column}"
    if cache_key not in st.session_state:
        with st.spinner("Building RAG index from dataset insights..."):
            pipeline = RAGPipeline(target_column=target_column)
            pipeline.build_index_from_csv(csv_path)
            st.session_state[cache_key] = pipeline
    return st.session_state[cache_key]


def get_llm():
    """Load and cache the LLM."""
    if "llm_pipe" not in st.session_state:
        with st.spinner("Loading AI model (first time may take a minute)..."):
            st.session_state["llm_pipe"] = load_llm()
    return st.session_state["llm_pipe"]


# Sidebar: dataset selection
st.sidebar.header("Dataset")
use_default = st.sidebar.checkbox("Use default housing dataset", value=True)

uploaded_file = None
if not use_default:
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Upload your own dataset to ask questions about it.",
    )

if use_default and DEFAULT_DATASET.exists():
    csv_path = str(DEFAULT_DATASET)
    df = pd.read_csv(csv_path)
    st.sidebar.success(f"Using default dataset: {len(df)} rows, {len(df.columns)} columns")
elif uploaded_file is not None:
    csv_path = str(project_root / "uploaded_dataset.csv")
    with open(csv_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    df = pd.read_csv(csv_path)
    st.sidebar.success(f"Uploaded: {len(df)} rows, {len(df.columns)} columns")
else:
    csv_path = None
    df = None

# Target column (for correlation-focused EDA)
target_column = "price"
if df is not None and "price" not in df.columns:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    target_column = st.sidebar.selectbox(
        "Target column (for correlations)",
        options=numeric_cols or list(df.columns),
        index=0,
    )

# Main area: show dataset preview and Q&A
if df is not None:
    with st.expander("Preview dataset"):
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Build RAG pipeline when we have a dataset
    try:
        rag = get_rag_pipeline(csv_path, target_column=target_column)
    except Exception as e:
        st.error(f"Failed to build RAG index: {e}")
        st.stop()

    # Question input and suggested questions
    st.subheader("Ask a question")
    suggested = [
        "What features affect house prices?",
        "Which feature has the highest correlation with price?",
        "Summarize the dataset insights.",
    ]
    for q in suggested:
        if st.button(f"📌 {q}", key=q):
            st.session_state["question"] = q

    question = st.text_input(
        "Your question",
        value=st.session_state.get("question", ""),
        placeholder="e.g. What features affect house prices?",
        key="question_input",
    )

    if question:
        # Retrieve context and generate answer
        context = rag.get_context_for_query(question, top_k=5)
        llm = get_llm()

        with st.spinner("Generating answer..."):
            answer = generate_answer(llm, question, context)

        st.markdown("### Answer")
        st.write(answer)

        with st.expander("Retrieved context used for this answer"):
            st.text(context)

else:
    st.info(
        "Upload a CSV file in the sidebar or check 'Use default housing dataset' to get started."
    )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How it works:** EDA and correlations are turned into text, "
    "embedded with SentenceTransformer, and stored in FAISS. "
    "Your question is used to retrieve relevant chunks and an LLM generates the answer."
)
