# RAG-Based AI Data Insight Assistant

An AI assistant that answers natural language questions about a dataset using **Data Science**, **Machine Learning**, and **Retrieval Augmented Generation (RAG)**.

## Features

- **Load CSV datasets** (default housing dataset or your own upload)
- **Exploratory Data Analysis (EDA)**: summary statistics, feature correlations, missing values
- **Text knowledge base**: EDA insights converted into text chunks for retrieval
- **RAG pipeline**: SentenceTransformer embeddings + FAISS vector store + retrieval
- **LLM answers**: HuggingFace transformer (e.g. FLAN-T5) generates answers from retrieved context
- **Streamlit UI**: upload CSV, ask questions, view AI-generated insights

## Project Structure

```
AI_Data_Assistant/
├── dataset/
│   └── housing.csv          # Default housing dataset
├── src/
│   ├── __init__.py
│   ├── data_processing.py   # EDA, stats, correlations, text chunks
│   ├── embedding.py         # SentenceTransformer embeddings
│   ├── retrieval.py         # FAISS index and retrieval
│   ├── rag_pipeline.py      # End-to-end RAG pipeline
│   └── llm_generator.py     # HuggingFace LLM generation
├── app.py                   # Streamlit application
├── requirements.txt
└── README.md
```

## Installation

1. **Clone or copy** this project and go to the project folder:

   ```bash
   cd AI_Data_Assistant
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

   The app will open in your browser (default: http://localhost:8501).

## Usage

1. **Dataset**: Use the default housing dataset or upload your own CSV in the sidebar.
2. **Ask questions** in natural language, for example:
   - *"What features affect house prices?"*
   - *"Which feature has the highest correlation with price?"*
   - *"Summarize the dataset insights."*
3. The app retrieves relevant insight chunks and uses the LLM to generate an answer. You can expand "Retrieved context" to see the text used.

## How It Works

1. **Data processing** (`data_processing.py`): Loads the CSV, computes summary statistics and correlations (e.g. with `price`), and turns these into text chunks.
2. **Embeddings** (`embedding.py`): SentenceTransformer (e.g. `all-MiniLM-L6-v2`) converts each chunk into a vector.
3. **FAISS** (`retrieval.py`): Vectors are stored in a FAISS index for fast similarity search.
4. **RAG pipeline** (`rag_pipeline.py`): Your question is embedded and the top-k closest chunks are retrieved as context.
5. **LLM** (`llm_generator.py`): A HuggingFace model (e.g. FLAN-T5) gets the question plus context and generates the final answer.

## Requirements

- Python 3.8+
- Sufficient RAM for SentenceTransformer and a small transformer model (e.g. FLAN-T5-small). GPU is optional.

## License

Use and modify as needed for learning and projects.
