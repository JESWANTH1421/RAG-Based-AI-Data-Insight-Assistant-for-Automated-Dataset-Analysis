"""
Data Processing Module for RAG-Based AI Data Insight Assistant.

This module performs exploratory data analysis (EDA), generates summary statistics,
calculates feature correlations, and converts these insights into text knowledge
for use in the RAG pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load a CSV dataset using pandas.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        DataFrame containing the loaded dataset.
    """
    df = pd.read_csv(csv_path)
    return df


def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for the dataset (count, mean, std, min, quartiles, max).

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame of summary statistics.
    """
    return df.describe()


def get_feature_correlations(df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate Pearson correlation matrix for numeric columns.
    If target_column is provided, returns correlations with that column only.

    Args:
        df: Input DataFrame.
        target_column: Optional column name to correlate against (e.g., 'price').

    Returns:
        Correlation matrix or series.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    if target_column and target_column in corr_matrix.columns:
        return corr_matrix[target_column].sort_values(ascending=False)
    return corr_matrix


def get_missing_values_summary(df: pd.DataFrame) -> pd.Series:
    """
    Summarize missing values per column.

    Args:
        df: Input DataFrame.

    Returns:
        Series with missing counts per column.
    """
    return df.isnull().sum()


def get_column_info(df: pd.DataFrame) -> List[str]:
    """
    Generate human-readable column information (dtypes, non-null counts).

    Args:
        df: Input DataFrame.

    Returns:
        List of text lines describing columns.
    """
    lines = []
    lines.append(f"Dataset has {len(df)} rows and {len(df.columns)} columns.")
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].notna().sum()
        lines.append(f"Column '{col}': type {dtype}, non-null count {non_null}.")
    return lines


def insights_to_text_chunks(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    max_chunk_chars: int = 500,
) -> List[str]:
    """
    Convert dataset insights (EDA, summary stats, correlations) into text chunks
    suitable for RAG retrieval. Each chunk is a self-contained piece of knowledge.

    Args:
        df: Input DataFrame.
        target_column: Optional target column (e.g., 'price') for correlation focus.
        max_chunk_chars: Approximate maximum characters per chunk.

    Returns:
        List of text chunks.
    """
    chunks = []

    # 1. Dataset overview
    overview = (
        f"The dataset contains {len(df)} records and {len(df.columns)} features: "
        f"{', '.join(df.columns)}. "
    )
    chunks.append(overview)

    # 2. Column info
    col_info = get_column_info(df)
    chunks.append(" ".join(col_info))

    # 3. Summary statistics as text (key stats for numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        stats_df = get_summary_statistics(df)
        stats_lines = ["Summary statistics (numeric columns):"]
        for col in numeric_cols[:10]:  # limit to avoid huge chunks
            if col in stats_df.columns:
                mean_val = stats_df[col].get("mean", np.nan)
                std_val = stats_df[col].get("std", np.nan)
                min_val = stats_df[col].get("min", np.nan)
                max_val = stats_df[col].get("max", np.nan)
                stats_lines.append(
                    f"{col}: mean={mean_val:.2f}, std={std_val:.2f}, min={min_val:.2f}, max={max_val:.2f}"
                )
        chunks.append(" ".join(stats_lines))

    # 4. Correlation with target (e.g., price)
    if target_column and target_column in df.columns:
        corr_series = get_feature_correlations(df, target_column=target_column)
        corr_lines = [f"Correlation with {target_column}:"]
        for feat, val in corr_series.items():
            if feat != target_column and not np.isnan(val):
                corr_lines.append(f"{feat}: {val:.3f}")
        chunks.append(" ".join(corr_lines))

    # 5. Full correlation matrix summary (top correlations with target)
    if target_column and target_column in df.columns:
        corr_series = get_feature_correlations(df, target_column=target_column)
        sorted_corr = (
            corr_series.drop(target_column, errors="ignore")
            .sort_values(key=lambda x: x.abs(), ascending=False)
        )
        top_positive = sorted_corr.head(3)
        top_neg = sorted_corr.tail(3)
        chunk_pos = (
            f"Features most positively correlated with {target_column}: "
            + ", ".join([f"{k}({v:.3f})" for k, v in top_positive.items()])
        )
        chunk_neg = (
            f"Features most negatively correlated with {target_column}: "
            + ", ".join([f"{k}({v:.3f})" for k, v in top_neg.items()])
        )
        chunks.append(chunk_pos)
        chunks.append(chunk_neg)

    # 6. Missing values summary
    missing = get_missing_values_summary(df)
    if missing.sum() > 0:
        missing_lines = ["Missing values: "] + [
            f"{col}: {cnt}" for col, cnt in missing.items() if cnt > 0
        ]
        chunks.append(" ".join(missing_lines))
    else:
        chunks.append("There are no missing values in the dataset.")

    # 7. Dataset insights summary (for questions like "summarize the dataset")
    insight_summary = (
        f"Dataset insight summary: The housing dataset has numeric features including "
        f"{', '.join(numeric_cols)}. "
        f"Key statistics and correlations have been computed. "
        f"Use correlation with price to understand which features affect house prices."
    )
    chunks.append(insight_summary)

    # Split any chunk that exceeds max_chunk_chars into smaller pieces
    final_chunks = []
    for c in chunks:
        if len(c) <= max_chunk_chars:
            final_chunks.append(c)
        else:
            # Simple split by sentence or by length
            start = 0
            while start < len(c):
                end = min(start + max_chunk_chars, len(c))
                if end < len(c):
                    # Try to break at sentence or space
                    break_at = c.rfind(". ", start, end + 1)
                    if break_at > start:
                        end = break_at + 1
                    else:
                        break_at = c.rfind(" ", start, end + 1)
                        if break_at > start:
                            end = break_at + 1
                final_chunks.append(c[start:end].strip())
                start = end

    return [ch for ch in final_chunks if ch]


def process_dataset(
    csv_path: str,
    target_column: Optional[str] = "price",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Full pipeline: load CSV, run EDA, and produce text chunks for RAG.

    Args:
        csv_path: Path to the CSV file.
        target_column: Column to use as target for correlations (e.g., 'price').

    Returns:
        Tuple of (DataFrame, list of text chunks).
    """
    df = load_dataset(csv_path)
    chunks = insights_to_text_chunks(df, target_column=target_column)
    return df, chunks
