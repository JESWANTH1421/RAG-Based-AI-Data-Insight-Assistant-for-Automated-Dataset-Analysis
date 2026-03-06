"""
LLM Response Generation for RAG-Based AI Data Insight Assistant.

Uses a HuggingFace transformer model to generate answers by combining
retrieved RAG context with the user question.
"""

from typing import Optional, Dict, Any


# Default: small model that runs on CPU without too much memory
DEFAULT_MODEL_NAME = "google/flan-t5-small"


def load_llm(model_name: str = DEFAULT_MODEL_NAME) -> Dict[str, Any]:
    """
    Load a HuggingFace Seq2Seq model (e.g., FLAN-T5) for text generation.

    Note: Transformers v5 removed/changed some pipeline task names (like
    'text2text-generation'). To keep compatibility, we load the model and call
    `generate()` directly.

    Args:
        model_name: HuggingFace model id.

    Returns:
        Dict with 'model', 'tokenizer', and 'device'.
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    return {"model": model, "tokenizer": tokenizer, "device": device}


def generate_answer(
    llm: Dict[str, Any],
    question: str,
    context: str,
    model_name: Optional[str] = None,
    max_new_tokens: int = 150,
) -> str:
    """
    Generate an AI answer by feeding the model the question plus retrieved context.

    Args:
        llm: Dict returned by `load_llm()` containing model/tokenizer/device.
        question: User's natural language question.
        context: Retrieved RAG context (concatenated chunks).
        model_name: Unused (kept for API compatibility).
        max_new_tokens: Maximum length of generated answer.

    Returns:
        Generated answer string.
    """
    # Build prompt: instruct the model to answer based on context
    prompt = (
        f"Based on the following dataset insights, answer the question. "
        f"If the answer is not in the context, say so briefly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    # Keep prompt within reasonable length (models often have 512 token limit)
    if len(prompt) > 2800:
        prompt = prompt[:2800] + "\n\nAnswer:"

    try:
        import torch

        model = llm["model"]
        tokenizer = llm["tokenizer"]
        device = llm["device"]

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=2,
                do_sample=False,
                early_stopping=True,
            )

        text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return text if text else "I could not generate an answer. Please try rephrasing your question."
    except Exception as e:
        return f"I could not generate an answer due to an error: {e}"
