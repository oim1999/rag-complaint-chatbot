"""
Prompt templates for the RAG generator.
All prompt strings are defined here so they can be tuned independently
of pipeline logic.
"""

RAG_SYSTEM_PROMPT = (
    "You are a financial analyst assistant for CrediTrust Financial. "
    "Your role is to help internal teams understand customer complaints "
    "across four product lines: Credit Cards, Personal Loans, "
    "Savings Accounts, and Money Transfers. "
    "Answer questions based strictly on the complaint excerpts provided. "
    "Do not use any knowledge outside of these excerpts. "
    "If the context does not contain enough information to answer the "
    "question, say clearly: 'The available complaints do not contain "
    "enough information to answer this question.'"
)

RAG_PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust Financial.
Your task is to answer questions about customer complaints.
Use ONLY the following retrieved complaint excerpts to formulate your answer.
If the context does not contain enough information to answer the question,
state clearly that you do not have enough information — do not invent details.

Context:
{context}

Question:
{question}

Answer:"""


def build_prompt(question: str, chunks: list) -> str:
    """
    Assemble the full RAG prompt from a user question and retrieved chunks.

    Parameters
    ----------
    question : str
        The user's plain-English question.
    chunks : list
        List of dicts returned by retrieve_chunks(), each with 'text',
        'product_category', and 'complaint_id' keys.

    Returns
    -------
    str
        Formatted prompt string ready to pass to the LLM.
    """
    if not chunks:
        context_text = "No relevant complaint excerpts were retrieved."
    else:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            product = chunk.get("product_category", "Unknown")
            cid = chunk.get("complaint_id", "N/A")
            text = chunk.get("text", "")
            context_parts.append(
                f"[Excerpt {i} | Product: {product} | Complaint ID: {cid}]\n{text}"
            )
        context_text = "\n\n---\n\n".join(context_parts)

    return RAG_PROMPT_TEMPLATE.format(
        context=context_text,
        question=question,
    )