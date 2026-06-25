"""
Task 3 — Qualitative evaluation of the RAG pipeline.

Run this module directly to execute all evaluation questions and print
a Markdown table ready to paste into the final report:

    python -m src.evaluator
"""
from typing import List, Dict
from src.rag_pipeline import answer_question

# 5-10 representative questions covering all four product categories
EVALUATION_QUESTIONS = [
    "What are the most common reasons customers complain about credit cards?",
    "Why are customers dissatisfied with the billing and payment process for credit cards?",
    "What issues do customers report with personal loan repayment or interest rates?",
    "What problems do customers face when opening or managing savings accounts?",
    "What are the main complaints related to money transfers failing or being delayed?",
    "Are there complaints about companies not responding to customer disputes?",
    "What fraud or unauthorized transaction issues do customers report?",
    "How do customers describe their experience trying to resolve complaints with companies?",
    "Are there complaints about incorrect information on credit reports?",
    "What issues do customers raise about account closures or fund access?",
]


def run_evaluation(
    questions: List[str] = None,
    k: int = 5,
) -> List[Dict]:
    """
    Run each question through the RAG pipeline and collect results.

    Parameters
    ----------
    questions : List[str] or None
        Questions to evaluate. Defaults to EVALUATION_QUESTIONS.
    k : int
        Number of chunks to retrieve per question.

    Returns
    -------
    List[Dict]
        Each dict:
            'question'  : str
            'answer'    : str
            'sources'   : list of chunk dicts
            'error'     : str or None
            'score'     : int  — fill manually after review (1-5)
            'comments'  : str  — fill manually after review
    """
    if questions is None:
        questions = EVALUATION_QUESTIONS

    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Evaluating: {q[:70]}...")
        result = answer_question(q, k=k)
        result["score"] = 0        # fill manually
        result["comments"] = ""    # fill manually
        results.append(result)
        print(f"           Answer: {result['answer'][:100]}...")
        print()

    return results


def format_sources_short(sources: list, n: int = 2) -> str:
    """Return a short Markdown string showing the first n source excerpts."""
    if not sources:
        return "_No sources retrieved._"
    lines = []
    for src in sources[:n]:
        cid = src.get("complaint_id", "N/A")
        product = src.get("product_category", "?")
        text = src.get("text", "")[:120].replace("\n", " ")
        lines.append(f"**#{cid}** ({product}): _{text}..._")
    return "<br>".join(lines)


def print_evaluation_table(results: List[Dict]) -> None:
    """
    Print a Markdown evaluation table to stdout.
    Paste the output directly into your final report.

    Columns:
        Question | Generated Answer | Retrieved Sources | Score | Comments
    """
    header = (
        "| # | Question | Generated Answer | "
        "Retrieved Sources (top 2) | Quality Score (1-5) | Comments |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows = []
    for i, r in enumerate(results, 1):
        q = r["question"].replace("|", "\\|")
        a = r["answer"][:300].replace("\n", " ").replace("|", "\\|") + "..."
        sources = format_sources_short(r["sources"], n=2)
        score = r["score"] if r["score"] else "[FILL]"
        comments = r["comments"] if r["comments"] else "[FILL]"
        rows.append(f"| {i} | {q} | {a} | {sources} | {score} | {comments} |")

    print(header + "\n".join(rows))


if __name__ == "__main__":
    print("Running RAG evaluation...\n")
    results = run_evaluation()
    print("\n" + "="*80)
    print("EVALUATION TABLE (paste into report)\n")
    print_evaluation_table(results)