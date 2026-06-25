"""
Task 4 — Interactive Chat Interface.
Tested against Gradio 6.19.0.

Prerequisites:
    1. Run notebooks/03_build_vector_store_from_parquet.ipynb to build ChromaDB
    2. Windows:  set HF_TOKEN=your_huggingface_token
       Mac/Linux: export HF_TOKEN=your_huggingface_token

Run:
    python app.py
    Open: http://localhost:7860
"""
import gradio as gr
from src.rag_pipeline import answer_question_streaming
from src.config import TARGET_PRODUCTS


# ── Format retrieved chunks for the sources panel ─────────────────────────────

def format_sources(sources: list) -> str:
    """Convert retrieved chunk dicts into readable Markdown for display."""
    if not sources:
        return "_No sources retrieved._"

    lines = ["### Retrieved Source Complaints\n"]
    for i, src in enumerate(sources, 1):
        product  = src.get("product_category", "Unknown")
        cid      = src.get("complaint_id", "N/A")
        issue    = src.get("issue", "")
        company  = src.get("company", "")
        distance = src.get("distance", "")
        text     = src.get("text", "")

        header = f"**Source {i}** — Product: {product} | Complaint ID: {cid}"
        if company:
            header += f" | Company: {company}"
        if distance != "":
            try:
                header += f" | Similarity: {1 - float(distance):.2%}"
            except (ValueError, TypeError):
                pass

        lines.append(header)
        if issue:
            lines.append(f"*Issue: {issue}*")
        lines.append(f"\n> {text}\n")
        lines.append("---")

    return "\n".join(lines)


# ── Gradio 6.19 chat handler ───────────────────────────────────────────────────
# History format in Gradio 6 is a list of dicts:
# [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

def respond(question: str, product_filter: str, history: list):
    """Stream RAG responses token by token into the chatbot."""
    if history is None:
        history = []

    question = question.strip() if question else ""

    if not question:
        history = history + [{"role": "assistant", "content": "Please enter a question."}]
        yield history, "_No sources to display._"
        return

    filter_val = None if product_filter == "All Products" else product_filter

    # Append user message and a loading placeholder for the assistant
    history = history + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": "Retrieving relevant complaints..."},
    ]
    yield history, "_Searching the complaint database..._"

    sources_md    = "_Generating answer..._"
    final_sources = []

    try:
        for partial_answer, sources in answer_question_streaming(
            question, k=5, product_filter=filter_val
        ):
            # Replace last assistant message with the growing answer
            history = history[:-1] + [{"role": "assistant", "content": partial_answer}]

            if sources and not final_sources:
                final_sources = sources
                sources_md    = format_sources(sources)

            yield history, sources_md

    except Exception as e:
        history = history[:-1] + [{"role": "assistant", "content": f"Error: {e}"}]
        yield history, "_Could not retrieve sources._"
        return

    yield history, format_sources(final_sources)


def clear_conversation():
    """Reset all UI components."""
    return (
        [],
        "",
        "All Products",
        "_Sources will appear here after you ask a question._",
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────

DESCRIPTION = """
## CrediTrust Financial — Complaint Analyzer

Ask plain-English questions about customer complaints across
**Credit Cards, Personal Loans, Savings Accounts, and Money Transfers**.

**Example questions:**
- *What are the most common credit card complaints?*
- *Why are customers unhappy with money transfers?*
- *What problems do customers have with savings accounts?*
"""

with gr.Blocks(title="CrediTrust Complaint Analyzer") as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Row():

        # ── Left column: inputs ───────────────────────────────────────────────
        with gr.Column(scale=1, min_width=280):

            product_filter = gr.Dropdown(
                choices=["All Products"] + TARGET_PRODUCTS,
                value="All Products",
                label="Filter by Product (optional)",
                info="Restrict retrieval to one product category.",
            )

            question_box = gr.Textbox(
                label="Your Question",
                placeholder="e.g. Why are customers unhappy with credit cards?",
                lines=3,
            )

            with gr.Row():
                submit_btn = gr.Button("Ask", variant="primary", scale=3)
                clear_btn  = gr.Button("Clear", variant="secondary", scale=1)

            gr.Markdown(
                "_Model: `all-MiniLM-L6-v2` retrieval + "
                "`Mistral-7B-Instruct` generation_"
            )

        # ── Right column: outputs ─────────────────────────────────────────────
        with gr.Column(scale=2):

            # Gradio 6.19 Chatbot — only use confirmed supported kwargs
            chatbot = gr.Chatbot(
                label="AI Answer",
                height=400,
                autoscroll=True,
                placeholder="Your answer will appear here...",
            )

            sources_display = gr.Markdown(
                value="_Sources will appear here after you ask a question._",
                label="Retrieved Sources",
            )

    # ── Wire events ───────────────────────────────────────────────────────────

    submit_btn.click(
        fn=respond,
        inputs=[question_box, product_filter, chatbot],
        outputs=[chatbot, sources_display],
    )

    question_box.submit(
        fn=respond,
        inputs=[question_box, product_filter, chatbot],
        outputs=[chatbot, sources_display],
    )

    clear_btn.click(
        fn=clear_conversation,
        inputs=[],
        outputs=[chatbot, question_box, product_filter, sources_display],
    )


if __name__ == "__main__":
    # In Gradio 6, theme belongs in launch() not Blocks()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue"),
    )
