import os
from typing import Dict, Any, Optional, Iterator

from huggingface_hub import InferenceClient

from src.retriever import retrieve_chunks
from src.prompt_templates import build_prompt, RAG_SYSTEM_PROMPT


LLM_MODEL_ID = os.environ.get("HF_LLM_MODEL_ID", "deepseek-ai/DeepSeek-R1:fastest")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.2"))


def _get_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            token = os.environ.get("HF_TOKEN", "").strip()
        except ImportError:
            pass

    if not token:
        raise EnvironmentError(
            "HF_TOKEN not found. Set it in your environment or .env file."
        )

    return token


def _client() -> InferenceClient:
    return InferenceClient(api_key=_get_token())


def _messages(prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _call_llm(prompt: str) -> str:
    client = _client()
    completion = client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=_messages(prompt),
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
    return completion.choices[0].message.content.strip()


def _call_llm_streaming(prompt: str) -> Iterator[str]:
    client = _client()
    stream = client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=_messages(prompt),
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        stream=True,
    )

    for chunk in stream:
        try:
            text = chunk.choices[0].delta.content
        except Exception:
            text = None
        if text:
            yield text


def answer_question(
    question: str,
    k: int = 5,
    product_filter: Optional[str] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "question": question,
        "answer": "",
        "sources": [],
        "error": None,
    }

    try:
        chunks = retrieve_chunks(question, k=k, product_filter=product_filter)
        result["sources"] = chunks
    except Exception as e:
        result["error"] = f"Retrieval failed: {e}"
        result["answer"] = (
            "I was unable to retrieve relevant complaints. "
            "Please check that the vector store is built correctly."
        )
        return result

    prompt = build_prompt(question, chunks)

    try:
        result["answer"] = _call_llm(prompt)
    except Exception as e:
        result["error"] = str(e)
        result["answer"] = f"I was unable to generate an answer. Details: {e}"

    return result


def answer_question_streaming(
    question: str,
    k: int = 5,
    product_filter: Optional[str] = None,
) -> Iterator[tuple]:
    try:
        chunks = retrieve_chunks(question, k=k, product_filter=product_filter)
    except Exception as e:
        yield (f"Retrieval failed: {e}", [])
        return

    prompt = build_prompt(question, chunks)
    accumulated = ""

    try:
        for token in _call_llm_streaming(prompt):
            accumulated += token
            yield accumulated, chunks
    except Exception as e:
        yield (f"I was unable to generate an answer. Details: {e}", chunks)
        return

    if not accumulated:
        yield "No response generated. Please try again.", chunks