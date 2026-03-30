from pathlib import Path


def create_augmented_prompt(user_query: str, retrieved_docs: list, max_docs: int = 5) -> str:
    """
    Combines top retrieved documents and the user's query into a structured prompt.
    Returns a plain string — no LLM call happens here.
    """
    context_parts = []
    for i, doc in enumerate(retrieved_docs[:max_docs]):
        context_parts.append(f"Source [{i+1}]:\n{doc.page_content}")

    context = "\n\n".join(context_parts)

    prompt = f"""You are an expert AI Powered Vehicle Assistant. Your goal is to help users understand their vehicle manual accurately and safely.

Instructions:
1. Use ONLY the provided context below to answer the user's question.
2. If the answer is not contained in the context, say: "I am sorry, but I couldn't find information about that in your vehicle manual."
3. Cite your sources using the numbering format like [1], [2], etc. inside your answer.

Context:
{context}

User Question:
{user_query}

Answer:"""
    return prompt


def build_sources(retrieved_docs: list, max_docs: int = 5) -> list[dict]:
    """
    Returns a list of source metadata dicts for the top retrieved documents.
    Page numbers are adjusted: PyPDFLoader is 0-indexed; +2 aligns with PDF viewer labels.
    """
    sources = []
    for i, doc in enumerate(retrieved_docs[:max_docs]):
        raw_page = doc.metadata.get("page")
        page = (int(raw_page) + 1) if raw_page is not None else "?"
        sources.append({
            "index": i + 1,
            "source": Path(doc.metadata.get("source", "unknown")).name,
            "page": page,
            "excerpt": doc.page_content[:200],
        })
    return sources
