import os
import openai
from dotenv import load_dotenv
from augmentation import create_augmented_prompt

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_answer(user_query: str, retrieved_docs: list) -> str:
    """
    Sends the augmented prompt to OpenAI GPT-4o and returns the generated answer.
    """
    if not retrieved_docs:
        return "I couldn't find any relevant sections in the manual for your query."

    prompt = create_augmented_prompt(user_query, retrieved_docs)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful vehicle manual assistant that always cites sources from the context provided.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error communicating with OpenAI: {str(e)}"


if __name__ == "__main__":
    from retrieval import retrieve, get_available_models, setup_hybrid_retriever
    from augmentation import build_sources
    from pathlib import Path

    print("🚗 Welcome to DriveAssist-AI!")

    models = get_available_models()
    if not models:
        print("❌ No manuals found in the database. Please run ingest.py first.")
        exit()

    print("\nPlease select your vehicle model:")
    for i, m in enumerate(models):
        print(f"[{i+1}] {m}")

    try:
        selection = int(input("\nEnter number: "))
        selected_model = models[selection - 1]
        print(f"✅ Selected: {selected_model}")
    except (ValueError, IndexError):
        print("⚠️ Invalid selection. Defaulting to all manuals.")
        selected_model = None

    retriever = setup_hybrid_retriever(selected_model)

    while True:
        print("\n" + "-" * 30)
        query = input("Ask a question about your vehicle (or type 'exit' to quit): ").strip()

        if query.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break

        if not query:
            continue

        docs = retrieve(query, retriever)
        answer = generate_answer(query, docs)
        sources = build_sources(docs)

        print("\n" + "=" * 50)
        print("FINAL AI RESPONSE:")
        print("=" * 50)
        print(answer)
        print("\n" + "-" * 50)
        print("SOURCES USED:")
        for s in sources:
            print(f"[{s['index']}] {s['source']} (Page {s['page']})")
        print("=" * 50)
