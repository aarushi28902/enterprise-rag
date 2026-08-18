"""Test RAG system on Taylor Swift questions."""
import httpx

QUESTIONS = [
    "When was Folklore released and how was it announced?",
    "What is the fictional love triangle in Folklore?",
    "What are the names of the three characters in the love triangle?",
    "Which Evermore song features Bon Iver?",
    "Why are Folklore and Evermore called sister albums?",
]


def ask(question: str) -> str:
    response = httpx.post(
        "http://localhost:8000/api/v1/query/stream",
        json={"query": question, "top_k": 4},
        timeout=60.0,
    )
    result = ""
    for line in response.text.split("\n"):
        if line.startswith("data: ") and "[DONE]" not in line and "[ERROR]" not in line:
            result += line[6:]
    return result.strip()


def main():
    for q in QUESTIONS:
        print(f"\nQ: {q}")
        print(f"A: {ask(q)}")
        print("-" * 60)


if __name__ == "__main__":
    main()
