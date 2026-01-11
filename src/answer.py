import json
from pathlib import Path
from retrieve import EmbeddingRetriever
from gigachat import GigaChat
import os
from dotenv import load_dotenv

load_dotenv()


def build_prompt(question_text, kind, context):
    format_rules = {
        "number": "'N/A' OR ONLY one number (no commas, no separators, no percent sign, ONLY digits (0-9), example: 45000.6)",
        "name": "ONLY the name",
        "names": "'N/A' OR a list of names, separated by commas",
        "boolean": "ONLY true or false",
    }

    missing_info_rules = {
        "boolean": "If the information is not explicitly stated in the context, return false.",
        "number": "If the information is not explicitly stated in the context, return 'N/A'.",
        "names": "If the information is not explicitly stated in the context, return 'N/A'.",
        "name": (
            "Return name of the product or a company."
            "Exclude companies with missing data from the comparison. "
            "If only one company remains, return its name. "
            "Do NOT return 'N/A'."
        ),
    }

    return f"""
You are an assistant that answers questions ONLY using the provided context.
Do NOT use any external knowledge.
Do NOT add explanations or extra text.

Context:
{context}

Question:
{question_text}

Rules:
- Follow the answer format.
- {missing_info_rules[kind]}

Answer format:
- {format_rules[kind]}

Answer:
"""


def build_context(chunks, max_chars=3500):
    context_parts = []
    total = 0

    for ch in chunks:
        part = f"Source page {ch['page']}:\n{ch['text']}\n\n"

        if total + len(part) > max_chars:
            break
        context_parts.append(part)
        total += len(part)

    return "\n".join(context_parts)


def normalize_answer(raw_answer: str, kind: str):
    if raw_answer.strip().lower() in ("n/a", "na"):
        return "N/A"

    if kind == "boolean":
        return "true" if raw_answer.strip().lower() in ("yes", "true") else "false"

    if kind == "number":
        return raw_answer.replace(",", "").replace("%", "")

    return raw_answer.strip()


def build_references(chunks):
    refs = []
    seen = set()

    for ch in chunks:
        key = (ch["file"], ch["page"])
        if key in seen:
            continue
        seen.add(key)

        refs.append(
            {"pdf_sha1": ch["file"].replace(".json", ""), "page_index": ch["page"] - 1}
        )

    return refs


def run_rag(
    output_path_str: str = "submission_Shagimardanova_v0.json",
    SUBMISSION_NAME: str = "Shagimardanova_v0",
):
    questions_path = Path("data/questions.json")
    output_path = Path(output_path_str)
    TEAM_EMAIL = "st119018@student.spbu.ru"

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    retriever = EmbeddingRetriever()
    answers = []

    gigachat = GigaChat(
        credentials=os.environ["GIGACHAT_TOKEN"],
        model="GigaChat",
        verify_ssl_certs=False,
    )

    for q in questions:
        question_text = q["text"]
        kind = q["kind"]
        chunks = retriever.retrieve(question_text)

        context = build_context(chunks)
        prompt = build_prompt(question_text, kind, context)
        raw_answer = gigachat.chat(prompt).choices[0].message.content.strip()
        value = normalize_answer(raw_answer, kind)
        references = build_references(chunks)

        answers.append(
            {"question_text": question_text, "value": value, "references": references}
        )

    submission = {
        "team_email": TEAM_EMAIL,
        "submission_name": SUBMISSION_NAME,
        "answers": answers,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

    print(f"Submission saved to {output_path}")
