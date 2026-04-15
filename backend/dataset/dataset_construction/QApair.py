import argparse
import json
import os
import re
from pathlib import Path

from openai import OpenAI

from backend.dataset.ingestion import data_ingestion as ingestion


def collect_documents(input_path):
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    documents = []
    if path.is_file():
        documents.extend(ingestion.load_file(path))
        return documents

    for file_path in sorted(path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in ingestion.SUPPORTED_EXTENSIONS:
            documents.extend(ingestion.load_file(file_path))

    return documents


def group_documents_by_source(documents): #groups pages together of a document
    grouped = {}
    for document in documents:
        source = document["source"]
        grouped.setdefault(
            source,
            {
                "source": source,
                "file_name": document["file_name"],
                "segments": [],
            },
        )
        grouped[source]["segments"].append(
            {
                "page": document["page"],
                "text": document["text"],
            }
        )

    normalized = []
    for item in grouped.values():
        ordered_segments = sorted(item["segments"], key=lambda segment: page_sort_key(segment["page"]))
        full_text = "\n\n".join(segment["text"] for segment in ordered_segments if segment["text"].strip())
        if not full_text:
            continue
        normalized.append(
            {
                "source": item["source"],
                "file_name": item["file_name"],
                "text": full_text,
            }
        )

    return normalized


def page_sort_key(page):
    if isinstance(page, int):
        return (0, page)
    return (1, str(page))


def build_qa_record(metadata, question, answer):
    return {
        "dataset_type": "qa",
        "source": metadata["source"],
        "file_name": metadata["file_name"],
        "messages": [
            {
                "role": "system",
                "content": "You are a domain assistant. Answer using only the provided source document context.",
            },
            {
                "role": "user",
                "content": question,
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ],
    }


def extract_json_array(content):
    text = (content or "").strip()  #remove whitespace from the start and end of the string
    if not text:
        return []

    try: #checks for valid JSON
        payload = json.loads(text) #parses
        return payload if isinstance(payload, list) else []
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        fenced = fence_match.group(1).strip()
        try:
            payload = json.loads(fenced)
            return payload if isinstance(payload, list) else []
        except json.JSONDecodeError:
            text = fenced

    array_match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if not array_match:
        return []

    candidate = array_match.group(0)
    repaired = re.sub(r"(\}\s*)(\{)", r"\1,\2", candidate)
    repaired = re.sub(r",\s*]", "]", repaired)
    try:
        payload = json.loads(repaired)
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def normalize_entries(entries):
    normalized = []
    seen_questions = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        question = str(entry.get("question", "")).strip()
        answer = str(entry.get("answer", "")).strip()
        if not question or not answer:
            continue

        question_key = " ".join(question.lower().split())
        if question_key in seen_questions:
            continue
        seen_questions.add(question_key)
        normalized.append({"question": question, "answer": answer})
    return normalized


def generate_qa_entries(client, model_name, document_text, num_pairs, max_retries):
    prompt = f"""Generate {num_pairs} grounded question-answer pairs based only on the document text.
Return ONLY valid JSON in this exact format:
[
  {{
    "question": "Question text",
    "answer": "Answer text"
  }}
]
Rules:
- Use only information supported by the document text.
- Do not include markdown fences.
- Do not include commentary or explanation.
- Do not include text before or after the JSON.
- Separate every object in the array with a comma.
- Write specific, useful questions. Avoid duplicates.
- Write concise, evidence-grounded answers.

Document text:
---
{document_text}
---
"""

    last_response_text = ""
    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": "You create grounded QA pairs for legal and business documents and must return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        last_response_text = response.choices[0].message.content or ""
        entries = normalize_entries(extract_json_array(last_response_text))
        if entries:
            return prompt, last_response_text, entries

        prompt = f"""Your previous response was not valid JSON or did not contain usable question-answer pairs.
Fix the response and return ONLY valid JSON in this exact format:
[
  {{
    "question": "Question text",
    "answer": "Answer text"
  }}
]
Remember:
- no markdown fences
- no commentary
- no extra text
- commas between objects

Document text:
---
{document_text}
---
"""

    return prompt, last_response_text, []


def build_qa_dataset(
    input_path,
    output_path,
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    api_base="http://localhost:8000/v1",
    num_pairs=10,
    chunk_size=5000,
    chunk_overlap=200,
    max_contexts_per_document=None,
    max_retries=2,
):
    documents = collect_documents(input_path)
    if not documents:
        raise ValueError(f"No supported files found under '{input_path}'.")

    grouped_documents = group_documents_by_source(documents)
    if not grouped_documents:
        raise ValueError("No normalized document text was created for QA generation.")

    output_file = Path(output_path).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(
        base_url=api_base,
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    )

    records_written = 0
    with output_file.open("w", encoding="utf-8") as file_handle:
        for document in grouped_documents:
            text = document["text"].strip()
            if not text:
                continue

            if len(text) <= chunk_size:
                contexts = [text]
            else:
                contexts = ingestion.split_text(text, chunk_size, chunk_overlap)
                if max_contexts_per_document is not None and max_contexts_per_document > 0:
                    contexts = contexts[:max_contexts_per_document]

            if not contexts:
                continue

            for context in contexts:
                _, _, entries = generate_qa_entries(
                    client=client,
                    model_name=model_name,
                    document_text=context,
                    num_pairs=num_pairs,
                    max_retries=max_retries,
                )

                for entry in entries:
                    record = build_qa_record(document, entry["question"], entry["answer"])
                    file_handle.write(json.dumps(record, ensure_ascii=False) + "\n") #converts python object to JSON string and writes to file
                    records_written += 1

    return {
        "documents_processed": len(grouped_documents),
        "records_written": records_written,
        "output_path": str(output_file),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a QA dataset from source documents using an OpenAI-compatible LLM endpoint."
    )
    parser.add_argument("--input", required=True, help="Path to a file or directory of source documents.")
    parser.add_argument("--output", required=True, help="Path to the merged QA JSONL output.")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model served by your OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:8000/v1",
        help="Base URL for the OpenAI-compatible endpoint.",
    )
    parser.add_argument("--num-pairs", type=int, default=10, help="Target QA pairs to request per context window.")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Context window size for each generation call.")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between context windows.")
    parser.add_argument(
        "--max-contexts-per-document",
        type=int,
        default=None,
        help="Maximum context windows to use from each normalized document. Default uses the full file.",
    )
    parser.add_argument("--max-retries", type=int, default=2, help="Retries for malformed model output.")
    return parser.parse_args()


def main():
    args = parse_args()
    result = build_qa_dataset(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        api_base=args.api_base,
        num_pairs=args.num_pairs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_contexts_per_document=args.max_contexts_per_document,
        max_retries=args.max_retries,
    )
    print(
        f"Wrote {result['records_written']} QA records from "
        f"{result['documents_processed']} normalized documents to {result['output_path']}."
    )


if __name__ == "__main__":
    main()
