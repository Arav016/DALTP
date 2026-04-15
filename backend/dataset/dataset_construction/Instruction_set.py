import argparse
import json
import os
from pathlib import Path
from openai import OpenAI
from backend.dataset.dataset_construction import QApair as qa_builder


TASK_TEMPLATES = {
    "summarize": {
        "system": "You are a domain assistant. Use only the provided document text.",
        "user": "Summarize the key obligations, rights, and important terms in the following document text:\n\n{context}",
    },
    "extract": {
        "system": "You are a domain assistant. Extract only information supported by the provided document text.",
        "user": "Extract the key contractual details from the following document text, including parties, obligations, payment terms, dates, and termination conditions when present:\n\n{context}",
    },
    "explain": {
        "system": "You are a domain assistant. Explain document language in clear business-friendly terms using only the provided text.",
        "user": "Explain the following document text in simpler language while preserving its meaning:\n\n{context}",
    },
    "classify": {
        "system": "You are a domain assistant. Classify text using only the provided document context.",
        "user": "Classify the type of clause or document segment represented by the following text and briefly justify the classification:\n\n{context}",
    },
    "risk_spotting": {
        "system": "You are a domain assistant. Identify only risks or ambiguities that are grounded in the provided text.",
        "user": "Identify potential risks, ambiguities, or unclear obligations in the following document text:\n\n{context}",
    },
    "compare": {
        "system": "You are a domain assistant. Compare ideas only when the provided text supports the comparison.",
        "user": "Compare the major obligations, restrictions, and risk signals that appear across the following document text:\n\n{context}",
    },
}


def build_instruction_dataset(
    input_path,
    output_path,
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    api_base="http://localhost:8000/v1",
    context_size=5000,
    context_overlap=200,
    max_contexts_per_document=None,
    task_types=None,
):
    documents = qa_builder.collect_documents(input_path) #collects chunks of data
    if not documents:
        raise ValueError(f"No supported files found under '{input_path}'.")

    grouped_documents = qa_builder.group_documents_by_source(documents) #groups documents by their source
    if not grouped_documents:
        raise ValueError("No normalized document text was created for instruction generation.")

    client = OpenAI(
        base_url=api_base,
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    )

    selected_tasks = resolve_task_types(task_types)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    records_written = 0
    with output_file.open("w", encoding="utf-8") as file_handle:
        for document in grouped_documents:
            text = document["text"].strip()
            if not text:
                continue

            if len(text) <= context_size:
                contexts = [text]
            else:
                contexts = qa_builder.ingestion.split_text(text, context_size, context_overlap)
                if max_contexts_per_document is not None and max_contexts_per_document > 0:
                    contexts = contexts[:max_contexts_per_document]

            if not contexts:
                continue

            for context_index, context in enumerate(contexts):
                for task_type in selected_tasks:
                    template = TASK_TEMPLATES[task_type]
                    messages = [
                        {"role": "system", "content": template["system"]},
                        {"role": "user", "content": template["user"].format(context=context)},
                    ]
                    assistant_text = generate_instruction_response(client, model_name, messages)
                    if not assistant_text:
                        continue

                    record = {
                        "dataset_type": "instruction",
                        "task_type": task_type,
                        "source": document["source"],
                        "file_name": document["file_name"],
                        "context_index": context_index,
                        "messages": messages
                        + [
                            {
                                "role": "assistant",
                                "content": assistant_text,
                            }
                        ],
                    }
                    file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records_written += 1

    return {
        "documents_processed": len(grouped_documents),
        "records_written": records_written,
        "output_path": str(output_file),
    }


def resolve_task_types(task_types):
    if not task_types:
        return list(TASK_TEMPLATES.keys())

    unknown = [task_type for task_type in task_types if task_type not in TASK_TEMPLATES]
    if unknown:
        raise ValueError(f"Unsupported task types: {', '.join(unknown)}")
    return task_types


def generate_instruction_response(client, model_name, messages):
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.4,
    )
    content = response.choices[0].message.content
    return content.strip() if content else ""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an instruction-tuning dataset from source documents."
    )
    parser.add_argument("--input", required=True, help="Path to a file or directory of source documents.")
    parser.add_argument("--output", required=True, help="Path to the output JSONL file.")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model served by your vLLM OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:8000/v1",
        help="Base URL for the vLLM OpenAI-compatible endpoint.",
    )
    parser.add_argument("--context-size", type=int, default=5000, help="Context window size for each instruction example.")
    parser.add_argument("--context-overlap", type=int, default=200, help="Overlap between consecutive context windows.")
    parser.add_argument(
        "--max-contexts-per-document",
        type=int,
        default=None,
        help="Maximum context windows to use from each document. Default uses the full file.",
    )
    parser.add_argument(
        "--task-types",
        default="summarize,extract,explain,classify,risk_spotting,compare",
        help="Comma-separated list of task types to generate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    task_types = [task.strip() for task in args.task_types.split(",") if task.strip()]
    result = build_instruction_dataset(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        api_base=args.api_base,
        context_size=args.context_size,
        context_overlap=args.context_overlap,
        max_contexts_per_document=args.max_contexts_per_document,
        task_types=task_types,
    )
    print(
        f"Wrote {result['records_written']} instruction records from "
        f"{result['documents_processed']} normalized documents to {result['output_path']}."
    )


if __name__ == "__main__":
    main()
