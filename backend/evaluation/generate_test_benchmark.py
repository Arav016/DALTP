import argparse
import json
from pathlib import Path

from backend.dataset.dataset_construction import QApair
from backend.dataset.ingestion import data_ingestion


def load_jsonl(path):
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    records = []
    with file_path.open("r", encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {file_path}:{line_number}: {exc}") from exc
    return records


def build_benchmark_from_qa_records(qa_records):
    benchmark_records = []
    for index, record in enumerate(qa_records):
        messages = record.get("messages", [])
        if len(messages) < 3:
            continue

        question = str(messages[1].get("content", "")).strip()
        reference = str(messages[2].get("content", "")).strip()
        if not question or not reference:
            continue

        benchmark_records.append(
            {
                "id": f"sample_{index}",
                "question": question,
                "reference": reference,
                "source": record.get("source", ""),
                "file_name": record.get("file_name", ""),
            }
        )

    if not benchmark_records:
        raise ValueError("No benchmark records were created from the generated QA dataset.")

    return benchmark_records


def write_jsonl(records, output_path):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file_handle:
        for record in records:
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_benchmark_from_qa_file(qa_dataset_path, benchmark_output_path):
    qa_records = load_jsonl(qa_dataset_path)
    benchmark_records = build_benchmark_from_qa_records(qa_records)
    write_jsonl(benchmark_records, benchmark_output_path)
    return benchmark_records


def maybe_ingest_documents(input_path, collection_name, ingest_to_qdrant):
    if not ingest_to_qdrant:
        return
    if not collection_name:
        raise ValueError("--collection is required when --ingest-to-qdrant is used.")

    data_ingestion.ingest_documents(
        input_path=input_path,
        collection_name=collection_name,
    )


def generate_test_benchmark(
    input_path,
    qa_dataset_path,
    benchmark_path,
    qa_generation_model="meta-llama/Llama-3.1-8B-Instruct",
    qa_api_base="http://localhost:8000/v1",
    num_pairs=10,
    chunk_size=5000,
    chunk_overlap=200,
    max_contexts_per_document=None,
    max_retries=2,
    collection=None,
    ingest_to_qdrant=False,
):
    maybe_ingest_documents(
        input_path=input_path,
        collection_name=collection,
        ingest_to_qdrant=ingest_to_qdrant,
    )

    QApair.build_qa_dataset(
        input_path=input_path,
        output_path=qa_dataset_path,
        model_name=qa_generation_model,
        api_base=qa_api_base,
        num_pairs=num_pairs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_contexts_per_document=max_contexts_per_document,
        max_retries=max_retries,
    )

    benchmark_records = build_benchmark_from_qa_file(qa_dataset_path, benchmark_path)
    return {
        "qa_dataset_path": str(Path(qa_dataset_path)),
        "benchmark_path": str(Path(benchmark_path)),
        "benchmark_size": len(benchmark_records),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a held-out QA dataset and benchmark file from new test documents."
    )
    parser.add_argument("--input", required=True, help="Path to a new held-out file or directory of files.")
    parser.add_argument("--qa-output", required=True, help="Output JSONL path for generated QA pairs.")
    parser.add_argument("--benchmark-output", required=True, help="Output JSONL path for the benchmark file.")
    parser.add_argument(
        "--qa-generation-model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="OpenAI-compatible model name used to generate held-out QA pairs.",
    )
    parser.add_argument(
        "--qa-api-base",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible endpoint used for QA generation.",
    )
    parser.add_argument("--num-pairs", type=int, default=10, help="QA pairs requested per context window.")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Context size for QA generation.")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Context overlap for QA generation.")
    parser.add_argument(
        "--max-contexts-per-document",
        type=int,
        default=None,
        help="Maximum contexts used from each document. Default uses the full document.",
    )
    parser.add_argument("--max-retries", type=int, default=2, help="Retries for malformed QA generation output.")
    parser.add_argument("--collection", help="Qdrant collection name for optional ingestion.")
    parser.add_argument(
        "--ingest-to-qdrant",
        action="store_true",
        help="Ingest the held-out documents into the provided Qdrant collection before evaluation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = generate_test_benchmark(
        input_path=args.input,
        qa_dataset_path=args.qa_output,
        benchmark_path=args.benchmark_output,
        qa_generation_model=args.qa_generation_model,
        qa_api_base=args.qa_api_base,
        num_pairs=args.num_pairs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_contexts_per_document=args.max_contexts_per_document,
        max_retries=args.max_retries,
        collection=args.collection,
        ingest_to_qdrant=args.ingest_to_qdrant,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
