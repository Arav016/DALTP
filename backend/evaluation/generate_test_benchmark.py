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


def build_grouped_documents_from_corpus_records(corpus_records):
    grouped = {}
    for record in corpus_records:
        source = record.get("source") or record.get("file_name") or "corpus-source"
        grouped.setdefault(
            source,
            {
                "source": source,
                "file_name": record.get("file_name", ""),
                "segments": [],
            },
        )
        grouped[source]["segments"].append(
            {
                "page": record.get("page", ""),
                "text": str(record.get("text", "")).strip(),
            }
        )

    normalized = []
    for item in grouped.values():
        ordered_segments = sorted(item["segments"], key=lambda segment: QApair.page_sort_key(segment["page"]))
        full_text = "\n\n".join(segment["text"] for segment in ordered_segments if segment["text"])
        if not full_text:
            continue
        normalized.append(
            {
                "source": item["source"],
                "file_name": item["file_name"] or Path(item["source"]).name,
                "text": full_text,
            }
        )
    return normalized


def build_qa_dataset_from_grouped_documents(
    grouped_documents,
    qa_dataset_path,
    qa_generation_model,
    qa_api_base,
    num_pairs,
    chunk_size,
    chunk_overlap,
    max_contexts_per_document,
    max_retries,
):
    output_file = Path(qa_dataset_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    client = QApair.build_generation_client(api_base=qa_api_base)

    records_written = 0
    with output_file.open("w", encoding="utf-8") as file_handle:
        for document in grouped_documents:
            text = document["text"].strip()
            if not text:
                continue

            if len(text) <= chunk_size:
                contexts = [text]
            else:
                contexts = data_ingestion.split_text(text, chunk_size, chunk_overlap)
                if max_contexts_per_document is not None and max_contexts_per_document > 0:
                    contexts = contexts[:max_contexts_per_document]

            for context in contexts:
                _, _, entries = QApair.generate_qa_entries(
                    client=client,
                    model_name=qa_generation_model,
                    document_text=context,
                    num_pairs=num_pairs,
                    max_retries=max_retries,
                )
                for entry in entries:
                    file_handle.write(
                        json.dumps(QApair.build_qa_record(document, entry["question"], entry["answer"]), ensure_ascii=False) + "\n"
                    )
                    records_written += 1

    return {
        "documents_processed": len(grouped_documents),
        "records_written": records_written,
        "output_path": str(output_file),
    }


def maybe_ingest_documents(input_path, collection_name, ingest_to_pgvector):
    if not ingest_to_pgvector:
        return
    if not collection_name:
        raise ValueError("--collection is required when --ingest-to-pgvector is used.")

    data_ingestion.ingest_documents(
        input_path=input_path,
        collection_name=collection_name,
    )


def generate_test_benchmark(
    input_path,
    qa_dataset_path,
    benchmark_path,
    corpus_jsonl_path=None,
    qa_generation_model=None,
    qa_api_base=QApair.default_generation_api_base(),
    num_pairs=10,
    chunk_size=5000,
    chunk_overlap=200,
    max_contexts_per_document=None,
    max_retries=2,
    collection=None,
    ingest_to_pgvector=False,
):
    if qa_generation_model is None:
        qa_generation_model = QApair.default_generation_model()
    if not input_path and not corpus_jsonl_path:
        raise ValueError("Provide either input_path or corpus_jsonl_path for benchmark generation.")
    if corpus_jsonl_path:
        corpus_records = load_jsonl(corpus_jsonl_path)
        grouped_documents = build_grouped_documents_from_corpus_records(corpus_records)
        if not grouped_documents:
            raise ValueError("No usable grouped documents were reconstructed from the corpus dataset.")
        build_qa_dataset_from_grouped_documents(
            grouped_documents=grouped_documents,
            qa_dataset_path=qa_dataset_path,
            qa_generation_model=qa_generation_model,
            qa_api_base=qa_api_base,
            num_pairs=num_pairs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_contexts_per_document=max_contexts_per_document,
            max_retries=max_retries,
        )
    else:
        maybe_ingest_documents(
            input_path=input_path,
            collection_name=collection,
            ingest_to_pgvector=ingest_to_pgvector,
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
    parser.add_argument("--input", help="Path to a new held-out file or directory of files.")
    parser.add_argument("--corpus-jsonl", help="Path to a stored corpus dataset JSONL file.")
    parser.add_argument("--qa-output", required=True, help="Output JSONL path for generated QA pairs.")
    parser.add_argument("--benchmark-output", required=True, help="Output JSONL path for the benchmark file.")
    parser.add_argument(
        "--qa-generation-model",
        default=QApair.default_generation_model(),
        help="Model or deployment name used to generate held-out QA pairs.",
    )
    parser.add_argument(
        "--qa-api-base",
        default=QApair.default_generation_api_base(),
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
    parser.add_argument("--collection", help="pgvector namespace for optional ingestion.")
    parser.add_argument(
        "--ingest-to-pgvector",
        action="store_true",
        help="Ingest the held-out documents into the provided pgvector namespace before evaluation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input and not args.corpus_jsonl:
        raise ValueError("Provide either --input or --corpus-jsonl.")
    summary = generate_test_benchmark(
        input_path=args.input,
        qa_dataset_path=args.qa_output,
        benchmark_path=args.benchmark_output,
        corpus_jsonl_path=args.corpus_jsonl,
        qa_generation_model=args.qa_generation_model,
        qa_api_base=args.qa_api_base,
        num_pairs=args.num_pairs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_contexts_per_document=args.max_contexts_per_document,
        max_retries=args.max_retries,
        collection=args.collection,
        ingest_to_pgvector=args.ingest_to_pgvector,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
