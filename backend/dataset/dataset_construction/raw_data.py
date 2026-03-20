import argparse
import json
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
INGESTION_DIR = CURRENT_DIR.parent / "ingestion"
if str(INGESTION_DIR) not in sys.path:
    sys.path.insert(0, str(INGESTION_DIR))

import ingestion.data_ingestion as ingestion


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


def build_raw_dataset(input_path, output_path, chunk_size=1000, chunk_overlap=150):
    documents = collect_documents(input_path)
    if not documents:
        raise ValueError(f"No supported files found under '{input_path}'.")

    chunks = ingestion.chunk_documents(documents, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError("No chunks were created from the input documents.")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file_handle:
        for chunk in chunks:
            record = {
                "dataset_type": "corpus",
                "source": chunk["source"],
                "file_name": chunk["file_name"],
                "page": chunk["page"],
                "text": chunk["text"],
            }
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "documents_processed": len(documents),
        "records_written": len(chunks),
        "output_path": str(output_file),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a raw corpus dataset from ingested chunk data."
    )
    parser.add_argument("--input", required=True, help="Path to a file or directory of source documents.")
    parser.add_argument("--output", required=True, help="Path to the output JSONL file.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size used for corpus records.")
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Chunk overlap used for corpus records.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = build_raw_dataset(
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(
        f"Wrote {result['records_written']} raw corpus records from "
        f"{result['documents_processed']} document units to {result['output_path']}."
    )


if __name__ == "__main__":
    main()
