import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.dataset.ingestion import data_ingestion as ingestion


SDK_SUFFIXES = ("_qa_pairs", "_cleaned", "_curated")


def collect_documents(input_path):
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    documents = []
    if path.is_file():
        documents.extend(load_supported_file(path))
        return documents

    for file_path in sorted(path.rglob("*")):
        if file_path.is_file() and is_supported(file_path):
            documents.extend(load_supported_file(file_path))

    return documents


def is_supported(file_path):
    return file_path.suffix.lower() in ingestion.SUPPORTED_EXTENSIONS or file_path.suffix.lower() == ".doc"


def load_supported_file(file_path):
    if file_path.suffix.lower() == ".doc":
        return [load_doc(file_path)]
    return ingestion.load_file(file_path)


def load_doc(file_path):
    try:
        import textract  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Legacy .doc support requires the optional 'textract' package."
        ) from exc

    text = textract.process(str(file_path)).decode("utf-8", errors="ignore").strip()
    if not text:
        raise ValueError(f"No readable text found in DOC file: {file_path}")

    return {
        "text": text,
        "source": str(file_path),
        "file_name": file_path.name,
        "page": 1,
    }


def group_documents_by_source(documents):
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


def prepare_sdk_inputs(documents, parsed_dir):
    parsed_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for document in documents:
        doc_id = build_doc_id(document["source"])
        txt_path = parsed_dir / f"{doc_id}.txt"
        txt_path.write_text(document["text"], encoding="utf-8")
        manifest[doc_id] = {
            "source": document["source"],
            "file_name": document["file_name"],
            "parsed_text_path": str(txt_path),
        }

    manifest_path = parsed_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def build_doc_id(source):
    source_path = Path(source)
    digest = hashlib.md5(source.encode("utf-8")).hexdigest()[:10]
    safe_stem = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in source_path.stem)
    return f"{safe_stem}_{digest}"


def write_sdk_config(config_path, model_name, api_base, num_pairs, chunk_size, chunk_overlap, threshold):
    config_text = f"""llm:
  provider: "vllm"

vllm:
  api_base: "{api_base}"
  model: "{model_name}"
  sleep_time: 0.2

generation:
  temperature: 0.7
  chunk_size: {chunk_size}
  chunk_overlap: {chunk_overlap}
  num_pairs: {num_pairs}
  max_context_length: 8000

curate:
  threshold: {threshold}
  batch_size: 8
"""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_text, encoding="utf-8")


def run_sdk_command(command, workdir):
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        cwd=workdir,
    )
    return completed.stdout


def build_qa_dataset(
    input_path,
    workspace_dir,
    output_path,
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    api_base="http://localhost:8000/v1",
    num_pairs=20,
    chunk_size=4000,
    chunk_overlap=100,
    curate_threshold=7.5,
    skip_curate=False,
):
    documents = collect_documents(input_path)
    if not documents:
        raise ValueError(f"No supported files found under '{input_path}'.")

    grouped_documents = group_documents_by_source(documents)
    if not grouped_documents:
        raise ValueError("No normalized document text was created for QA generation.")

    workspace = Path(workspace_dir)
    data_dir = workspace / "data"
    parsed_dir = data_dir / "parsed"
    generated_dir = data_dir / "generated"
    curated_dir = data_dir / "curated"
    output_file = Path(output_path)

    generated_dir.mkdir(parents=True, exist_ok=True)
    curated_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = prepare_sdk_inputs(grouped_documents, parsed_dir)
    config_path = workspace / "synthetic_data_kit_config.yaml"
    write_sdk_config(
        config_path=config_path,
        model_name=model_name,
        api_base=api_base,
        num_pairs=num_pairs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        threshold=curate_threshold,
    )

    run_sdk_command(
        [
            "synthetic-data-kit",
            "-c",
            str(config_path),
            "create",
            str(parsed_dir),
            "--type",
            "qa",
        ],
        workspace,
    )

    qa_source_dir = generated_dir
    if not skip_curate:
        run_sdk_command(
            [
                "synthetic-data-kit",
                "-c",
                str(config_path),
                "curate",
                str(generated_dir),
                "--threshold",
                str(curate_threshold),
            ],
            workspace,
        )
        qa_source_dir = curated_dir

    output_file.parent.mkdir(parents=True, exist_ok=True)
    records_written = merge_qa_outputs(qa_source_dir, manifest_path, output_file)

    return {
        "documents_processed": len(grouped_documents),
        "records_written": records_written,
        "workspace_dir": str(workspace),
        "output_path": str(output_file),
    }


def merge_qa_outputs(qa_source_dir, manifest_path, output_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    qa_dir = Path(qa_source_dir)
    if not qa_dir.exists():
        raise FileNotFoundError(f"Synthetic Data Kit output directory not found: {qa_dir}")

    records_written = 0
    with Path(output_path).open("w", encoding="utf-8") as file_handle:
        for qa_file in sorted(qa_dir.glob("*.json")):
            doc_id = recover_doc_id(qa_file.stem)
            metadata = manifest.get(doc_id)
            if not metadata:
                continue

            entries = load_qa_entries(qa_file)
            for entry in entries:
                question = entry.get("question", "").strip()
                answer = entry.get("answer", "").strip()
                if not question or not answer:
                    continue

                record = {
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
                file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_written += 1

    return records_written


def recover_doc_id(file_stem):
    for suffix in SDK_SUFFIXES:
        if file_stem.endswith(suffix):
            return file_stem[: -len(suffix)]
    return file_stem


def load_qa_entries(qa_file):
    payload = json.loads(qa_file.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "items", "examples", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a QA dataset from source documents using Synthetic Data Kit."
    )
    parser.add_argument("--input", required=True, help="Path to a file or directory of source documents.")
    parser.add_argument("--workspace", required=True, help="Workspace directory for parsed/generated SDK files.")
    parser.add_argument("--output", required=True, help="Path to the merged QA JSONL output.")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model served by your vLLM endpoint for Synthetic Data Kit generation.",
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:8000/v1",
        help="Base URL for the vLLM OpenAI-compatible endpoint.",
    )
    parser.add_argument("--num-pairs", type=int, default=20, help="Target QA pairs per document.")
    parser.add_argument("--chunk-size", type=int, default=4000, help="Chunk size passed to Synthetic Data Kit.")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap passed to Synthetic Data Kit.")
    parser.add_argument("--curate-threshold", type=float, default=7.5, help="Quality threshold for curation.")
    parser.add_argument(
        "--skip-curate",
        action="store_true",
        help="Skip Synthetic Data Kit curation and merge raw generated QA pairs directly.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = build_qa_dataset(
        input_path=args.input,
        workspace_dir=args.workspace,
        output_path=args.output,
        model_name=args.model,
        api_base=args.api_base,
        num_pairs=args.num_pairs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        curate_threshold=args.curate_threshold,
        skip_curate=args.skip_curate,
    )
    print(
        f"Wrote {result['records_written']} QA records from "
        f"{result['documents_processed']} normalized documents to {result['output_path']}."
    )


if __name__ == "__main__":
    main()
