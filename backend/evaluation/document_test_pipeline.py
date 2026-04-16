import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from backend.evaluation import compare_model_outputs, generate_predictions, generate_test_benchmark


@dataclass
class PredictionRunConfig:
    benchmark: str
    output: str
    system_name: str
    mode: str
    base_model: str
    adapter_path: str | None
    collection: str | None
    top_k: int
    max_new_tokens: int
    temperature: float
    dtype: str
    load_in_4bit: bool
    quantization_compute_dtype: str
    bnb_4bit_quant_type: str
    bnb_4bit_use_double_quant: bool
    trust_remote_code: bool


def run_prediction_generation(config):
    generate_predictions.validate_mode_args(config)
    benchmark_samples = generate_predictions.build_benchmark_samples(config.benchmark)
    model, tokenizer = generate_predictions.load_generation_model(config)

    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file_handle:
        for sample in benchmark_samples:
            retrieved_chunks = (
                generate_predictions.retrieve_context(sample["question"], config.collection, config.top_k)
                if config.mode in {"rag", "fine_tuned_rag"}
                else []
            )
            messages = generate_predictions.build_messages(sample["question"], retrieved_chunks)
            prediction = generate_predictions.generate_answer(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
            )
            record = generate_predictions.build_prediction_record(
                sample=sample,
                prediction=prediction,
                system_name=config.system_name,
                mode=config.mode,
                retrieved_chunks=retrieved_chunks,
            )
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path


def build_prediction_plan(args, benchmark_path, output_dir):
    modes = [
        ("base_model", "base"),
        ("rag_model", "rag"),
        ("fine_tuned_model", "fine_tuned"),
        ("fine_tuned_rag_model", "fine_tuned_rag"),
    ]

    plan = []
    for system_name, mode in modes:
        if mode in {"rag", "fine_tuned_rag"} and not args.collection:
            continue
        if mode in {"fine_tuned", "fine_tuned_rag"} and not args.adapter_path:
            continue

        plan.append(
            PredictionRunConfig(
                benchmark=str(benchmark_path),
                output=str(output_dir / f"{system_name}_predictions.jsonl"),
                system_name=system_name,
                mode=mode,
                base_model=args.base_model,
                adapter_path=args.adapter_path,
                collection=args.collection,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                dtype=args.dtype,
                load_in_4bit=args.load_in_4bit,
                quantization_compute_dtype=args.quantization_compute_dtype,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
                trust_remote_code=args.trust_remote_code,
            )
        )
    return plan


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a held-out QA benchmark from new documents, run multiple model variants, and compare their outputs."
    )
    parser.add_argument("--input", required=True, help="Path to a new held-out file or directory of files.")
    parser.add_argument("--output-dir", required=True, help="Directory where benchmark, predictions, and reports will be written.")
    parser.add_argument("--base-model", required=True, help="Base Hugging Face model name or path.")
    parser.add_argument("--qa-generation-model", default="meta-llama/Llama-3.1-8B-Instruct", help="OpenAI-compatible model name used to generate held-out QA benchmark pairs.")
    parser.add_argument("--qa-api-base", default="http://localhost:8000/v1", help="OpenAI-compatible endpoint used for QA generation.")
    parser.add_argument("--adapter-path", help="Fine-tuned LoRA adapter path for fine-tuned modes.")
    parser.add_argument("--collection", help="Qdrant collection name used for RAG retrieval.")
    parser.add_argument("--ingest-to-qdrant", action="store_true", help="Ingest the held-out documents into the provided Qdrant collection before running RAG modes.")
    parser.add_argument("--num-pairs", type=int, default=10, help="QA pairs requested per context window when building the held-out benchmark.")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Context size for held-out QA generation.")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Context overlap for held-out QA generation.")
    parser.add_argument("--max-contexts-per-document", type=int, default=None, help="Maximum contexts used from each held-out document. Default uses the full document.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries for malformed QA generation output.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks retrieved from Qdrant for RAG modes.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generated answer tokens for prediction runs.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature for prediction runs.")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16", help="Model loading dtype for prediction runs.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load the generation model in 4-bit mode.")
    parser.add_argument("--quantization-compute-dtype", choices=["bfloat16", "float16"], default="bfloat16", help="Compute dtype for 4-bit quantization.")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4", help="bitsandbytes quantization type.")
    parser.add_argument("--bnb-4bit-use-double-quant", action="store_true", help="Enable bitsandbytes double quantization.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True when loading generation models.")
    parser.add_argument("--bertscore-model", default="microsoft/deberta-xlarge-mnli", help="Encoder model used by BERTScore.")
    parser.add_argument("--fact-threshold", type=float, default=0.6, help="Fact coverage threshold.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qa_dataset_path = output_dir / "held_out_qa_dataset.jsonl"
    benchmark_path = output_dir / "benchmark.jsonl"

    benchmark_summary = generate_test_benchmark.generate_test_benchmark(
        input_path=args.input,
        qa_dataset_path=qa_dataset_path,
        benchmark_path=benchmark_path,
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

    system_prediction_files = {}
    prediction_plan = build_prediction_plan(args, benchmark_path, output_dir)
    for prediction_config in prediction_plan:
        prediction_file = run_prediction_generation(prediction_config)
        system_prediction_files[prediction_config.system_name] = prediction_file

    if not system_prediction_files:
        raise ValueError("No prediction files were generated. Check collection/adapter arguments.")

    summary = compare_model_outputs.score_prediction_files(
        benchmark_path=benchmark_path,
        prediction_files=system_prediction_files,
        output_dir=output_dir,
        bertscore_model=args.bertscore_model,
        fact_threshold=args.fact_threshold,
    )
    summary["benchmark_generation"] = benchmark_summary

    print(json.dumps(summary, indent=2))
    print(f"Wrote held-out test pipeline outputs to {output_dir}")


if __name__ == "__main__":
    main()
