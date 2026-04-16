import argparse
import json
import re
from pathlib import Path

from bert_score import score as bertscore_score
from rouge_score import rouge_scorer


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


def normalize_text(text):
    return " ".join(str(text or "").split()).strip()


def split_units(text):
    normalized = str(text or "").replace("\r", "\n")
    lines = [line.strip(" -*\t") for line in normalized.splitlines()]
    bullet_units = [line for line in lines if line]
    if len(bullet_units) > 1:
        return bullet_units

    sentence_units = [
        unit.strip()
        for unit in re.split(r"(?<=[.!?])\s+|\n+", normalized)
        if unit.strip()
    ]
    return sentence_units or ([normalize_text(text)] if normalize_text(text) else [])


def fact_coverage_score(reference, prediction, scorer, match_threshold):
    ref_units = split_units(reference)
    pred_units = split_units(prediction)
    if not ref_units or not pred_units:
        return 0.0

    covered = 0
    prediction_text = normalize_text(prediction)
    for ref_unit in ref_units:
        best = 0.0
        for pred_unit in pred_units:
            rouge = scorer.score(ref_unit, pred_unit)["rougeL"]
            best = max(best, rouge.recall, rouge.fmeasure)

        if best < match_threshold and prediction_text:
            rouge = scorer.score(ref_unit, prediction_text)["rougeL"]
            best = max(best, rouge.recall, rouge.fmeasure)

        if best >= match_threshold:
            covered += 1

    return covered / len(ref_units)


def build_benchmark_map(records):
    benchmark = {}
    for index, record in enumerate(records):
        sample_id = record.get("id") or record.get("sample_id") or record.get("question_id") or f"sample_{index}"
        question = normalize_text(record.get("question", ""))
        reference = normalize_text(
            record.get("reference")
            or record.get("reference_answer")
            or record.get("ground_truth")
            or record.get("answer")
        )
        if not reference:
            raise ValueError(f"Benchmark sample '{sample_id}' is missing a reference answer.")
        benchmark[sample_id] = {
            "id": sample_id,
            "question": question,
            "reference": reference,
            "metadata": {
                key: value
                for key, value in record.items()
                if key not in {"id", "sample_id", "question_id", "question", "reference", "reference_answer", "ground_truth", "answer"}
            },
        }
    return benchmark


def build_prediction_map(records):
    predictions = {}
    for index, record in enumerate(records):
        sample_id = record.get("id") or record.get("sample_id") or record.get("question_id") or f"sample_{index}"
        prediction = normalize_text(
            record.get("prediction")
            or record.get("response")
            or record.get("answer")
            or record.get("output")
        )
        predictions[sample_id] = prediction
    return predictions


def evaluate_system(system_name, prediction_map, benchmark_map, bertscore_model, fact_threshold):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    ordered_ids = [sample_id for sample_id in benchmark_map if sample_id in prediction_map]
    if not ordered_ids:
        raise ValueError(f"No overlapping sample ids found for system '{system_name}'.")

    references = [benchmark_map[sample_id]["reference"] for sample_id in ordered_ids]
    predictions = [prediction_map[sample_id] for sample_id in ordered_ids]

    _, _, bert_f1 = bertscore_score(
        predictions,
        references,
        lang="en",
        model_type=bertscore_model,
        verbose=False,
    )

    examples = []
    metric_totals = {
        "rouge_l": 0.0,
        "bertscore_f1": 0.0,
        "fact_coverage": 0.0,
    }

    for index, sample_id in enumerate(ordered_ids):
        reference = benchmark_map[sample_id]["reference"]
        prediction = prediction_map[sample_id]
        rouge_l = scorer.score(reference, prediction)["rougeL"].fmeasure
        fact_coverage = fact_coverage_score(reference, prediction, scorer, fact_threshold)
        bert_f1_value = float(bert_f1[index])

        metric_totals["rouge_l"] += rouge_l
        metric_totals["bertscore_f1"] += bert_f1_value
        metric_totals["fact_coverage"] += fact_coverage

        examples.append(
            {
                "id": sample_id,
                "question": benchmark_map[sample_id]["question"],
                "reference": reference,
                "prediction": prediction,
                "metrics": {
                    "ROUGE-L": round(rouge_l, 4),
                    "BERTScore F1": round(bert_f1_value, 4),
                    "Fact Coverage": round(fact_coverage, 4),
                },
            }
        )

    count = len(ordered_ids)
    summary = {
        "samples_scored": count,
        "ROUGE-L": round(metric_totals["rouge_l"] / count, 4),
        "BERTScore F1": round(metric_totals["bertscore_f1"] / count, 4),
        "Fact Coverage": round(metric_totals["fact_coverage"] / count, 4),
    }

    return {
        "system_name": system_name,
        "summary": summary,
        "examples": examples,
    }


def score_prediction_files(
    benchmark_path,
    prediction_files,
    output_dir,
    bertscore_model,
    fact_threshold,
):
    benchmark_map = build_benchmark_map(load_jsonl(benchmark_path))

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    summary = {}
    per_system_results = {}
    for system_name, prediction_path in prediction_files.items():
        prediction_map = build_prediction_map(load_jsonl(prediction_path))
        result = evaluate_system(
            system_name=system_name,
            prediction_map=prediction_map,
            benchmark_map=benchmark_map,
            bertscore_model=bertscore_model,
            fact_threshold=fact_threshold,
        )
        per_system_results[system_name] = result
        summary[system_name] = result["summary"]

        per_example_path = output_directory / f"{system_name}_per_example.jsonl"
        with per_example_path.open("w", encoding="utf-8") as file_handle:
            for example in result["examples"]:
                file_handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    combined_output_path = output_directory / "combined_model_outputs.jsonl"
    with combined_output_path.open("w", encoding="utf-8") as file_handle:
        for sample_id, benchmark in benchmark_map.items():
            record = {
                "id": sample_id,
                "question": benchmark["question"],
                "actual_answer": benchmark["reference"],
                "model_outputs": {},
            }
            for system_name, result in per_system_results.items():
                example_lookup = {example["id"]: example for example in result["examples"]}
                example = example_lookup.get(sample_id)
                if not example:
                    continue
                record["model_outputs"][system_name] = {
                    "prediction": example["prediction"],
                    "metrics": example["metrics"],
                }
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_payload = {
        "benchmark_size": len(benchmark_map),
        "systems": summary,
        "prediction_files": {system_name: str(path) for system_name, path in prediction_files.items()},
        "combined_output_file": str(combined_output_path),
    }
    summary_path = output_directory / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_payload


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare model prediction files against a benchmark and write per-example and combined reports."
    )
    parser.add_argument("--benchmark", required=True, help="JSONL benchmark file with id/question/reference fields.")
    parser.add_argument("--base-predictions", help="JSONL predictions from the base model.")
    parser.add_argument("--rag-predictions", help="JSONL predictions from the RAG model.")
    parser.add_argument("--fine-tuned-predictions", help="JSONL predictions from the fine-tuned model.")
    parser.add_argument("--fine-tuned-rag-predictions", help="JSONL predictions from the fine-tuned + RAG model.")
    parser.add_argument("--output-dir", required=True, help="Directory where evaluation files will be written.")
    parser.add_argument(
        "--bertscore-model",
        default="microsoft/deberta-xlarge-mnli",
        help="Encoder model used by BERTScore.",
    )
    parser.add_argument(
        "--fact-threshold",
        type=float,
        default=0.6,
        help="Minimum overlap threshold used when deciding whether a reference fact is covered.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    prediction_files = {
        "base_model": args.base_predictions,
        "rag_model": args.rag_predictions,
        "fine_tuned_model": args.fine_tuned_predictions,
        "fine_tuned_rag_model": args.fine_tuned_rag_predictions,
    }
    prediction_files = {name: path for name, path in prediction_files.items() if path}
    if not prediction_files:
        raise ValueError("Provide at least one prediction file to score.")

    summary = score_prediction_files(
        benchmark_path=args.benchmark,
        prediction_files=prediction_files,
        output_dir=args.output_dir,
        bertscore_model=args.bertscore_model,
        fact_threshold=args.fact_threshold,
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote comparison outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
