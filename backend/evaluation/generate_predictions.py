import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from backend.dataset.ingestion import embed_db, embeddings


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


def build_model_kwargs(args):
    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "device_map": "auto",
    }

    if args.dtype == "bfloat16":
        model_kwargs["dtype"] = torch.bfloat16
    elif args.dtype == "float16":
        model_kwargs["dtype"] = torch.float16
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    if args.load_in_4bit:
        if args.quantization_compute_dtype == "bfloat16":
            quant_dtype = torch.bfloat16
        elif args.quantization_compute_dtype == "float16":
            quant_dtype = torch.float16
        else:
            raise ValueError(f"Unsupported quantization compute dtype: {args.quantization_compute_dtype}")

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=quant_dtype,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )

    return model_kwargs


def load_generation_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        **build_model_kwargs(args),
    )

    if args.mode in {"fine_tuned", "fine_tuned_rag"}:
        if not args.adapter_path:
            raise ValueError(f"--adapter-path is required for mode '{args.mode}'.")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()
    return model, tokenizer


def validate_mode_args(args):
    if args.mode in {"rag", "fine_tuned_rag"} and not args.collection:
        raise ValueError(f"--collection is required for mode '{args.mode}'.")
    if args.mode in {"fine_tuned", "fine_tuned_rag"} and not args.adapter_path:
        raise ValueError(f"--adapter-path is required for mode '{args.mode}'.")


def retrieve_context(question, collection_name, top_k):
    query_vector = embeddings.generate_embeddings([question])[0]
    if hasattr(embed_db.client, "search"):
        results = embed_db.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )
    elif hasattr(embed_db.client, "query_points"):
        response = embed_db.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        results = getattr(response, "points", response)
    else:
        raise AttributeError("The configured Qdrant client does not support search or query_points.")

    retrieved_chunks = []
    for result in results:
        payload = result.payload or {}
        text = normalize_text(payload.get("text", ""))
        if not text:
            continue
        retrieved_chunks.append(
            {
                "text": text,
                "source": payload.get("source", ""),
                "file_name": payload.get("file_name", ""),
                "page": payload.get("page", ""),
            }
        )
    return retrieved_chunks


def build_messages(question, retrieved_chunks):
    if retrieved_chunks:
        context_blocks = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            label = f"[Context {index}]"
            metadata = f"source={chunk['file_name']} page={chunk['page']}"
            context_blocks.append(f"{label} {metadata}\n{chunk['text']}")

        context_text = "\n\n".join(context_blocks)
        user_prompt = (
            "Answer the question using only the retrieved context below. "
            "If the answer is not supported by the context, say so.\n\n"
            f"{context_text}\n\n"
            f"Question: {question}"
        )
        system_prompt = "You are a careful domain assistant. Stay grounded in the provided context."
    else:
        user_prompt = question
        system_prompt = "You are a helpful domain assistant."

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_answer(model, tokenizer, messages, max_new_tokens, temperature):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
    else:
        generation_kwargs["do_sample"] = False

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()


def build_prediction_record(sample, prediction, system_name, mode, retrieved_chunks):
    record = {
        "id": sample["id"],
        "question": sample["question"],
        "prediction": prediction,
        "system_name": system_name,
        "mode": mode,
    }
    if retrieved_chunks:
        record["retrieved_context"] = [
            {
                "source": chunk["source"],
                "file_name": chunk["file_name"],
                "page": chunk["page"],
                "text": chunk["text"],
            }
            for chunk in retrieved_chunks
        ]
    return record


def build_benchmark_samples(path):
    samples = []
    for index, record in enumerate(load_jsonl(path)):
        sample_id = record.get("id") or record.get("sample_id") or record.get("question_id") or f"sample_{index}"
        question = normalize_text(record.get("question", ""))
        if not question:
            raise ValueError(f"Benchmark sample '{sample_id}' is missing a question.")
        samples.append({"id": sample_id, "question": question, "raw": record})
    return samples


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate prediction JSONL files for base, RAG, fine-tuned, or fine-tuned+RAG systems."
    )
    parser.add_argument("--benchmark", required=True, help="Benchmark JSONL with id/question fields.")
    parser.add_argument("--output", required=True, help="Output JSONL path for generated predictions.")
    parser.add_argument("--system-name", required=True, help="Label written into the prediction file.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["base", "rag", "fine_tuned", "fine_tuned_rag"],
        help="Prediction mode to run.",
    )
    parser.add_argument("--base-model", required=True, help="Base Hugging Face model name or path.")
    parser.add_argument("--adapter-path", help="LoRA adapter path for fine-tuned modes.")
    parser.add_argument("--collection", help="Qdrant collection name used for retrieval in RAG modes.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieved chunks to attach for RAG modes.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generated tokens per answer.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. Use 0 for greedy decoding.")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
        help="Model loading dtype.",
    )
    parser.add_argument("--load-in-4bit", action="store_true", help="Load the model in 4-bit mode.")
    parser.add_argument(
        "--quantization-compute-dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
        help="Compute dtype for 4-bit quantization.",
    )
    parser.add_argument("--bnb-4bit-quant-type", default="nf4", help="bitsandbytes 4-bit quantization type.")
    parser.add_argument("--bnb-4bit-use-double-quant", action="store_true", help="Enable bitsandbytes double quantization.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True when loading the model.")
    return parser.parse_args()


def main():
    args = parse_args()
    validate_mode_args(args)

    benchmark_samples = build_benchmark_samples(args.benchmark)
    model, tokenizer = load_generation_model(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file_handle:
        for sample in benchmark_samples:
            retrieved_chunks = (
                retrieve_context(sample["question"], args.collection, args.top_k)
                if args.mode in {"rag", "fine_tuned_rag"}
                else []
            )
            messages = build_messages(sample["question"], retrieved_chunks)
            prediction = generate_answer(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            record = build_prediction_record(
                sample=sample,
                prediction=prediction,
                system_name=args.system_name,
                mode=args.mode,
                retrieved_chunks=retrieved_chunks,
            )
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(benchmark_samples)} predictions to {output_path}")


if __name__ == "__main__":
    main()
