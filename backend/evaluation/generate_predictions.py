import argparse
import json
import os
from pathlib import Path
from urllib import error, request

import torch
from openai import OpenAI
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


def default_openrouter_api_base():
    return os.getenv("OPENROUTER_API_BASE") or "https://openrouter.ai/api/v1"


def default_openrouter_eval_model():
    return os.getenv("OPENROUTER_MODEL") or "meta-llama/llama-3.1-8b-instruct"


def default_modal_eval_endpoint():
    return os.getenv("MODAL_EVAL_ENDPOINT") or ""


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
    if args.provider == "local_hf" and args.mode in {"fine_tuned", "fine_tuned_rag"} and not args.adapter_path:
        raise ValueError(f"--adapter-path is required for mode '{args.mode}' when using local_hf.")
    if args.provider == "modal" and args.mode not in {"fine_tuned", "fine_tuned_rag"}:
        raise ValueError("Modal evaluation is only supported for fine-tuned evaluation modes.")
    if args.provider == "modal" and args.mode in {"fine_tuned", "fine_tuned_rag"}:
        if not args.modal_endpoint:
            raise ValueError("Configure a Modal evaluation endpoint before running fine-tuned evaluation.")
        if not args.adapter_url:
            raise ValueError("Provide --adapter-url for Modal fine-tuned evaluation.")
    if args.provider == "openrouter" and not args.remote_model:
        raise ValueError("Provide a remote model name for OpenRouter evaluation.")


def retrieve_context(question, collection_name, top_k):
    query_vector = embeddings.generate_embeddings([question])[0]
    results = embed_db.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
    )

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


def generate_answer_local(model, tokenizer, messages, max_new_tokens, temperature):
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


def render_modal_prompt(messages):
    if not messages:
        return ""

    system_parts = [normalize_text(message.get("content", "")) for message in messages if message.get("role") == "system"]
    user_parts = [normalize_text(message.get("content", "")) for message in messages if message.get("role") == "user"]

    sections = []
    if system_parts:
        sections.append("System instructions:\n" + "\n\n".join(part for part in system_parts if part))
    if user_parts:
        sections.append("User request:\n" + "\n\n".join(part for part in user_parts if part))
    return "\n\n".join(section for section in sections if section).strip()


def build_benchmark_samples(path):
    samples = []
    for index, record in enumerate(load_jsonl(path)):
        sample_id = record.get("id") or record.get("sample_id") or record.get("question_id") or f"sample_{index}"
        question = normalize_text(record.get("question", ""))
        if not question:
            raise ValueError(f"Benchmark sample '{sample_id}' is missing a question.")
        samples.append({"id": sample_id, "question": question, "raw": record})
    return samples


def build_prepared_samples(args, benchmark_samples):
    prepared = []
    for sample in benchmark_samples:
        retrieved_chunks = (
            retrieve_context(sample["question"], args.collection, args.top_k)
            if args.mode in {"rag", "fine_tuned_rag"}
            else []
        )
        prepared.append(
            {
                "sample": sample,
                "retrieved_chunks": retrieved_chunks,
                "messages": build_messages(sample["question"], retrieved_chunks),
            }
        )
    return prepared


def build_openrouter_client(args):
    return OpenAI(
        base_url=args.api_base,
        api_key=args.api_key,
    )


def generate_predictions_openrouter(args, prepared_samples, output_path):
    client = build_openrouter_client(args)

    with output_path.open("w", encoding="utf-8") as file_handle:
        for prepared in prepared_samples:
            response = client.chat.completions.create(
                model=args.remote_model,
                temperature=args.temperature,
                max_tokens=args.max_new_tokens,
                messages=prepared["messages"],
            )
            prediction = (response.choices[0].message.content or "").strip()
            record = build_prediction_record(
                sample=prepared["sample"],
                prediction=prediction,
                system_name=args.system_name,
                mode=args.mode,
                retrieved_chunks=prepared["retrieved_chunks"],
            )
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_modal_payload(args, prepared_samples):
    payload = {
        "mode": args.mode,
        "system_name": args.system_name,
        "base_model": args.base_model,
        "model_id": args.model_id,
        "model_name": args.model_artifact_name,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        },
        "samples": [
            {
                "id": prepared["sample"]["id"],
                "question": prepared["sample"]["question"],
                "prompt": render_modal_prompt(prepared["messages"]),
                "retrieved_context": prepared["retrieved_chunks"],
            }
            for prepared in prepared_samples
        ],
    }
    if args.mode in {"fine_tuned", "fine_tuned_rag"}:
        payload["adapter"] = {
            "url": args.adapter_url,
            "cache_key": args.adapter_cache_key or args.model_id or args.model_artifact_name or args.adapter_url,
        }
        if args.peft_method:
            payload["peft_method"] = args.peft_method
        if args.lora_rank is not None:
            payload["lora_rank"] = args.lora_rank
    return payload


def parse_modal_predictions(response_payload):
    predictions = response_payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("Modal response did not include a valid 'predictions' list.")

    lookup = {}
    for item in predictions:
        if not isinstance(item, dict):
            continue
        sample_id = item.get("id") or item.get("sample_id")
        prediction = normalize_text(item.get("prediction") or item.get("answer") or item.get("output") or item.get("response"))
        if sample_id and prediction:
            lookup[str(sample_id)] = prediction
    if not lookup:
        raise ValueError("Modal response did not include usable predictions.")
    return lookup


def generate_predictions_modal(args, prepared_samples, output_path):
    payload = build_modal_payload(args, prepared_samples)
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    req = request.Request(args.modal_endpoint, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Modal evaluation endpoint returned HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach the Modal evaluation endpoint: {exc.reason}") from exc

    prediction_lookup = parse_modal_predictions(response_payload)
    with output_path.open("w", encoding="utf-8") as file_handle:
        for prepared in prepared_samples:
            sample_id = prepared["sample"]["id"]
            if sample_id not in prediction_lookup:
                raise ValueError(f"Modal response is missing a prediction for sample '{sample_id}'.")
            record = build_prediction_record(
                sample=prepared["sample"],
                prediction=prediction_lookup[sample_id],
                system_name=args.system_name,
                mode=args.mode,
                retrieved_chunks=prepared["retrieved_chunks"],
            )
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_predictions_local_hf(args, prepared_samples, output_path):
    model, tokenizer = load_generation_model(args)

    with output_path.open("w", encoding="utf-8") as file_handle:
        for prepared in prepared_samples:
            prediction = generate_answer_local(
                model=model,
                tokenizer=tokenizer,
                messages=prepared["messages"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            record = build_prediction_record(
                sample=prepared["sample"],
                prediction=prediction,
                system_name=args.system_name,
                mode=args.mode,
                retrieved_chunks=prepared["retrieved_chunks"],
            )
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    parser.add_argument(
        "--provider",
        choices=["local_hf", "openrouter", "modal"],
        default="local_hf",
        help="Inference provider used to produce predictions.",
    )
    parser.add_argument("--base-model", required=True, help="Base Hugging Face model name or path.")
    parser.add_argument("--remote-model", default=default_openrouter_eval_model(), help="Remote model id used for OpenRouter evaluation.")
    parser.add_argument("--api-base", default=default_openrouter_api_base(), help="OpenAI-compatible API base used for OpenRouter evaluation.")
    parser.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY") or "EMPTY", help="API key used for OpenRouter evaluation.")
    parser.add_argument("--adapter-path", help="LoRA adapter path for local fine-tuned modes.")
    parser.add_argument("--adapter-url", help="Presigned URL for the fine-tuned adapter archive used by Modal.")
    parser.add_argument("--adapter-cache-key", help="Stable cache key for the fine-tuned adapter used by Modal.")
    parser.add_argument("--collection", help="pgvector namespace used for retrieval in RAG modes.")
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
    parser.add_argument("--modal-endpoint", default=default_modal_eval_endpoint(), help="Modal HTTPS endpoint used for fine-tuned evaluation.")
    parser.add_argument("--model-id", help="Selected model registry artifact id.")
    parser.add_argument("--model-artifact-name", help="Selected model registry artifact name.")
    parser.add_argument("--peft-method", help="PEFT method metadata sent to Modal for fine-tuned runs.")
    parser.add_argument("--lora-rank", type=int, help="LoRA rank metadata sent to Modal for fine-tuned runs.")
    return parser.parse_args()


def main():
    args = parse_args()
    validate_mode_args(args)

    benchmark_samples = build_benchmark_samples(args.benchmark)
    prepared_samples = build_prepared_samples(args, benchmark_samples)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.provider == "openrouter":
        generate_predictions_openrouter(args, prepared_samples, output_path)
    elif args.provider == "modal":
        generate_predictions_modal(args, prepared_samples, output_path)
    else:
        generate_predictions_local_hf(args, prepared_samples, output_path)

    print(f"Wrote {len(benchmark_samples)} predictions to {output_path}")


if __name__ == "__main__":
    main()
