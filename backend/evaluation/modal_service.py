from __future__ import annotations

import io
import os
from pathlib import Path
import tempfile
import threading
import zipfile
from typing import Any

import modal
import requests
import torch
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


APP_NAME = "daltp-evaluation-service"
HF_CACHE_DIR = "/cache/huggingface"
MODAL_EVAL_GPU = "T4"

app = modal.App(APP_NAME)
hf_cache_volume = modal.Volume.from_name("daltp-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("daltp-llama")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi>=0.115.0,<1",
        "pydantic>=2,<3",
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "bitsandbytes",
        "requests",
        "sentencepiece",
    )
)
web_app = FastAPI(title="DALTP Modal Evaluation Service", version="0.1.0")

_MODEL_CACHE: dict[tuple[str, str | None, str | None], tuple[Any, Any]] = {}
_ADAPTER_CACHE: dict[str, str] = {}
_CACHE_LOCK = threading.RLock()


class AdapterRef(BaseModel):
    url: str = Field(min_length=10)
    cache_key: str | None = None


class GenerationConfig(BaseModel):
    max_new_tokens: int = 256
    temperature: float = 0.0


class EvalSample(BaseModel):
    id: str = Field(min_length=1)
    prompt: str = ""
    question: str | None = None
    messages: list[dict[str, Any]] | None = None
    retrieved_context: list[dict[str, Any]] | None = None


class EvalRequest(BaseModel):
    mode: str = Field(pattern="^(fine_tuned|fine_tuned_rag)$")
    system_name: str | None = None
    base_model: str = Field(min_length=2)
    model_id: str | None = None
    model_name: str | None = None
    adapter: AdapterRef | None = None
    peft_method: str | None = None
    lora_rank: int | None = None
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    samples: list[EvalSample] = Field(min_length=1)


def huggingface_token() -> str | None:
    token = os.getenv("HF_TOKEN") or ""
    return token or None


def load_in_4bit_enabled() -> bool:
    return os.getenv("MODAL_LOAD_IN_4BIT", "1").strip().lower() not in {"0", "false", "no"}


def model_dtype() -> torch.dtype:
    dtype_name = (os.getenv("MODAL_MODEL_DTYPE") or "bfloat16").strip().lower()
    if dtype_name == "float16":
        return torch.float16
    return torch.bfloat16


def quantization_compute_dtype() -> torch.dtype:
    dtype_name = (os.getenv("MODAL_QUANT_DTYPE") or "bfloat16").strip().lower()
    if dtype_name == "float16":
        return torch.float16
    return torch.bfloat16


def build_hf_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "cache_dir": os.getenv("HF_HOME", HF_CACHE_DIR),
        "trust_remote_code": False,
    }
    token = huggingface_token()
    if token:
        kwargs["token"] = token
    return kwargs


def build_model_kwargs() -> dict[str, Any]:
    kwargs = build_hf_kwargs()
    kwargs["device_map"] = "auto"
    kwargs["torch_dtype"] = model_dtype()
    if load_in_4bit_enabled():
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=os.getenv("MODAL_BNB_4BIT_QUANT_TYPE") or "nf4",
            bnb_4bit_compute_dtype=quantization_compute_dtype(),
            bnb_4bit_use_double_quant=(os.getenv("MODAL_BNB_DOUBLE_QUANT", "1").strip().lower() not in {"0", "false", "no"}),
        )
    return kwargs


def cache_key(base_model: str, adapter: AdapterRef | None) -> tuple[str, str | None]:
    if not adapter:
        return (base_model, None)
    return (base_model, adapter.cache_key or adapter.url)


def ensure_adapter_dir(adapter: AdapterRef) -> str:
    cache_key_value = adapter.cache_key or adapter.url
    with _CACHE_LOCK:
        cached_path = _ADAPTER_CACHE.get(cache_key_value)
        if cached_path and Path(cached_path).exists():
            return cached_path

        response = requests.get(adapter.url, timeout=120)
        response.raise_for_status()
        payload = response.content

        target_dir = tempfile.mkdtemp(prefix="daltp_adapter_")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(target_dir)

        adapter_dir = find_adapter_dir(Path(target_dir))
        _ADAPTER_CACHE[cache_key_value] = str(adapter_dir)
        return str(adapter_dir)


def find_adapter_dir(extracted_root: Path) -> Path:
    direct_config = extracted_root / "adapter_config.json"
    if direct_config.exists():
        return extracted_root

    matches = sorted(
        path.parent
        for path in extracted_root.rglob("adapter_config.json")
        if "__MACOSX" not in path.parts
    )
    if not matches:
        raise FileNotFoundError(
            "The adapter archive does not contain adapter_config.json. "
            "Upload the final LoRA adapter output folder, not checkpoint folders or the full training project."
        )
    return matches[0]


def load_runtime(base_model: str, adapter: AdapterRef | None):
    key = cache_key(base_model, adapter)
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, **build_hf_kwargs())
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(base_model, **build_model_kwargs())
        if adapter:
            adapter_dir = ensure_adapter_dir(adapter)
            model = PeftModel.from_pretrained(model, adapter_dir)

        model.eval()
        _MODEL_CACHE[key] = (model, tokenizer)
        return model, tokenizer


def build_input_prompt(sample: EvalSample) -> str:
    if sample.prompt.strip():
        return sample.prompt.strip()
    messages = sample.messages or []
    if not messages:
        raise ValueError(f"Sample '{sample.id}' is missing a prompt.")

    system_parts = [str(message.get("content", "")).strip() for message in messages if message.get("role") == "system"]
    user_parts = [str(message.get("content", "")).strip() for message in messages if message.get("role") == "user"]

    sections = []
    if system_parts:
        sections.append("System instructions:\n" + "\n\n".join(part for part in system_parts if part))
    if user_parts:
        sections.append("User request:\n" + "\n\n".join(part for part in user_parts if part))
    prompt = "\n\n".join(section for section in sections if section).strip()
    if not prompt:
        raise ValueError(f"Sample '{sample.id}' is missing usable prompt content.")
    return prompt


def generate_prediction_for_sample(model, tokenizer, sample: EvalSample, generation: GenerationConfig) -> str:
    prompt = build_input_prompt(sample)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": generation.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if generation.temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = generation.temperature
    else:
        generation_kwargs["do_sample"] = False

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()


@web_app.post("/generate")
async def generate_predictions(payload: EvalRequest) -> dict[str, Any]:
    adapter = payload.adapter
    if not adapter:
        raise HTTPException(status_code=400, detail="Fine-tuned evaluation requires an adapter archive URL.")

    try:
        model, tokenizer = load_runtime(payload.base_model, adapter)
        predictions = []
        for sample in payload.samples:
            prediction = generate_prediction_for_sample(model, tokenizer, sample, payload.generation)
            predictions.append({"id": sample.id, "prediction": prediction})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction generation failed: {exc}") from exc

    return {
        "predictions": predictions,
        "model": {
            "base_model": payload.base_model,
            "adapter_url": adapter.url if adapter else None,
            "adapter_cache_key": adapter.cache_key if adapter else None,
        },
    }


@app.function(
    image=image,
    gpu=MODAL_EVAL_GPU,
    timeout=60 * 60,
    scaledown_window=10 * 60,
    volumes={HF_CACHE_DIR: hf_cache_volume},
    secrets=[hf_secret],
)
@modal.asgi_app()
def modal_service():
    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    return web_app
