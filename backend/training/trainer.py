import argparse
import json
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


DEFAULT_CONFIG_PATH = Path("backend/training/configs/lora_llama3_8b_instruct.json")


def load_config(config_path):
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Training config does not exist: {config_file}")

    with config_file.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def load_jsonl_records(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    records = []
    with path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_training_dataset(config):
    dataset_files = config["datasets"]
    records = []
    dataset_counts = {}

    for dataset_type, dataset_path in dataset_files.items():
        entries = load_jsonl_records(dataset_path)
        accepted = 0
        for entry in entries:
            messages = entry.get("messages")
            if not messages:
                continue
            records.append(
                {
                    "messages": messages,
                    "dataset_type": dataset_type,
                    "source": entry.get("source", ""),
                    "file_name": entry.get("file_name", ""),
                }
            )
            accepted += 1
        dataset_counts[dataset_type] = accepted

    if not records:
        raise ValueError("No conversational records found in the configured datasets.")

    dataset = Dataset.from_list(records)
    eval_ratio = config["training"].get("eval_ratio", 0.1)
    if eval_ratio and 0 < eval_ratio < 1:
        split = dataset.train_test_split(test_size=eval_ratio, seed=config["training"].get("seed", 42))
        return split["train"], split["test"], dataset_counts

    return dataset, None, dataset_counts


def build_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_model(config):
    model_name = config["model"]["name"]
    training_config = config["training"]
    quantization = config.get("quantization", {})

    model_kwargs = {
        "trust_remote_code": config["model"].get("trust_remote_code", False),
    }

    dtype_name = training_config["dtype"]
    if dtype_name == "bfloat16":
        model_kwargs["dtype"] = torch.bfloat16
    elif dtype_name == "float16":
        model_kwargs["dtype"] = torch.float16
    else:
        raise ValueError(f"Unsupported training.dtype: {dtype_name}")


    if quantization.get("load_in_4bit"):
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quantization.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, quantization.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=quantization.get("bnb_4bit_use_double_quant", True),
        )
        model_kwargs["device_map"] = "auto"

    if quantization.get("load_in_4bit"):
        quant_dtype_name = quantization["bnb_4bit_compute_dtype"]
        if quant_dtype_name == "bfloat16":
            quant_dtype = torch.bfloat16
        elif quant_dtype_name == "float16":
            quant_dtype = torch.float16
        else:
            raise ValueError(f"Unsupported quantization.bnb_4bit_compute_dtype: {quant_dtype_name}")

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quantization.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=quant_dtype,
            bnb_4bit_use_double_quant=quantization.get("bnb_4bit_use_double_quant", True),
        )
        model_kwargs["device_map"] = "auto"


    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False
    return model


def build_lora_config(config):
    lora = config["lora"]
    return LoraConfig(
        r=lora["r"],
        lora_alpha=lora["lora_alpha"],
        lora_dropout=lora["lora_dropout"],
        target_modules=lora["target_modules"],
        task_type="CAUSAL_LM",
    )


def train_model(config):
    train_dataset, eval_dataset, dataset_counts = build_training_dataset(config)
    tokenizer = build_tokenizer(config["model"]["name"])
    model = build_model(config)
    peft_config = build_lora_config(config)
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Training datasets:")
    for dataset_type, count in dataset_counts.items():
        print(f"  - {dataset_type}: {count} records")
    print(f"Train records: {len(train_dataset)}")
    print(f"Eval records: {len(eval_dataset) if eval_dataset is not None else 0}")
    print(f"Output dir: {output_dir}")

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"].get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        lr_scheduler_type=config["training"].get("lr_scheduler_type", "cosine"),
        warmup_ratio=config["training"].get("warmup_ratio", 0.03),
        logging_steps=config["training"].get("logging_steps", 10),
        save_steps=config["training"].get("save_steps", 100),
        eval_steps=config["training"].get("eval_steps", 100),
        save_strategy=config["training"].get("save_strategy", "steps"),
        eval_strategy=config["training"].get("eval_strategy", "steps") if eval_dataset is not None else "no",
        bf16=config["training"].get("bf16", True),
        fp16=config["training"].get("fp16", False),
        gradient_checkpointing=config["training"].get("gradient_checkpointing", True),
        max_length=config["training"].get("max_length", 2048),
        assistant_only_loss=config["training"].get("assistant_only_loss", True),
        packing=config["training"].get("packing", False),
        dataset_text_field=None,
        seed=config["training"].get("seed", 42),
        report_to=config["training"].get("report_to", []),
        save_total_limit=config["training"].get("save_total_limit"),
        optim=config["training"].get(
            "optim",
            "paged_adamw_8bit" if config.get("quantization", {}).get("load_in_4bit") else "adamw_torch",
        ),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(config["training"]["output_dir"])


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune an open-source Llama model on QA and instruction datasets.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the JSON training config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    train_model(config)


if __name__ == "__main__":
    main()
