# DALTP

Domain Adaptive LLM Training Platform

The Domain-Adaptive LLM Training Platform is an end-to-end system that enables organizations to take any open-source large language model and fine-tune it on proprietary domain-specific data — without writing boilerplate training code.

## ✨ Features

- 🎯 Parameter-efficient fine-tuning via LoRA / QLoRA (fits on a single GPU)
- 📊 Multi-dimensional automated evaluation (perplexity, ROUGE, BERTScore, hallucination rate)
- 🔄 Continual learning flywheel — model improves from user corrections over time
- 🤖 Local synthetic dataset generation (no external API required)
- 🔒 Org-level access control with Firebase Auth
- 🗃️ Dataset versioning with DVC — full reproducibility across training runs
- 🚀 vLLM-powered inference API with hot-swappable LoRA adapters
- 📡 Real-time training job status via Firestore + WebSocket
- 🐳 Fully containerized — runs entirely on-premise, no data leaves your servers
- 📱 Streamlit dashboard for training, evaluation, and feedback management

---

## 🗂 Folder Structure

```
domain-llm-platform/
├── data/                    # Data ingestion & preprocessing
│   ├── ingestion/           # PDF, DOCX, JSON, CSV loaders
│   ├── preprocessing/       # Chunking, deduplication, filtering
│   └── versioning/          # DVC-tracked dataset snapshots
├── training/                # Fine-tuning pipeline
│   ├── configs/             # YAML training configs per run
│   ├── peft/                # LoRA, QLoRA, prefix tuning modules
│   ├── trainer.py           # Main training entrypoint
│   └── checkpointing/       # Checkpoint management
├── evaluation/              # Automated evaluation engine
│   ├── metrics/             # ROUGE, BERTScore, perplexity, etc.
│   ├── benchmarks/          # Domain-specific QA test sets
│   └── report_generator.py  # HTML/JSON eval reports
├── serving/                 # Inference API
│   ├── api/                 # FastAPI endpoints
│   ├── adapters/            # Hot-swappable LoRA adapter registry
│   └── inference_engine.py  # vLLM integration
├── feedback/                # Continual learning pipeline
│   ├── collector.py         # User correction ingestion
│   ├── replay_buffer.py     # Experience replay for fine-tuning
│   └── scheduler.py         # Triggered micro-training jobs
├── ui/                      # Streamlit dashboard
├── scripts/                 # CLI utilities
├── tests/                   # Unit & integration tests
├── docker/                  # Dockerfiles per service
├── configs/                 # Global platform config
└── README.md
```

## 👥 User Roles

| Role          | Access                                                                   |
| ------------- | ------------------------------------------------------------------------ |
| Admin         | Full platform access — training, evaluation, deployment, user management |
| ML Engineer   | Training runs, evaluation reports, adapter management                    |
| Domain Expert | Feedback UI, response correction, dataset review                         |
| Viewer        | Read-only access to evaluation reports and model metrics                 |

Demo credentials (development only):

```
Admin:        admin@daltp.dev
ML Engineer:  engineer@daltp.dev
Domain Expert: expert@daltp.dev
```

---

## 🗄️ Tech Stack

| Layer               | Technology                         | Purpose                               |
| ------------------- | ---------------------------------- | ------------------------------------- |
| Core ML             | PyTorch + HuggingFace Transformers | Model loading, training loop          |
| PEFT                | peft, bitsandbytes, trl            | LoRA/QLoRA adapters, DPO trainer      |
| Evaluation          | evaluate, bert-score, nltk         | Automated metrics                     |
| Serving             | vLLM + FastAPI                     | High-throughput inference API         |
| Primary DB          | PostgreSQL                         | Structured metadata, relational data  |
| Vector DB           | Qdrant                             | Semantic search over domain documents |
| Cache / Queue       | DragonflyDB (Redis-compatible)     | Async job queue, inference cache      |
| Dataset Versioning  | DVC + MinIO / Google Drive         | Reproducible training data            |
| Auth + Real-time    | Firebase Auth + Firestore          | User management, live job status      |
| Experiment Tracking | Weights & Biases / MLflow          | Run tracking, artifact registry       |
| Dashboard           | Streamlit                          | Training and feedback UI              |
| Containerization    | Docker + docker-compose            | Reproducible environments             |

---

## ⚙️ Training Configuration

Training runs are fully described by YAML config files. Example for Llama 3.1 8B on Colab:

```yaml
# configs/lora_llama3_8b.yaml
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
peft_method: qlora

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_use_double_quant: true

lora:
  r: 16
  lora_alpha: 32
  target_modules:
    [q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj]
  lora_dropout: 0.05

training:
  num_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2e-4
  lr_scheduler: cosine
  warmup_ratio: 0.03
  max_seq_length: 2048
  fp16: true
  gradient_checkpointing: true
  save_steps: 50
  output_dir: /content/drive/MyDrive/daltp_checkpoints

evaluation:
  eval_steps: 100
  metrics: [perplexity, rouge_l, bertscore, domain_accuracy, hallucination_rate]
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Arav016/DALTP.git
cd DALTP
```

# Create virtual environment

python -m venv venv && source venv/bin/activate

# Install dependencies

pip install -r requirements.txt

# Copy environment config

cp .env.example .env # Edit with your API keys and storage paths

# Launch supporting services (Redis, MinIO, Postgres)

docker-compose up -d

## Quick Start

# 1. Prepare your domain data

python scripts/ingest.py --source ./my_documents --output data/domain_v1

# 2. Launch a LoRA fine-tuning run

python training/trainer.py --config configs/lora_mistral7b.yaml --data data/domain_v1

# 3. Run automated evaluation

python evaluation/run_eval.py --checkpoint runs/mistral7b_lora_v1 --benchmark benchmarks/domain_qa

# 4. Start the inference API

python serving/api/main.py --adapter runs/mistral7b_lora_v1/final_adapter

## 🤝 Contributing

Pull requests are welcome!  
To contribute:

1. Fork the repo
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add a new feature"`)
4. Push and open a PR

## 📧 Contact

Created by [Arav016](https://github.com/Arav016)
