# DALTP

DALTP, the **Domain Adaptive LLM Training Platform**, is a practical end-to-end workflow for:

- constructing domain datasets from raw documents
- fine-tuning an open-source LLM with QLoRA
- evaluating `base`, `RAG`, `fine-tuned`, and `fine-tuned + RAG` variants
- reviewing results in a React frontend backed by a FastAPI API
- preparing **local** or **Colab-assisted** run bundles

## What Is Implemented

- **Dataset construction**
  - Raw corpus generation from document collections
  - QA-pair generation
  - Instruction-pair generation
- **Training**
  - QLoRA-style fine-tuning of `Llama 3.1 8B`
  - JSON-based trainer configuration
- **RAG support**
  - Qdrant-backed retrieval in evaluation and prediction generation
  - Embedding generation through the ingestion layer
- **Evaluation**
  - Held-out benchmark generation from unseen documents
  - Prediction generation for:
    - base
    - rag
    - fine_tuned
    - fine_tuned_rag
  - Scoring with:
    - `ROUGE-L`
    - `BERTScore F1`
    - `Fact Coverage`
- **Frontend + API**
  - React frontend for overview, dataset upload, run building, bundle download, and evaluation review
  - FastAPI backend for serving run summaries, examples, datasets, and generated run bundles
- **Execution modes**
  - Local execution when user hardware is capable enough
  - Colab-assisted workflow when local hardware is not sufficient

## Current Stack

- **Python**
- **PyTorch**
- **Hugging Face Transformers**
- **PEFT / bitsandbytes / TRL**
- **Qdrant**
- **FastAPI**
- **React + Vite**

## Repo Structure

```text
DALTP/
|- backend/
|  |- api/
|  |  |- app.py                  # FastAPI app for frontend data + run bundles
|  |  \- runtime/               # Generated uploads and run bundles
|  |- dataset/
|  |  |- dataset_construction/
|  |  |  |- raw_data.py
|  |  |  |- QApair.py
|  |  |  |- Instruction_set.py
|  |  |  \- generated_datasets/
|  |  \- ingestion/
|  |     |- data_ingestion.py
|  |     |- embeddings.py
|  |     \- embed_db.py
|  |- evaluation/
|  |  |- generate_test_benchmark.py
|  |  |- generate_predictions.py
|  |  |- compare_model_outputs.py
|  |  |- document_test_pipeline.py
|  |  \- outputs/
|  \- training/
|     |- trainer.py
|     |- configs/
|     \- outputs/
|- frontend/
|  |- src/
|  |  |- App.jsx
|  |  |- main.jsx
|  |  \- styles.css
|  |- index.html
|  |- package.json
|  \- vite.config.js
|- DALTP_dataset/               # Local document corpus used for experimentation
|- local_run_commands.md        # Useful local CLI commands
|- requirements.txt
\- README.md
```

## Core Workflow

### 1. Build datasets

Generate the three dataset types separately:

- **raw corpus**
- **qa dataset**
- **instruction dataset**

This separation matters:

- corpus text helps with domain language
- QA data teaches grounded answering behavior
- instruction data teaches task following and response structure

### 2. Fine-tune the model

The current trainer uses a JSON config such as:

- `backend/training/configs/lora_llama3_8b_instruct.json`

### 3. Build a held-out benchmark

For proper testing, held-out documents should be kept separate from the fine-tuning documents. The evaluation pipeline supports generating benchmark QA data from unseen documents.

### 4. Generate model outputs

Supported modes:

- `base`
- `rag`
- `fine_tuned`
- `fine_tuned_rag`

### 5. Compare the systems

The comparison layer scores predictions against the held-out benchmark using:

- `ROUGE-L`
- `BERTScore F1`
- `Fact Coverage`

### 6. Review through the frontend

The frontend lets you:

- inspect scored runs
- upload datasets
- prepare new training bundles
- choose local vs Colab-assisted execution
- download run bundles
- review example benchmark outputs side by side

## Local Development

Install Python dependencies from the repo root:

```bash
pip install -r requirements.txt
```

## Backend API

Run the FastAPI backend from the repo root:

```bash
uvicorn backend.api.app:app --reload
```

Main API responsibilities:

- dashboard metrics from scored outputs
- run summaries and example outputs
- dataset registry
- uploaded dataset storage
- run bundle generation
- bundle download and launch commands

## Frontend

Run the React frontend:

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server proxies `/api` requests to:

- `http://127.0.0.1:8000`


That file contains the commands currently used for:

- raw corpus construction
- QA dataset construction
- instruction dataset construction
- training dependency installation
- fine-tuning
- local vLLM serving

## Colab-Assisted Execution

DALTP currently supports a pragmatic hybrid workflow:

- if a user's device is capable enough, use **local execution**
- if not, use **Colab-assisted execution**

The frontend/API supports this by generating run bundles that contain:

- dataset artifacts
- config JSON
- local launch commands
- Colab-friendly launch commands
- bundle download archive

This keeps the frontend stable while allowing heavy jobs to move to remote GPU environments when needed.

## Current Evaluation Artifacts

The repo currently includes a scored evaluation run under:

- `backend/evaluation/outputs/test_run_01`

and fine-tuned adapter outputs under:

- `backend/training/outputs/llama3_8b_sft`