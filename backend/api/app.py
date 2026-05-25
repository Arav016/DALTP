from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.api import db_store, dataset_storage
from backend.dataset.dataset_construction import Instruction_set, QApair, raw_data
from backend.dataset.ingestion import data_ingestion
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field


API_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = API_DIR / "runtime"
AUTH_DIR = RUNTIME_DIR / "auth"
DATASETS_DIR = RUNTIME_DIR / "datasets"
BUNDLES_DIR = RUNTIME_DIR / "run_bundles"
JOBS_DIR = RUNTIME_DIR / "jobs"
MODELS_DIR = RUNTIME_DIR / "models"
RUN_MANIFEST_FILE = "daltp_run.json"
USERS_FILE = AUTH_DIR / "users.json"
SESSIONS_FILE = AUTH_DIR / "sessions.json"

BACKEND_DIR = API_DIR.parent
ROOT_DIR = BACKEND_DIR.parent
EVAL_OUTPUTS_DIR = BACKEND_DIR / "evaluation" / "outputs"
TRAINING_CONFIG_PATH = BACKEND_DIR / "training" / "configs" / "lora_llama3_8b_instruct.json"
LOCAL_COMMANDS_PATH = ROOT_DIR / "local_run_commands.md"
DEFAULT_GENERATION_API_BASE = QApair.default_generation_api_base()
DEFAULT_GENERATION_MODEL = QApair.default_generation_model()
UPLOADED_DATASET_EXTENSIONS = {".json", ".jsonl", ".csv", ".txt"}

class RegisterPayload(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    email: str = Field(min_length=5, max_length=160)
    password: str = Field(min_length=8, max_length=128)


class LoginPayload(BaseModel):
    email: str = Field(min_length=5, max_length=160)
    password: str = Field(min_length=8, max_length=128)


class DatasetUploadFile(BaseModel):
    name: str
    content: str
    encoding: str = "base64"
    size: int | None = None
    mimeType: str | None = None
    relativePath: str | None = None


class DatasetUploadPayload(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    kind: str = Field(pattern="^(qa|instruction|benchmark|corpus)$")
    files: list[DatasetUploadFile] = Field(min_length=1)


class DatasetGeneratePayload(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    kind: str = Field(pattern="^(qa|instruction|corpus)$")
    files: list[DatasetUploadFile] = Field(min_length=1)
    chunkSize: int = 1000
    chunkOverlap: int = 150
    qaNumPairs: int = 4
    qaChunkSize: int = 2500
    qaChunkOverlap: int = 150
    qaMaxContextsPerDocument: int | None = 2
    instructionContextSize: int = 2500
    instructionContextOverlap: int = 150
    instructionMaxContextsPerDocument: int | None = 2
    taskTypes: list[str] | None = None
    modelName: str = DEFAULT_GENERATION_MODEL
    apiBase: str = DEFAULT_GENERATION_API_BASE
    ingestToPgvector: bool = False
    collectionName: str | None = None


class DatasetVectorIngestPayload(BaseModel):
    collectionName: str = Field(min_length=2, max_length=120)
    chunkSize: int = 1000
    chunkOverlap: int = 150


class ManualConfigPayload(BaseModel):
    learningRate: float | None = None
    epochs: int | None = None
    batchSize: int | None = None
    gradientAccumulationSteps: int | None = None
    maxLength: int | None = None
    dtype: str | None = None
    loadIn4bit: bool | None = None


class RunBundlePayload(BaseModel):
    runName: str = Field(min_length=2, max_length=120)
    executionMode: str = Field(pattern="^(local|colab)$")
    baseModel: str
    peftMethod: str = Field(pattern="^(qlora|lora)$")
    loraRank: int
    qaDatasetId: str
    instructionDatasetId: str
    configMode: str = Field(pattern="^(preset|manual|upload)$")
    presetId: str | None = None
    manualConfig: ManualConfigPayload | None = None
    uploadedConfigText: str | None = None


class ModelImportPayload(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    source: str = Field(pattern="^(colab|manual)$")
    baseModel: str = Field(min_length=2, max_length=200)
    peftMethod: str = Field(pattern="^(qlora|lora)$")
    loraRank: int | None = None
    files: list[DatasetUploadFile] = Field(min_length=1)


class EvaluationJobPayload(BaseModel):
    runName: str = Field(min_length=2, max_length=120)
    benchmarkMode: str = Field(pattern="^(existing|generate)$")
    benchmarkDatasetId: str | None = None
    corpusDatasetId: str | None = None
    runBase: bool = True
    runRag: bool = True
    runFineTuned: bool = True
    runFineTunedRag: bool = True
    topK: int = 3
    maxNewTokens: int = 256
    temperature: float = 0.0
    qaGenerationModel: str = DEFAULT_GENERATION_MODEL
    qaApiBase: str = DEFAULT_GENERATION_API_BASE
    numPairs: int = 10
    chunkSize: int = 5000
    chunkOverlap: int = 200


app = FastAPI(title="DALTP API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clarify_generation_error(exc: Exception, api_base: str, kind: str) -> str:
    message = str(exc)
    if "404" in message and "Not Found" in message:
        return (
            f"{kind.upper()} generation could not reach a valid OpenRouter endpoint. "
            f"DALTP attempted to call '{api_base}'. Make sure your OpenRouter base URL is correct and that it exposes the chat completions route."
        )
    if "Connection error" in message or "actively refused it" in message or "ConnectError" in message:
        return (
            f"{kind.upper()} generation could not connect to the generation endpoint at '{api_base}'. "
            f"Make sure your OpenRouter base URL and network access are configured correctly before generating datasets."
        )
    return message


def normalize_relative_path(value: str | None, fallback: str) -> Path:
    raw = (value or fallback or "dataset.txt").replace("\\", "/").strip("/")
    candidate = Path(raw)
    safe_parts = [part for part in candidate.parts if part not in ("", ".", "..")]
    return Path(*safe_parts) if safe_parts else Path(fallback)


def slugify(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "daltp-run"


def training_output_dir_for_execution(run_name: str, execution_mode: str) -> str:
    run_slug = slugify(run_name)
    if execution_mode == "colab":
        return f"/content/daltp_outputs/{run_slug}"
    return str(Path("outputs") / run_slug)


def ensure_runtime_dirs() -> None:
    for path in (AUTH_DIR, DATASETS_DIR, BUNDLES_DIR, JOBS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
    if not USERS_FILE.exists():
        USERS_FILE.write_text("[]", encoding="utf-8")
    if not SESSIONS_FILE.exists():
        SESSIONS_FILE.write_text("{}", encoding="utf-8")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def safe_read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def upload_bytes(upload: DatasetUploadFile) -> bytes:
    if upload.encoding == "base64":
        return base64.b64decode(upload.content.encode("utf-8"))
    return upload.content.encode("utf-8")


def upload_text(upload: DatasetUploadFile) -> str:
    return upload_bytes(upload).decode("utf-8", errors="ignore")


def validate_uploaded_dataset_files(files: list[DatasetUploadFile]) -> None:
    unsupported_files: list[str] = []
    likely_source_docs: list[str] = []

    for index, upload in enumerate(files):
        file_name = upload.name or f"dataset-{index}.txt"
        relative_path = normalize_relative_path(upload.relativePath, file_name)
        suffix = Path(file_name).suffix.lower()
        mime_type = (upload.mimeType or "").lower()
        payload_bytes = upload_bytes(upload)

        if suffix not in UPLOADED_DATASET_EXTENSIONS:
            unsupported_files.append(relative_path.as_posix())
            continue

        # Guard against raw PDFs or other binary docs slipping through folder uploads.
        if mime_type == "application/pdf" or payload_bytes.startswith(b"%PDF"):
            likely_source_docs.append(relative_path.as_posix())

    if likely_source_docs or unsupported_files:
        rejected = likely_source_docs + unsupported_files
        raise HTTPException(
            status_code=400,
            detail=(
                "Upload dataset only accepts prepared dataset files (.json, .jsonl, .csv, .txt). "
                "Raw source documents like PDFs should go through 'Generate from documents' instead. "
                f"Rejected files: {', '.join(rejected[:8])}"
            ),
        )


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return round(path.stat().st_size / (1024 * 1024), 2)


def adapter_size_mb() -> float:
    adapter_dir = BACKEND_DIR / "training" / "outputs" / "llama3_8b_sft"
    if not adapter_dir.exists():
        return 0.0
    total = 0
    for item in adapter_dir.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return round(total / (1024 * 1024), 2)


def default_training_config() -> dict[str, Any]:
    payload = load_json(TRAINING_CONFIG_PATH, {})
    return payload if isinstance(payload, dict) else {}


def built_in_datasets() -> list[dict[str, Any]]:
    return []


def uploaded_dataset_dirs(user_id: str) -> list[Path]:
    owner_dir = DATASETS_DIR / user_id
    if not owner_dir.exists():
        return []
    return sorted([path for path in owner_dir.iterdir() if path.is_dir()], reverse=True)


def list_uploaded_datasets(user_id: str) -> list[dict[str, Any]]:
    datasets = db_store.list_datasets(user_id)
    for dataset in datasets:
        if dataset.get("lineCount") is not None and dataset.get("sizeMb") is not None:
            continue
        data_file = Path(dataset["path"])
        dataset["lineCount"] = line_count(data_file)
        dataset["sizeMb"] = file_size_mb(data_file)
        db_store.upsert_dataset(dataset)
    return datasets


def dataset_catalog(user: dict[str, Any]) -> list[dict[str, Any]]:
    return built_in_datasets() + list_uploaded_datasets(user["id"])


def dataset_by_id(dataset_id: str, user: dict[str, Any]) -> dict[str, Any]:
    for dataset in dataset_catalog(user):
        if dataset["id"] == dataset_id:
            return dataset
    raise HTTPException(status_code=404, detail="Dataset not found for this account.")


def normalized_name(value: str) -> str:
    return " ".join(part for part in str(value or "").strip().lower().split())


def dataset_name_exists(user: dict[str, Any], dataset_name: str) -> bool:
    target = normalized_name(dataset_name)
    return any(normalized_name(dataset.get("name", "")) == target for dataset in dataset_catalog(user))


def bundle_run_name_exists(user_id: str, run_name: str) -> bool:
    target = normalized_name(run_name)
    return any(normalized_name(bundle.get("runName", "")) == target for bundle in list_bundles(user_id))


def evaluation_run_name_exists(user_id: str, run_name: str) -> bool:
    target = normalized_name(run_name)
    return any(
        normalized_name(load_run_manifest(path).get("runName", "")) == target
        for path in list_run_dirs_for_user(user_id)
    )


def default_collection_name(dataset_name: str, user: dict[str, Any]) -> str:
    return f"{slugify(user['name'])}-{slugify(dataset_name)}"


def write_dataset_manifest(dataset_dir: Path, manifest: dict[str, Any]) -> None:
    write_json(dataset_dir / "manifest.json", manifest)


def ingest_dataset_source_to_vector_store(
    *,
    dataset_dir: Path,
    dataset_name: str,
    user: dict[str, Any],
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, Any]:
    source_root = dataset_dir / "source_files"
    if not source_root.exists():
        raise HTTPException(status_code=400, detail="Source documents are missing for this dataset.")

    try:
        ingest_summary = data_ingestion.ingest_documents(
            input_path=str(source_root),
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"pgvector ingestion failed: {exc}") from exc

    return {
        "ingested": True,
        "collectionName": collection_name,
        "chunkSize": chunk_size,
        "chunkOverlap": chunk_overlap,
        "documentsProcessed": ingest_summary.get("documents_processed", 0),
        "chunksCreated": ingest_summary.get("chunks_created", 0),
        "ingestedAt": now_iso(),
        "ingestedBy": user["id"],
        "datasetName": dataset_name,
        "backend": "pgvector",
    }


def delete_uploaded_dataset(dataset_id: str, user: dict[str, Any]) -> None:
    dataset = db_store.get_dataset(dataset_id, user["id"])
    if not dataset:
        raise HTTPException(status_code=404, detail="Uploaded dataset not found for this account.")

    dataset_storage.delete_dataset_artifact(dataset)
    dataset_dir_value = dataset.get("datasetDir")
    dataset_dir = Path(dataset_dir_value) if dataset_dir_value else (DATASETS_DIR / user["id"] / dataset_id)
    if dataset_dir.exists() and dataset_dir.is_dir():
        shutil.rmtree(dataset_dir)
    db_store.delete_dataset(dataset_id, user["id"])


def delete_run_bundle(bundle_id: str, user: dict[str, Any]) -> None:
    bundle_row = db_store.get_bundle(bundle_id, user["id"])
    if not bundle_row:
        raise HTTPException(status_code=404, detail="Run bundle not found for this account.")

    bundle_dir = Path(bundle_row["bundleDir"])
    archive_path = Path(bundle_row["archivePath"])
    if bundle_dir.exists() and bundle_dir.is_dir():
        shutil.rmtree(bundle_dir)
    archive_path.unlink(missing_ok=True)
    db_store.delete_bundle(bundle_id, user["id"])


def delete_model_artifact(model_id: str, user: dict[str, Any]) -> None:
    model = db_store.get_model(model_id, user["id"])
    if not model:
        raise HTTPException(status_code=404, detail="Model artifact not found for this account.")

    model_dir = Path(model["modelDir"])
    archive_path = Path(model["archivePath"])
    if model_dir.exists() and model_dir.is_dir():
        shutil.rmtree(model_dir)
    archive_path.unlink(missing_ok=True)
    db_store.delete_model(model_id, user["id"])


def uploaded_dataset_dir(dataset_id: str, user: dict[str, Any]) -> Path:
    owner_dir = DATASETS_DIR / user["id"]
    dataset_dir = owner_dir / dataset_id
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise HTTPException(status_code=404, detail="Uploaded dataset not found for this account.")
    return dataset_dir


def user_models_dir(user_id: str) -> Path:
    path = MODELS_DIR / user_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_model_artifact_dirs(user_id: str) -> list[Path]:
    owner_dir = user_models_dir(user_id)
    return sorted([path for path in owner_dir.iterdir() if path.is_dir()], reverse=True)


def write_model_manifest(model_dir: Path, manifest: dict[str, Any]) -> None:
    write_json(model_dir / "manifest.json", manifest)


def load_model_manifest(model_dir: Path) -> dict[str, Any]:
    payload = load_json(model_dir / "manifest.json", {})
    return payload if isinstance(payload, dict) else {}


def list_model_artifacts(user_id: str) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for model in db_store.list_models(user_id):
        archive = Path(model["archivePath"])
        artifacts.append(
            {
                "id": model["id"],
                "name": model["name"],
                "source": model["source"],
                "baseModel": model.get("baseModel"),
                "peftMethod": model.get("peftMethod", "qlora"),
                "loraRank": model.get("loraRank"),
                "runName": model.get("runName"),
                "ownerId": user_id,
                "createdAt": model.get("createdAt", now_iso()),
                "fileCount": len(model.get("files", [])),
                "archiveSizeMb": file_size_mb(archive),
                "downloadUrl": f"/api/models/{model['id']}/download",
            }
        )
    return artifacts


def model_artifact_by_id(model_id: str, user_id: str) -> tuple[dict[str, Any], Path]:
    model = db_store.get_model(model_id, user_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model artifact not found for this account.")
    model_dir = Path(model["modelDir"])
    artifact = {
        "id": model["id"],
        "name": model["name"],
        "source": model["source"],
        "baseModel": model.get("baseModel"),
        "peftMethod": model.get("peftMethod", "qlora"),
        "loraRank": model.get("loraRank"),
        "runName": model.get("runName"),
        "ownerId": user_id,
        "createdAt": model.get("createdAt", now_iso()),
        "fileCount": len(model.get("files", [])),
        "downloadUrl": f"/api/models/{model_id}/download",
        "archiveSizeMb": file_size_mb(model_dir.with_suffix(".zip")),
    }
    return artifact, model_dir


def model_artifact_name_exists(user_id: str, model_name: str) -> bool:
    target = normalized_name(model_name)
    return any(normalized_name(model.get("name", "")) == target for model in list_model_artifacts(user_id))


def build_directory_archive(source_dir: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(source_dir))


def import_model_artifact(payload: ModelImportPayload, user: dict[str, Any]) -> dict[str, Any]:
    model_name = payload.name.strip()
    if model_artifact_name_exists(user["id"], model_name):
        raise HTTPException(status_code=400, detail=f"A model artifact named '{model_name}' already exists in this account.")

    artifact_id = f"model-{slugify(model_name)}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    model_dir = user_models_dir(user["id"]) / artifact_id
    model_dir.mkdir(parents=True, exist_ok=True)
    files_dir = model_dir / "artifact_files"
    files_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[dict[str, Any]] = []
    for index, upload in enumerate(payload.files):
        file_name = upload.name or f"artifact-{index}"
        relative_path = normalize_relative_path(upload.relativePath, file_name)
        target_path = files_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(upload_bytes(upload))
        saved_files.append(
            {
                "name": file_name,
                "relativePath": relative_path.as_posix(),
                "size": upload.size or len(upload_bytes(upload)),
                "mimeType": upload.mimeType,
            }
        )

    manifest = {
        "id": artifact_id,
        "name": model_name,
        "source": payload.source,
        "baseModel": payload.baseModel.strip(),
        "peftMethod": payload.peftMethod,
        "loraRank": payload.loraRank,
        "ownerId": user["id"],
        "createdAt": now_iso(),
        "runName": None,
        "files": saved_files,
    }
    write_model_manifest(model_dir, manifest)
    db_store.upsert_model({**manifest, "modelDir": str(model_dir), "archivePath": str(model_dir.with_suffix(".zip"))})
    build_directory_archive(files_dir, model_dir.with_suffix(".zip"))
    artifact, _ = model_artifact_by_id(artifact_id, user["id"])
    return artifact


def register_local_model_artifact(*, bundle: dict[str, Any], user: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    model_name = bundle["runName"]
    if model_artifact_name_exists(user["id"], model_name):
        raise HTTPException(status_code=400, detail=f"A model artifact named '{model_name}' already exists in this account.")
    if not output_dir.exists():
        raise HTTPException(status_code=400, detail=f"Training output directory is missing: {output_dir}")

    artifact_id = f"model-{slugify(model_name)}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    model_dir = user_models_dir(user["id"]) / artifact_id
    model_dir.mkdir(parents=True, exist_ok=True)
    files_dir = model_dir / "artifact_files"
    shutil.copytree(output_dir, files_dir)

    saved_files: list[dict[str, Any]] = []
    for file_path in files_dir.rglob("*"):
        if file_path.is_file():
            saved_files.append(
                {
                    "name": file_path.name,
                    "relativePath": file_path.relative_to(files_dir).as_posix(),
                    "size": file_path.stat().st_size,
                    "mimeType": None,
                }
            )

    manifest = {
        "id": artifact_id,
        "name": model_name,
        "source": "local",
        "baseModel": bundle.get("baseModel"),
        "peftMethod": bundle.get("peftMethod", "qlora"),
        "loraRank": bundle.get("loraRank"),
        "ownerId": user["id"],
        "createdAt": now_iso(),
        "runName": bundle.get("runName"),
        "bundleId": bundle.get("id"),
        "files": saved_files,
    }
    write_model_manifest(model_dir, manifest)
    db_store.upsert_model({**manifest, "modelDir": str(model_dir), "archivePath": str(model_dir.with_suffix(".zip"))})
    build_directory_archive(files_dir, model_dir.with_suffix(".zip"))
    artifact, _ = model_artifact_by_id(artifact_id, user["id"])
    return artifact


def user_jobs_dir(user_id: str) -> Path:
    path = JOBS_DIR / user_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_jobs(user_id: str) -> list[dict[str, Any]]:
    return db_store.list_jobs(user_id)


def job_dir(job_id: str, user_id: str) -> Path:
    path = user_jobs_dir(user_id) / job_id
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job not found for this account.")
    return path


def append_job_log(path: Path, text: str) -> None:
    log_path = path / "job.log"
    with log_path.open("a", encoding="utf-8") as file_handle:
        file_handle.write(text)


def read_job_log(path: Path, max_lines: int = 120) -> str:
    log_path = path / "job.log"
    if not log_path.exists():
        return ""
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-max_lines:])


def update_job_manifest(path: Path, **changes: Any) -> dict[str, Any]:
    manifest = load_json(path / "manifest.json", {})
    manifest.update(changes)
    manifest["updatedAt"] = now_iso()
    write_json(path / "manifest.json", manifest)
    manifest["jobDir"] = str(path)
    db_store.upsert_job(manifest)
    return manifest


def create_job(user: dict[str, Any], job_type: str, title: str, metadata: dict[str, Any] | None = None) -> tuple[dict[str, Any], Path]:
    job_id = f"{job_type}-{datetime.now().strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(4)}"
    path = user_jobs_dir(user["id"]) / job_id
    path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "id": job_id,
        "type": job_type,
        "title": title,
        "status": "queued",
        "ownerId": user["id"],
        "createdAt": now_iso(),
        "updatedAt": now_iso(),
        "metadata": metadata or {},
        "logPath": str(path / "job.log"),
        "jobDir": str(path),
    }
    write_json(path / "manifest.json", manifest)
    db_store.upsert_job(manifest)
    return manifest, path


def get_job_manifest(job_id: str, user: dict[str, Any]) -> dict[str, Any]:
    manifest = db_store.get_job(job_id, user["id"])
    if not manifest:
        raise HTTPException(status_code=404, detail="Job not found for this account.")
    path = Path(manifest["jobDir"])
    manifest["logs"] = read_job_log(path)
    return manifest


def start_background_job(path: Path, runner) -> None:
    thread = threading.Thread(target=runner, args=(path,), daemon=True)
    thread.start()


def run_logged_command(command: list[str], *, cwd: Path, path: Path, step_label: str) -> None:
    append_job_log(path, f"\n$ {' '.join(command)}\n")
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if process.stdout is not None:
        for line in process.stdout:
            append_job_log(path, line)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{step_label} failed with exit code {return_code}.")


def load_users() -> list[dict[str, Any]]:
    return []


def save_users(users: list[dict[str, Any]]) -> None:
    return None


def load_sessions() -> dict[str, dict[str, Any]]:
    return {}


def save_sessions(sessions: dict[str, dict[str, Any]]) -> None:
    return None


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt_value = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_value.encode("utf-8"), 120_000)
    return base64.b64encode(digest).decode("utf-8"), salt_value


def verify_password(password: str, password_hash: str, password_salt: str) -> bool:
    candidate_hash, _ = hash_password(password, password_salt)
    return secrets.compare_digest(candidate_hash, password_hash)


def create_session_token(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    created_at = now_iso()
    db_store.create_session(token, user_id, created_at, created_at)
    return token


def workspace_summary_for_user(user: dict[str, Any]) -> dict[str, Any]:
    uploaded = list_uploaded_datasets(user["id"])
    bundles = list_bundles(user["id"])
    models = sorted({bundle["baseModel"] for bundle in bundles if bundle.get("baseModel")})
    return {
        "uploadedDatasetCount": len(uploaded),
        "bundleCount": len(bundles),
        "recentModels": models[:4],
    }


def get_current_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Authentication required.")

    token = authorization.split(" ", 1)[1].strip()
    session = db_store.get_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Session is invalid or expired.")

    user = db_store.get_user_by_id(session["userId"])
    if not user:
        db_store.delete_session(token)
        raise HTTPException(status_code=401, detail="User account no longer exists.")

    db_store.touch_session(token, now_iso())
    return user


def get_run_dir(run_id: str) -> Path:
    return EVAL_OUTPUTS_DIR / run_id


def list_run_dirs() -> list[Path]:
    if not EVAL_OUTPUTS_DIR.exists():
        return []
    return sorted([path for path in EVAL_OUTPUTS_DIR.iterdir() if path.is_dir()], reverse=True)


def load_run_manifest(run_dir: Path) -> dict[str, Any]:
    payload = load_json(run_dir / RUN_MANIFEST_FILE, {})
    return payload if isinstance(payload, dict) else {}


def run_dir_for_user(run_id: str, user_id: str) -> Path:
    run_entry = db_store.get_eval_run(run_id, user_id)
    if not run_entry:
        raise HTTPException(status_code=404, detail="Run not found.")
    run_dir = Path(run_entry["runDir"])
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found for this account.")
    return run_dir


def list_run_dirs_for_user(user_id: str) -> list[Path]:
    visible_runs: list[Path] = []
    for run in db_store.list_eval_runs(user_id):
        run_dir = Path(run["runDir"])
        if run_dir.exists():
            visible_runs.append(run_dir)
    return visible_runs


def find_scored_dir(run_dir: Path) -> Path:
    scored_dir = run_dir / "scored"
    if scored_dir.exists():
        return scored_dir
    return run_dir


def summary_from_run(run_dir: Path) -> dict[str, Any]:
    scored_dir = find_scored_dir(run_dir)
    summary_path = scored_dir / "comparison_summary.json"
    summary_json = load_json(summary_path, {})
    manifest = load_run_manifest(run_dir)
    systems = []
    for system_name, metrics in (summary_json.get("systems") or {}).items():
        systems.append(
            {
                "name": system_name,
                "label": system_name.replace("_", " "),
                "metrics": metrics,
            }
        )
    return {
        "runId": run_dir.name,
        "runName": manifest.get("runName", run_dir.name),
        "benchmarkSize": summary_json.get("benchmark_size", 0),
        "systems": systems,
    }


def list_bundles(user_id: str) -> list[dict[str, Any]]:
    bundles: list[dict[str, Any]] = []
    for bundle in db_store.list_bundles(user_id):
        archive = Path(bundle["archivePath"])
        bundles.append(
            {
                "id": bundle["id"],
                "runName": bundle["runName"],
                "executionMode": bundle["executionMode"],
                "baseModel": bundle.get("baseModel"),
                "peftMethod": bundle.get("peftMethod", "qlora"),
                "loraRank": bundle.get("loraRank", 16),
                "qaDatasetId": bundle.get("qaDatasetId"),
                "instructionDatasetId": bundle.get("instructionDatasetId"),
                "configMode": bundle.get("configMode"),
                "createdAt": bundle.get("createdAt", now_iso()),
                "archiveSizeMb": file_size_mb(archive),
                "commands": bundle.get("commands", {"local": [], "colab": []}),
                "downloadUrl": f"/api/run-bundles/{bundle['id']}/download",
            }
        )
    return bundles


def bundle_by_id(bundle_id: str, user_id: str) -> tuple[dict[str, Any], Path]:
    bundle_row = db_store.get_bundle(bundle_id, user_id)
    if not bundle_row:
        raise HTTPException(status_code=404, detail="Run bundle not found for this account.")
    bundle_dir = Path(bundle_row["bundleDir"])
    bundle = {
        "id": bundle_row["id"],
        "runName": bundle_row["runName"],
        "executionMode": bundle_row["executionMode"],
        "baseModel": bundle_row.get("baseModel"),
        "peftMethod": bundle_row.get("peftMethod", "qlora"),
        "loraRank": bundle_row.get("loraRank", 16),
        "qaDatasetId": bundle_row.get("qaDatasetId"),
        "instructionDatasetId": bundle_row.get("instructionDatasetId"),
        "configMode": bundle_row.get("configMode"),
        "createdAt": bundle_row.get("createdAt", now_iso()),
        "commands": bundle_row.get("commands", {"local": [], "colab": []}),
        "downloadUrl": f"/api/run-bundles/{bundle_id}/download",
        "archiveSizeMb": file_size_mb(bundle_dir.with_suffix(".zip")),
    }
    return bundle, bundle_dir


def options_payload() -> dict[str, Any]:
    return {
        "models": [
            {
                "id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "name": "Llama 3.1 8B",
                "provider": "Meta",
                "params": "8B params",
                "vramHint": "16GB VRAM with QLoRA",
            },
        ],
        "peftMethods": [
            {"id": "qlora", "label": "QLoRA (4-bit)"},
            {"id": "lora", "label": "LoRA"},
        ],
        "loraRanks": [8, 16, 32],
        "executionModes": [
            {"id": "local", "label": "Local", "description": "Run directly on the user machine with DALTP installed."},
            {"id": "colab", "label": "Colab-assisted", "description": "Prepare a bundle for a Google Colab GPU session."},
        ],
        "configPresets": [
            {
                "id": "balanced-qlora",
                "label": "Balanced QLoRA",
                "description": "Default DALTP preset tuned for practical Colab runs.",
                "manualDefaults": {
                    "learningRate": 0.0001,
                    "epochs": 3,
                    "batchSize": 1,
                    "gradientAccumulationSteps": 8,
                    "maxLength": 2048,
                    "dtype": "bfloat16",
                    "loadIn4bit": True,
                },
            },
            {
                "id": "fast-iteration",
                "label": "Fast iteration",
                "description": "Shorter run for quick workflow validation.",
                "manualDefaults": {
                    "learningRate": 0.00015,
                    "epochs": 1,
                    "batchSize": 1,
                    "gradientAccumulationSteps": 4,
                    "maxLength": 1024,
                    "dtype": "bfloat16",
                    "loadIn4bit": True,
                },
            },
        ],
    }


def preset_by_id(preset_id: str | None) -> dict[str, Any] | None:
    for preset in options_payload()["configPresets"]:
        if preset["id"] == preset_id:
            return preset
    return None


def build_dashboard_payload(user: dict[str, Any]) -> dict[str, Any]:
    run_dirs = list_run_dirs_for_user(user["id"])
    recent_runs = []
    best_bert = 0.0
    for run_dir in run_dirs:
        summary = summary_from_run(run_dir)
        manifest = load_run_manifest(run_dir)
        systems = summary["systems"]
        best_system = max(systems, key=lambda system: system["metrics"].get("BERTScore F1", 0), default=None)
        if best_system:
            best_bert = max(best_bert, best_system["metrics"].get("BERTScore F1", 0))
        recent_runs.append(
            {
                "id": run_dir.name,
                "name": manifest.get("runName", run_dir.name),
                "model": "Llama 3.1 8B",
                "status": "Done" if systems else "Pending",
                "progress": 100 if systems else 0,
                "metric": f"{best_system['metrics'].get('ROUGE-L', 0):.4f}" if best_system else "--",
                "metricLabel": f"best ROUGE-L ({best_system['name'].replace('_', ' ')})" if best_system else "awaiting scored outputs",
                "benchmarkSize": summary.get("benchmarkSize", 0),
                "updatedAt": datetime.fromtimestamp(run_dir.stat().st_mtime, timezone.utc).isoformat(),
            }
        )

    workspace = workspace_summary_for_user(user)
    return {
        "stats": [
            {"label": "Evaluation runs", "value": len(run_dirs), "subtext": "saved DALTP output runs", "tone": "green"},
            {"label": "Best BERTScore F1", "value": f"{best_bert:.4f}" if best_bert else "0.0000", "subtext": "across scored runs", "tone": "green"},
            {"label": "My datasets", "value": len(dataset_catalog(user)), "subtext": f"{workspace['uploadedDatasetCount']} uploaded by you", "tone": "green"},
            {"label": "Prepared bundles", "value": workspace["bundleCount"], "subtext": "local + Colab launch packages", "tone": "amber"},
        ],
        "recentRuns": recent_runs[:6],
        "defaultRunId": recent_runs[0]["id"] if recent_runs else None,
        "adapterSizeMb": adapter_size_mb(),
        "datasetCount": len(dataset_catalog(user)),
        "workspace": workspace,
    }


def create_uploaded_dataset(payload: DatasetUploadPayload, user: dict[str, Any]) -> dict[str, Any]:
    dataset_storage.require_supabase_dataset_storage()
    dataset_name = payload.name.strip()
    if dataset_name_exists(user, dataset_name):
        raise HTTPException(status_code=400, detail=f"A dataset named '{dataset_name}' already exists in this account.")
    validate_uploaded_dataset_files(payload.files)

    owner_dir = DATASETS_DIR / user["id"]
    owner_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = f"user-{slugify(dataset_name)}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    dataset_dir = owner_dir / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    source_dir = dataset_dir / "source_files"
    source_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    combined_chunks: list[str] = []
    first_extension = ".txt"
    for index, upload in enumerate(payload.files):
        file_name = upload.name or f"{payload.kind}-{index}.txt"
        relative_path = normalize_relative_path(upload.relativePath, file_name)
        target_path = source_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(upload_bytes(upload))
        content = upload_text(upload).rstrip()
        if content:
            combined_chunks.append(content)
        saved_files.append(
            {
                "name": file_name,
                "relativePath": relative_path.as_posix(),
                "size": upload.size or len(upload_bytes(upload)),
                "mimeType": upload.mimeType,
            }
        )

        extension = Path(file_name).suffix.lower()
        if index == 0 and extension in {".jsonl", ".json", ".csv", ".txt"}:
            first_extension = extension

    stored_file_name = f"dataset{first_extension or '.txt'}"
    combined_path = dataset_dir / stored_file_name
    combined_payload = "\n".join(chunk for chunk in combined_chunks if chunk)
    if combined_payload and not combined_payload.endswith("\n"):
        combined_payload += "\n"
    combined_path.write_text(combined_payload, encoding="utf-8")

    manifest = {
        "id": dataset_id,
        "name": dataset_name,
        "kind": payload.kind,
        "description": f"User-uploaded {payload.kind} dataset.",
        "source": "uploaded",
        "ownerId": user["id"],
        "storedFileName": stored_file_name,
        "createdAt": now_iso(),
        "files": saved_files,
        "vectorStore": None,
    }
    write_dataset_manifest(dataset_dir, manifest)
    combined_line_count = line_count(combined_path)
    combined_size_mb = file_size_mb(combined_path)
    storage_metadata = dataset_storage.upload_dataset_artifact(
        combined_path,
        user_id=user["id"],
        dataset_id=dataset_id,
        stored_file_name=stored_file_name,
        mime_type="text/plain",
    )
    manifest.update(storage_metadata)
    write_dataset_manifest(dataset_dir, manifest)
    db_store.upsert_dataset(
        {
            **manifest,
            "path": storage_metadata["path"],
            "datasetDir": str(dataset_dir),
            "lineCount": combined_line_count,
            "sizeMb": combined_size_mb,
            "generator": None,
        }
    )
    combined_path.unlink(missing_ok=True)

    return {
        "id": dataset_id,
        "name": manifest["name"],
        "kind": payload.kind,
        "description": manifest["description"],
        "source": "uploaded",
        "ownerId": user["id"],
        "path": storage_metadata["path"],
        "lineCount": combined_line_count,
        "sizeMb": combined_size_mb,
        "createdAt": manifest["createdAt"],
        "files": saved_files,
        "vectorStore": manifest["vectorStore"],
        "storageProvider": manifest.get("storageProvider", "supabase"),
        "storageBucket": manifest.get("storageBucket", ""),
    }


def create_source_documents(files: list[DatasetUploadFile], target_dir: Path, fallback_prefix: str) -> list[dict[str, Any]]:
    source_dir = target_dir / "source_files"
    source_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for index, upload in enumerate(files):
        file_name = upload.name or f"{fallback_prefix}-{index}.txt"
        relative_path = normalize_relative_path(upload.relativePath, file_name)
        target_path = source_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(upload_bytes(upload))
        saved_files.append(
            {
                "name": file_name,
                "relativePath": relative_path.as_posix(),
                "size": upload.size or len(upload_bytes(upload)),
                "mimeType": upload.mimeType,
            }
        )

    return saved_files


def create_generated_dataset(payload: DatasetGeneratePayload, user: dict[str, Any]) -> dict[str, Any]:
    dataset_storage.require_supabase_dataset_storage()
    dataset_name = payload.name.strip()
    if dataset_name_exists(user, dataset_name):
        raise HTTPException(status_code=400, detail=f"A dataset named '{dataset_name}' already exists in this account.")

    owner_dir = DATASETS_DIR / user["id"]
    owner_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = f"user-{slugify(dataset_name)}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    dataset_dir = owner_dir / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    saved_files = create_source_documents(payload.files, dataset_dir, payload.kind)
    source_root = dataset_dir / "source_files"
    output_path = dataset_dir / "dataset.jsonl"
    vector_store_payload = None

    if payload.kind == "corpus":
        raw_data.build_raw_dataset(
            input_path=str(source_root),
            output_path=str(output_path),
            chunk_size=payload.chunkSize,
            chunk_overlap=payload.chunkOverlap,
        )
        generator_settings = {
            "chunkSize": payload.chunkSize,
            "chunkOverlap": payload.chunkOverlap,
        }
        if payload.ingestToPgvector:
            target_collection = (payload.collectionName or default_collection_name(dataset_name, user)).strip()
            try:
                vector_store_payload = ingest_dataset_source_to_vector_store(
                    dataset_dir=dataset_dir,
                    dataset_name=dataset_name,
                    user=user,
                    collection_name=target_collection,
                    chunk_size=payload.chunkSize,
                    chunk_overlap=payload.chunkOverlap,
                )
            except HTTPException as exc:
                vector_store_payload = {
                    "ingested": False,
                    "collectionName": target_collection,
                    "chunkSize": payload.chunkSize,
                    "chunkOverlap": payload.chunkOverlap,
                    "error": exc.detail if isinstance(exc.detail, str) else str(exc.detail),
                    "lastAttemptedAt": now_iso(),
                    "datasetName": dataset_name,
                    "backend": "pgvector",
                }
    elif payload.kind == "qa":
        try:
            QApair.build_qa_dataset(
                input_path=str(source_root),
                output_path=str(output_path),
                model_name=payload.modelName,
                api_base=payload.apiBase,
                num_pairs=payload.qaNumPairs,
                chunk_size=payload.qaChunkSize,
                chunk_overlap=payload.qaChunkOverlap,
                max_contexts_per_document=payload.qaMaxContextsPerDocument,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=clarify_generation_error(exc, payload.apiBase, "qa"),
            ) from exc
        generator_settings = {
            "modelName": payload.modelName,
            "apiBase": payload.apiBase,
            "numPairs": payload.qaNumPairs,
            "chunkSize": payload.qaChunkSize,
            "chunkOverlap": payload.qaChunkOverlap,
            "maxContextsPerDocument": payload.qaMaxContextsPerDocument,
        }
    else:
        try:
            Instruction_set.build_instruction_dataset(
                input_path=str(source_root),
                output_path=str(output_path),
                model_name=payload.modelName,
                api_base=payload.apiBase,
                context_size=payload.instructionContextSize,
                context_overlap=payload.instructionContextOverlap,
                max_contexts_per_document=payload.instructionMaxContextsPerDocument,
                task_types=payload.taskTypes,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=clarify_generation_error(exc, payload.apiBase, "instruction"),
            ) from exc
        generator_settings = {
            "modelName": payload.modelName,
            "apiBase": payload.apiBase,
            "contextSize": payload.instructionContextSize,
            "contextOverlap": payload.instructionContextOverlap,
            "maxContextsPerDocument": payload.instructionMaxContextsPerDocument,
            "taskTypes": payload.taskTypes or [],
        }

    manifest = {
        "id": dataset_id,
        "name": dataset_name,
        "kind": payload.kind,
        "description": f"Generated {payload.kind} dataset from uploaded source documents.",
        "source": "generated",
        "ownerId": user["id"],
        "storedFileName": output_path.name,
        "createdAt": now_iso(),
        "files": saved_files,
        "generator": {
            "mode": "generated",
            "settings": generator_settings,
        },
        "vectorStore": vector_store_payload,
    }
    write_dataset_manifest(dataset_dir, manifest)
    output_line_count = line_count(output_path)
    output_size_mb = file_size_mb(output_path)
    storage_metadata = dataset_storage.upload_dataset_artifact(
        output_path,
        user_id=user["id"],
        dataset_id=dataset_id,
        stored_file_name=output_path.name,
        mime_type="application/json",
    )
    manifest.update(storage_metadata)
    write_dataset_manifest(dataset_dir, manifest)
    db_store.upsert_dataset(
        {
            **manifest,
            "path": storage_metadata["path"],
            "datasetDir": str(dataset_dir),
            "lineCount": output_line_count,
            "sizeMb": output_size_mb,
            "generator": manifest["generator"],
        }
    )
    output_path.unlink(missing_ok=True)

    return {
        "id": dataset_id,
        "name": manifest["name"],
        "kind": payload.kind,
        "description": manifest["description"],
        "source": "generated",
        "ownerId": user["id"],
        "path": storage_metadata["path"],
        "lineCount": output_line_count,
        "sizeMb": output_size_mb,
        "createdAt": manifest["createdAt"],
        "files": saved_files,
        "vectorStore": manifest["vectorStore"],
        "storageProvider": manifest.get("storageProvider", "supabase"),
        "storageBucket": manifest.get("storageBucket", ""),
    }


def ingest_dataset_to_vector_store(dataset_id: str, payload: DatasetVectorIngestPayload, user: dict[str, Any]) -> dict[str, Any]:
    dataset = dataset_by_id(dataset_id, user)
    if dataset["kind"] != "corpus":
        raise HTTPException(status_code=400, detail="Only corpus datasets can be ingested into pgvector storage.")

    dataset_dir = uploaded_dataset_dir(dataset_id, user)
    manifest = load_json(dataset_dir / "manifest.json", {})
    manifest["vectorStore"] = ingest_dataset_source_to_vector_store(
        dataset_dir=dataset_dir,
        dataset_name=dataset["name"],
        user=user,
        collection_name=payload.collectionName.strip(),
        chunk_size=payload.chunkSize,
        chunk_overlap=payload.chunkOverlap,
    )
    write_dataset_manifest(dataset_dir, manifest)
    db_store.upsert_dataset(
        {
            **manifest,
            "path": dataset["path"],
            "datasetDir": str(dataset_dir),
            "lineCount": dataset.get("lineCount"),
            "sizeMb": dataset.get("sizeMb"),
            "storageProvider": dataset.get("storageProvider", "supabase"),
            "storageBucket": dataset.get("storageBucket", ""),
            "generator": manifest.get("generator"),
        }
    )
    return dataset_by_id(dataset_id, user)


def copy_dataset_into_bundle(dataset: dict[str, Any], datasets_dir: Path) -> dict[str, Any]:
    stored_file_name = dataset.get("storedFileName") or Path(dataset["path"]).name
    suffix = Path(stored_file_name).suffix or ".jsonl"
    bundle_file_name = f"{dataset['kind']}_{slugify(dataset['name'])}_{dataset['id']}{suffix}"
    target_path = datasets_dir / bundle_file_name
    dataset_storage.copy_dataset_artifact_to_path(dataset, target_path)
    return {
        "id": dataset["id"],
        "name": dataset["name"],
        "kind": dataset["kind"],
        "source": dataset["source"],
        "relativePath": f"datasets/{bundle_file_name}",
        "lineCount": dataset.get("lineCount", 0),
        "sizeMb": dataset.get("sizeMb", 0),
    }


def deep_merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_config_from_payload(payload: RunBundlePayload, copied_datasets: list[dict[str, Any]]) -> dict[str, Any]:
    base = default_training_config()
    if not base:
        base = {"training": {}, "model": {}, "datasets": {}, "quantization": {}, "lora": {}}

    config = json.loads(json.dumps(base))
    config.setdefault("model", {})
    config.setdefault("training", {})
    config.setdefault("datasets", {})
    config.setdefault("quantization", {})
    config.setdefault("lora", {})

    config["model"]["name"] = payload.baseModel
    config["lora"]["r"] = payload.loraRank
    config["training"]["output_dir"] = training_output_dir_for_execution(payload.runName, payload.executionMode)

    dataset_mapping = {entry["kind"]: entry["relativePath"] for entry in copied_datasets}
    if "qa" in dataset_mapping:
        config["datasets"]["qa"] = dataset_mapping["qa"]
    if "instruction" in dataset_mapping:
        config["datasets"]["instruction"] = dataset_mapping["instruction"]

    if payload.configMode == "preset":
        preset = preset_by_id(payload.presetId)
        if preset:
            defaults = preset["manualDefaults"]
            config["training"].update(
                {
                    "learning_rate": defaults["learningRate"],
                    "num_train_epochs": defaults["epochs"],
                    "per_device_train_batch_size": defaults["batchSize"],
                    "gradient_accumulation_steps": defaults["gradientAccumulationSteps"],
                    "max_length": defaults["maxLength"],
                    "dtype": defaults["dtype"],
                }
            )
            config["training"]["bf16"] = defaults["dtype"] == "bfloat16"
            config["training"]["fp16"] = defaults["dtype"] == "float16"
            config["quantization"]["load_in_4bit"] = defaults["loadIn4bit"] if payload.peftMethod == "qlora" else False
    elif payload.configMode == "manual" and payload.manualConfig:
        manual = payload.manualConfig.model_dump(exclude_none=True)
        if "learningRate" in manual:
            config["training"]["learning_rate"] = manual["learningRate"]
        if "epochs" in manual:
            config["training"]["num_train_epochs"] = manual["epochs"]
        if "batchSize" in manual:
            config["training"]["per_device_train_batch_size"] = manual["batchSize"]
        if "gradientAccumulationSteps" in manual:
            config["training"]["gradient_accumulation_steps"] = manual["gradientAccumulationSteps"]
        if "maxLength" in manual:
            config["training"]["max_length"] = manual["maxLength"]
        if "dtype" in manual:
            config["training"]["dtype"] = manual["dtype"]
            config["training"]["bf16"] = manual["dtype"] == "bfloat16"
            config["training"]["fp16"] = manual["dtype"] == "float16"
        if "loadIn4bit" in manual:
            config["quantization"]["load_in_4bit"] = manual["loadIn4bit"] if payload.peftMethod == "qlora" else False
    elif payload.configMode == "upload" and payload.uploadedConfigText:
        try:
            uploaded_config = json.loads(payload.uploadedConfigText)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Uploaded config is not valid JSON: {exc}") from exc
        if not isinstance(uploaded_config, dict):
            raise HTTPException(status_code=400, detail="Uploaded config must be a JSON object.")
        config = deep_merge_config(config, uploaded_config)

    if payload.peftMethod == "lora":
        config["quantization"]["load_in_4bit"] = False

    config.setdefault("model", {})
    config.setdefault("training", {})
    config.setdefault("datasets", {})
    config.setdefault("quantization", {})
    config.setdefault("lora", {})

    config["datasets"]["qa"] = dataset_mapping["qa"]
    config["datasets"]["instruction"] = dataset_mapping["instruction"]
    config["training"]["output_dir"] = training_output_dir_for_execution(payload.runName, payload.executionMode)
    config.setdefault("metadata", {})
    config["metadata"]["run_name"] = payload.runName
    config["metadata"]["selected_base_model"] = payload.baseModel
    config["metadata"]["selected_peft_method"] = payload.peftMethod
    config["metadata"]["selected_lora_rank"] = payload.loraRank

    return config


def create_bundle_manifest(payload: RunBundlePayload, user: dict[str, Any]) -> dict[str, Any]:
    run_name = payload.runName.strip()
    if bundle_run_name_exists(user["id"], run_name):
        raise HTTPException(status_code=400, detail=f"A prepared run named '{run_name}' already exists in this account.")

    owner_dir = BUNDLES_DIR / user["id"]
    owner_dir.mkdir(parents=True, exist_ok=True)

    bundle_id = f"{slugify(run_name)}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    bundle_dir = owner_dir / bundle_id
    datasets_dir = bundle_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    qa_dataset = dataset_by_id(payload.qaDatasetId, user)
    instruction_dataset = dataset_by_id(payload.instructionDatasetId, user)

    copied_datasets = [
        copy_dataset_into_bundle(qa_dataset, datasets_dir),
        copy_dataset_into_bundle(instruction_dataset, datasets_dir),
    ]

    config = build_config_from_payload(payload, copied_datasets)
    config_path = bundle_dir / "training_config.json"
    write_json(config_path, config)

    local_commands = [
        "pip install -r requirements.txt",
        "python -m backend.training.trainer --config training_config.json",
    ]
    colab_commands = [
        "pip install -r requirements.txt",
        "python -m backend.training.trainer --config /content/training_config.json",
    ]

    instructions = {
        "local": [
            "Place the extracted bundle in the DALTP repo root.",
            "Review training_config.json and confirm the dataset paths inside the datasets folder.",
            "Run the local command set from the bundle root or adapt the paths into your DALTP workspace.",
        ],
        "colab": [
            "Upload the extracted bundle into your Colab session or mount Drive.",
            "Copy training_config.json and the datasets folder into the DALTP repo or /content workspace.",
            "Run the Colab command set after installing the required dependencies.",
        ],
    }

    manifest = {
        "id": bundle_id,
        "runName": run_name,
        "executionMode": payload.executionMode,
        "baseModel": payload.baseModel,
        "peftMethod": payload.peftMethod,
        "loraRank": payload.loraRank,
        "qaDatasetId": payload.qaDatasetId,
        "instructionDatasetId": payload.instructionDatasetId,
        "configMode": payload.configMode,
        "presetId": payload.presetId,
        "ownerId": user["id"],
        "createdAt": now_iso(),
        "datasets": copied_datasets,
        "commands": {"local": local_commands, "colab": colab_commands},
        "instructions": instructions,
    }
    write_json(bundle_dir / "manifest.json", manifest)
    db_store.upsert_bundle(
        {
            **manifest,
            "bundleDir": str(bundle_dir),
            "archivePath": str(bundle_dir.with_suffix(".zip")),
        }
    )

    archive_path = bundle_dir.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in bundle_dir.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(bundle_dir))

    return {
        "id": bundle_id,
        "runName": manifest["runName"],
        "executionMode": manifest["executionMode"],
        "baseModel": manifest["baseModel"],
        "peftMethod": manifest["peftMethod"],
        "loraRank": manifest["loraRank"],
        "qaDatasetId": manifest["qaDatasetId"],
        "instructionDatasetId": manifest["instructionDatasetId"],
        "configMode": manifest["configMode"],
        "createdAt": manifest["createdAt"],
        "archiveSizeMb": file_size_mb(archive_path),
        "commands": manifest["commands"],
        "downloadUrl": f"/api/run-bundles/{bundle_id}/download",
    }


def start_dataset_generation_job(payload: DatasetGeneratePayload, user: dict[str, Any]) -> dict[str, Any]:
    job, path = create_job(
        user,
        "dataset-generation",
        f"Generate {payload.kind} dataset",
        {"datasetName": payload.name.strip(), "kind": payload.kind},
    )

    def runner(job_path: Path) -> None:
        update_job_manifest(job_path, status="running")
        append_job_log(job_path, f"Generating {payload.kind} dataset '{payload.name.strip()}'...\n")
        try:
            dataset = create_generated_dataset(payload, user)
            update_job_manifest(job_path, status="completed", result={"dataset": dataset})
            append_job_log(job_path, f"Completed dataset generation for {dataset['name']}.\n")
        except Exception as exc:
            append_job_log(job_path, traceback.format_exc() + "\n")
            update_job_manifest(job_path, status="failed", error=str(exc))

    start_background_job(path, runner)
    return job


def start_vector_ingestion_job(dataset_id: str, payload: DatasetVectorIngestPayload, user: dict[str, Any]) -> dict[str, Any]:
    dataset = dataset_by_id(dataset_id, user)
    job, path = create_job(
        user,
        "vector-ingestion",
        f"Ingest {dataset['name']} to pgvector",
        {"datasetId": dataset_id, "collectionName": payload.collectionName},
    )

    def runner(job_path: Path) -> None:
        update_job_manifest(job_path, status="running")
        append_job_log(job_path, f"Ingesting corpus dataset '{dataset['name']}' into pgvector namespace {payload.collectionName}...\n")
        try:
            refreshed = ingest_dataset_to_vector_store(dataset_id, payload, user)
            update_job_manifest(job_path, status="completed", result={"dataset": refreshed})
            append_job_log(job_path, f"Completed pgvector ingestion for {refreshed['name']}.\n")
        except Exception as exc:
            append_job_log(job_path, traceback.format_exc() + "\n")
            update_job_manifest(job_path, status="failed", error=str(exc))

    start_background_job(path, runner)
    return job


def start_local_training_job(bundle_id: str, user: dict[str, Any]) -> dict[str, Any]:
    bundle, bundle_dir = bundle_by_id(bundle_id, user["id"])
    if bundle["executionMode"] != "local":
        raise HTTPException(status_code=400, detail="Only local bundles can be launched directly from DALTP.")

    job, path = create_job(
        user,
        "training",
        f"Train {bundle['runName']}",
        {"bundleId": bundle_id, "runName": bundle["runName"]},
    )
    config_path = bundle_dir / "training_config.json"

    def runner(job_path: Path) -> None:
        update_job_manifest(job_path, status="running")
        append_job_log(job_path, f"Starting local training for bundle {bundle['runName']}...\n")
        try:
            run_logged_command(
                [sys.executable, "-m", "backend.training.trainer", "--config", str(config_path)],
                cwd=ROOT_DIR,
                path=job_path,
                step_label="Local training",
            )
            config_payload = load_json(config_path, {})
            output_dir = Path(config_payload.get("training", {}).get("output_dir", ""))
            if output_dir and not output_dir.is_absolute():
                output_dir = (config_path.parent / output_dir).resolve()
            artifact = register_local_model_artifact(bundle=bundle, user=user, output_dir=output_dir)
            update_job_manifest(job_path, status="completed", result={"bundleId": bundle_id, "model": artifact})
            append_job_log(job_path, f"Local training completed. Registered model artifact '{artifact['name']}'.\n")
        except Exception as exc:
            append_job_log(job_path, traceback.format_exc() + "\n")
            update_job_manifest(job_path, status="failed", error=str(exc))

    start_background_job(path, runner)
    return job


def start_evaluation_job(payload: EvaluationJobPayload, user: dict[str, Any]) -> dict[str, Any]:
    run_name = payload.runName.strip()
    selected_modes = []
    if payload.runBase:
        selected_modes.append("base")
    if payload.runRag:
        selected_modes.append("rag")
    if payload.runFineTuned:
        selected_modes.append("fine_tuned")
    if payload.runFineTunedRag:
        selected_modes.append("fine_tuned_rag")
    if not selected_modes:
        raise HTTPException(status_code=400, detail="Select at least one evaluation mode.")

    corpus_dataset = dataset_by_id(payload.corpusDatasetId, user) if payload.corpusDatasetId else None
    collection_name = corpus_dataset.get("vectorStore", {}).get("collectionName") if corpus_dataset else None
    if any(mode in {"rag", "fine_tuned_rag"} for mode in selected_modes) and not collection_name:
        raise HTTPException(status_code=400, detail="RAG evaluation needs a corpus dataset that has been ingested into pgvector storage.")
    if payload.benchmarkMode == "existing" and not payload.benchmarkDatasetId:
        raise HTTPException(status_code=400, detail="Choose an existing benchmark dataset.")
    if payload.benchmarkMode == "generate" and not payload.corpusDatasetId:
        raise HTTPException(status_code=400, detail="Choose a corpus dataset to generate a benchmark from.")
    if evaluation_run_name_exists(user["id"], run_name):
        raise HTTPException(status_code=400, detail=f"An evaluation run named '{run_name}' already exists.")

    run_id = f"{slugify(run_name)}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    run_dir = EVAL_OUTPUTS_DIR / run_id
    job, path = create_job(
        user,
        "evaluation",
        f"Evaluate {run_name}",
        {"runId": run_id, "modes": selected_modes, "benchmarkMode": payload.benchmarkMode},
    )

    def runner(job_path: Path) -> None:
        update_job_manifest(job_path, status="running")
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            run_dir / RUN_MANIFEST_FILE,
            {
                "id": run_id,
                "runName": run_name,
                "ownerId": user["id"],
                "createdAt": now_iso(),
                "createdBy": user["email"],
                "type": "evaluation",
            },
        )
        db_store.upsert_eval_run(
            run_id=run_id,
            owner_id=user["id"],
            run_name=run_name,
            created_at=now_iso(),
            created_by=user["email"],
            run_dir=str(run_dir),
        )
        benchmark_path = run_dir / "benchmark.jsonl"
        qa_output_path = run_dir / "held_out_qa_dataset.jsonl"
        prediction_files: dict[str, Path] = {}
        try:
            if payload.benchmarkMode == "existing":
                benchmark_dataset = dataset_by_id(payload.benchmarkDatasetId or "", user)
                shutil.copy2(benchmark_dataset["path"], benchmark_path)
                append_job_log(job_path, f"Using existing benchmark dataset {benchmark_dataset['name']}.\n")
            else:
                corpus_dir = uploaded_dataset_dir(payload.corpusDatasetId or "", user)
                source_root = corpus_dir / "source_files"
                append_job_log(job_path, "Generating held-out benchmark from source documents...\n")
                run_logged_command(
                    [
                        sys.executable,
                        "-m",
                        "backend.evaluation.generate_test_benchmark",
                        "--input",
                        str(source_root),
                        "--qa-output",
                        str(qa_output_path),
                        "--benchmark-output",
                        str(benchmark_path),
                        "--qa-generation-model",
                        payload.qaGenerationModel,
                        "--qa-api-base",
                        payload.qaApiBase,
                        "--num-pairs",
                        str(payload.numPairs),
                        "--chunk-size",
                        str(payload.chunkSize),
                        "--chunk-overlap",
                        str(payload.chunkOverlap),
                    ],
                    cwd=ROOT_DIR,
                    path=job_path,
                    step_label="Benchmark generation",
                )

            for mode in selected_modes:
                output_path = run_dir / f"{mode}_predictions.jsonl"
                command = [
                    sys.executable,
                    "-m",
                    "backend.evaluation.generate_predictions",
                    "--benchmark",
                    str(benchmark_path),
                    "--output",
                    str(output_path),
                    "--system-name",
                    f"{mode}_model",
                    "--mode",
                    mode,
                    "--base-model",
                    "meta-llama/Meta-Llama-3.1-8B-Instruct",
                    "--top-k",
                    str(payload.topK),
                    "--max-new-tokens",
                    str(payload.maxNewTokens),
                    "--temperature",
                    str(payload.temperature),
                    "--load-in-4bit",
                ]
                if mode in {"rag", "fine_tuned_rag"}:
                    command.extend(["--collection", collection_name])
                if mode in {"fine_tuned", "fine_tuned_rag"}:
                    command.extend(["--adapter-path", str(BACKEND_DIR / "training" / "outputs" / "llama3_8b_sft")])
                append_job_log(job_path, f"Running prediction mode {mode}...\n")
                run_logged_command(command, cwd=ROOT_DIR, path=job_path, step_label=f"Prediction generation ({mode})")
                prediction_files[mode] = output_path

            compare_command = [
                sys.executable,
                "-m",
                "backend.evaluation.compare_model_outputs",
                "--benchmark",
                str(benchmark_path),
                "--output-dir",
                str(run_dir / "scored"),
            ]
            mode_flag_map = {
                "base": "--base-predictions",
                "rag": "--rag-predictions",
                "fine_tuned": "--fine-tuned-predictions",
                "fine_tuned_rag": "--fine-tuned-rag-predictions",
            }
            for mode_name, path_value in prediction_files.items():
                compare_command.extend([mode_flag_map[mode_name], str(path_value)])

            append_job_log(job_path, "Scoring model outputs...\n")
            run_logged_command(compare_command, cwd=ROOT_DIR, path=job_path, step_label="Evaluation scoring")
            update_job_manifest(
                job_path,
                status="completed",
                result={"runId": run_id, "outputDir": str(run_dir), "collectionName": collection_name},
            )
            append_job_log(job_path, f"Evaluation completed for run {run_id}.\n")
        except Exception as exc:
            append_job_log(job_path, traceback.format_exc() + "\n")
            update_job_manifest(job_path, status="failed", error=str(exc))

    start_background_job(path, runner)
    return job


@app.on_event("startup")
def startup() -> None:
    ensure_runtime_dirs()
    db_store.initialize_metadata_store()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/auth/register")
def register(payload: RegisterPayload) -> dict[str, Any]:
    if "@" not in payload.email:
        raise HTTPException(status_code=400, detail="Please provide a valid email address.")
    existing = db_store.get_user_by_email(payload.email.lower())
    if existing:
        raise HTTPException(status_code=409, detail="An account with that email already exists.")

    password_hash, password_salt = hash_password(payload.password)
    user = {
        "id": f"user-{secrets.token_hex(8)}",
        "name": payload.name.strip(),
        "email": payload.email.lower(),
        "passwordHash": password_hash,
        "passwordSalt": password_salt,
        "createdAt": now_iso(),
    }
    db_store.upsert_user(user)

    token = create_session_token(user["id"])
    return {
        "token": token,
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "createdAt": user["createdAt"],
            "workspace": workspace_summary_for_user(user),
        },
    }


@app.post("/api/auth/login")
def login(payload: LoginPayload) -> dict[str, Any]:
    if "@" not in payload.email:
        raise HTTPException(status_code=400, detail="Please provide a valid email address.")
    user = db_store.get_user_by_email(payload.email.lower())
    if not user or not verify_password(payload.password, user["passwordHash"], user["passwordSalt"]):
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")

    token = create_session_token(user["id"])
    return {
        "token": token,
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "createdAt": user["createdAt"],
            "workspace": workspace_summary_for_user(user),
        },
    }


@app.get("/api/auth/me")
def auth_me(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return {
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "createdAt": user["createdAt"],
            "workspace": workspace_summary_for_user(user),
        }
    }


@app.post("/api/auth/logout")
def logout(user: dict[str, Any] = Depends(get_current_user), authorization: str | None = Header(default=None)) -> dict[str, str]:
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
        db_store.delete_session(token)
    return {"status": "logged_out"}


@app.get("/api/platform/options")
def get_platform_options(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return options_payload()


@app.get("/api/dashboard")
def get_dashboard(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return build_dashboard_payload(user)


@app.get("/api/runs")
def get_runs(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    runs = []
    for run_dir in list_run_dirs_for_user(user["id"]):
        scored_dir = find_scored_dir(run_dir)
        manifest = load_run_manifest(run_dir)
        runs.append(
            {
                "id": run_dir.name,
                "name": manifest.get("runName", run_dir.name),
                "updatedAt": datetime.fromtimestamp(run_dir.stat().st_mtime, timezone.utc).isoformat(),
                "hasScoredOutputs": (scored_dir / "comparison_summary.json").exists(),
            }
        )
    return {"runs": runs}


@app.get("/api/runs/{run_id}/summary")
def get_run_summary(run_id: str, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    run_dir = run_dir_for_user(run_id, user["id"])
    return summary_from_run(run_dir)


@app.get("/api/datasets")
def get_datasets(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return {"datasets": dataset_catalog(user)}


@app.get("/api/models")
def get_models(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return {"models": list_model_artifacts(user["id"])}


@app.post("/api/models/import")
def import_model(payload: ModelImportPayload, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    artifact = import_model_artifact(payload, user)
    return {"model": artifact}


@app.get("/api/models/{model_id}/download")
def download_model(model_id: str, user: dict[str, Any] = Depends(get_current_user)) -> FileResponse:
    artifact, model_dir = model_artifact_by_id(model_id, user["id"])
    archive = model_dir.with_suffix(".zip")
    if not archive.exists():
        raise HTTPException(status_code=404, detail="Model archive is missing.")
    return FileResponse(path=archive, filename=f"{slugify(artifact['name'])}.zip", media_type="application/zip")


@app.delete("/api/models/{model_id}")
def remove_model(model_id: str, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, str]:
    delete_model_artifact(model_id, user)
    return {"status": "deleted"}


@app.post("/api/datasets/upload")
def upload_dataset(payload: DatasetUploadPayload, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    dataset = create_uploaded_dataset(payload, user)
    return {"dataset": dataset}


@app.post("/api/datasets/generate")
def generate_dataset(payload: DatasetGeneratePayload, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    job = start_dataset_generation_job(payload, user)
    return {"job": job}


@app.post("/api/datasets/{dataset_id}/ingest-pgvector")
def ingest_dataset_collection(
    dataset_id: str,
    payload: DatasetVectorIngestPayload,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    job = start_vector_ingestion_job(dataset_id, payload, user)
    return {"job": job}


@app.get("/api/datasets/{dataset_id}/download")
def download_dataset(dataset_id: str, user: dict[str, Any] = Depends(get_current_user)) -> Response:
    dataset = dataset_by_id(dataset_id, user)
    stored_file_name = dataset.get("storedFileName") or Path(dataset["path"]).name
    payload = dataset_storage.download_dataset_artifact_bytes(dataset)
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{stored_file_name}"'},
    )


@app.delete("/api/datasets/{dataset_id}")
def remove_dataset(dataset_id: str, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, str]:
    delete_uploaded_dataset(dataset_id, user)
    return {"status": "deleted"}


@app.get("/api/run-bundles")
def get_run_bundles(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return {"bundles": list_bundles(user["id"])}


@app.post("/api/run-bundles")
def create_run_bundle(payload: RunBundlePayload, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    bundle = create_bundle_manifest(payload, user)
    return {"bundle": bundle}


@app.get("/api/run-bundles/{bundle_id}")
def get_run_bundle(bundle_id: str, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    bundle, _bundle_dir = bundle_by_id(bundle_id, user["id"])
    return {"bundle": bundle}


@app.post("/api/run-bundles/{bundle_id}/launch-local")
def launch_local_bundle(bundle_id: str, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    job = start_local_training_job(bundle_id, user)
    return {"job": job}


@app.get("/api/run-bundles/{bundle_id}/download")
def download_run_bundle(bundle_id: str, user: dict[str, Any] = Depends(get_current_user)) -> FileResponse:
    _bundle, bundle_dir = bundle_by_id(bundle_id, user["id"])
    archive = bundle_dir.with_suffix(".zip")
    if not archive.exists():
        raise HTTPException(status_code=404, detail="Bundle archive is missing.")
    return FileResponse(path=archive, filename=archive.name, media_type="application/zip")


@app.delete("/api/run-bundles/{bundle_id}")
def remove_run_bundle(bundle_id: str, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, str]:
    delete_run_bundle(bundle_id, user)
    return {"status": "deleted"}


@app.get("/api/local-commands")
def get_local_commands(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, str]:
    return {"markdown": safe_read_text(LOCAL_COMMANDS_PATH)}


@app.get("/api/jobs")
def get_jobs(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return {"jobs": list_jobs(user["id"])}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return {"job": get_job_manifest(job_id, user)}


@app.post("/api/evaluation/jobs")
def create_evaluation_job(payload: EvaluationJobPayload, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    job = start_evaluation_job(payload, user)
    return {"job": job}
