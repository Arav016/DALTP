from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.api import bundle_storage, db_store, dataset_storage, evaluation_storage, model_storage
from backend.dataset.dataset_construction import Instruction_set, QApair, raw_data
from backend.dataset.ingestion import data_ingestion
from backend.evaluation import generate_test_benchmark
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field


API_DIR = Path(__file__).resolve().parent
RUN_MANIFEST_FILE = "daltp_run.json"

BACKEND_DIR = API_DIR.parent
ROOT_DIR = BACKEND_DIR.parent
TRAINING_CONFIG_PATH = BACKEND_DIR / "training" / "configs" / "lora_llama3_8b_instruct.json"
DEFAULT_GENERATION_API_BASE = QApair.default_generation_api_base()
DEFAULT_GENERATION_MODEL = QApair.default_generation_model()
DEFAULT_EVALUATION_OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL") or "meta-llama/llama-3.1-8b-instruct"
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
    kind: str = Field(pattern="^(qa|instruction|benchmark|corpus)$")
    files: list[DatasetUploadFile] = Field(default_factory=list)
    corpusDatasetId: str | None = None
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


class BenchmarkGenerationPayload(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    qaGenerationModel: str = DEFAULT_GENERATION_MODEL
    qaApiBase: str = DEFAULT_GENERATION_API_BASE
    numPairs: int = 10
    chunkSize: int = 5000
    chunkOverlap: int = 200
    maxContextsPerDocument: int | None = None


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
    executionMode: str = Field(default="colab", pattern="^colab$")
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
    source: str = Field(default="imported", pattern="^imported$")
    baseModel: str = Field(min_length=2, max_length=200)
    peftMethod: str = Field(pattern="^(qlora|lora)$")
    loraRank: int | None = None
    files: list[DatasetUploadFile] = Field(min_length=1)


class EvaluationJobPayload(BaseModel):
    runName: str = Field(min_length=2, max_length=120)
    benchmarkMode: str = Field(default="existing", pattern="^(existing|generate)$")
    benchmarkDatasetId: str | None = None
    corpusDatasetId: str | None = None
    modelId: str | None = None
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


def cors_allowed_origins() -> list[str]:
    configured = os.getenv("DALTP_CORS_ORIGINS") or ""
    origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    return origins or ["http://localhost:5173", "http://127.0.0.1:5173"]


app = FastAPI(title="DALTP API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clarify_generation_error(exc: Exception, api_base: str, kind: str) -> str:
    message = str(exc)
    if "404" in message and "Not Found" in message:
        return f"{kind.upper()} generation could not reach the AI generation service. Please try again in a few minutes."
    if "Connection error" in message or "actively refused it" in message or "ConnectError" in message:
        return f"{kind.upper()} generation could not complete because the AI generation service was unavailable. Please try again later."
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


def training_output_dir_for_execution(run_name: str) -> str:
    run_slug = slugify(run_name)
    return f"/content/daltp_outputs/{run_slug}"


def default_evaluation_openrouter_model() -> str:
    return os.getenv("OPENROUTER_MODEL") or DEFAULT_EVALUATION_OPENROUTER_MODEL


def default_evaluation_openrouter_api_base() -> str:
    return os.getenv("OPENROUTER_API_BASE") or DEFAULT_GENERATION_API_BASE


def modal_evaluation_endpoint() -> str:
    return os.getenv("MODAL_EVAL_ENDPOINT") or ""


def openrouter_eval_api_key_configured() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY"))


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


def remove_tree_if_exists(path: Path) -> None:
    if path.exists() and path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


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


def list_uploaded_datasets(user_id: str) -> list[dict[str, Any]]:
    datasets = db_store.list_datasets(user_id)
    for dataset in datasets:
        if dataset.get("lineCount") is not None and dataset.get("sizeMb") is not None:
            continue
        payload = dataset_storage.download_dataset_artifact_bytes(dataset)
        text = payload.decode("utf-8", errors="ignore")
        dataset["lineCount"] = sum(1 for line in text.splitlines() if line.strip())
        dataset["sizeMb"] = round(len(payload) / (1024 * 1024), 2)
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
        normalized_name(load_run_manifest(run).get("runName", run.get("runName", ""))) == target
        for run in list_runs_for_user(user_id)
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
        raise HTTPException(
            status_code=400,
            detail="DALTP could not prepare this corpus dataset for RAG search. Please make sure it contains readable text and try again.",
        ) from exc

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
    db_store.delete_dataset(dataset_id, user["id"])


def delete_run_bundle(bundle_id: str, user: dict[str, Any]) -> None:
    bundle_row = db_store.get_bundle(bundle_id, user["id"])
    if not bundle_row:
        raise HTTPException(status_code=404, detail="Run bundle not found for this account.")

    if bundle_row.get("storageProvider") == "supabase":
        bundle_storage.delete_bundle_archive(bundle_row)
    db_store.delete_bundle(bundle_id, user["id"])


def delete_model_artifact(model_id: str, user: dict[str, Any]) -> None:
    model = db_store.get_model(model_id, user["id"])
    if not model:
        raise HTTPException(status_code=404, detail="Model artifact not found for this account.")

    if (model.get("storageProvider") or "azure_blob") == "azure_blob":
        model_storage.delete_model_archive(model)
    db_store.delete_model(model_id, user["id"])


def write_model_manifest(model_dir: Path, manifest: dict[str, Any]) -> None:
    write_json(model_dir / "manifest.json", manifest)


def list_model_artifacts(user_id: str) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for model in db_store.list_models(user_id):
        archive_size_mb = model.get("archiveSizeMb")
        if archive_size_mb is None:
            total_size = sum(int(file.get("size") or 0) for file in model.get("files", []))
            archive_size_mb = round(total_size / (1024 * 1024), 2)
        artifacts.append(
            {
                "id": model["id"],
                "name": model["name"],
                "source": model["source"],
                "baseModel": model.get("baseModel"),
                "peftMethod": model.get("peftMethod", "qlora"),
                "loraRank": model.get("loraRank"),
                "runName": model.get("runName"),
                "storageProvider": model.get("storageProvider", "azure_blob"),
                "storageBucket": model.get("storageBucket", ""),
                "archivePath": model.get("archivePath"),
                "ownerId": user_id,
                "createdAt": model.get("createdAt", now_iso()),
                "fileCount": len(model.get("files", [])),
                "archiveSizeMb": archive_size_mb,
                "downloadUrl": f"/api/models/{model['id']}/download",
            }
        )
    return artifacts


def model_artifact_by_id(model_id: str, user_id: str) -> dict[str, Any]:
    model = db_store.get_model(model_id, user_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model artifact not found for this account.")
    archive_size_mb = model.get("archiveSizeMb")
    if archive_size_mb is None:
        total_size = sum(int(file.get("size") or 0) for file in model.get("files", []))
        archive_size_mb = round(total_size / (1024 * 1024), 2)
    artifact = {
        "id": model["id"],
        "name": model["name"],
        "source": model["source"],
        "baseModel": model.get("baseModel"),
        "peftMethod": model.get("peftMethod", "qlora"),
        "loraRank": model.get("loraRank"),
        "runName": model.get("runName"),
        "storageProvider": model.get("storageProvider", "azure_blob"),
        "storageBucket": model.get("storageBucket", ""),
        "archivePath": model.get("archivePath"),
        "modelDir": model.get("modelDir"),
        "ownerId": user_id,
        "createdAt": model.get("createdAt", now_iso()),
        "fileCount": len(model.get("files", [])),
        "downloadUrl": f"/api/models/{model_id}/download",
        "archiveSizeMb": archive_size_mb,
    }
    return artifact


def model_artifact_name_exists(user_id: str, model_name: str) -> bool:
    target = normalized_name(model_name)
    return any(normalized_name(model.get("name", "")) == target for model in list_model_artifacts(user_id))


def build_directory_archive(source_dir: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(source_dir))


def persist_evaluation_run_artifact(
    *,
    run_id: str,
    user: dict[str, Any],
    run_name: str,
    run_dir: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any] | None,
    created_at: str,
) -> dict[str, Any]:
    evaluation_storage.require_supabase_evaluation_storage()
    archive_path = run_dir.with_suffix(".zip")
    build_directory_archive(run_dir, archive_path)
    storage_metadata = evaluation_storage.upload_evaluation_archive(
        archive_path,
        user_id=user["id"],
        run_id=run_id,
    )
    db_store.upsert_eval_run(
        run_id=run_id,
        owner_id=user["id"],
        run_name=run_name,
        created_at=created_at,
        updated_at=now_iso(),
        created_by=user["email"],
        run_dir="",
        storage_provider=storage_metadata["storageProvider"],
        storage_bucket=storage_metadata["storageBucket"],
        archive_path=storage_metadata["archivePath"],
        manifest=manifest,
        summary=summary,
    )
    archive_path.unlink(missing_ok=True)
    remove_tree_if_exists(run_dir)
    return storage_metadata


def create_model_import_workspace(
    *,
    user: dict[str, Any],
    model_name: str,
    base_model: str,
    peft_method: str,
    lora_rank: int | None,
) -> tuple[str, Path, Path, dict[str, Any]]:
    model_name = model_name.strip()
    if model_artifact_name_exists(user["id"], model_name):
        raise HTTPException(status_code=400, detail=f"A model artifact named '{model_name}' already exists in this account.")

    artifact_id = f"model-{slugify(model_name)}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    model_dir = Path(tempfile.mkdtemp(prefix=f"daltp_models_{user['id']}_")) / artifact_id
    model_dir.mkdir(parents=True, exist_ok=True)
    files_dir = model_dir / "artifact_files"
    files_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "id": artifact_id,
        "name": model_name,
        "source": "imported",
        "baseModel": base_model.strip(),
        "peftMethod": peft_method,
        "loraRank": lora_rank,
        "ownerId": user["id"],
        "createdAt": now_iso(),
        "runName": None,
    }
    return artifact_id, model_dir, files_dir, manifest


def finalize_model_import(model_dir: Path, manifest: dict[str, Any], saved_files: list[dict[str, Any]], user_id: str) -> dict[str, Any]:
    manifest["files"] = saved_files
    write_model_manifest(model_dir, manifest)
    archive_path = model_dir.with_suffix(".zip")
    build_directory_archive(model_dir / "artifact_files", archive_path)
    storage_metadata = model_storage.upload_model_archive(
        archive_path,
        user_id=user_id,
        model_id=manifest["id"],
    )
    archive_size_mb = round(archive_path.stat().st_size / (1024 * 1024), 2)
    db_store.upsert_model(
        {
            **manifest,
            "storageProvider": storage_metadata["storageProvider"],
            "storageBucket": storage_metadata["storageBucket"],
            "modelDir": "",
            "archivePath": storage_metadata["archivePath"],
            "archiveSizeMb": archive_size_mb,
        }
    )
    archive_path.unlink(missing_ok=True)
    remove_tree_if_exists(model_dir.parent)
    artifact = model_artifact_by_id(manifest["id"], user_id)
    return artifact


def extract_zip_to_model_files(archive_source: Any, files_dir: Path) -> list[dict[str, Any]]:
    saved_files: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(archive_source) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                relative_path = normalize_relative_path(member.filename, f"artifact-{len(saved_files)}")
                target_path = files_dir / relative_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(archive.read(member))
                saved_files.append(
                    {
                        "name": Path(relative_path).name,
                        "relativePath": relative_path.as_posix(),
                        "size": member.file_size,
                        "mimeType": None,
                    }
                )
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="Uploaded model archive is not a valid zip file.") from exc
    return saved_files


def import_model_artifact(payload: ModelImportPayload, user: dict[str, Any]) -> dict[str, Any]:
    model_name = payload.name.strip()
    artifact_id, model_dir, files_dir, manifest = create_model_import_workspace(
        user=user,
        model_name=model_name,
        base_model=payload.baseModel,
        peft_method=payload.peftMethod,
        lora_rank=payload.loraRank,
    )

    saved_files: list[dict[str, Any]] = []
    if len(payload.files) == 1 and Path(payload.files[0].name or "").suffix.lower() == ".zip":
        archive_upload = payload.files[0]
        archive_bytes = upload_bytes(archive_upload)
        saved_files = extract_zip_to_model_files(io.BytesIO(archive_bytes), files_dir)
    else:
        for index, upload in enumerate(payload.files):
            file_name = upload.name or f"artifact-{index}"
            relative_path = normalize_relative_path(upload.relativePath, file_name)
            target_path = files_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            file_bytes = upload_bytes(upload)
            target_path.write_bytes(file_bytes)
            saved_files.append(
                {
                    "name": file_name,
                    "relativePath": relative_path.as_posix(),
                    "size": upload.size or len(file_bytes),
                    "mimeType": upload.mimeType,
                }
            )

    return finalize_model_import(model_dir, manifest, saved_files, user["id"])


async def import_model_archive_upload(
    *,
    name: str,
    base_model: str,
    peft_method: str,
    lora_rank: int | None,
    archive: UploadFile,
    user: dict[str, Any],
) -> dict[str, Any]:
    artifact_id, model_dir, files_dir, manifest = create_model_import_workspace(
        user=user,
        model_name=name,
        base_model=base_model,
        peft_method=peft_method,
        lora_rank=lora_rank,
    )
    saved_files = extract_zip_to_model_files(archive.file, files_dir)
    return finalize_model_import(model_dir, manifest, saved_files, user["id"])


def list_jobs(user_id: str) -> list[dict[str, Any]]:
    return [job_for_client(job) for job in db_store.list_jobs(user_id)]


def sanitize_job_log_for_display(text: str) -> str:
    sanitized = str(text or "")
    if not sanitized:
        return ""

    # Job logs are useful, but the UI should not expose machine-local paths,
    # signed URL query strings, or token-shaped secrets.
    sanitized = re.sub(r"[A-Za-z]:\\[^\s\"']+", "[local path]", sanitized)
    sanitized = re.sub(r"(?<!:)//[^\s\"']+", "//[redacted]", sanitized)
    sanitized = re.sub(r"\b(?:hf|sk|rk)_[A-Za-z0-9]{12,}\b", "[redacted token]", sanitized)
    sanitized = re.sub(r"Bearer\s+[A-Za-z0-9._-]+", "Bearer [redacted]", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"(https?://[^\s\"'?]+)\?[^\s\"']+", r"\1?[redacted]", sanitized)
    sanitized = re.sub(r"(?m)^Traceback \(most recent call last\):(?:\n\s+File .*)+", "Technical details were hidden for safety.", sanitized)
    return sanitized.strip()


def job_for_client(job: dict[str, Any]) -> dict[str, Any]:
    return {
        **job,
        "logs": sanitize_job_log_for_display(job.get("logs", "")),
        "logPath": "",
        "jobDir": "",
    }


def job_handle(job_id: str, owner_id: str) -> dict[str, str]:
    return {"id": job_id, "ownerId": owner_id}


def job_id_from_ref(job_ref: dict[str, str] | str) -> str:
    return job_ref["id"] if isinstance(job_ref, dict) else str(job_ref)


def append_job_log(job_ref: dict[str, str], text: str) -> None:
    db_store.append_job_log(job_ref["id"], job_ref["ownerId"], sanitize_job_log_for_display(text) + ("\n" if text.endswith("\n") else ""), now_iso())


def read_job_log(job_ref: dict[str, str], max_lines: int = 120) -> str:
    job = db_store.get_job(job_ref["id"], job_ref["ownerId"])
    if not job:
        return ""
    lines = str(job.get("logs") or "").splitlines()
    return "\n".join(lines[-max_lines:])


def update_job_manifest(job_ref: dict[str, str], **changes: Any) -> dict[str, Any]:
    manifest = db_store.get_job(job_ref["id"], job_ref["ownerId"])
    if not manifest:
        raise RuntimeError(f"Job {job_ref['id']} no longer exists.")
    manifest.update(changes)
    manifest["updatedAt"] = now_iso()
    db_store.upsert_job(manifest)
    return manifest


def create_job(user: dict[str, Any], job_type: str, title: str, metadata: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, str]]:
    job_id = f"{job_type}-{datetime.now().strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(4)}"
    manifest = {
        "id": job_id,
        "type": job_type,
        "title": title,
        "status": "queued",
        "ownerId": user["id"],
        "createdAt": now_iso(),
        "updatedAt": now_iso(),
        "metadata": metadata or {},
        "logs": "",
        "logPath": f"postgres://daltp_jobs/{job_id}/logs",
        "jobDir": f"postgres://daltp_jobs/{job_id}",
    }
    db_store.upsert_job(manifest)
    return manifest, job_handle(job_id, user["id"])


def get_job_manifest(job_id: str, user: dict[str, Any]) -> dict[str, Any]:
    manifest = db_store.get_job(job_id, user["id"])
    if not manifest:
        raise HTTPException(status_code=404, detail="Job not found for this account.")
    return manifest


def delete_job_record(job_id: str, user: dict[str, Any]) -> None:
    manifest = db_store.get_job(job_id, user["id"])
    if not manifest:
        raise HTTPException(status_code=404, detail="Job not found for this account.")
    if manifest.get("status") in {"queued", "running"}:
        raise HTTPException(status_code=400, detail="Running jobs cannot be removed yet.")
    db_store.delete_job(job_id, user["id"])


def start_background_job(job_ref: dict[str, str], runner) -> None:
    thread = threading.Thread(target=runner, args=(job_ref,), daemon=True)
    thread.start()


def run_logged_command(command: list[str], *, cwd: Path, path: dict[str, str], step_label: str) -> None:
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
            cleaned_line = line.strip()
            progress_prefixes = (
                "Generated ",
                "Scoring ",
                "Running BERTScore",
                "Finished BERTScore",
                "Wrote ",
            )
            if cleaned_line.startswith(progress_prefixes) or "Traceback" in cleaned_line or "Error" in cleaned_line:
                append_job_log(path, line)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{step_label} failed with exit code {return_code}.")


def sanitize_user_visible_error(text: str) -> str:
    sanitized = str(text or "").strip()
    if not sanitized:
        return "The operation failed."

    # Avoid surfacing machine-specific filesystem details directly in user-facing copy.
    sanitized = re.sub(r"[A-Za-z]:\\[^\s\"]+", "[local path]", sanitized)
    sanitized = re.sub(r"/(?:[^/\s]+/)+[^/\s]+", "[local path]", sanitized)

    # Avoid accidentally surfacing token-looking strings.
    sanitized = re.sub(r"\b(?:hf|sk|rk)_[A-Za-z0-9]{12,}\b", "[redacted token]", sanitized)
    sanitized = re.sub(r"Bearer\s+[A-Za-z0-9._-]+", "Bearer [redacted]", sanitized, flags=re.IGNORECASE)

    return " ".join(sanitized.split())


def summarize_evaluation_failure(exc: Exception, job_path: dict[str, str]) -> str:
    raw_message = str(exc)
    log_excerpt = read_job_log(job_path, max_lines=80)
    combined = f"{raw_message}\n{log_excerpt}".lower()

    if "modal evaluation endpoint" in combined or "could not reach the modal evaluation endpoint" in combined:
        return (
            "DALTP could not reach the service that runs fine-tuned model predictions. "
            "Please try again in a few minutes. If this keeps happening, the fine-tuned model service may need to be reconnected."
        )

    if "modal response" in combined:
        return (
            "The fine-tuned model service returned an answer DALTP could not read. "
            "Please try again. If it happens again, re-import the model artifact before running evaluation."
        )

    if "openrouter" in combined and ("401" in combined or "403" in combined or "api key" in combined or "authentication" in combined):
        return (
            "DALTP could not access the base model prediction service. "
            "Please ask the workspace owner to check the model access settings, then try again."
        )

    if "openrouter" in combined and ("429" in combined or "rate limit" in combined):
        return (
            "The base model prediction service is temporarily rate-limiting requests. "
            "Please wait a little and try again, or run fewer model types at once."
        )

    if "gated repo" in combined or ("401" in combined and "huggingface" in combined) or "repositorynotfounderror" in combined:
        return (
            "DALTP could not load the base model needed for fine-tuned evaluation. "
            "Please ask the workspace owner to confirm that the model is available to this workspace."
        )

    if "not a valid model identifier" in combined or "could not find" in combined and "hugging face" in combined:
        return (
            "DALTP could not find the base model linked to this model artifact. "
            "Please re-import the model artifact with the correct base model selected."
        )

    if "cuda out of memory" in combined or "outofmemoryerror" in combined or "not enough memory" in combined:
        return (
            "Evaluation ran out of memory while running the selected model. "
            "Try fewer model types, use a smaller benchmark dataset, or try again later."
        )

    if "bitsandbytes" in combined or "4-bit" in combined:
        if "cuda" in combined or "gpu" in combined or "compiled without gpu support" in combined:
            return (
                "DALTP could not start the fine-tuned model in 4-bit mode. "
                "Please try again later or re-import the model artifact."
            )

    if "adapter" in combined and ("does not exist" in combined or "not found" in combined):
        return (
            "DALTP could not load the selected fine-tuned model files. "
            "Please re-import that model artifact and run evaluation again."
        )

    if "benchmark generation failed" in combined or "held-out benchmark" in combined:
        return (
            "DALTP could not create a benchmark from the selected corpus dataset. "
            "Please choose a corpus dataset with enough readable text, or use an existing benchmark dataset."
        )

    if "prediction generation (base)" in combined:
        return (
            "DALTP could not finish generating answers for the base model. "
            "Please try again later, or run evaluation with fewer benchmark samples."
        )

    if "prediction generation (rag)" in combined:
        return (
            "DALTP could not finish generating answers for the RAG model. "
            "Please make sure the selected corpus dataset has been ingested for search, then try again."
        )

    if "prediction generation (fine_tuned" in combined:
        return (
            "DALTP could not finish generating answers for the fine-tuned model. "
            "Please re-check the selected model artifact, then try again."
        )

    if "evaluation scoring failed" in combined:
        return (
            "DALTP generated the model answers but could not calculate the final scores. "
            "Please try again with a smaller benchmark dataset. If it still fails, one or more benchmark answers may be missing or too large to score."
        )

    return sanitize_user_visible_error(raw_message)


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


def load_run_manifest(run: dict[str, Any]) -> dict[str, Any]:
    payload = run.get("manifest") or {}
    return payload if isinstance(payload, dict) else {}


def run_record_for_user(run_id: str, user_id: str) -> dict[str, Any]:
    run_entry = db_store.get_eval_run(run_id, user_id)
    if not run_entry:
        raise HTTPException(status_code=404, detail="Run not found.")
    return run_entry


def is_supabase_evaluation_run(run: dict[str, Any]) -> bool:
    return run.get("storageProvider") == "supabase" and bool(run.get("storageBucket")) and bool(run.get("archivePath"))


def list_runs_for_user(user_id: str) -> list[dict[str, Any]]:
    return [run for run in db_store.list_eval_runs(user_id) if is_supabase_evaluation_run(run)]


def summary_from_run(run: dict[str, Any]) -> dict[str, Any]:
    summary_json = run.get("summary") or {}
    manifest = load_run_manifest(run)
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
        "runId": run["id"],
        "runName": manifest.get("runName", run["runName"]),
        "benchmarkSize": summary_json.get("benchmark_size", 0),
        "createdAt": run.get("createdAt"),
        "updatedAt": run.get("updatedAt"),
        "hasScoredOutputs": bool(systems),
        "systems": systems,
    }


def ensure_scored_evaluation_run(run: dict[str, Any]) -> None:
    if not is_supabase_evaluation_run(run):
        raise HTTPException(status_code=404, detail="This evaluation report is not available for download yet.")
    if summary_from_run(run).get("hasScoredOutputs"):
        return
    raise HTTPException(status_code=404, detail="Evaluation report is not available until scoring completes successfully.")


def delete_evaluation_run(run_id: str, user: dict[str, Any]) -> None:
    run = run_record_for_user(run_id, user["id"])
    evaluation_storage.delete_evaluation_archive(run)
    db_store.delete_eval_run(run_id, user["id"])


def list_bundles(user_id: str) -> list[dict[str, Any]]:
    bundles: list[dict[str, Any]] = []
    for bundle in db_store.list_bundles(user_id):
        archive = Path(bundle["archivePath"] or "")
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
                "archiveSizeMb": file_size_mb(archive) if archive.exists() else 0,
                "commands": bundle.get("commands", {"colab": []}),
                "downloadUrl": f"/api/run-bundles/{bundle['id']}/download",
            }
        )
    return bundles


def bundle_by_id(bundle_id: str, user_id: str) -> tuple[dict[str, Any], Path]:
    bundle_row = db_store.get_bundle(bundle_id, user_id)
    if not bundle_row:
        raise HTTPException(status_code=404, detail="Run bundle not found for this account.")
    bundle_dir_value = bundle_row.get("bundleDir") or ""
    bundle_dir = Path(bundle_dir_value) if bundle_dir_value else Path()
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
        "commands": bundle_row.get("commands", {"colab": []}),
        "downloadUrl": f"/api/run-bundles/{bundle_id}/download",
        "storageProvider": bundle_row.get("storageProvider"),
        "storageBucket": bundle_row.get("storageBucket", ""),
        "archivePath": bundle_row.get("archivePath"),
        "archiveSizeMb": file_size_mb(bundle_dir.with_suffix(".zip")) if bundle_dir_value else 0,
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
    runs = list_runs_for_user(user["id"])
    scored_runs = []
    recent_runs = []
    best_bert = 0.0
    for run in runs:
        summary = summary_from_run(run)
        manifest = load_run_manifest(run)
        systems = summary["systems"]
        if not systems:
            continue
        scored_runs.append(run)
        best_system = max(systems, key=lambda system: system["metrics"].get("BERTScore F1", 0), default=None)
        if best_system:
            best_bert = max(best_bert, best_system["metrics"].get("BERTScore F1", 0))
        recent_runs.append(
            {
                "id": run["id"],
                "name": manifest.get("runName", run["runName"]),
                "model": "Llama 3.1 8B",
                "status": "Done" if systems else "Pending",
                "progress": 100 if systems else 0,
                "metric": f"{best_system['metrics'].get('ROUGE-L', 0):.4f}" if best_system else "--",
                "metricLabel": f"best ROUGE-L ({best_system['name'].replace('_', ' ')})" if best_system else "awaiting scored outputs",
                "benchmarkSize": summary.get("benchmarkSize", 0),
                "updatedAt": run.get("updatedAt") or run.get("createdAt"),
            }
        )

    workspace = workspace_summary_for_user(user)
    return {
        "stats": [
            {"label": "Scored runs", "value": len(scored_runs), "subtext": "completed evaluation reports", "tone": "green"},
            {"label": "Best BERTScore F1", "value": f"{best_bert:.4f}" if best_bert else "0.0000", "subtext": "across scored runs", "tone": "green"},
            {"label": "My datasets", "value": len(dataset_catalog(user)), "subtext": f"{workspace['uploadedDatasetCount']} uploaded by you", "tone": "green"},
            {"label": "Prepared bundles", "value": workspace["bundleCount"], "subtext": "Colab launch packages", "tone": "amber"},
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

    dataset_id = f"user-{slugify(dataset_name)}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    with tempfile.TemporaryDirectory(prefix="daltp_dataset_") as temp_dir:
        dataset_dir = Path(temp_dir) / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []
        combined_chunks: list[str] = []
        first_extension = ".txt"
        for index, upload in enumerate(payload.files):
            file_name = upload.name or f"{payload.kind}-{index}.txt"
            relative_path = normalize_relative_path(upload.relativePath, file_name)
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
    db_store.upsert_dataset(
        {
            **manifest,
            "path": storage_metadata["path"],
            "datasetDir": "",
            "lineCount": combined_line_count,
            "sizeMb": combined_size_mb,
            "generator": None,
        }
    )

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
    if payload.kind == "benchmark":
        if not payload.corpusDatasetId:
            raise HTTPException(status_code=400, detail="Choose a corpus dataset before generating a benchmark dataset.")
        return create_benchmark_dataset_from_corpus(
            payload.corpusDatasetId,
            BenchmarkGenerationPayload(
                name=payload.name,
                qaGenerationModel=payload.modelName,
                qaApiBase=payload.apiBase,
                numPairs=payload.qaNumPairs,
                chunkSize=payload.qaChunkSize,
                chunkOverlap=payload.qaChunkOverlap,
                maxContextsPerDocument=payload.qaMaxContextsPerDocument,
            ),
            user,
        )

    dataset_name = payload.name.strip()
    if dataset_name_exists(user, dataset_name):
        raise HTTPException(status_code=400, detail=f"A dataset named '{dataset_name}' already exists in this account.")
    if not payload.files:
        raise HTTPException(status_code=400, detail="Choose source documents before generating a dataset.")

    owner_dir = Path(tempfile.mkdtemp(prefix=f"daltp_datasets_{user['id']}_"))
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
    elif payload.kind == "benchmark":
        held_out_qa_path = dataset_dir / "held_out_qa_dataset.jsonl"
        try:
            generate_test_benchmark.generate_test_benchmark(
                input_path=str(source_root),
                qa_dataset_path=str(held_out_qa_path),
                benchmark_path=str(output_path),
                qa_generation_model=payload.modelName,
                qa_api_base=payload.apiBase,
                num_pairs=payload.qaNumPairs,
                chunk_size=payload.qaChunkSize,
                chunk_overlap=payload.qaChunkOverlap,
                max_contexts_per_document=payload.qaMaxContextsPerDocument,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=clarify_generation_error(exc, payload.apiBase, "benchmark"),
            ) from exc
        held_out_qa_path.unlink(missing_ok=True)
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
            "datasetDir": "",
            "lineCount": output_line_count,
            "sizeMb": output_size_mb,
            "generator": manifest["generator"],
        }
    )
    remove_tree_if_exists(owner_dir)

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


def create_benchmark_dataset_from_corpus(
    corpus_dataset_id: str,
    payload: BenchmarkGenerationPayload,
    user: dict[str, Any],
) -> dict[str, Any]:
    dataset_storage.require_supabase_dataset_storage()
    corpus_dataset = dataset_by_id(corpus_dataset_id, user)
    if corpus_dataset["kind"] != "corpus":
        raise HTTPException(status_code=400, detail="Only corpus datasets can be used to generate benchmark datasets.")

    dataset_name = payload.name.strip()
    if dataset_name_exists(user, dataset_name):
        raise HTTPException(status_code=400, detail=f"A dataset named '{dataset_name}' already exists in this account.")

    owner_dir = Path(tempfile.mkdtemp(prefix=f"daltp_datasets_{user['id']}_"))
    dataset_id = f"user-{slugify(dataset_name)}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    dataset_dir = owner_dir / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    corpus_jsonl_path = dataset_dir / "corpus_dataset.jsonl"
    qa_output_path = dataset_dir / "held_out_qa_dataset.jsonl"
    benchmark_output_path = dataset_dir / "benchmark.jsonl"
    dataset_storage.copy_dataset_artifact_to_path(corpus_dataset, corpus_jsonl_path)

    try:
        generate_test_benchmark.generate_test_benchmark(
            input_path=None,
            corpus_jsonl_path=str(corpus_jsonl_path),
            qa_dataset_path=str(qa_output_path),
            benchmark_path=str(benchmark_output_path),
            qa_generation_model=payload.qaGenerationModel,
            qa_api_base=payload.qaApiBase,
            num_pairs=payload.numPairs,
            chunk_size=payload.chunkSize,
            chunk_overlap=payload.chunkOverlap,
            max_contexts_per_document=payload.maxContextsPerDocument,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=clarify_generation_error(exc, payload.qaApiBase, "benchmark"),
        ) from exc

    saved_files = [
        {
            "name": corpus_dataset["name"],
            "datasetId": corpus_dataset["id"],
            "kind": corpus_dataset["kind"],
            "storageProvider": corpus_dataset.get("storageProvider", "supabase"),
        }
    ]
    manifest = {
        "id": dataset_id,
        "name": dataset_name,
        "kind": "benchmark",
        "description": f"Generated benchmark dataset from corpus dataset {corpus_dataset['name']}.",
        "source": "generated",
        "ownerId": user["id"],
        "storedFileName": benchmark_output_path.name,
        "createdAt": now_iso(),
        "files": saved_files,
        "vectorStore": None,
        "generator": {
            "mode": "benchmark_from_corpus",
            "sourceDatasetId": corpus_dataset["id"],
            "sourceDatasetName": corpus_dataset["name"],
            "settings": {
                "modelName": payload.qaGenerationModel,
                "apiBase": payload.qaApiBase,
                "numPairs": payload.numPairs,
                "chunkSize": payload.chunkSize,
                "chunkOverlap": payload.chunkOverlap,
                "maxContextsPerDocument": payload.maxContextsPerDocument,
            },
        },
    }
    write_dataset_manifest(dataset_dir, manifest)
    output_line_count = line_count(benchmark_output_path)
    output_size_mb = file_size_mb(benchmark_output_path)
    storage_metadata = dataset_storage.upload_dataset_artifact(
        benchmark_output_path,
        user_id=user["id"],
        dataset_id=dataset_id,
        stored_file_name=benchmark_output_path.name,
        mime_type="application/json",
    )
    manifest.update(storage_metadata)
    write_dataset_manifest(dataset_dir, manifest)
    db_store.upsert_dataset(
        {
            **manifest,
            "path": storage_metadata["path"],
            "datasetDir": "",
            "lineCount": output_line_count,
            "sizeMb": output_size_mb,
            "generator": manifest["generator"],
        }
    )

    remove_tree_if_exists(owner_dir)

    return {
        "id": dataset_id,
        "name": manifest["name"],
        "kind": manifest["kind"],
        "description": manifest["description"],
        "source": manifest["source"],
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
        raise HTTPException(status_code=400, detail="Only corpus datasets can be prepared for RAG search.")

    with tempfile.TemporaryDirectory(prefix="daltp_pgvector_") as temp_dir:
        corpus_path = Path(temp_dir) / "corpus.jsonl"
        dataset_storage.copy_dataset_artifact_to_path(dataset, corpus_path)
        ingest_summary = data_ingestion.ingest_jsonl_dataset(
            input_path=str(corpus_path),
            collection_name=payload.collectionName.strip(),
        )

    manifest = {
        "id": dataset["id"],
        "name": dataset["name"],
        "kind": dataset["kind"],
        "description": dataset.get("description", ""),
        "source": dataset.get("source", "generated"),
        "ownerId": user["id"],
        "storedFileName": dataset.get("storedFileName") or Path(dataset["path"]).name,
        "createdAt": dataset.get("createdAt", now_iso()),
        "files": dataset.get("files", []),
        "generator": dataset.get("generator"),
        "vectorStore": {
            "ingested": True,
            "collectionName": payload.collectionName.strip(),
            "chunkSize": payload.chunkSize,
            "chunkOverlap": payload.chunkOverlap,
            "documentsProcessed": ingest_summary.get("documents_processed", 0),
            "chunksCreated": ingest_summary.get("chunks_created", 0),
            "ingestedAt": now_iso(),
            "ingestedBy": user["id"],
            "datasetName": dataset["name"],
            "backend": "pgvector",
        },
    }
    db_store.upsert_dataset(
        {
            **manifest,
            "path": dataset["path"],
            "datasetDir": "",
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
    config["training"]["output_dir"] = training_output_dir_for_execution(payload.runName)

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
    config["training"]["output_dir"] = training_output_dir_for_execution(payload.runName)
    config.setdefault("metadata", {})
    config["metadata"]["run_name"] = payload.runName
    config["metadata"]["selected_base_model"] = payload.baseModel
    config["metadata"]["selected_peft_method"] = payload.peftMethod
    config["metadata"]["selected_lora_rank"] = payload.loraRank

    return config


def create_bundle_manifest(payload: RunBundlePayload, user: dict[str, Any]) -> dict[str, Any]:
    bundle_storage.require_supabase_bundle_storage()
    run_name = payload.runName.strip()
    if bundle_run_name_exists(user["id"], run_name):
        raise HTTPException(status_code=400, detail=f"A prepared run named '{run_name}' already exists in this account.")

    owner_dir = Path(tempfile.mkdtemp(prefix=f"daltp_bundles_{user['id']}_"))

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

    colab_commands = [
        "pip install -r requirements.txt",
        "python -m backend.training.trainer --config /content/training_config.json",
    ]

    instructions = {
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
        "commands": {"colab": colab_commands},
        "instructions": instructions,
    }
    write_json(bundle_dir / "manifest.json", manifest)
    archive_path = bundle_dir.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in bundle_dir.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(bundle_dir))
    storage_metadata = bundle_storage.upload_bundle_archive(
        archive_path,
        user_id=user["id"],
        bundle_id=bundle_id,
    )
    db_store.upsert_bundle(
        {
            **manifest,
            "bundleDir": "",
            "archivePath": storage_metadata["archivePath"],
            "storageProvider": storage_metadata["storageProvider"],
            "storageBucket": storage_metadata["storageBucket"],
        }
    )
    archive_size_mb = file_size_mb(archive_path)
    remove_tree_if_exists(owner_dir)

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
        "archiveSizeMb": archive_size_mb,
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
            update_job_manifest(job_path, status="failed", error=summarize_evaluation_failure(exc, job_path))

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
    selected_model = None
    if any(mode in {"fine_tuned", "fine_tuned_rag"} for mode in selected_modes):
        if not payload.modelId:
            raise HTTPException(status_code=400, detail="Choose a model artifact for fine-tuned evaluation.")
        selected_model = model_artifact_by_id(payload.modelId, user["id"])
        if (selected_model.get("storageProvider") or "azure_blob") != "azure_blob" or not selected_model.get("archivePath"):
            raise HTTPException(
                status_code=400,
                detail="The selected model artifact is missing the files needed for fine-tuned evaluation. Please re-import the model artifact and try again.",
            )
    elif payload.modelId:
        selected_model = model_artifact_by_id(payload.modelId, user["id"])

    if any(mode in {"rag", "fine_tuned_rag"} for mode in selected_modes) and not collection_name:
        raise HTTPException(status_code=400, detail="RAG evaluation needs a corpus dataset that has been prepared for search. Please ingest the corpus dataset first, then try again.")
    if not payload.benchmarkDatasetId:
        raise HTTPException(status_code=400, detail="Choose a benchmark dataset.")
    if evaluation_run_name_exists(user["id"], run_name):
        raise HTTPException(status_code=400, detail=f"An evaluation run named '{run_name}' already exists.")
    if any(mode in {"base", "rag"} for mode in selected_modes) and not openrouter_eval_api_key_configured():
        raise HTTPException(
            status_code=400,
            detail="Base and RAG evaluation are not available yet because the base model prediction service is not connected. Please ask the workspace owner to finish setup.",
        )
    if any(mode in {"fine_tuned", "fine_tuned_rag"} for mode in selected_modes) and not modal_evaluation_endpoint():
        raise HTTPException(
            status_code=400,
            detail="Fine-tuned evaluation is not available yet because the fine-tuned model service is not connected. Please ask the workspace owner to finish setup.",
        )

    run_id = f"{slugify(run_name)}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    run_dir = Path(tempfile.mkdtemp(prefix=f"daltp_eval_{user['id']}_")) / run_id
    created_at = now_iso()
    openrouter_eval_model = default_evaluation_openrouter_model()
    job, path = create_job(
        user,
        "evaluation",
        f"Evaluate {run_name}",
        {
            "runId": run_id,
            "modes": selected_modes,
            "benchmarkMode": "existing",
            "selectedModes": selected_modes,
            "completedModes": [],
            "currentStep": "queued",
            "totalModes": len(selected_modes),
            "benchmarkDatasetId": payload.benchmarkDatasetId,
            "corpusDatasetId": payload.corpusDatasetId,
            "modelId": selected_model["id"] if selected_model else None,
            "modelName": selected_model["name"] if selected_model else None,
            "providers": {
                "base": "openrouter" if payload.runBase else None,
                "rag": "openrouter" if payload.runRag else None,
                "fineTuned": "modal" if payload.runFineTuned else None,
                "fineTunedRag": "modal" if payload.runFineTunedRag else None,
            },
        },
    )

    def runner(job_path: Path) -> None:
        base_metadata = {
            "runId": run_id,
            "modes": selected_modes,
            "benchmarkMode": "existing",
            "selectedModes": selected_modes,
            "completedModes": [],
            "currentStep": "running",
            "totalModes": len(selected_modes),
            "benchmarkDatasetId": payload.benchmarkDatasetId,
            "corpusDatasetId": payload.corpusDatasetId,
            "modelId": selected_model["id"] if selected_model else None,
            "modelName": selected_model["name"] if selected_model else None,
            "providers": {
                "base": "openrouter" if payload.runBase else None,
                "rag": "openrouter" if payload.runRag else None,
                "fineTuned": "modal" if payload.runFineTuned else None,
                "fineTunedRag": "modal" if payload.runFineTunedRag else None,
            },
        }
        update_job_manifest(job_path, status="running", metadata=base_metadata)
        run_dir.mkdir(parents=True, exist_ok=True)
        run_manifest = {
            "id": run_id,
            "runName": run_name,
            "ownerId": user["id"],
            "createdAt": created_at,
            "createdBy": user["email"],
            "type": "evaluation",
            "modelId": selected_model["id"] if selected_model else None,
            "modelName": selected_model["name"] if selected_model else None,
            "providers": {
                "base": "openrouter" if "base" in selected_modes else None,
                "rag": "openrouter" if "rag" in selected_modes else None,
                "fineTuned": "modal" if "fine_tuned" in selected_modes else None,
                "fineTunedRag": "modal" if "fine_tuned_rag" in selected_modes else None,
            },
        }
        write_json(run_dir / RUN_MANIFEST_FILE, run_manifest)
        db_store.upsert_eval_run(
            run_id=run_id,
            owner_id=user["id"],
            run_name=run_name,
            created_at=created_at,
            updated_at=created_at,
            created_by=user["email"],
            run_dir=str(run_dir),
            manifest=run_manifest,
        )
        benchmark_path = run_dir / "benchmark.jsonl"
        qa_output_path = run_dir / "held_out_qa_dataset.jsonl"
        prediction_files: dict[str, Path] = {}
        base_model_name = (selected_model or {}).get("baseModel") or "meta-llama/Meta-Llama-3.1-8B-Instruct"
        try:
            adapter_url = None
            if selected_model and any(mode in {"fine_tuned", "fine_tuned_rag"} for mode in selected_modes):
                adapter_url = model_storage.create_model_archive_signed_url(selected_model)

            benchmark_dataset = dataset_by_id(payload.benchmarkDatasetId or "", user)
            dataset_storage.copy_dataset_artifact_to_path(benchmark_dataset, benchmark_path)
            append_job_log(job_path, f"Using existing benchmark dataset {benchmark_dataset['name']}.\n")

            for mode in selected_modes:
                base_metadata["currentStep"] = f"running_{mode}"
                update_job_manifest(job_path, metadata=base_metadata)
                output_path = run_dir / f"{mode}_predictions.jsonl"
                provider = "modal" if mode in {"fine_tuned", "fine_tuned_rag"} else "openrouter"
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
                    "--provider",
                    provider,
                    "--base-model",
                    base_model_name,
                    "--top-k",
                    str(payload.topK),
                    "--max-new-tokens",
                    str(payload.maxNewTokens),
                    "--temperature",
                    str(payload.temperature),
                ]
                if provider == "openrouter":
                    command.extend(
                        [
                            "--remote-model",
                            openrouter_eval_model,
                            "--api-base",
                            default_evaluation_openrouter_api_base(),
                        ]
                    )
                if mode in {"rag", "fine_tuned_rag"}:
                    command.extend(["--collection", collection_name])
                if mode in {"fine_tuned", "fine_tuned_rag"}:
                    command.extend(
                        [
                            "--model-id",
                            selected_model["id"],
                            "--model-artifact-name",
                            selected_model["name"],
                            "--adapter-url",
                            adapter_url,
                            "--adapter-cache-key",
                            selected_model["archivePath"],
                            "--peft-method",
                            selected_model.get("peftMethod") or "qlora",
                        ]
                    )
                    if selected_model.get("loraRank") is not None:
                        command.extend(["--lora-rank", str(selected_model["loraRank"])])
                append_job_log(job_path, f"Running prediction mode {mode}...\n")
                run_logged_command(command, cwd=ROOT_DIR, path=job_path, step_label=f"Prediction generation ({mode})")
                prediction_files[mode] = output_path
                base_metadata["completedModes"] = [*base_metadata["completedModes"], mode]
                base_metadata["currentStep"] = f"completed_{mode}"
                update_job_manifest(job_path, metadata=base_metadata)

            base_metadata["currentStep"] = "scoring"
            update_job_manifest(job_path, metadata=base_metadata)
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
            summary_payload = load_json(run_dir / "scored" / "comparison_summary.json", {})
            update_job_manifest(
                job_path,
                status="completed",
                metadata={**base_metadata, "currentStep": "completed"},
                result={"runId": run_id, "collectionName": collection_name},
            )
            persist_evaluation_run_artifact(
                run_id=run_id,
                user=user,
                run_name=run_name,
                run_dir=run_dir,
                manifest=run_manifest,
                summary=summary_payload if isinstance(summary_payload, dict) else None,
                created_at=created_at,
            )
            append_job_log(job_path, f"Evaluation completed for run {run_id}.\n")
        except Exception as exc:
            append_job_log(job_path, traceback.format_exc() + "\n")
            if run_dir.exists():
                try:
                    persist_evaluation_run_artifact(
                        run_id=run_id,
                        user=user,
                        run_name=run_name,
                        run_dir=run_dir,
                        manifest=run_manifest,
                        summary=None,
                        created_at=created_at,
                    )
                except Exception as persist_exc:
                    append_job_log(job_path, f"Failed to persist evaluation artifact: {persist_exc}\n")
            update_job_manifest(job_path, status="failed", error=summarize_evaluation_failure(exc, job_path))

    start_background_job(path, runner)
    return job


@app.on_event("startup")
def startup() -> None:
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
    for run in list_runs_for_user(user["id"]):
        summary = run.get("summary") or {}
        if not summary.get("systems"):
            continue
        manifest = load_run_manifest(run)
        runs.append(
            {
                "id": run["id"],
                "name": manifest.get("runName", run["runName"]),
                "updatedAt": run.get("updatedAt") or run.get("createdAt"),
                "hasScoredOutputs": True,
                "downloadUrl": f"/api/runs/{run['id']}/download",
            }
        )
    return {"runs": runs}


@app.get("/api/runs/{run_id}/summary")
def get_run_summary(run_id: str, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    run = run_record_for_user(run_id, user["id"])
    return summary_from_run(run)


@app.get("/api/runs/{run_id}/download")
def download_run_archive(run_id: str, user: dict[str, Any] = Depends(get_current_user)) -> Response:
    run = run_record_for_user(run_id, user["id"])
    ensure_scored_evaluation_run(run)
    payload = evaluation_storage.download_evaluation_archive_bytes(run)
    file_name = f"{slugify(load_run_manifest(run).get('runName', run['runName']))}.zip"
    return Response(
        content=payload,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{file_name}"'},
    )


@app.delete("/api/runs/{run_id}")
def remove_run(run_id: str, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, str]:
    delete_evaluation_run(run_id, user)
    return {"status": "deleted"}


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


@app.post("/api/models/import-archive")
async def import_model_archive(
    name: str = Form(...),
    source: str = Form(default="imported"),
    baseModel: str = Form(...),
    peftMethod: str = Form(...),
    loraRank: int | None = Form(default=None),
    archive: UploadFile = File(...),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    if Path(archive.filename or "").suffix.lower() != ".zip":
        raise HTTPException(status_code=400, detail="Upload a .zip archive for model import.")
    artifact = await import_model_archive_upload(
        name=name,
        base_model=baseModel,
        peft_method=peftMethod,
        lora_rank=loraRank,
        archive=archive,
        user=user,
    )
    return {"model": artifact}


@app.get("/api/models/{model_id}/download")
def download_model(model_id: str, user: dict[str, Any] = Depends(get_current_user)) -> Response:
    artifact = model_artifact_by_id(model_id, user["id"])
    if artifact.get("storageProvider") != "azure_blob":
        raise HTTPException(status_code=404, detail="This model artifact is not available for download.")
    payload = model_storage.download_model_archive_bytes(artifact)
    return Response(
        content=payload,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{slugify(artifact["name"])}.zip"'},
    )


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


@app.get("/api/run-bundles/{bundle_id}/download")
def download_run_bundle(bundle_id: str, user: dict[str, Any] = Depends(get_current_user)) -> Response:
    bundle, _bundle_dir = bundle_by_id(bundle_id, user["id"])
    if bundle.get("storageProvider") == "supabase":
        payload = bundle_storage.download_bundle_archive_bytes(bundle)
        file_name = f"{slugify(bundle['runName'])}.zip"
        return Response(
            content=payload,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{file_name}"'},
        )

    raise HTTPException(status_code=404, detail="This run bundle is not available for download.")


@app.delete("/api/run-bundles/{bundle_id}")
def remove_run_bundle(bundle_id: str, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, str]:
    delete_run_bundle(bundle_id, user)
    return {"status": "deleted"}


@app.get("/api/jobs")
def get_jobs(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return {"jobs": list_jobs(user["id"])}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return {"job": job_for_client(get_job_manifest(job_id, user))}


@app.delete("/api/jobs/{job_id}")
def remove_job(job_id: str, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, str]:
    delete_job_record(job_id, user)
    return {"status": "deleted"}


@app.post("/api/evaluation/jobs")
def create_evaluation_job(payload: EvaluationJobPayload, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    job = start_evaluation_job(payload, user)
    return {"job": job}
