from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from fastapi import HTTPException

load_dotenv()

def _normalize_supabase_url(value: str) -> str:
    cleaned = (value or "").strip().rstrip("/")
    for suffix in ("/rest/v1", "/storage/v1"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    return cleaned


SUPABASE_URL = _normalize_supabase_url(os.getenv("SUPABASE_URL") or "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or ""
SUPABASE_DATASETS_BUCKET = os.getenv("SUPABASE_DATASETS_BUCKET") or ""


def is_supabase_dataset_storage_enabled() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and SUPABASE_DATASETS_BUCKET)


def require_supabase_dataset_storage() -> None:
    if is_supabase_dataset_storage_enabled():
        return
    raise HTTPException(
        status_code=500,
        detail="Dataset storage is not available yet. Please ask the workspace owner to finish storage setup before creating datasets.",
    )


def dataset_storage_metadata(*, user_id: str, dataset_id: str, stored_file_name: str) -> dict[str, str]:
    object_key = f"{user_id}/{dataset_id}/{stored_file_name}"
    return {
        "storageProvider": "supabase",
        "storageBucket": SUPABASE_DATASETS_BUCKET,
        "path": object_key,
    }


def _storage_api_url(path: str) -> str:
    return f"{SUPABASE_URL}/storage/v1{path}"


def _supabase_headers(content_type: str | None = None) -> dict[str, str]:
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    }
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def _raise_storage_error(exc: Exception, action: str) -> None:
    if isinstance(exc, HTTPError):
        try:
            detail = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            detail = str(exc)
        raise HTTPException(status_code=500, detail=f"DALTP could not {action} the dataset file. Please try again later.") from exc
    if isinstance(exc, URLError):
        raise HTTPException(status_code=500, detail="DALTP could not reach dataset storage. Please try again later.") from exc
    raise HTTPException(status_code=500, detail=f"DALTP could not {action} the dataset file. Please try again later.") from exc


def upload_dataset_artifact(local_path: Path, *, user_id: str, dataset_id: str, stored_file_name: str, mime_type: str | None = None) -> dict[str, str]:
    require_supabase_dataset_storage()
    metadata = dataset_storage_metadata(user_id=user_id, dataset_id=dataset_id, stored_file_name=stored_file_name)
    object_key = metadata["path"]
    upload_url = _storage_api_url(f"/object/{quote(SUPABASE_DATASETS_BUCKET, safe='')}/{quote(object_key, safe='/')}")
    guessed_type = mime_type or mimetypes.guess_type(stored_file_name)[0] or "application/octet-stream"
    payload = local_path.read_bytes()
    request = Request(
        upload_url,
        data=payload,
        headers={**_supabase_headers(guessed_type), "x-upsert": "true"},
        method="POST",
    )
    try:
        with urlopen(request):
            return metadata
    except Exception as exc:  # pragma: no cover
        _raise_storage_error(exc, "upload")
    return metadata


def download_dataset_artifact_bytes(dataset: dict[str, Any]) -> bytes:
    require_supabase_dataset_storage()
    storage_provider = dataset.get("storageProvider", "supabase")
    if storage_provider != "supabase":
        raise HTTPException(
            status_code=500,
            detail=f"The file for dataset '{dataset['name']}' is not available. Please re-upload or regenerate the dataset.",
        )

    bucket = dataset.get("storageBucket") or SUPABASE_DATASETS_BUCKET
    object_key = dataset["path"]
    download_url = _storage_api_url(f"/object/authenticated/{quote(bucket, safe='')}/{quote(object_key, safe='/')}")
    request = Request(download_url, headers=_supabase_headers(), method="GET")
    try:
        with urlopen(request) as response:
            return response.read()
    except Exception as exc:  # pragma: no cover
        _raise_storage_error(exc, "download")
    return b""


def copy_dataset_artifact_to_path(dataset: dict[str, Any], target_path: Path) -> None:
    target_path.write_bytes(download_dataset_artifact_bytes(dataset))


def delete_dataset_artifact(dataset: dict[str, Any]) -> None:
    storage_provider = dataset.get("storageProvider", "supabase")
    if storage_provider != "supabase":
        return

    bucket = dataset.get("storageBucket") or SUPABASE_DATASETS_BUCKET
    object_key = dataset["path"]
    delete_url = _storage_api_url(f"/object/{quote(bucket, safe='')}")
    payload = json.dumps({"prefixes": [object_key]}).encode("utf-8")
    request = Request(delete_url, data=payload, headers=_supabase_headers("application/json"), method="DELETE")
    try:
        with urlopen(request):
            return
    except Exception as exc:  # pragma: no cover
        _raise_storage_error(exc, "delete")
