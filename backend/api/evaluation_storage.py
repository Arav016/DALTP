from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from fastapi import HTTPException

from backend.api.dataset_storage import _normalize_supabase_url

load_dotenv()

SUPABASE_URL = _normalize_supabase_url(os.getenv("SUPABASE_URL") or "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or ""
SUPABASE_EVALUATIONS_BUCKET = os.getenv("SUPABASE_EVALUATIONS_BUCKET") or ""


def is_supabase_evaluation_storage_enabled() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and SUPABASE_EVALUATIONS_BUCKET)


def require_supabase_evaluation_storage() -> None:
    if is_supabase_evaluation_storage_enabled():
        return
    raise HTTPException(
        status_code=500,
        detail="Evaluation report storage is not available yet. Please ask the workspace owner to finish storage setup before running evaluation.",
    )


def evaluation_storage_metadata(*, user_id: str, run_id: str, stored_file_name: str = "evaluation_run.zip") -> dict[str, str]:
    object_key = f"{user_id}/{run_id}/{stored_file_name}"
    return {
        "storageProvider": "supabase",
        "storageBucket": SUPABASE_EVALUATIONS_BUCKET,
        "archivePath": object_key,
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
        raise HTTPException(status_code=500, detail=f"DALTP could not {action} the evaluation report. Please try again later.") from exc
    if isinstance(exc, URLError):
        raise HTTPException(status_code=500, detail=f"DALTP could not reach evaluation report storage. Please try again later.") from exc
    raise HTTPException(status_code=500, detail=f"DALTP could not {action} the evaluation report. Please try again later.") from exc


def upload_evaluation_archive(local_path: Path, *, user_id: str, run_id: str) -> dict[str, str]:
    require_supabase_evaluation_storage()
    metadata = evaluation_storage_metadata(user_id=user_id, run_id=run_id)
    object_key = metadata["archivePath"]
    upload_url = _storage_api_url(f"/object/{quote(SUPABASE_EVALUATIONS_BUCKET, safe='')}/{quote(object_key, safe='/')}")
    request = Request(
        upload_url,
        data=local_path.read_bytes(),
        headers={**_supabase_headers("application/zip"), "x-upsert": "true"},
        method="POST",
    )
    try:
        with urlopen(request):
            return metadata
    except Exception as exc:  # pragma: no cover
        _raise_storage_error(exc, "upload")
    return metadata


def download_evaluation_archive_bytes(run: dict[str, Any]) -> bytes:
    require_supabase_evaluation_storage()
    if (run.get("storageProvider") or "supabase") != "supabase":
        raise HTTPException(status_code=500, detail=f"The evaluation report for '{run['runName']}' is not available for download.")

    bucket = run.get("storageBucket") or SUPABASE_EVALUATIONS_BUCKET
    object_key = run.get("archivePath") or ""
    download_url = _storage_api_url(f"/object/authenticated/{quote(bucket, safe='')}/{quote(object_key, safe='/')}")
    request = Request(download_url, headers=_supabase_headers(), method="GET")
    try:
        with urlopen(request) as response:
            return response.read()
    except Exception as exc:  # pragma: no cover
        _raise_storage_error(exc, "download")
    return b""


def delete_evaluation_archive(run: dict[str, Any]) -> None:
    if (run.get("storageProvider") or "supabase") != "supabase":
        return

    bucket = run.get("storageBucket") or SUPABASE_EVALUATIONS_BUCKET
    object_key = run.get("archivePath") or ""
    if not object_key:
        return
    delete_url = _storage_api_url(f"/object/{quote(bucket, safe='')}")
    payload = json.dumps({"prefixes": [object_key]}).encode("utf-8")
    request = Request(delete_url, data=payload, headers=_supabase_headers("application/json"), method="DELETE")
    try:
        with urlopen(request):
            return
    except Exception as exc:  # pragma: no cover
        _raise_storage_error(exc, "delete")
