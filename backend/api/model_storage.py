from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from azure.core.exceptions import AzureError, ResourceNotFoundError
from azure.storage.blob import BlobSasPermissions, BlobServiceClient, generate_blob_sas
from dotenv import load_dotenv
from fastapi import HTTPException

load_dotenv()

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING") or ""
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME") or ""
AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME") or ""
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY") or ""
STORAGE_PROVIDER = "azure_blob"


def is_azure_model_storage_enabled() -> bool:
    return bool(
        AZURE_STORAGE_CONNECTION_STRING
        and AZURE_STORAGE_CONTAINER_NAME
        and AZURE_STORAGE_ACCOUNT_NAME
        and AZURE_STORAGE_ACCOUNT_KEY
    )


def require_azure_model_storage() -> None:
    if is_azure_model_storage_enabled():
        return
    raise HTTPException(
        status_code=500,
        detail="Model storage is not available yet. Please ask the workspace owner to finish storage setup before importing model artifacts.",
    )


def model_storage_metadata(*, user_id: str, model_id: str, stored_file_name: str = "model_artifact.zip") -> dict[str, str]:
    object_key = f"models/{user_id}/{model_id}/{stored_file_name}"
    return {
        "storageProvider": STORAGE_PROVIDER,
        "storageBucket": AZURE_STORAGE_CONTAINER_NAME,
        "archivePath": object_key,
    }


def _blob_service_client() -> BlobServiceClient:
    require_azure_model_storage()
    return BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)


def _archive_label(model: dict[str, Any]) -> str:
    return f"model artifact '{model.get('name') or model.get('id') or 'unknown'}'"


def _raise_storage_error(exc: Exception, action: str) -> None:
    if isinstance(exc, HTTPException):
        raise exc
    if isinstance(exc, ResourceNotFoundError):
        raise HTTPException(status_code=404, detail="The selected model artifact files could not be found. Please re-import the model artifact.") from exc
    if isinstance(exc, AzureError):
        raise HTTPException(status_code=500, detail=f"DALTP could not {action} the model artifact. Please try again later.") from exc
    raise HTTPException(status_code=500, detail=f"DALTP could not {action} the model artifact. Please try again later.") from exc


def upload_model_archive(local_path: Path, *, user_id: str, model_id: str) -> dict[str, str]:
    metadata = model_storage_metadata(user_id=user_id, model_id=model_id)
    try:
        blob_client = _blob_service_client().get_blob_client(
            container=AZURE_STORAGE_CONTAINER_NAME,
            blob=metadata["archivePath"],
        )
        with local_path.open("rb") as file_handle:
            blob_client.upload_blob(file_handle, overwrite=True)
        return metadata
    except Exception as exc:  # pragma: no cover
        _raise_storage_error(exc, "upload")
    return metadata


def download_model_archive_bytes(model: dict[str, Any]) -> bytes:
    require_azure_model_storage()
    if (model.get("storageProvider") or STORAGE_PROVIDER) != STORAGE_PROVIDER:
        raise HTTPException(status_code=500, detail=f"{_archive_label(model)} is not available in model storage. Please re-import the model artifact.")

    blob_name = model.get("archivePath") or ""
    container = model.get("storageBucket") or AZURE_STORAGE_CONTAINER_NAME
    if not blob_name:
        raise HTTPException(status_code=500, detail=f"{_archive_label(model)} is missing its files. Please re-import the model artifact.")

    try:
        blob_client = _blob_service_client().get_blob_client(container=container, blob=blob_name)
        return blob_client.download_blob().readall()
    except Exception as exc:  # pragma: no cover
        _raise_storage_error(exc, "download")
    return b""


def create_model_archive_signed_url(model: dict[str, Any], *, expires_in: int = 60 * 30) -> str:
    require_azure_model_storage()
    if (model.get("storageProvider") or STORAGE_PROVIDER) != STORAGE_PROVIDER:
        raise HTTPException(status_code=500, detail=f"{_archive_label(model)} is not available in model storage. Please re-import the model artifact.")

    blob_name = model.get("archivePath") or ""
    container = model.get("storageBucket") or AZURE_STORAGE_CONTAINER_NAME
    if not blob_name:
        raise HTTPException(status_code=500, detail=f"{_archive_label(model)} is missing its files. Please re-import the model artifact.")

    try:
        sas_token = generate_blob_sas(
            account_name=AZURE_STORAGE_ACCOUNT_NAME,
            account_key=AZURE_STORAGE_ACCOUNT_KEY,
            container_name=container,
            blob_name=blob_name,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(seconds=expires_in),
        )
    except Exception as exc:  # pragma: no cover
        _raise_storage_error(exc, "signed-url generation")

    return f"https://{AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{container}/{blob_name}?{sas_token}"


def delete_model_archive(model: dict[str, Any]) -> None:
    if (model.get("storageProvider") or STORAGE_PROVIDER) != STORAGE_PROVIDER:
        return

    blob_name = model.get("archivePath") or ""
    container = model.get("storageBucket") or AZURE_STORAGE_CONTAINER_NAME
    if not blob_name:
        return

    try:
        blob_client = _blob_service_client().get_blob_client(container=container, blob=blob_name)
        blob_client.delete_blob(delete_snapshots="include")
    except ResourceNotFoundError:
        return
    except Exception as exc:  # pragma: no cover
        _raise_storage_error(exc, "delete")
