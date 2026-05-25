from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
API_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = API_DIR / "runtime"
AUTH_DIR = RUNTIME_DIR / "auth"
DATASETS_DIR = RUNTIME_DIR / "datasets"
BUNDLES_DIR = RUNTIME_DIR / "run_bundles"
JOBS_DIR = RUNTIME_DIR / "jobs"
MODELS_DIR = RUNTIME_DIR / "models"
USERS_FILE = AUTH_DIR / "users.json"
SESSIONS_FILE = AUTH_DIR / "sessions.json"


def _require_psycopg():
    if psycopg is None:
        raise RuntimeError("psycopg is not installed. Add psycopg[binary] to your environment.")
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL or POSTGRES_URL must be configured for DALTP metadata storage.")


def connection():
    _require_psycopg()
    return psycopg.connect(DATABASE_URL, autocommit=True)


def _json_dump(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value)


def _json_load(value: Any, default: Any):
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def ensure_tables() -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS daltp_users (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS daltp_sessions (
                    token TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES daltp_users(id) ON DELETE CASCADE,
                    created_at TIMESTAMPTZ NOT NULL,
                    last_seen_at TIMESTAMPTZ NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS daltp_datasets (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL REFERENCES daltp_users(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    description TEXT NOT NULL,
                    source TEXT NOT NULL,
                    stored_file_name TEXT,
                    storage_provider TEXT,
                    storage_bucket TEXT,
                    dataset_path TEXT NOT NULL,
                    dataset_dir TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    line_count INTEGER,
                    size_mb DOUBLE PRECISION,
                    files_json TEXT,
                    generator_json TEXT,
                    vector_store_json TEXT
                );
                """
            )
            cur.execute("ALTER TABLE daltp_datasets ADD COLUMN IF NOT EXISTS line_count INTEGER;")
            cur.execute("ALTER TABLE daltp_datasets ADD COLUMN IF NOT EXISTS size_mb DOUBLE PRECISION;")
            cur.execute("ALTER TABLE daltp_datasets ADD COLUMN IF NOT EXISTS storage_provider TEXT;")
            cur.execute("ALTER TABLE daltp_datasets ADD COLUMN IF NOT EXISTS storage_bucket TEXT;")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_daltp_datasets_owner_created ON daltp_datasets (owner_id, created_at DESC);")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS daltp_bundles (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL REFERENCES daltp_users(id) ON DELETE CASCADE,
                    run_name TEXT NOT NULL,
                    execution_mode TEXT NOT NULL,
                    base_model TEXT,
                    peft_method TEXT,
                    lora_rank INTEGER,
                    qa_dataset_id TEXT,
                    instruction_dataset_id TEXT,
                    benchmark_dataset_id TEXT,
                    config_mode TEXT,
                    preset_id TEXT,
                    created_at TIMESTAMPTZ NOT NULL,
                    bundle_dir TEXT NOT NULL,
                    archive_path TEXT NOT NULL,
                    commands_json TEXT,
                    instructions_json TEXT
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_daltp_bundles_owner_created ON daltp_bundles (owner_id, created_at DESC);")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS daltp_jobs (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL REFERENCES daltp_users(id) ON DELETE CASCADE,
                    type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    metadata_json TEXT,
                    result_json TEXT,
                    error TEXT,
                    log_path TEXT NOT NULL,
                    job_dir TEXT NOT NULL
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_daltp_jobs_owner_updated ON daltp_jobs (owner_id, updated_at DESC);")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS daltp_models (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL REFERENCES daltp_users(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    base_model TEXT,
                    peft_method TEXT,
                    lora_rank INTEGER,
                    run_name TEXT,
                    bundle_id TEXT,
                    created_at TIMESTAMPTZ NOT NULL,
                    files_json TEXT,
                    model_dir TEXT NOT NULL,
                    archive_path TEXT NOT NULL
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_daltp_models_owner_created ON daltp_models (owner_id, created_at DESC);")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS daltp_eval_runs (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL REFERENCES daltp_users(id) ON DELETE CASCADE,
                    run_name TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    created_by TEXT,
                    run_dir TEXT NOT NULL
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_daltp_eval_runs_owner_created ON daltp_eval_runs (owner_id, created_at DESC);")


def upsert_user(user: dict[str, Any]) -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO daltp_users (id, name, email, password_hash, password_salt, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    email = EXCLUDED.email,
                    password_hash = EXCLUDED.password_hash,
                    password_salt = EXCLUDED.password_salt,
                    created_at = EXCLUDED.created_at;
                """,
                (
                    user["id"],
                    user["name"],
                    user["email"],
                    user["passwordHash"],
                    user["passwordSalt"],
                    user["createdAt"],
                ),
            )


def get_user_by_email(email: str) -> dict[str, Any] | None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, email, password_hash, password_salt, created_at
                FROM daltp_users
                WHERE lower(email) = lower(%s)
                LIMIT 1;
                """,
                (email,),
            )
            row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "name": row[1],
        "email": row[2],
        "passwordHash": row[3],
        "passwordSalt": row[4],
        "createdAt": row[5].isoformat() if hasattr(row[5], "isoformat") else row[5],
    }


def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, email, password_hash, password_salt, created_at
                FROM daltp_users
                WHERE id = %s
                LIMIT 1;
                """,
                (user_id,),
            )
            row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "name": row[1],
        "email": row[2],
        "passwordHash": row[3],
        "passwordSalt": row[4],
        "createdAt": row[5].isoformat() if hasattr(row[5], "isoformat") else row[5],
    }


def user_exists(user_id: str) -> bool:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM daltp_users
                WHERE id = %s
                LIMIT 1;
                """,
                (user_id,),
            )
            row = cur.fetchone()
    return bool(row)


def create_session(token: str, user_id: str, created_at: str, last_seen_at: str) -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO daltp_sessions (token, user_id, created_at, last_seen_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (token) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    created_at = EXCLUDED.created_at,
                    last_seen_at = EXCLUDED.last_seen_at;
                """,
                (token, user_id, created_at, last_seen_at),
            )


def get_session(token: str) -> dict[str, Any] | None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT token, user_id, created_at, last_seen_at
                FROM daltp_sessions
                WHERE token = %s
                LIMIT 1;
                """,
                (token,),
            )
            row = cur.fetchone()
    if not row:
        return None
    return {
        "token": row[0],
        "userId": row[1],
        "createdAt": row[2].isoformat() if hasattr(row[2], "isoformat") else row[2],
        "lastSeenAt": row[3].isoformat() if hasattr(row[3], "isoformat") else row[3],
    }


def touch_session(token: str, last_seen_at: str) -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE daltp_sessions SET last_seen_at = %s WHERE token = %s;", (last_seen_at, token))


def delete_session(token: str) -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM daltp_sessions WHERE token = %s;", (token,))


def upsert_dataset(metadata: dict[str, Any]) -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO daltp_datasets
                    (id, owner_id, name, kind, description, source, stored_file_name, storage_provider, storage_bucket, dataset_path, dataset_dir, created_at, line_count, size_mb, files_json, generator_json, vector_store_json)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    owner_id = EXCLUDED.owner_id,
                    name = EXCLUDED.name,
                    kind = EXCLUDED.kind,
                    description = EXCLUDED.description,
                    source = EXCLUDED.source,
                    stored_file_name = EXCLUDED.stored_file_name,
                    storage_provider = EXCLUDED.storage_provider,
                    storage_bucket = EXCLUDED.storage_bucket,
                    dataset_path = EXCLUDED.dataset_path,
                    dataset_dir = EXCLUDED.dataset_dir,
                    created_at = EXCLUDED.created_at,
                    line_count = EXCLUDED.line_count,
                    size_mb = EXCLUDED.size_mb,
                    files_json = EXCLUDED.files_json,
                    generator_json = EXCLUDED.generator_json,
                    vector_store_json = EXCLUDED.vector_store_json;
                """,
                (
                    metadata["id"],
                    metadata["ownerId"],
                    metadata["name"],
                    metadata["kind"],
                    metadata["description"],
                    metadata["source"],
                    metadata.get("storedFileName"),
                    metadata.get("storageProvider", "supabase"),
                    metadata.get("storageBucket", ""),
                    metadata["path"],
                    metadata["datasetDir"],
                    metadata["createdAt"],
                    metadata.get("lineCount"),
                    metadata.get("sizeMb"),
                    _json_dump(metadata.get("files", [])),
                    _json_dump(metadata.get("generator")),
                    _json_dump(metadata.get("vectorStore")),
                ),
            )


def list_datasets(owner_id: str) -> list[dict[str, Any]]:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, owner_id, name, kind, description, source, stored_file_name, storage_provider, storage_bucket, dataset_path, dataset_dir, created_at, line_count, size_mb, files_json, vector_store_json
                FROM daltp_datasets
                WHERE owner_id = %s
                ORDER BY created_at DESC;
                """,
                (owner_id,),
            )
            rows = cur.fetchall()
    results = []
    for row in rows:
        results.append(
            {
                "id": row[0],
                "ownerId": row[1],
                "name": row[2],
                "kind": row[3],
                "description": row[4],
                "source": row[5],
                "storedFileName": row[6],
                "storageProvider": row[7] or "supabase",
                "storageBucket": row[8] or "",
                "path": row[9],
                "datasetDir": row[10],
                "createdAt": row[11].isoformat() if hasattr(row[11], "isoformat") else row[11],
                "lineCount": row[12],
                "sizeMb": row[13],
                "files": _json_load(row[14], []),
                "vectorStore": _json_load(row[15], None),
            }
        )
    return results


def get_dataset(dataset_id: str, owner_id: str) -> dict[str, Any] | None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, owner_id, name, kind, description, source, stored_file_name, storage_provider, storage_bucket, dataset_path, dataset_dir, created_at, line_count, size_mb, files_json, vector_store_json
                FROM daltp_datasets
                WHERE id = %s AND owner_id = %s
                LIMIT 1;
                """,
                (dataset_id, owner_id),
            )
            row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "ownerId": row[1],
        "name": row[2],
        "kind": row[3],
        "description": row[4],
        "source": row[5],
        "storedFileName": row[6],
        "storageProvider": row[7] or "supabase",
        "storageBucket": row[8] or "",
        "path": row[9],
        "datasetDir": row[10],
        "createdAt": row[11].isoformat() if hasattr(row[11], "isoformat") else row[11],
        "lineCount": row[12],
        "sizeMb": row[13],
        "files": _json_load(row[14], []),
        "vectorStore": _json_load(row[15], None),
    }


def delete_dataset(dataset_id: str, owner_id: str) -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM daltp_datasets WHERE id = %s AND owner_id = %s;", (dataset_id, owner_id))


def upsert_bundle(metadata: dict[str, Any]) -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO daltp_bundles
                    (id, owner_id, run_name, execution_mode, base_model, peft_method, lora_rank, qa_dataset_id, instruction_dataset_id, benchmark_dataset_id, config_mode, preset_id, created_at, bundle_dir, archive_path, commands_json, instructions_json)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    owner_id = EXCLUDED.owner_id,
                    run_name = EXCLUDED.run_name,
                    execution_mode = EXCLUDED.execution_mode,
                    base_model = EXCLUDED.base_model,
                    peft_method = EXCLUDED.peft_method,
                    lora_rank = EXCLUDED.lora_rank,
                    qa_dataset_id = EXCLUDED.qa_dataset_id,
                    instruction_dataset_id = EXCLUDED.instruction_dataset_id,
                    benchmark_dataset_id = EXCLUDED.benchmark_dataset_id,
                    config_mode = EXCLUDED.config_mode,
                    preset_id = EXCLUDED.preset_id,
                    created_at = EXCLUDED.created_at,
                    bundle_dir = EXCLUDED.bundle_dir,
                    archive_path = EXCLUDED.archive_path,
                    commands_json = EXCLUDED.commands_json,
                    instructions_json = EXCLUDED.instructions_json;
                """,
                (
                    metadata["id"],
                    metadata["ownerId"],
                    metadata["runName"],
                    metadata["executionMode"],
                    metadata.get("baseModel"),
                    metadata.get("peftMethod"),
                    metadata.get("loraRank"),
                    metadata.get("qaDatasetId"),
                    metadata.get("instructionDatasetId"),
                    None,
                    metadata.get("configMode"),
                    metadata.get("presetId"),
                    metadata["createdAt"],
                    metadata["bundleDir"],
                    metadata["archivePath"],
                    _json_dump(metadata.get("commands", {"local": [], "colab": []})),
                    _json_dump(metadata.get("instructions", {})),
                ),
            )


def list_bundles(owner_id: str) -> list[dict[str, Any]]:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, run_name, execution_mode, base_model, peft_method, lora_rank, qa_dataset_id, instruction_dataset_id, benchmark_dataset_id, config_mode, created_at, archive_path, commands_json
                FROM daltp_bundles
                WHERE owner_id = %s
                ORDER BY created_at DESC;
                """,
                (owner_id,),
            )
            rows = cur.fetchall()
    return [
        {
            "id": row[0],
            "runName": row[1],
            "executionMode": row[2],
            "baseModel": row[3],
            "peftMethod": row[4],
            "loraRank": row[5],
            "qaDatasetId": row[6],
            "instructionDatasetId": row[7],
            "configMode": row[9],
            "createdAt": row[10].isoformat() if hasattr(row[10], "isoformat") else row[10],
            "archivePath": row[11],
            "commands": _json_load(row[12], {"local": [], "colab": []}),
        }
        for row in rows
    ]


def get_bundle(bundle_id: str, owner_id: str) -> dict[str, Any] | None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, run_name, execution_mode, base_model, peft_method, lora_rank, qa_dataset_id, instruction_dataset_id, benchmark_dataset_id, config_mode, created_at, archive_path, commands_json, bundle_dir
                FROM daltp_bundles
                WHERE id = %s AND owner_id = %s
                LIMIT 1;
                """,
                (bundle_id, owner_id),
            )
            row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "runName": row[1],
        "executionMode": row[2],
        "baseModel": row[3],
        "peftMethod": row[4],
        "loraRank": row[5],
        "qaDatasetId": row[6],
        "instructionDatasetId": row[7],
        "configMode": row[9],
        "createdAt": row[10].isoformat() if hasattr(row[10], "isoformat") else row[10],
        "archivePath": row[11],
        "commands": _json_load(row[12], {"local": [], "colab": []}),
        "bundleDir": row[13],
    }


def delete_bundle(bundle_id: str, owner_id: str) -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM daltp_bundles WHERE id = %s AND owner_id = %s;", (bundle_id, owner_id))


def upsert_job(metadata: dict[str, Any]) -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO daltp_jobs
                    (id, owner_id, type, title, status, created_at, updated_at, metadata_json, result_json, error, log_path, job_dir)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    owner_id = EXCLUDED.owner_id,
                    type = EXCLUDED.type,
                    title = EXCLUDED.title,
                    status = EXCLUDED.status,
                    created_at = EXCLUDED.created_at,
                    updated_at = EXCLUDED.updated_at,
                    metadata_json = EXCLUDED.metadata_json,
                    result_json = EXCLUDED.result_json,
                    error = EXCLUDED.error,
                    log_path = EXCLUDED.log_path,
                    job_dir = EXCLUDED.job_dir;
                """,
                (
                    metadata["id"],
                    metadata["ownerId"],
                    metadata["type"],
                    metadata["title"],
                    metadata["status"],
                    metadata["createdAt"],
                    metadata["updatedAt"],
                    _json_dump(metadata.get("metadata", {})),
                    _json_dump(metadata.get("result")),
                    metadata.get("error"),
                    metadata["logPath"],
                    metadata["jobDir"],
                ),
            )


def list_jobs(owner_id: str) -> list[dict[str, Any]]:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, type, title, status, owner_id, created_at, updated_at, metadata_json, result_json, error, log_path, job_dir
                FROM daltp_jobs
                WHERE owner_id = %s
                ORDER BY updated_at DESC;
                """,
                (owner_id,),
            )
            rows = cur.fetchall()
    return [
        {
            "id": row[0],
            "type": row[1],
            "title": row[2],
            "status": row[3],
            "ownerId": row[4],
            "createdAt": row[5].isoformat() if hasattr(row[5], "isoformat") else row[5],
            "updatedAt": row[6].isoformat() if hasattr(row[6], "isoformat") else row[6],
            "metadata": _json_load(row[7], {}),
            "result": _json_load(row[8], None),
            "error": row[9],
            "logPath": row[10],
            "jobDir": row[11],
        }
        for row in rows
    ]


def get_job(job_id: str, owner_id: str) -> dict[str, Any] | None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, type, title, status, owner_id, created_at, updated_at, metadata_json, result_json, error, log_path, job_dir
                FROM daltp_jobs
                WHERE id = %s AND owner_id = %s
                LIMIT 1;
                """,
                (job_id, owner_id),
            )
            row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "type": row[1],
        "title": row[2],
        "status": row[3],
        "ownerId": row[4],
        "createdAt": row[5].isoformat() if hasattr(row[5], "isoformat") else row[5],
        "updatedAt": row[6].isoformat() if hasattr(row[6], "isoformat") else row[6],
        "metadata": _json_load(row[7], {}),
        "result": _json_load(row[8], None),
        "error": row[9],
        "logPath": row[10],
        "jobDir": row[11],
    }


def upsert_model(metadata: dict[str, Any]) -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO daltp_models
                    (id, owner_id, name, source, base_model, peft_method, lora_rank, run_name, bundle_id, created_at, files_json, model_dir, archive_path)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    owner_id = EXCLUDED.owner_id,
                    name = EXCLUDED.name,
                    source = EXCLUDED.source,
                    base_model = EXCLUDED.base_model,
                    peft_method = EXCLUDED.peft_method,
                    lora_rank = EXCLUDED.lora_rank,
                    run_name = EXCLUDED.run_name,
                    bundle_id = EXCLUDED.bundle_id,
                    created_at = EXCLUDED.created_at,
                    files_json = EXCLUDED.files_json,
                    model_dir = EXCLUDED.model_dir,
                    archive_path = EXCLUDED.archive_path;
                """,
                (
                    metadata["id"],
                    metadata["ownerId"],
                    metadata["name"],
                    metadata["source"],
                    metadata.get("baseModel"),
                    metadata.get("peftMethod"),
                    metadata.get("loraRank"),
                    metadata.get("runName"),
                    metadata.get("bundleId"),
                    metadata["createdAt"],
                    _json_dump(metadata.get("files", [])),
                    metadata["modelDir"],
                    metadata["archivePath"],
                ),
            )


def list_models(owner_id: str) -> list[dict[str, Any]]:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, source, base_model, peft_method, lora_rank, run_name, created_at, files_json, archive_path, model_dir
                FROM daltp_models
                WHERE owner_id = %s
                ORDER BY created_at DESC;
                """,
                (owner_id,),
            )
            rows = cur.fetchall()
    return [
        {
            "id": row[0],
            "name": row[1],
            "source": row[2],
            "baseModel": row[3],
            "peftMethod": row[4],
            "loraRank": row[5],
            "runName": row[6],
            "createdAt": row[7].isoformat() if hasattr(row[7], "isoformat") else row[7],
            "files": _json_load(row[8], []),
            "archivePath": row[9],
            "modelDir": row[10],
        }
        for row in rows
    ]


def get_model(model_id: str, owner_id: str) -> dict[str, Any] | None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, source, base_model, peft_method, lora_rank, run_name, created_at, files_json, archive_path, model_dir
                FROM daltp_models
                WHERE id = %s AND owner_id = %s
                LIMIT 1;
                """,
                (model_id, owner_id),
            )
            row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "name": row[1],
        "source": row[2],
        "baseModel": row[3],
        "peftMethod": row[4],
        "loraRank": row[5],
        "runName": row[6],
        "createdAt": row[7].isoformat() if hasattr(row[7], "isoformat") else row[7],
        "files": _json_load(row[8], []),
        "archivePath": row[9],
        "modelDir": row[10],
    }


def delete_model(model_id: str, owner_id: str) -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM daltp_models WHERE id = %s AND owner_id = %s;", (model_id, owner_id))


def upsert_eval_run(run_id: str, owner_id: str, run_name: str, created_at: str, created_by: str | None, run_dir: str) -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO daltp_eval_runs (id, owner_id, run_name, created_at, created_by, run_dir)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    owner_id = EXCLUDED.owner_id,
                    run_name = EXCLUDED.run_name,
                    created_at = EXCLUDED.created_at,
                    created_by = EXCLUDED.created_by,
                    run_dir = EXCLUDED.run_dir;
                """,
                (run_id, owner_id, run_name, created_at, created_by, run_dir),
            )


def list_eval_runs(owner_id: str) -> list[dict[str, Any]]:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, run_name, created_at, created_by, run_dir FROM daltp_eval_runs WHERE owner_id = %s ORDER BY created_at DESC;",
                (owner_id,),
            )
            rows = cur.fetchall()
    return [
        {
            "id": row[0],
            "runName": row[1],
            "createdAt": row[2].isoformat() if hasattr(row[2], "isoformat") else row[2],
            "createdBy": row[3],
            "runDir": row[4],
        }
        for row in rows
    ]


def get_eval_run(run_id: str, owner_id: str) -> dict[str, Any] | None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, run_name, created_at, created_by, run_dir FROM daltp_eval_runs WHERE id = %s AND owner_id = %s LIMIT 1;",
                (run_id, owner_id),
            )
            row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "runName": row[1],
        "createdAt": row[2].isoformat() if hasattr(row[2], "isoformat") else row[2],
        "createdBy": row[3],
        "runDir": row[4],
    }


def migrate_legacy_runtime() -> None:
    users = _read_json(USERS_FILE, [])
    for user in users if isinstance(users, list) else []:
        if all(key in user for key in ("id", "name", "email", "passwordHash", "passwordSalt", "createdAt")):
            upsert_user(user)

    sessions = _read_json(SESSIONS_FILE, {})
    for token, session in (sessions.items() if isinstance(sessions, dict) else []):
        if isinstance(session, dict) and session.get("userId") and user_exists(session["userId"]):
            create_session(token, session["userId"], session.get("createdAt", ""), session.get("lastSeenAt", session.get("createdAt", "")))

    for owner_dir in BUNDLES_DIR.iterdir() if BUNDLES_DIR.exists() else []:
        if not owner_dir.is_dir():
            continue
        owner_id = owner_dir.name
        if not user_exists(owner_id):
            continue
        for bundle_dir in owner_dir.iterdir():
            if not bundle_dir.is_dir():
                continue
            manifest = _read_json(bundle_dir / "manifest.json", {})
            if not manifest:
                continue
            upsert_bundle(
                {
                    "id": manifest.get("id", bundle_dir.name),
                    "ownerId": manifest.get("ownerId", owner_id),
                    "runName": manifest.get("runName", bundle_dir.name),
                    "executionMode": manifest.get("executionMode", "colab"),
                    "baseModel": manifest.get("baseModel"),
                    "peftMethod": manifest.get("peftMethod"),
                    "loraRank": manifest.get("loraRank"),
                    "qaDatasetId": manifest.get("qaDatasetId"),
                    "instructionDatasetId": manifest.get("instructionDatasetId"),
                    "configMode": manifest.get("configMode"),
                    "presetId": manifest.get("presetId"),
                    "createdAt": manifest.get("createdAt", ""),
                    "bundleDir": str(bundle_dir),
                    "archivePath": str(bundle_dir.with_suffix(".zip")),
                    "commands": manifest.get("commands", {"local": [], "colab": []}),
                    "instructions": manifest.get("instructions", {}),
                }
            )

    for owner_dir in JOBS_DIR.iterdir() if JOBS_DIR.exists() else []:
        if not owner_dir.is_dir():
            continue
        owner_id = owner_dir.name
        if not user_exists(owner_id):
            continue
        for job_dir in owner_dir.iterdir():
            if not job_dir.is_dir():
                continue
            manifest = _read_json(job_dir / "manifest.json", {})
            if not manifest:
                continue
            manifest["ownerId"] = manifest.get("ownerId", owner_id)
            manifest["jobDir"] = str(job_dir)
            upsert_job(manifest)

    for owner_dir in MODELS_DIR.iterdir() if MODELS_DIR.exists() else []:
        if not owner_dir.is_dir():
            continue
        owner_id = owner_dir.name
        if not user_exists(owner_id):
            continue
        for model_dir in owner_dir.iterdir():
            if not model_dir.is_dir():
                continue
            manifest = _read_json(model_dir / "manifest.json", {})
            if not manifest:
                continue
            upsert_model(
                {
                    "id": manifest.get("id", model_dir.name),
                    "ownerId": manifest.get("ownerId", owner_id),
                    "name": manifest.get("name", model_dir.name),
                    "source": manifest.get("source", "local"),
                    "baseModel": manifest.get("baseModel"),
                    "peftMethod": manifest.get("peftMethod"),
                    "loraRank": manifest.get("loraRank"),
                    "runName": manifest.get("runName"),
                    "bundleId": manifest.get("bundleId"),
                    "createdAt": manifest.get("createdAt", ""),
                    "files": manifest.get("files", []),
                    "modelDir": str(model_dir),
                    "archivePath": str(model_dir.with_suffix(".zip")),
                }
            )

    eval_outputs_dir = API_DIR.parent / "evaluation" / "outputs"
    run_manifest_name = "daltp_run.json"
    for run_dir in eval_outputs_dir.iterdir() if eval_outputs_dir.exists() else []:
        if not run_dir.is_dir():
            continue
        manifest = _read_json(run_dir / run_manifest_name, {})
        if not manifest or not manifest.get("ownerId") or not user_exists(manifest["ownerId"]):
            continue
        upsert_eval_run(
            run_id=manifest.get("id", run_dir.name),
            owner_id=manifest["ownerId"],
            run_name=manifest.get("runName", run_dir.name),
            created_at=manifest.get("createdAt", ""),
            created_by=manifest.get("createdBy"),
            run_dir=str(run_dir),
        )


def initialize_metadata_store() -> None:
    ensure_tables()
    migrate_legacy_runtime()
