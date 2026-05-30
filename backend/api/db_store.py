from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv

try:
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")

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
                    storage_provider TEXT,
                    storage_bucket TEXT,
                    commands_json TEXT,
                    instructions_json TEXT
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_daltp_bundles_owner_created ON daltp_bundles (owner_id, created_at DESC);")
            cur.execute("ALTER TABLE daltp_bundles ADD COLUMN IF NOT EXISTS storage_provider TEXT;")
            cur.execute("ALTER TABLE daltp_bundles ADD COLUMN IF NOT EXISTS storage_bucket TEXT;")
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
                    logs_text TEXT,
                    log_path TEXT NOT NULL,
                    job_dir TEXT NOT NULL
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_daltp_jobs_owner_updated ON daltp_jobs (owner_id, updated_at DESC);")
            cur.execute("ALTER TABLE daltp_jobs ADD COLUMN IF NOT EXISTS logs_text TEXT;")
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
                    storage_provider TEXT,
                    storage_bucket TEXT,
                    model_dir TEXT NOT NULL,
                    archive_path TEXT NOT NULL,
                    archive_size_mb DOUBLE PRECISION
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_daltp_models_owner_created ON daltp_models (owner_id, created_at DESC);")
            cur.execute("ALTER TABLE daltp_models ADD COLUMN IF NOT EXISTS storage_provider TEXT;")
            cur.execute("ALTER TABLE daltp_models ADD COLUMN IF NOT EXISTS storage_bucket TEXT;")
            cur.execute("ALTER TABLE daltp_models ADD COLUMN IF NOT EXISTS archive_size_mb DOUBLE PRECISION;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS daltp_eval_runs (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL REFERENCES daltp_users(id) ON DELETE CASCADE,
                    run_name TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ,
                    created_by TEXT,
                    run_dir TEXT NOT NULL,
                    storage_provider TEXT,
                    storage_bucket TEXT,
                    archive_path TEXT,
                    manifest_json TEXT,
                    summary_json TEXT
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_daltp_eval_runs_owner_created ON daltp_eval_runs (owner_id, created_at DESC);")
            cur.execute("ALTER TABLE daltp_eval_runs ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;")
            cur.execute("ALTER TABLE daltp_eval_runs ADD COLUMN IF NOT EXISTS storage_provider TEXT;")
            cur.execute("ALTER TABLE daltp_eval_runs ADD COLUMN IF NOT EXISTS storage_bucket TEXT;")
            cur.execute("ALTER TABLE daltp_eval_runs ADD COLUMN IF NOT EXISTS archive_path TEXT;")
            cur.execute("ALTER TABLE daltp_eval_runs ADD COLUMN IF NOT EXISTS manifest_json TEXT;")
            cur.execute("ALTER TABLE daltp_eval_runs ADD COLUMN IF NOT EXISTS summary_json TEXT;")


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
                SELECT id, owner_id, name, kind, description, source, stored_file_name, storage_provider, storage_bucket, dataset_path, dataset_dir, created_at, line_count, size_mb, files_json, generator_json, vector_store_json
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
                "generator": _json_load(row[15], None),
                "vectorStore": _json_load(row[16], None),
            }
        )
    return results


def get_dataset(dataset_id: str, owner_id: str) -> dict[str, Any] | None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, owner_id, name, kind, description, source, stored_file_name, storage_provider, storage_bucket, dataset_path, dataset_dir, created_at, line_count, size_mb, files_json, generator_json, vector_store_json
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
        "generator": _json_load(row[15], None),
        "vectorStore": _json_load(row[16], None),
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
                    (id, owner_id, run_name, execution_mode, base_model, peft_method, lora_rank, qa_dataset_id, instruction_dataset_id, benchmark_dataset_id, config_mode, preset_id, created_at, bundle_dir, archive_path, storage_provider, storage_bucket, commands_json, instructions_json)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                    storage_provider = EXCLUDED.storage_provider,
                    storage_bucket = EXCLUDED.storage_bucket,
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
                    metadata.get("storageProvider"),
                    metadata.get("storageBucket"),
                    _json_dump(metadata.get("commands", {"colab": []})),
                    _json_dump(metadata.get("instructions", {})),
                ),
            )


def list_bundles(owner_id: str) -> list[dict[str, Any]]:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, run_name, execution_mode, base_model, peft_method, lora_rank, qa_dataset_id, instruction_dataset_id, benchmark_dataset_id, config_mode, created_at, archive_path, commands_json, storage_provider, storage_bucket
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
            "commands": _json_load(row[12], {"colab": []}),
            "storageProvider": row[13],
            "storageBucket": row[14],
        }
        for row in rows
    ]


def get_bundle(bundle_id: str, owner_id: str) -> dict[str, Any] | None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, run_name, execution_mode, base_model, peft_method, lora_rank, qa_dataset_id, instruction_dataset_id, benchmark_dataset_id, config_mode, created_at, archive_path, commands_json, bundle_dir, storage_provider, storage_bucket
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
        "commands": _json_load(row[12], {"colab": []}),
        "bundleDir": row[13],
        "storageProvider": row[14],
        "storageBucket": row[15],
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
                    (id, owner_id, type, title, status, created_at, updated_at, metadata_json, result_json, error, logs_text, log_path, job_dir)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                    logs_text = EXCLUDED.logs_text,
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
                    metadata.get("logs", ""),
                    metadata["logPath"],
                    metadata["jobDir"],
                ),
            )


def append_job_log(job_id: str, owner_id: str, text: str, updated_at: str) -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE daltp_jobs
                SET logs_text = COALESCE(logs_text, '') || %s,
                    updated_at = %s
                WHERE id = %s AND owner_id = %s;
                """,
                (text, updated_at, job_id, owner_id),
            )


def list_jobs(owner_id: str) -> list[dict[str, Any]]:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, type, title, status, owner_id, created_at, updated_at, metadata_json, result_json, error, logs_text, log_path, job_dir
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
            "logs": row[10] or "",
            "logPath": row[11],
            "jobDir": row[12],
        }
        for row in rows
    ]


def get_job(job_id: str, owner_id: str) -> dict[str, Any] | None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, type, title, status, owner_id, created_at, updated_at, metadata_json, result_json, error, logs_text, log_path, job_dir
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
        "logs": row[10] or "",
        "logPath": row[11],
        "jobDir": row[12],
    }


def delete_job(job_id: str, owner_id: str) -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM daltp_jobs WHERE id = %s AND owner_id = %s;", (job_id, owner_id))


def upsert_model(metadata: dict[str, Any]) -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO daltp_models
                    (id, owner_id, name, source, base_model, peft_method, lora_rank, run_name, bundle_id, created_at, files_json, storage_provider, storage_bucket, model_dir, archive_path, archive_size_mb)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                    storage_provider = EXCLUDED.storage_provider,
                    storage_bucket = EXCLUDED.storage_bucket,
                    model_dir = EXCLUDED.model_dir,
                    archive_path = EXCLUDED.archive_path,
                    archive_size_mb = EXCLUDED.archive_size_mb;
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
                    metadata.get("storageProvider", "supabase"),
                    metadata.get("storageBucket", ""),
                    metadata["modelDir"],
                    metadata["archivePath"],
                    metadata.get("archiveSizeMb"),
                ),
            )


def list_models(owner_id: str) -> list[dict[str, Any]]:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, source, base_model, peft_method, lora_rank, run_name, created_at, files_json, storage_provider, storage_bucket, archive_path, model_dir, archive_size_mb
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
            "storageProvider": row[9] or "azure_blob",
            "storageBucket": row[10] or "",
            "archivePath": row[11],
            "modelDir": row[12],
            "archiveSizeMb": row[13],
        }
        for row in rows
    ]


def get_model(model_id: str, owner_id: str) -> dict[str, Any] | None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, source, base_model, peft_method, lora_rank, run_name, created_at, files_json, storage_provider, storage_bucket, archive_path, model_dir, archive_size_mb
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
        "storageProvider": row[9] or "azure_blob",
        "storageBucket": row[10] or "",
        "archivePath": row[11],
        "modelDir": row[12],
        "archiveSizeMb": row[13],
    }


def delete_model(model_id: str, owner_id: str) -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM daltp_models WHERE id = %s AND owner_id = %s;", (model_id, owner_id))


def upsert_eval_run(
    run_id: str,
    owner_id: str,
    run_name: str,
    created_at: str,
    created_by: str | None,
    run_dir: str,
    *,
    updated_at: str | None = None,
    storage_provider: str | None = None,
    storage_bucket: str | None = None,
    archive_path: str | None = None,
    manifest: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
) -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO daltp_eval_runs
                    (id, owner_id, run_name, created_at, updated_at, created_by, run_dir, storage_provider, storage_bucket, archive_path, manifest_json, summary_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    owner_id = EXCLUDED.owner_id,
                    run_name = EXCLUDED.run_name,
                    created_at = EXCLUDED.created_at,
                    updated_at = EXCLUDED.updated_at,
                    created_by = EXCLUDED.created_by,
                    run_dir = EXCLUDED.run_dir,
                    storage_provider = EXCLUDED.storage_provider,
                    storage_bucket = EXCLUDED.storage_bucket,
                    archive_path = EXCLUDED.archive_path,
                    manifest_json = EXCLUDED.manifest_json,
                    summary_json = EXCLUDED.summary_json;
                """,
                (
                    run_id,
                    owner_id,
                    run_name,
                    created_at,
                    updated_at or created_at,
                    created_by,
                    run_dir,
                    storage_provider,
                    storage_bucket,
                    archive_path,
                    _json_dump(manifest),
                    _json_dump(summary),
                ),
            )


def list_eval_runs(owner_id: str) -> list[dict[str, Any]]:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, run_name, created_at, updated_at, created_by, run_dir, storage_provider, storage_bucket, archive_path, manifest_json, summary_json
                FROM daltp_eval_runs
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
            "createdAt": row[2].isoformat() if hasattr(row[2], "isoformat") else row[2],
            "updatedAt": row[3].isoformat() if hasattr(row[3], "isoformat") else row[3],
            "createdBy": row[4],
            "runDir": row[5],
            "storageProvider": row[6],
            "storageBucket": row[7],
            "archivePath": row[8],
            "manifest": _json_load(row[9], {}),
            "summary": _json_load(row[10], None),
        }
        for row in rows
    ]


def get_eval_run(run_id: str, owner_id: str) -> dict[str, Any] | None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, run_name, created_at, updated_at, created_by, run_dir, storage_provider, storage_bucket, archive_path, manifest_json, summary_json
                FROM daltp_eval_runs
                WHERE id = %s AND owner_id = %s
                LIMIT 1;
                """,
                (run_id, owner_id),
            )
            row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "runName": row[1],
        "createdAt": row[2].isoformat() if hasattr(row[2], "isoformat") else row[2],
        "updatedAt": row[3].isoformat() if hasattr(row[3], "isoformat") else row[3],
        "createdBy": row[4],
        "runDir": row[5],
        "storageProvider": row[6],
        "storageBucket": row[7],
        "archivePath": row[8],
        "manifest": _json_load(row[9], {}),
        "summary": _json_load(row[10], None),
    }


def delete_eval_run(run_id: str, owner_id: str) -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM daltp_eval_runs WHERE id = %s AND owner_id = %s;", (run_id, owner_id))


def initialize_metadata_store() -> None:
    ensure_tables()
