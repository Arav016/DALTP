import hashlib
import os
import uuid
from dataclasses import dataclass

from dotenv import load_dotenv

try:
    import psycopg
except ImportError:  # pragma: no cover - handled at runtime when dependency is missing
    psycopg = None

load_dotenv()

POSTGRES_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
PGVECTOR_TABLE = os.getenv("PGVECTOR_TABLE_NAME", "daltp_embeddings")
PGVECTOR_COLLECTIONS_TABLE = os.getenv("PGVECTOR_COLLECTIONS_TABLE_NAME", "daltp_embedding_collections")


@dataclass
class SearchResult:
    payload: dict
    score: float | None = None


def _require_psycopg():
    if psycopg is None:
        raise RuntimeError("psycopg is not installed. Add psycopg[binary] to your environment before using pgvector.")
    if not POSTGRES_URL:
        raise RuntimeError("DATABASE_URL or POSTGRES_URL must be configured before using pgvector.")


def get_connection():
    _require_psycopg()
    return psycopg.connect(POSTGRES_URL, autocommit=True)


def vector_literal(values):
    return "[" + ",".join(format(float(value), ".10f") for value in values) + "]"


def ensure_store():
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {PGVECTOR_COLLECTIONS_TABLE} (
                    name TEXT PRIMARY KEY,
                    vector_size INTEGER NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {PGVECTOR_TABLE} (
                    id TEXT PRIMARY KEY,
                    collection_name TEXT NOT NULL,
                    embedding VECTOR,
                    text_content TEXT NOT NULL,
                    source TEXT,
                    file_name TEXT,
                    page TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{PGVECTOR_TABLE}_collection_name ON {PGVECTOR_TABLE} (collection_name);"
            )
            try:
                cursor.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{PGVECTOR_TABLE}_embedding_hnsw
                    ON {PGVECTOR_TABLE}
                    USING hnsw (embedding vector_cosine_ops);
                    """
                )
            except Exception:
                # Index creation support depends on the pgvector/Postgres version.
                pass


def check_collection(collection_name, vector_size):
    ensure_store()
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT vector_size FROM {PGVECTOR_COLLECTIONS_TABLE} WHERE name = %s;",
                (collection_name,),
            )
            existing = cursor.fetchone()
            if existing:
                existing_size = int(existing[0])
                if existing_size != int(vector_size):
                    raise ValueError(
                        f"Collection '{collection_name}' already exists with vector size {existing_size}, "
                        f"but the current embeddings use size {vector_size}."
                    )
                return
            cursor.execute(
                f"INSERT INTO {PGVECTOR_COLLECTIONS_TABLE} (name, vector_size) VALUES (%s, %s);",
                (collection_name, int(vector_size)),
            )


def upsert(collection_name, chunks, embeddings):
    ensure_store()
    with get_connection() as connection:
        with connection.cursor() as cursor:
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                cursor.execute(
                    f"""
                    INSERT INTO {PGVECTOR_TABLE}
                        (id, collection_name, embedding, text_content, source, file_name, page)
                    VALUES
                        (%s, %s, %s::vector, %s, %s, %s, %s)
                    ON CONFLICT (id)
                    DO UPDATE SET
                        collection_name = EXCLUDED.collection_name,
                        embedding = EXCLUDED.embedding,
                        text_content = EXCLUDED.text_content,
                        source = EXCLUDED.source,
                        file_name = EXCLUDED.file_name,
                        page = EXCLUDED.page;
                    """,
                    (
                        build_point_id(chunk),
                        collection_name,
                        vector_literal(embedding),
                        chunk["text"],
                        chunk["source"],
                        chunk["file_name"],
                        str(chunk["page"]),
                    ),
                )


def search(collection_name, query_vector, limit=3):
    ensure_store()
    with get_connection() as connection:
        with connection.cursor() as cursor:
            query_literal = vector_literal(query_vector)
            cursor.execute(
                f"""
                SELECT
                    text_content,
                    source,
                    file_name,
                    page,
                    1 - (embedding <=> %s::vector) AS score
                FROM {PGVECTOR_TABLE}
                WHERE collection_name = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (query_literal, collection_name, query_literal, limit),
            )
            rows = cursor.fetchall()

    return [
        SearchResult(
            payload={
                "text": row[0],
                "source": row[1],
                "file_name": row[2],
                "page": row[3],
            },
            score=float(row[4]) if row[4] is not None else None,
        )
        for row in rows
    ]


def build_point_id(chunk):
    raw_key = f"{chunk['source']}:{chunk['page']}:{chunk['text']}"
    digest = hashlib.md5(raw_key.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))
