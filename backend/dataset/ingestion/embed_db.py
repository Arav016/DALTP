import hashlib
import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()

qdrant_endpoint = os.getenv("Qdrant_endpoint")
qdrant_api_key = os.getenv("Qdrant_api_key")
client = QdrantClient(url=qdrant_endpoint, api_key=qdrant_api_key)

def check_collection(collection_name, vector_size):
        collections =client.get_collections().collections
        existing_names = set()
        for collection in collections:
            existing_names.add(collection.name)
        
        if collection_name in existing_names:
            print(f"Collection '{collection_name}' already exists.")
            return

        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        print(f"Collection '{collection_name}' created successfully.")

def upsert(collection_name, chunks, embeddings):
        points = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            points.append(
                models.PointStruct(
                    id=build_point_id(chunk),
                    vector=embedding,
                    payload={
                        "text": chunk["text"],
                        "source": chunk["source"],
                        "file_name": chunk["file_name"],
                        "page": chunk["page"],
                    },
                )
            )

        client.upsert(collection_name=collection_name, points=points)

def build_point_id(chunk):
    raw_key = f"{chunk['source']}:{chunk['page']}:{chunk['text']}" #makes unque key for chunk
    digest = hashlib.md5(raw_key.encode("utf-8")).hexdigest() # hash key for chunk
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest)) #key for chunk in qdrant