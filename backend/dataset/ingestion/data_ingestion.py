import argparse
import csv
import hashlib
import os
import uuid
import fitz
import embed_db
import embeddings as embed
import pandas as pd
from pathlib import Path
from typing import Iterable
from docx import Document as DocxDocument
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AzureOpenAI, OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".xlsx"}

def ingest_documents(
    input_path,
    collection_name,
    chunk_size= 1000,
    chunk_overlap= 150,
    batch_size= 64,
    ):
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    documents = []

    if path.is_file():
        documents = list(load_file(path))
    else:
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                documents.extend(load_file(file_path))

    if not documents:
        raise ValueError(f"No supported files found under '{input_path}'.")
    
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("No text chunks were created from the supplied documents.")

    first_embedding = embed.generate_embeddings([chunks[0]["text"]])[0]
    embed_db.check_collection(collection_name, len(first_embedding))
    

    for batch in batched(chunks, batch_size):
        texts = [chunk["text"] for chunk in batch]
        embeddings = embed.generate_embeddings(texts)
        embed_db.upsert(collection_name, batch, embeddings)

    return {"documents_processed": len(documents),
        "chunks_created": len(chunks),
        "collection_name": collection_name,
    }

def load_file(file_path):

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return list(load_pdf(file_path))
        
    if suffix == ".docx":
        return [load_docx(file_path)]   
    
    if suffix == ".txt":
        return [load_txt(file_path)]

    if suffix == ".csv":
        return list(load_csv(file_path))

    if suffix == ".xlsx":
        return list(load_xlsx(file_path))

    raise ValueError(f"Unsupported file type: {file_path.suffix}")

def load_pdf(file_path):
    documents=[]
    with fitz.open(file_path) as pdf_file:
        for page_number, page in enumerate(pdf_file, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            documents.append({
                "text": text,
                "source": str(file_path),
                "file_name": file_path.name,
                "page": page_number,
            })
    return documents

def load_docx(file_path):
    document = DocxDocument(file_path)
    paragraphs = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            paragraphs.append(text)

    text = "\n".join(paragraphs).strip()
    if not text:
        raise ValueError(f"No readable text found in DOCX file: {file_path}")

    return {
        "text": text,
        "source": str(file_path),
        "file_name": file_path.name,
        "page": 1,
    }

def load_txt(file_path):
    text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        raise ValueError(f"No readable text found in TXT file: {file_path}")

    return {
        "text": text,
        "source": str(file_path),
        "file_name": file_path.name,
        "page": 1,
    }


def load_csv(file_path):
    with file_path.open("r", encoding="utf-8", errors="ignore", newline="") as csv_file:
        reader = csv.DictReader(csv_file) # reads each row as a dict with column names as keys
        document=[]
        for row_number, row in enumerate(reader, start=1):
            cleaned = {}
            text=[]
            for key, value in row.items():
                if value not in (None, ""):
                    cleaned[key] = value
            if not cleaned:
                continue
            for column, value in cleaned.items():
                text.append(f"{column}: {value}")
            text="\n".join(text)
            document.append({
                "text": text,
                "source": str(file_path),
                "file_name": file_path.name,
                "page": row_number,
            })
    return document

def load_xlsx(file_path):
    workbook = pd.read_excel(file_path, sheet_name=None) #reads all sheets
    document=[]
    for sheet_name, dataframe in workbook.items(): #process each sheet separately
        normalized = dataframe.fillna("") #replace NaN with empty string

        # iterrows gives (index,row) in df
        for row_number, (_, row) in enumerate(normalized.iterrows(), start=1):
            cleaned = {}
            text=[]
            for column, value in row.items():
                if str(value).strip():
                    cleaned[column] = value
            if not cleaned:
                continue
            for column, value in cleaned.items():
                text.append(f"{column}: {value}")
            text = "\n".join(text)
            document.append({
                "text": text,
                "source": str(file_path),
                "file_name": file_path.name,
                "page": f"{sheet_name}:{row_number}",
            })
    return document


def chunk_documents(documents: list[dict],chunk_size,chunk_overlap,):
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks= []
    for document in documents:
        chunk_texts = split_text(document["text"], chunk_size, chunk_overlap)
        for chunk_text in chunk_texts:
            chunks.append(
                {
                    "text": chunk_text,
                    "source": document["source"],
                    "file_name": document["file_name"],
                    "page": document["page"],
                }
            )

    return chunks


def split_text(text, split_size, split_overlap):
    cleaned_text = " ".join(text.split())
    if not cleaned_text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=split_size,
        chunk_overlap=split_overlap,
    )
    return splitter.split_text(cleaned_text)


def batched(items, batch_size):
    batches=[]
    for index in range(0, len(items), batch_size):
        batches.append(items[index : index + batch_size])
    return batches


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract documents, create chunks, embed them, and store them in Qdrant."
    )
    parser.add_argument("--input", required=True, help="Path to a file or directory to ingest.")
    parser.add_argument("--collection", required=True, help="Qdrant collection name.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Character length of each chunk.")
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Number of overlapping characters between consecutive chunks.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = ingest_documents(
        input_path=args.input,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
    )
    print(
        f"Ingested {result['documents_processed']} document units into "
        f"{result['chunks_created']} chunks in '{result['collection_name']}'."
    )


if __name__ == "__main__":
    main()
