import os
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import numpy as np
from typing import List, Dict, Any
import pandas as pd
import ir_datasets
import ollama
import tenseal as ts # pip install tenseal
from cryptography.fernet import Fernet
from he_vector_db.store import HEVectorStore
from settings import DB_PATH, FERNET_KEY_PATH, CONTEXT_SECRET, MODEL, SAMPLE_SIZE, METRICS_PATH  ,DOC_EMBEDDINGS_FILE, DOCID_LIST_FILE
from settings import POLY_MOD_DEGREE, COEFF_MOD_BIT_SIZES, GLOBAL_SCALE, RANDOM_SEED, MAX_WORKERS
# Ensure ollama is running and the model is available



def build_and_serialize_context(secret_path):
    context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree =POLY_MOD_DEGREE,
            coeff_mod_bit_sizes = COEFF_MOD_BIT_SIZES
            )


    context.generate_galois_keys()
    context.global_scale = GLOBAL_SCALE
    secret_context = context.serialize(save_secret_key = True)
    # Save the serialized context to the specified path     
    os.makedirs(os.path.dirname(secret_path), exist_ok=True)
    with open(secret_path, "wb") as f:
        f.write(secret_context)


def generate_embedding(doc):
    """
    Worker function: takes a doc object, returns (doc_id, embedding, content, elapsed_time).
    """
    content = (doc.title or "") + "\n" + (doc.text or "")
    start = time.perf_counter()
    embedding = ollama.embeddings(model=MODEL, prompt=content)["embedding"]
    elapsed = time.perf_counter() - start
    return doc.doc_id, embedding, content, elapsed

def ingest_documents(
    db_path: str,
    context_path: str,
    fernet_key_path: str,
    doc_embeddings_file: str,
    docid_list_file: str,
    sample_size: int,
    metrics_file: str 
):
    """
    Ingest precomputed document embeddings into HEVectorStore.
    Reads embeddings from `doc_embeddings_file`, no real-time generation.
    """
    print("Initializing HEVectorStore and DB...")
    print(f"Context path: {context_path}")
    print(f"Database path: {db_path}")
    print(f"Fernet key path: {fernet_key_path}")
    print(f"Doc embeddings file: {doc_embeddings_file}")
    print(f"Docid list file: {docid_list_file}")
    print(f"Metrics file: {metrics_file}")
    print(f"Sample size: {sample_size}")
    
    t0 = time.perf_counter()
    # Load or create Fernet key
    store = HEVectorStore(db_path=db_path, context_path = context_path, id_key_path=fernet_key_path)

    init_time = time.perf_counter() - t0
    print(f"DB init took {init_time:.3f}s")

    # Skip if already ingested
    if store.count() > 0:
        print("이미 임베딩된 데이터가 존재합니다. Ingestion을 건너뜁니다.")
        return
    print("No existing embeddings found, proceeding with ingestion...")


    with open(docid_list_file, "r", encoding="utf-8") as f:
        doc_ids: List[str] = json.load(f)
    if len(doc_ids) != sample_size:
        print(f"Warning: docid_list has {len(doc_ids)} IDs, but sample_size={sample_size}.")
        # choose to truncate or pad; here we truncate:
        doc_ids = doc_ids[:sample_size]
    print(f"Ingesting {len(doc_ids)} documents (sample_size={sample_size})")

    # 4) Load embeddings
    with open(doc_embeddings_file, "r", encoding="utf-8") as f:
        all_docs = json.load(f)
    # Map doc_id → record for fast lookup
    docs_map = {d["doc_id"]: d for d in all_docs}
    docs = [docs_map[d] for d in doc_ids if d in docs_map]
    if len(docs) != sample_size:
        print(f"Warning: Loaded {len(docs)} documents, expected {sample_size}.")
        # choose to truncate or pad; here we truncate:
        docs = docs[:sample_size]
    print(f"Loaded {len(docs)} documents with embeddings.")
    metrics = {"total_time": 0.0}
    start_all = time.perf_counter()
    for doc in docs:
        start = time.perf_counter()
        store.add(
            ids=[doc["doc_id"]],
            embeddings=[doc["embedding"]],
            documents=[doc["content"]],
        )
        elapsed = time.perf_counter() - start
        metrics["total_time"] += elapsed
        print(f"Inserted {doc['doc_id']} in {elapsed:.3f}s")
    metrics["wall_clock_time"] = time.perf_counter() - start_all
    # 6) Save metrics
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as mf:
        json.dump(metrics, mf, indent=2)

    print(f"Metrics saved to {metrics_file}")
    store.close()

if __name__ == "__main__":

    # 1) Build and serialize context    
    build_and_serialize_context(CONTEXT_SECRET)
    print(f"Context serialized to {CONTEXT_SECRET}")  


    ingest_documents(DB_PATH, CONTEXT_SECRET, FERNET_KEY_PATH,  DOC_EMBEDDINGS_FILE, DOCID_LIST_FILE,SAMPLE_SIZE, METRICS_PATH)
