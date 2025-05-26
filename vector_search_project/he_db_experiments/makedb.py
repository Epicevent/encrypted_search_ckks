# makedb.py

import os
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from typing import List
import ir_datasets
import ollama
import tenseal as ts  # pip install tenseal
from cryptography.fernet import Fernet
from he_vector_db.store import HEVectorStore

# import updated settings
from settings import (
    SAMPLE_SIZES,
    get_embedding_path,
    get_docid_path,
    get_enc_db_dir,
    get_metrics_path,
    FERNET_KEY_PATH,
    CONTEXT_SECRET,
    POLY_MOD_DEGREE,
    COEFF_MOD_BIT_SIZES,
    GLOBAL_SCALE
)


def build_and_serialize_context(secret_path: str):
    # 이미 컨텍스트 파일이 존재하면 스킵
    if os.path.exists(secret_path):
        print(f"Context file already exists at {secret_path}, skipping generation.")
        return

    # 새 컨텍스트 생성
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MOD_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES
    )
    context.generate_galois_keys()
    context.global_scale = GLOBAL_SCALE
    secret_ctx = context.serialize(save_secret_key=True)

    # 디렉터리 생성 및 파일 저장
    os.makedirs(os.path.dirname(secret_path), exist_ok=True)
    with open(secret_path, "wb") as f:
        f.write(secret_ctx)
    print(f"Context serialized and saved to {secret_path}")


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
    """
    print(f"Initializing HEVectorStore on {db_path}...")
    store = HEVectorStore(db_path=db_path, context_path=context_path, id_key_path=fernet_key_path)

    if store.count() > 0:
        print("Existing embeddings found; skipping ingestion.")
        return

    # load IDs
    with open(docid_list_file, "r", encoding="utf-8") as f:
        doc_ids: List[str] = json.load(f)
    doc_ids = doc_ids[:sample_size]

    # load embeddings
    with open(doc_embeddings_file, "r", encoding="utf-8") as f:
        all_docs = json.load(f)
    docs_map = {d["doc_id"]: d for d in all_docs}
    docs = [docs_map[d] for d in doc_ids if d in docs_map]

    metrics = {"total_time": 0.0}
    start_all = time.perf_counter()
    for rec in docs:
        t0 = time.perf_counter()
        store.add(ids=[rec["doc_id"]], embeddings=[rec["embedding"]], documents=[rec["content"]])
        elapsed = time.perf_counter() - t0
        metrics["total_time"] += elapsed
        print(f"Inserted {rec['doc_id']} in {elapsed:.3f}s")
    metrics["wall_clock_time"] = time.perf_counter() - start_all

    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as mf:
        json.dump(metrics, mf, indent=2)
    print(f"Metrics saved to {metrics_file}")
    store.close()


if __name__ == "__main__":
    # 1) Build context once
    build_and_serialize_context(CONTEXT_SECRET)
    print(f"Context written to {CONTEXT_SECRET}")

    # 2) Ingest for each sample size
    for size in SAMPLE_SIZES:
        db_dir = get_enc_db_dir(size)
        db_file = os.path.join(db_dir, "vectors.db")
        emb_file = get_embedding_path(size)
        id_file = get_docid_path(size)
        metrics_file = get_metrics_path(size)

        print(f"\n=== Ingesting sample_size={size} into {db_file} ===")
        ingest_documents(
            db_path=db_file,
            context_path=CONTEXT_SECRET,
            fernet_key_path=FERNET_KEY_PATH,
            doc_embeddings_file=emb_file,
            docid_list_file=id_file,
            sample_size=size,
            metrics_file=metrics_file
        )