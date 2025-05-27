# makedb.py — Build HEVectorStore DBs using settings.py

import os
import json
import time
from typing import List
import tenseal as ts
from he_vector_db.store import HEVectorStore
from settings import (
    SAMPLE_SIZES,
    get_doc_embeddings_path,
    get_docid_list_path,
    get_he_db_path,
    get_metrics_path,
    FERNET_KEY_PATH,
    CONTEXT_SECRET,
    POLY_MOD_DEGREE,
    COEFF_MOD_BIT_SIZES,
    GLOBAL_SCALE,
)


def build_and_serialize_context(secret_path: str):
    if os.path.exists(secret_path):
        print(f"Context already exists at {secret_path}, skipping.")
        return

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MOD_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES
    )
    context.generate_galois_keys()
    context.global_scale = GLOBAL_SCALE
    secret_ctx = context.serialize(save_secret_key=True)

    os.makedirs(os.path.dirname(secret_path), exist_ok=True)
    with open(secret_path, "wb") as f:
        f.write(secret_ctx)
    print(f"Context saved to {secret_path}")


def ingest_documents(
    db_path: str,
    context_path: str,
    fernet_key_path: str,
    doc_embeddings_file: str,
    docid_list_file: str,
    sample_size: int,
    metrics_file: str
):
    print(f"🔐 Initializing HEVectorStore @ {db_path}")
    store = HEVectorStore(db_path=db_path, context_path=context_path, id_key_path=fernet_key_path)

    if store.count() > 0:
        print("📦 Existing embeddings found. Skipping ingestion.")
        return

    # Load IDs
    with open(docid_list_file, "r", encoding="utf-8") as f:
        doc_ids: List[str] = json.load(f)
    doc_ids = doc_ids[:sample_size]

    # Load embeddings
    with open(doc_embeddings_file, "r", encoding="utf-8") as f:
        all_docs = json.load(f)
    docs_map = {d["doc_id"]: d for d in all_docs}
    docs = [docs_map[d] for d in doc_ids if d in docs_map]

    metrics = {"total_time": 0.0}
    start_all = time.perf_counter()

    print(f"🚀 Ingesting {len(docs)} documents...")
    for rec in docs:
        t0 = time.perf_counter()
        store.add(
            ids=[rec["doc_id"]],
            embeddings=[rec["embedding"]],
            documents=[rec["content"]]
        )
        elapsed = time.perf_counter() - t0
        metrics["total_time"] += elapsed
        print(f"  + {rec['doc_id']} ({elapsed:.3f}s)")

    metrics["wall_clock_time"] = time.perf_counter() - start_all

    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as mf:
        json.dump(metrics, mf, indent=2)
    print(f"✅ Metrics saved to {metrics_file}")

    store.close()


if __name__ == "__main__":
    # 1. Context 생성
    build_and_serialize_context(CONTEXT_SECRET)

    # 2. 각 크기별 DB 생성
    for size in SAMPLE_SIZES:
        db_path       = get_he_db_path(size)
        doc_emb_path  = get_doc_embeddings_path(size)
        docid_path    = get_docid_list_path(size)
        metrics_path  = get_metrics_path(size)
        # ✅ DB 디렉터리 미리 생성
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        print(f"\n=== Sample Size: {size} ===")
        ingest_documents(
            db_path=db_path,
            context_path=CONTEXT_SECRET,
            fernet_key_path=FERNET_KEY_PATH,
            doc_embeddings_file=doc_emb_path,
            docid_list_file=docid_path,
            sample_size=size,
            metrics_file=metrics_path
        )
