#!/usr/bin/env python3
import os
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import ir_datasets
import ollama
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

from settings import (
    DOCID_LIST_FILE,
    PLAIN_DATA_DIR,
    METRICS_PLAIN_JSON,    # e.g. "./plain_data/plain_metrics.json"
    MODEL,
    COLLECTION_NAME,
    SAMPLE_SIZE,
    RANDOM_SEED,
    MAX_WORKERS
)

def main():
    # 1) Load or sample doc IDs
    if os.path.exists(DOCID_LIST_FILE):
        with open(DOCID_LIST_FILE, "r", encoding="utf-8") as f:
            doc_ids = json.load(f)
        print(f"Loaded {len(doc_ids)} doc IDs from {DOCID_LIST_FILE}")
    else:
        # fallback: sample from dataset
        random.seed(RANDOM_SEED)
        ds = ir_datasets.load("miracl/ko/dev")
        relevant = {q.doc_id for q in ds.qrels_iter()}
        all_ids = [d.doc_id for d in ds.docs_iter()]
        neg = list(set(all_ids) - relevant)
        n_neg = SAMPLE_SIZE - len(relevant)
        if n_neg < 0:
            raise ValueError("SAMPLE_SIZE smaller than number of relevant docs")
        neg_ids = random.sample(neg, k=n_neg)
        doc_ids = list(relevant) + neg_ids
        os.makedirs(os.path.dirname(DOCID_LIST_FILE), exist_ok=True)
        with open(DOCID_LIST_FILE, "w", encoding="utf-8") as f:
            json.dump(doc_ids, f, ensure_ascii=False, indent=2)
        print(f"Sampled and saved {len(doc_ids)} doc IDs to {DOCID_LIST_FILE}")

    # 2) Build doc text map
    ds = ir_datasets.load("miracl/ko/dev")
    doc_map = {
        d.doc_id: (d.title or "") + "\n" + (d.text or "")
        for d in ds.docs_iter()
        if d.doc_id in set(doc_ids)
    }

    # 3) Init ChromaDB client + collection
    os.makedirs(PLAIN_DATA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(
        path=PLAIN_DATA_DIR,
        settings=Settings(anonymized_telemetry=False),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    # recreate collection each run
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # 4) Embed & add in parallel
    workers = MAX_WORKERS or (os.cpu_count() or 4)
    metrics = {
        "threads_used": workers,
        "total_embedding_time": 0.0,
        "wall_clock_time": 0.0
    }
    print(f"Using {workers} threads for plain DB ingestion…")
    start_all = time.perf_counter()

    def embed_and_add(doc_id):
        text = doc_map.get(doc_id)
        if text is None:
            return None, 0.0
        t0 = time.perf_counter()
        emb = ollama.embeddings(model=MODEL, prompt=text)["embedding"]
        t1 = time.perf_counter()
        # add to Chroma
        collection.add(
            ids=[doc_id],
            embeddings=[emb],
            metadatas=[{"doc_id": doc_id}],
            documents=[text]
        )
        return doc_id, (time.perf_counter() - t0)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(embed_and_add, did): did for did in doc_ids}
        for fut in as_completed(futures):
            doc_id, emb_time = fut.result()
            if doc_id:
                metrics["total_embedding_time"] += emb_time
                print(f"Inserted {doc_id} (embed_time={emb_time:.3f}s)")

    metrics["wall_clock_time"] = time.perf_counter() - start_all
    client.persist()

    # 5) Save metrics
    os.makedirs(os.path.dirname(METRICS_PLAIN_JSON), exist_ok=True)
    with open(METRICS_PLAIN_JSON, "w", encoding="utf-8") as mf:
        json.dump(metrics, mf, ensure_ascii=False, indent=2)

    print(f"Plain DB built under '{PLAIN_DATA_DIR}'")
    print(f"Metrics saved to '{METRICS_PLAIN_JSON}'")

if __name__ == "__main__":
    main()
