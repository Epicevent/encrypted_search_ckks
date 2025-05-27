
#!/usr/bin/env python3
import os
import json
import time
import sqlite3
from tqdm import tqdm
from cryptography.fernet import Fernet
import pandas as pd
from he_vector_db.store import HEVectorStore

from settings import (
    SAMPLE_SIZES,
    get_query_path,
    get_enc_db_dir,
    get_eval_path,
    get_docid_path,
    get_metrics_path,
    FERNET_KEY_PATH,
    CONTEXT_SECRET,
    MAX_WORKERS,
    N_RESULTS,
    QUERY_NUM
)


def evaluate_queries(
    embeddings_file: str,
    store: HEVectorStore,
    fernet: Fernet,
    n_results: int,
    max_workers: int,
    limit: int = None
):
    """
    Load query embeddings, perform parallel encrypted vector queries,
    and return formatted results.
    """
    with open(embeddings_file, "r", encoding="utf-8") as f:
        queries = json.load(f)
    if limit is not None:
        queries = queries[:limit]
    embeddings = [q["embedding"] for q in queries]
    query_ids = [q["query_id"] for q in queries]
    if not embeddings:
        print("No embeddings found in the file.")
        return []
    if len(embeddings) != len(query_ids):
        raise ValueError("Mismatch between number of embeddings and query IDs.")
    print(f"Loaded {len(embeddings)} query embeddings from {embeddings_file}")

    start = time.perf_counter()
    all_hits = store.query(
        embeddings=embeddings,
        n_results=n_results,
        max_workers=max_workers
    )
    wall_time = time.perf_counter() - start
    print(f"Parallel search for {len(embeddings)} queries took {wall_time:.2f}s")

    results = []
    for qid, hits in zip(query_ids, all_hits):
        hit_list = []
        for rank, (enc_id, _, score) in enumerate(hits, start=1):
            try:
                doc_id = fernet.decrypt(enc_id).decode()
            except Exception:
                doc_id = enc_id.decode() if hasattr(enc_id, 'decode') else str(enc_id)
            hit_list.append({"rank": rank, "doc_id": doc_id, "score": score})
        results.append({"query_id": qid, "results": hit_list})
    return results


def dump_all_docids(
    db_path: str,
    fernet: Fernet,
    out_path: str
):
    """
    Dump and decrypt all document IDs stored in the HEVectorStore SQLite DB.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id FROM vectors")
    rows = cur.fetchall()
    conn.close()

    doc_ids = []
    for (enc_id,) in rows:
        try:
            doc_id = fernet.decrypt(enc_id).decode()
        except Exception:
            doc_id = enc_id.decode() if hasattr(enc_id, 'decode') else str(enc_id)
        doc_ids.append(doc_id)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc_ids, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(doc_ids)} doc ids to {out_path}")


def main():
    # Load Fernet key and initialize decryptor
    with open(FERNET_KEY_PATH, "rb") as f:
        key_bytes = f.read()
    fernet = Fernet(key_bytes)

    # Loop over each sample size
    for size in SAMPLE_SIZES:
        print(f"\n=== Evaluating sample_size={size} ===")
        db_dir = get_enc_db_dir(size)
        db_path = os.path.join(db_dir, "vectors.db")
        store = HEVectorStore(
            context_path=CONTEXT_SECRET,
            db_path=db_path,
            id_key_path=FERNET_KEY_PATH
        )

        # Perform query evaluation
        query_file = get_query_path(size)
        eval_results = evaluate_queries(
            embeddings_file=query_file,
            store=store,
            fernet=fernet,
            n_results=N_RESULTS,
            max_workers=MAX_WORKERS,
            limit=QUERY_NUM
        )

        # Update metrics
        metrics_file = get_metrics_path(size)
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        if os.path.exists(metrics_file):
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        else:
            metrics = {}
        metrics["eval_time_sec"] = round(time.time() - time.time(), 4)
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        # Save evaluation results
        eval_file = get_eval_path(size)
        os.makedirs(os.path.dirname(eval_file), exist_ok=True)
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {eval_file}")

        # Dump all doc IDs
        docid_out = get_docid_path(size)
        dump_all_docids(
            db_path=db_path,
            fernet=fernet,
            out_path=docid_out
        )

        store.close()

if __name__ == "__main__":
    main()

