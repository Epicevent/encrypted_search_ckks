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
    QUERY_EMBEDDINGS_FILE,
    DB_PATH,
    FERNET_KEY_PATH,
    CONTEXT_SECRET,
    EVAL_RESULTS_FILE,
    DOCID_LIST_FILE,
    MAX_WORKERS,
    N_RESULTS,
    METRICS_PATH,
    QUERY_NUM
)

def evaluate_queries(
    embeddings_file: str,
    store: HEVectorStore,
    fernet: Fernet,
    n_results: int,
    max_workers: int,
    limit: int = None  # Optional limit on number of queries to process
):
    # 1) 쿼리 임베딩 로드
    with open(embeddings_file, "r", encoding="utf-8") as f:
        queries = json.load(f)
    # 2) Optionally truncate to first `limit`
    if limit is not None:
        queries = queries[:limit]
    embeddings = [q["embedding"] for q in queries]
    query_ids  = [q["query_id"]  for q in queries]

    # 2) 병렬 검색
    start = time.perf_counter()
    all_hits = store.query(
        embeddings=embeddings,
        n_results=n_results,
        max_workers=max_workers
    )
    wall_time = time.perf_counter() - start
    print(f"Parallel search for {len(embeddings)} queries took {wall_time:.2f}s")

    # 3) 결과 포맷
    results = []
    for qid, hits in zip(query_ids, all_hits):
        hit_list = []
        for rank, (enc_id, _, score) in enumerate(hits, start=1):
            try:
                doc_id = fernet.decrypt(enc_id).decode()
            except:
                doc_id = enc_id.decode() if isinstance(enc_id, (bytes, bytearray)) else str(enc_id)
            hit_list.append({"rank": rank, "doc_id": doc_id, "score": score})
        results.append({"query_id": qid, "results": hit_list})

    return results

def dump_all_docids(
    db_path: str,
    fernet: Fernet,
    out_path: str
):
    # SQLite에서 모든 암호화된 ID 로드
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id FROM vectors")
    rows = cur.fetchall()
    conn.close()

    # 복호화
    doc_ids = []
    for (enc_id,) in rows:
        try:
            doc_id = fernet.decrypt(enc_id).decode()
        except:
            doc_id = enc_id.decode() if isinstance(enc_id, (bytes, bytearray)) else str(enc_id)
        doc_ids.append(doc_id)

    # 저장
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc_ids, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(doc_ids)} doc ids to {out_path}")

def main():
    # 0) 준비: Fernet, HEVectorStore
    with open(FERNET_KEY_PATH, "rb") as f:
        key_bytes = f.read()
    fernet = Fernet(key_bytes)

    store = HEVectorStore(
        context_path=CONTEXT_SECRET,
        db_path=DB_PATH,
        id_key_path=FERNET_KEY_PATH
    )

    # 1) 평가 실행
    start = time.perf_counter()
    eval_results = evaluate_queries(
        embeddings_file=QUERY_EMBEDDINGS_FILE,
        store=store,
        fernet=fernet,
        n_results=N_RESULTS,
        max_workers=MAX_WORKERS,
        limit= QUERY_NUM # None means evaluate all queries
    )
    eval_time = time.perf_counter() - start
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    metrics["eval_time_sec"]      = round(eval_time, 4)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    # 2) 평가 결과 저장
    os.makedirs(os.path.dirname(EVAL_RESULTS_FILE), exist_ok=True)
    with open(EVAL_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {EVAL_RESULTS_FILE}")

    # 3) DB에 들어있는 모든 doc_id 추출·복호화·저장
    dump_all_docids(
        db_path=DB_PATH,
        fernet=fernet,
        out_path=DOCID_LIST_FILE
    )

    store.close()

if __name__ == "__main__":
    main()
