# chroma_eval.py — Evaluate ChromaDB using settings.py

import os
import json
import time
import chromadb
from chromadb.config import Settings
from settings import (
    SAMPLE_SIZES,
    N_RESULTS,
    QUERY_NUM,
    get_query_embeddings_path,
    get_eval_path,
    get_metrics_path,
    get_docid_list_path,
    get_plain_db_dir,
)

COLL_NAME = "docs"

def evaluate_queries(collection, query_embeddings_file, n_results, limit=None):
    with open(query_embeddings_file, "r", encoding="utf-8") as f:
        queries = json.load(f)
    if limit:
        queries = queries[:limit]

    embeddings = [q["embedding"] for q in queries]
    query_ids  = [q["query_id"]  for q in queries]

    results = []
    start = time.perf_counter()
    for qid, vec in zip(query_ids, embeddings):
        hits = collection.query(query_embeddings=[vec], n_results=n_results)
        dids = hits['ids'][0]
        scores = hits['distances'][0]
        results.append({
            "query_id": qid,
            "results": [
                {"rank": i + 1, "doc_id": did, "score": score}
                for i, (did, score) in enumerate(zip(dids, scores))
            ]
        })
    wall_time = time.perf_counter() - start
    return results, wall_time

def dump_docids(collection, out_path):
    docs = collection.get(include=["documents"])["documents"]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

def main():
    for size in SAMPLE_SIZES:
        db_path = get_plain_db_dir(size)
        print(f"\n▶ 크기 {size} 평가 시작 (DB: {db_path})")

        client = chromadb.PersistentClient(path=db_path, settings=Settings())
        collection = client.get_collection(name=COLL_NAME)

        results, t = evaluate_queries(
            collection=collection,
            query_embeddings_file=get_query_embeddings_path(size),
            n_results=N_RESULTS,
            limit=QUERY_NUM
        )

        # 평가 결과 저장
        os.makedirs(os.path.dirname(get_eval_path(size)), exist_ok=True)

        with open(get_eval_path(size), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        with open(get_metrics_path(size), "w", encoding="utf-8") as f:
            json.dump({"eval_time_sec": round(t, 4)}, f, indent=2)

        dump_docids(collection, get_docid_list_path(size))

        print(f"✅ {size} 완료: {t:.2f}초")

if __name__ == "__main__":
    main()
