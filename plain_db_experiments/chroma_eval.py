import os
import json
import time
import chromadb
from chromadb.config import Settings

from settings import (
    QUERY_EMBEDDINGS_FILE,
    N_RESULTS,
    QUERY_NUM
)

SIZES = [10000, 50000, 100000]
BASE_DB_PREFIX = "chroma_db_"
COLL_NAME = "docs"
RESULTS_DIR = "results"

def evaluate_queries(collection, embeddings_file, n_results, limit=None):
    with open(embeddings_file, "r", encoding="utf-8") as f:
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
    for size in SIZES:
        db_path = f"{BASE_DB_PREFIX}{size}"
        print(f"\n▶ 크기 {size} 평가 시작 (DB: {db_path})")

        client = chromadb.PersistentClient(path=db_path, settings=Settings())
        collection = client.get_collection(name=COLL_NAME)

        # 평가
        results, t = evaluate_queries(
            collection=collection,
            embeddings_file=QUERY_EMBEDDINGS_FILE,
            n_results=N_RESULTS,
            limit=QUERY_NUM
        )

        # 파일 경로 설정
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, f"eval_results_{size}.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        with open(os.path.join(RESULTS_DIR, f"metrics_{size}.json"), "w", encoding="utf-8") as f:
            json.dump({"eval_time_sec": round(t, 4)}, f, indent=2)

        dump_docids(collection, os.path.join(RESULTS_DIR, f"docid_list_{size}.json"))

        print(f"✅ {size} 완료: {t:.2f}초")

if __name__ == "__main__":
    main()
