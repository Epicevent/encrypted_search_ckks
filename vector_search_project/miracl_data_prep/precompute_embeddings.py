
import os
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import ir_datasets
import ollama

from settings import (
    SAMPLE_SIZES,
    DATASET_NAME,
    MODEL,
    MAX_WORKERS,
    get_embedding_path,
    get_query_path,
    get_docid_path
)


def run_precompute_all(
    sample_size: int,
    dataset_name: str = DATASET_NAME,
    random_seed: int = None,
    docid_list_file: str = None,
    doc_embeddings_file: str = None,
    query_embeddings_file: str = None,
    model: str = None,
    max_workers: int = None
):
    """
    주어진 sample_size와 dataset_name에 따라 문서 및 쿼리 임베딩을 수행하고 파일로 저장합니다.
    """
    # 기본 설정
    random_seed = random_seed or random_seed
    model = model or MODEL
    max_workers = max_workers or MAX_WORKERS

    # 파일 경로 동적 생성
    docid_list_file = docid_list_file or get_docid_path(sample_size)
    doc_embeddings_file = doc_embeddings_file or get_embedding_path(sample_size)
    query_embeddings_file = query_embeddings_file or get_query_path(sample_size)

    ds = ir_datasets.load(dataset_name)

    # 1) Sample IDs
    random.seed(random_seed)
    relevant = {q.doc_id for q in ds.qrels_iter()}
    all_ids = [d.doc_id for d in ds.docs_iter()]
    relevant &= set(all_ids)
    if len(relevant) > sample_size:
        raise ValueError("sample_size smaller than number of relevant docs")
    negatives = random.sample(list(set(all_ids) - relevant), k=sample_size - len(relevant))
    doc_ids = sorted(relevant) + negatives

    os.makedirs(os.path.dirname(docid_list_file), exist_ok=True)
    with open(docid_list_file, "w", encoding="utf-8") as f:
        json.dump(doc_ids, f, ensure_ascii=False, indent=2)
    print(f"[1] Saved {len(doc_ids)} IDs → {docid_list_file}")

    # 2) One-pass: collect needed docs
    needed = set(doc_ids)
    to_embed = []
    for doc in ds.docs_iter():
        if doc.doc_id in needed:
            to_embed.append(doc)
            if len(to_embed) == len(needed):
                break

    missing = needed - {d.doc_id for d in to_embed}
    if missing:
        raise RuntimeError(f"Missing docs: {missing}")
    print(f"[2] Loaded {len(to_embed)} docs for embedding")

    # 3) Embed Documents with tqdm
    workers = max_workers or (os.cpu_count() or 4)
    doc_records = []

    def embed_doc(doc):
        text = f"{doc.title or ''}\n{doc.text or ''}"
        t0 = time.perf_counter()
        emb = ollama.embeddings(model=model, prompt=text)["embedding"]
        return {
            "doc_id": doc.doc_id,
            "content": text,
            "embedding": emb,
            "embed_time": time.perf_counter() - t0
        }

    print(f"[3] Embedding {len(to_embed)} docs with {workers} workers…")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(embed_doc, d): d.doc_id for d in to_embed}
        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        desc="Docs → embed",
                        unit="doc"):
            rec = fut.result()
            doc_records.append(rec)

    # 4) Save document embeddings
    os.makedirs(os.path.dirname(doc_embeddings_file), exist_ok=True)
    with open(doc_embeddings_file, "w", encoding="utf-8") as f:
        json.dump(doc_records, f, ensure_ascii=False, indent=2)
    print(f"[4] Saved {len(doc_records)} doc embeddings → {doc_embeddings_file}")

    # 5) Embed Queries with tqdm
    queries = list(ds.queries_iter())
    query_records = []

    def embed_query(q):
        t0 = time.perf_counter()
        emb = ollama.embeddings(model=model, prompt=q.text)["embedding"]
        return {
            "query_id": q.query_id,
            "text": q.text,
            "embedding": emb,
            "embed_time": time.perf_counter() - t0
        }

    print(f"[5] Embedding {len(queries)} queries…")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(embed_query, q): q.query_id for q in queries}
        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        desc="Queries → embed",
                        unit="qry"):
            rec = fut.result()
            query_records.append(rec)

    # 6) Save query embeddings
    os.makedirs(os.path.dirname(query_embeddings_file), exist_ok=True)
    with open(query_embeddings_file, "w", encoding="utf-8") as f:
        json.dump(query_records, f, ensure_ascii=False, indent=2)
    print(f"[6] Saved {len(query_records)} query embeddings → {query_embeddings_file}")


if __name__ == "__main__":
    # 반복할 샘플 사이즈
    for size in SAMPLE_SIZES:
        print(f"\n=== run_precompute_all(sample_size={size}) ===")
        run_precompute_all(
            sample_size=size,
            dataset_name=DATASET_NAME
        )
