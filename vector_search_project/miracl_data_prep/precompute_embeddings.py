
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
    get_query_path
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
 
    doc_embeddings_file = doc_embeddings_file or get_embedding_path(sample_size)
    query_embeddings_file = query_embeddings_file or get_query_path(sample_size)

    ds = ir_datasets.load(dataset_name)

    random.seed(random_seed)


    # ▶️ 빠른 count API 활용
    DOC_TOTAL   = ds.docs_count()
    QRELS_TOTAL = ds.qrels_count()

    print(f"✅ Total documents: {DOC_TOTAL:,}")
    print(f"✅ Total qrels: {QRELS_TOTAL:,}")

    # 1. qrels 로딩
    qrels_list = list(ds.qrels_iter())
    relevant = {q.doc_id for q in tqdm(qrels_list, desc="Loading qrels", total=QRELS_TOTAL)}

    # 2. 전체 문서 ID 수집 (조금 느릴 수 있음 → tqdm으로)
    all_ids = [d.doc_id for d in tqdm(ds.docs_iter(), desc="Collecting all doc IDs", total=DOC_TOTAL)]

    # 3. relevant 필터링
    relevant &= set(all_ids)
    if len(relevant) > sample_size:
        raise ValueError("sample_size smaller than number of relevant docs")

    negatives = random.sample(list(set(all_ids) - relevant), k=sample_size - len(relevant))
    doc_ids = sorted(relevant) + negatives

    # 4. 바로 문서 임베딩 (메모리에 올리지 않음)
    needed = set(doc_ids)
    print(f"[2]📦 Embedding {len(needed)} documents via docs_store().get_many_iter()...")

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


    docs = list(ds.docs_store().get_many_iter(needed))  # 미리 리스트로 변환

    doc_records = []
    print(f"[2] Embedding {len(docs)} docs sequentially…")

    for i, doc in enumerate(docs, 1):
        rec = embed_doc(doc)
        doc_records.append(rec)

        # 매 10개마다 한 번씩 출력 (원하면 1개마다도 가능)
        if i % 10 == 0 or i == len(docs):
            tqdm.write(f"✅ Embedded {i}/{len(docs)} documents")



    # 3) Save document embeddings
    os.makedirs(os.path.dirname(doc_embeddings_file), exist_ok=True)
    with open(doc_embeddings_file, "w", encoding="utf-8") as f:
        json.dump(doc_records, f, ensure_ascii=False, indent=2)
    print(f"[3] Saved {len(doc_records)} doc embeddings → {doc_embeddings_file}")

    # 4) Embed Queries with tqdm
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

    print(f"[4] Embedding {len(queries)} queries…")
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
    print(f"[5] Saved {len(query_records)} query embeddings → {query_embeddings_file}")


if __name__ == "__main__":
    # 반복할 샘플 사이즈
    for size in SAMPLE_SIZES:
        print(f"\n=== run_precompute_all(sample_size={size}) ===")
        run_precompute_all(
            sample_size=size,
            dataset_name=DATASET_NAME
        )
