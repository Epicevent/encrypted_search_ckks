# chroma.py — Create Chroma DB using settings.py

import os
import json
import time
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from settings import (
    SAMPLE_SIZES,
    get_doc_embeddings_path,
    get_plain_db_dir,
)

# ── 설정 ─────────────────────────────
COLL_NAME  = "docs"
BATCH_SIZE = 1000
TIME_LOG   = "embedding_load_times.txt"

# ── 시간 로그 초기화 ─────────────────
with open(TIME_LOG, "w", encoding="utf-8") as f:
    f.write("=== Embedding Load Times ===\n")

# ── 각 크기별로 DB 생성 ──────────────
for size in SAMPLE_SIZES:
    db_path = get_plain_db_dir(size)
    os.makedirs(db_path, exist_ok=True)

    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    try:
        collection = client.get_collection(name=COLL_NAME)
    except Exception:
        print(f"Creating new collection: {COLL_NAME} in {db_path}")
        collection = client.create_collection(
            name=COLL_NAME,
            metadata={"hnsw:space": "cosine"}  # 거리 메트릭 지정
        )


    embedding_file = get_doc_embeddings_path(size)

    start = time.time()
    with open(embedding_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    total = len(items)
    print(f"🚀 {size}개 문서 로딩 시작 (총 {total}개)")

    for i in range(0, total, BATCH_SIZE):
        batch = items[i:i+BATCH_SIZE]
        ids        = [item["doc_id"]   for item in batch]
        contents   = [item["content"]  for item in batch]
        embeddings = [item["embedding"] for item in batch]

        collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings
        )

        print(f"  ▶ {i+BATCH_SIZE if i+BATCH_SIZE < total else total}/{total} 완료")

    elapsed = time.time() - start
    with open(TIME_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{size}] count={total}, time={elapsed:.2f}s\n")

    print(f"✅ {size:,}개 문서 저장 완료 → {db_path} ({elapsed:.2f}s)")

print("\n🎉 모든 DB 저장이 성공적으로 완료되었습니다.")
