import sqlite3
import json
import numpy as np
import tenseal as ts
from tenseal import CKKSVector
from cryptography.fernet import Fernet
from settings import (
    DB_PATH,
    CONTEXT_SECRET,
    FERNET_KEY_PATH,
    QUERY_EMBEDDINGS_FILE,
    POLY_MOD_DEGREE,
    COEFF_MOD_BIT_SIZES,
    GLOBAL_SCALE
)

# 사용자 지정: 비교할 doc_id와 query_id
TARGET_DOC_ID = "317339#6"
TARGET_QUERY_ID = "2"

# 1) 복호화용 CKKS context 로드
with open(CONTEXT_SECRET, "rb") as f:
    ctx_bytes = f.read()
ctx = ts.context_from(ctx_bytes)

# 2) Fernet 키 로드
with open(FERNET_KEY_PATH, "rb") as f:
    key_bytes = f.read()
fernet = Fernet(key_bytes)

# 3) 쿼리 임베딩 로드 및 정규화
with open(QUERY_EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
    queries = json.load(f)
q = next(q for q in queries if str(q["query_id"]) == TARGET_QUERY_ID)
q_vec = np.array(q["embedding"], dtype=float)
qn = q_vec / (np.linalg.norm(q_vec) + 1e-12)

# 4) DB 순회하며 목표 문서 찾기
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("SELECT id, ciphertext FROM vectors")
found = False
for enc_id, blob in cur:
    # 4a) ID 복호화
    try:
        doc_id = fernet.decrypt(enc_id).decode()
    except:
        doc_id = enc_id.decode() if isinstance(enc_id, bytes) else str(enc_id)
    # 4b) 목표 문서이면 inner-product 계산
    if doc_id == TARGET_DOC_ID:
        found = True
        enc_d = CKKSVector.load(ctx, blob)
        enc_q = ts.ckks_vector(ctx, qn.tolist())
        raw = enc_q.dot(enc_d).decrypt()[0]
        print(f"Matched doc_id={doc_id}")
        print(f"Computed CKKS inner-product (cosine): {raw:.6f}")
        break

conn.close()
if not found:
    print(f"Doc_id={TARGET_DOC_ID} not found in DB.")
