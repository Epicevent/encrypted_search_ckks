import json
import tenseal as ts
from tenseal import CKKSVector
import numpy as np
from settings import (
    DOC_EMBEDDINGS_FILE,
    QUERY_EMBEDDINGS_FILE,
    POLY_MOD_DEGREE,
    COEFF_MOD_BIT_SIZES,
    GLOBAL_SCALE
)

# ─── 사용자 지정: 비교하고 싶은 doc_id, query_id ───
TARGET_DOC_ID   = "1162363#12"
TARGET_QUERY_ID = "2"
# ───────────────────────────────────────────────────

# 1) CKKS Context 생성 (테스트용)
ctx = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=POLY_MOD_DEGREE,
    coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES
)
ctx.generate_galois_keys()
ctx.global_scale = GLOBAL_SCALE

# 2) JSON에서 해당 문서/쿼리 로드
with open(DOC_EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
    docs = json.load(f)
with open(QUERY_EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
    queries = json.load(f)

# 찾아보기
d = next(filter(lambda x: x["doc_id"] == TARGET_DOC_ID, docs), None)
q = next(filter(lambda x: str(x["query_id"]) == str(TARGET_QUERY_ID), queries), None)

if d is None or q is None:
    raise ValueError(f"doc_id={TARGET_DOC_ID} 또는 query_id={TARGET_QUERY_ID}를 찾을 수 없습니다.")

d_vec = np.array(d["embedding"], dtype=float)
q_vec = np.array(q["embedding"], dtype=float)

# 3) 평문 내적
plain_score = float(np.dot(q_vec, d_vec))

# 4) 정규화 후 평문 내적 (코사인 유사도 기준)
qn = q_vec / (np.linalg.norm(q_vec) + 1e-12)
dn = d_vec / (np.linalg.norm(d_vec) + 1e-12)
plain_norm_score = float(np.dot(qn, dn))

# 5) CKKS 암호화·내적·복호화
enc_q = ts.ckks_vector(ctx, qn.tolist())
enc_d = ts.ckks_vector(ctx, dn.tolist())
raw = enc_q.dot(enc_d).decrypt()[0]
cosine_score = raw 

# 6) 출력
print("=== Comparison for doc_id={} & query_id={} ===".format(TARGET_DOC_ID, TARGET_QUERY_ID))
print(f"Document content excerpt: {d['content'][:50]!r}…")
print(f"Query text excerpt    : {q['text'][:50]!r}…\n")
print(f"1) Plain dot product           : {plain_score:.6f}")
print(f"2) Plain cosine (normalized)   : {plain_norm_score:.6f}")
print(f"3) CKKS raw inner-product      : {raw:.6f}")
print(f"4) CKKS post-scale cosine      : {cosine_score:.6f}")
