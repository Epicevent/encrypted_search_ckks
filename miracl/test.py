import json
import tenseal as ts
from tenseal import CKKSVector
import numpy as np
import ollama
from settings import (
    DOC_EMBEDDINGS_FILE,
    QUERY_EMBEDDINGS_FILE,
    POLY_MOD_DEGREE,
    COEFF_MOD_BIT_SIZES,
    GLOBAL_SCALE,
    MODEL
)

# ─── 사용자 지정: 비교하고 싶은 doc_id, query_id ───
TARGET_DOC_ID   = "1016438#9"
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
cosine_score = raw  # 이미 스케일 보정된 값

# 6) 기존 임베딩 결과 출력
print("=== Comparison for doc_id={} & query_id={} ===".format(TARGET_DOC_ID, TARGET_QUERY_ID))
print(f"Document content excerpt: {d['content'][:50]!r}…")
print(f"Query text excerpt    : {q['text'][:50]!r}…\n")
print(f"1) Plain dot product           : {plain_score:.6f}")
print(f"2) Plain cosine (normalized)   : {plain_norm_score:.6f}")
print(f"3) CKKS raw inner-product      : {raw:.6f}")
print(f"4) CKKS post-scale cosine      : {cosine_score:.6f}")

# ─── 7) Ollama API로 실시간 임베딩 비교 ───
print("\n--- Live Ollama Embedding Comparison ---")
# 7.1) 실시간 임베딩 생성
live_d = ollama.embeddings(model=MODEL, prompt=d["content"])["embedding"]
live_q = ollama.embeddings(model=MODEL, prompt=q["text"])["embedding"]
live_d_vec = np.array(live_d, dtype=float)
live_q_vec = np.array(live_q, dtype=float)

# 7.2) 평문 내적 및 코사인
live_plain = float(np.dot(live_q_vec, live_d_vec))
lqn = live_q_vec / (np.linalg.norm(live_q_vec) + 1e-12)
ldn = live_d_vec / (np.linalg.norm(live_d_vec) + 1e-12)
live_cosine = float(np.dot(lqn, ldn))

print(f"5) Live plain dot product           : {live_plain:.6f}")
print(f"6) Live plain cosine (normalized)   : {live_cosine:.6f}")
