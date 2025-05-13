from sentence_transformers import SentenceTransformer
import tenseal as ts
import time

# Sample sentences
sentences = [
   "The quick brown fox jumps over the lazy dog.",
   "I watched the sunset over the ocean.",
   "Artificial intelligence is transforming industries.",
   "The library was quiet and smelled like old books.",
   "He dreamed of traveling to distant galaxies.",
   "Innovation drives progress in the tech world.",
   "The chef prepared a delicious meal for the guests.",
   "Climate change poses a significant threat to global biodiversity.",
   "The athlete trained rigorously for the upcoming marathon.",
   "Music has the power to evoke deep emotional responses."
]

# Models to compare: MiniLM (384-dim) vs MPNet (768-dim)
model_names = {
    "MiniLM (384d)": "all-MiniLM-L6-v2",
    "roberta (1024d)": "sentence-transformers/stsb-roberta-large"
}

results = []

for label, model_id in model_names.items():
    # Load and encode embeddings
    model = SentenceTransformer(model_id)
    embeddings = model.encode(sentences)
    print(embeddings)
    query_embedding = model.encode(["artificial intelligence"])[0]

    # Setup TenSEAL context (CKKS)
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.generate_galois_keys()
    context.global_scale = 2**40
    secret = context.serialize(save_secret_key=True)
    context.make_context_public()
    context = ts.context_from(secret)

    # Encrypt query vector
    enc_query = ts.ckks_vector(context, query_embedding.tolist())

    # Measure encrypted dot-product loop
    start = time.time()
    for emb in embeddings:
        enc_vec = ts.ckks_vector(context, emb.tolist())
        _ = enc_query.dot(enc_vec).decrypt()[0]
    elapsed = time.time() - start

    results.append({"Model": label, "Time (s)": elapsed})

# Display results as a DataFrame
import pandas as pd

# (앞에서 results 리스트를 채우셨다면)
df = pd.DataFrame(results)

# 1) 간단히 화면에 출력
print(df)

# 2) 필요하다면 CSV로 저장
df.to_csv("encrypted_search_times.csv", index=False)