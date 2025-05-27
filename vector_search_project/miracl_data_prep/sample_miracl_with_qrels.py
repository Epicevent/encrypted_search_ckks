import ir_datasets
from tqdm import tqdm
import random


def sample_with_qrels(ds, sample_size, random_seed=42):
    random.seed(random_seed)

    # 1. qrels 문서 ID 확보
    qrels_list = list(ds.qrels_iter())
    qrels_ids = {q.doc_id for q in qrels_list}
    QRELS_TOTAL = len(qrels_ids)
    assert QRELS_TOTAL <= sample_size, f"QRELS 수({QRELS_TOTAL}) > sample_size({sample_size})"

    # 2. 나머지 문서 reservoir 샘플링
    reservoir = []
    total_seen = 0
    for doc in tqdm(ds.docs_iter(), desc="Reservoir sampling"):
        if doc.doc_id in qrels_ids:
            continue
        if len(reservoir) < sample_size - QRELS_TOTAL:
            reservoir.append(doc.doc_id)
        else:
            j = random.randint(0, total_seen)
            if j < (sample_size - QRELS_TOTAL):
                reservoir[j] = doc.doc_id
        total_seen += 1

    final_ids = sorted(list(qrels_ids) + reservoir)
    return final_ids


def main():
    dataset_name = "miracl/en/dev"
    sample_size = 10000
    seed = 42

    print(f"📦 Loading dataset: {dataset_name}")
    ds = ir_datasets.load(dataset_name)

    print(f"🔍 Sampling {sample_size} documents with qrels priority...")
    sampled_ids = sample_with_qrels(ds, sample_size, random_seed=seed)

    print(f"\n✅ Done. Sampled {len(sampled_ids)} document IDs (including all {len(set(ds.qrels_iter()))} qrels).")
    print("예시 ID 10개:", sampled_ids[:10])

    # 필요 시 저장
    with open(f"sampled_docids_{sample_size}.json", "w", encoding="utf-8") as f:
        import json
        json.dump(sampled_ids, f, indent=2)
        print(f"💾 저장 완료: sampled_docids_{sample_size}.json")


if __name__ == "__main__":
    main()
