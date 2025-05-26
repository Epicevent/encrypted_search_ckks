#!/usr/bin/env python3
# ndcg5_with_rels.py

import json
import sys
import os

import ir_datasets
import numpy as np
import pandas as pd


def dcg_at_k(rels, k):
    return sum((2**rels[i] - 1) / np.log2(i + 2) for i in range(min(len(rels), k)))


def ndcg_at_k(rels, k=5):
    dcg = dcg_at_k(rels, k)
    ideal = sorted(rels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def compute_ndcg5_with_rels(
    input_path: str,
    output_csv_path: str,
    dataset_name: str = "miracl/en/dev",
    k: int = 5
):
    """
    Load eval_results JSON, compute NDCG@k per query, and save CSV.
    Each row contains: query_id, list of binary relevance labels, NDCG@k.
    Adds an AVERAGE row at the end.
    """
    # 1) Load eval_results
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found.")
    with open(input_path, 'r', encoding='utf-8') as f:
        eval_results = json.load(f)
    if not eval_results:
        raise ValueError("No evaluation results to process.")

    # 2) Load qrels
    dataset = ir_datasets.load(dataset_name)
    qrels = {}
    for qrel in dataset.qrels_iter():
        qid = str(qrel.query_id)
        qrels.setdefault(qid, {})[qrel.doc_id] = qrel.relevance

    # 3) Process each entry
    records = []
    for entry in eval_results:
        qid = str(entry.get('query_id'))
        hits = entry.get('results', [])
        doc_ids = [h.get('doc_id') for h in hits]
        rels = [1 if qrels.get(qid, {}).get(did, 0) > 0 else 0 for did in doc_ids]
        score = ndcg_at_k(rels, k)
        records.append({
            'query_id': qid,
            'rels': rels,
            f'NDCG@{k}': score
        })

    if not records:
        raise ValueError("After processing, no records to compute.")

    # 4) Build DataFrame and append average row
    df = pd.DataFrame(records, columns=['query_id', 'rels', f'NDCG@{k}'])
    df[f'NDCG@{k}'] = df[f'NDCG@{k}'].astype(float)
    avg_score = df[f'NDCG@{k}'].mean()
    df.loc[len(df)] = ['AVERAGE', ['-'] * k, avg_score]

    # 5) Save CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False, float_format='%.6f')

    return df


if __name__ == "__main__":
    # allow CLI usage
    if len(sys.argv) != 3:
        print(
            "Usage: python ndcg5_with_rels.py <eval_results.json> <output_metrics.csv>",
            file=sys.stderr
        )
        sys.exit(1)

    input_path = sys.argv[1]
    output_csv = sys.argv[2]
    df = compute_ndcg5_with_rels(input_path, output_csv)
    print(df.to_string(index=False))
    print(f"\nSaved NDCG@5 metrics with relevances to {output_csv}")
