#!/bin/bash
set -euo pipefail

# Absolute path to script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Config 경로를 스크립트 기준으로 고정
CONFIG="$SCRIPT_DIR/../config/config.yaml"
SCRIPT="$SCRIPT_DIR/ndcg5_with_rels.py"

read -r -a sizes <<< "$(CONFIG=$CONFIG python3 - << 'EOF'
import os
import yaml

cfg_path = os.environ['CONFIG']
with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

print(' '.join(map(str, cfg['input']['sample_sizes'])))
EOF
)"

for size in "${sizes[@]}"; do
  eval_json="$SCRIPT_DIR/../results/eval_results_${size}.json"
  output_csv="$SCRIPT_DIR/../results/ndcg5_${size}.csv"
  echo "[NDCG@5] sample_size=${size} → ${output_csv}"
  python3 "$SCRIPT" "$eval_json" "$output_csv"
done

echo "NDCG@5 evaluation complete for sample sizes: ${sizes[*]}"
