set -euo pipefail

# Load sample sizes from config.yaml
CONFIG=config/config.yaml
read -r -a sizes <<< "$(python3 - << 'EOF'
import yaml
cfg = yaml.safe_load(open('$CONFIG', 'r', encoding='utf-8'))
print(' '.join(map(str, cfg['input']['sample_sizes'])))
EOF
)"

# Path to evaluation script
SCRIPT=evaluation/ndcg5_with_rels.py

for size in "${sizes[@]}"; do
  eval_json="results/eval_results_${size}.json"
  output_csv="results/ndcg5_${size}.csv"
  echo "[NDCG@5] sample_size=${size} → ${output_csv}"
  python3 "$SCRIPT" "$eval_json" "$output_csv"
done

echo "NDCG@5 evaluation complete for sample sizes: ${sizes[*]}"