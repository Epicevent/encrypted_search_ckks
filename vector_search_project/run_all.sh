#!/usr/bin/env bash

set -euo pipefail

# 1. MIRACL data preprocessing
echo "[1/5] Precompute embeddings (miracl_data_prep)"
python miracl_data_prep/precompute_embeddings.py

# 2. Plain DB experiment: build Chroma DB
echo "[2/5] Plain DB experiment: build Chroma DB"
python plain_db_experiments/chroma.py

# 3. Plain DB experiment: evaluation
echo "[3/5] Plain DB experiment: evaluation"
python plain_db_experiments/chorma_eval.py

# 4. HE DB experiment: build encrypted DB
echo "[4/5] HE DB experiment: build encrypted DB"
python he_db_experiments/makedb.py

# 5. HE DB experiment: evaluation
echo "[5/5] HE DB experiment: evaluation"
python he_db_experiments/eval.py

echo "All experiments completed successfully."