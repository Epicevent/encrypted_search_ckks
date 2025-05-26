# encrypted_search_ckks  
Modular Python toolkit for benchmarking CKKS‑encrypted vector similarity search with TenSEAL and SentenceTransformers.  

## Directory Structure  
```
vector_search_project/
├── config/
│   └── config.yaml               # All pipeline settings
├── data/
│   ├── keys/                     # Fernet & CKKS key files
│   ├── embeddings/               # Precomputed JSON embeddings
│   ├── encrypted_dbs/            # Encrypted DB folders (HE)
│   └── plain_dbs/                # Plain DB folders
├── miracl_data_prep/             # Precompute embeddings from MIRACL
├── plain_db_experiments/         # Plain-text vector search scripts
├── he_db_experiments/            # Homomorphic-encrypted DB scripts
├── evaluation/
│   ├── ndcg5_with_rels.py        # Compute NDCG@5 from eval JSON
│   └── run_eval.sh               # Shell script to run NDCG evaluation
├── run_all.sh                    # Orchestrates full pipeline
└── README.md                     # This file
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YourOrg/encrypted_search_ckks.git
   cd encrypted_search_ckks
   ```
2. **Create & activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Install in editable (development) mode**
   ```bash
   pip install -e .
   ```
5. **Verify installation**
   ```bash
   python -c "import encrypted_search_ckks; print(encrypted_search_ckks.__version__)"
   ```

## Usage

### Full end-to-end

Make both orchestration scripts executable and run them:
```bash
chmod +x run_all.sh
chmod +x evaluation/run_eval.sh

# 1) Preprocessing, DB builds & evaluations
./run_all.sh

# 2) NDCG@5 evaluation
evaluation/run_eval.sh
```

### Manual steps

- **Precompute embeddings**
  ```bash
  python miracl_data_prep/precompute_embeddings.py
  ```
- **Plain DB experiment**
  ```bash
  python plain_db_experiments/chroma.py
  python plain_db_experiments/chorma_eval.py
  ```
- **HE DB experiment**
  ```bash
  python he_db_experiments/makedb.py
  python he_db_experiments/eval.py
  ```
- **NDCG@5 Evaluation**
  ```bash
  python evaluation/ndcg5_with_rels.py \
    results/eval_results_<size>.json \
    results/ndcg5_<size>.csv
  ```

> **Tip:** Sample sizes and all paths are centrally managed in `config/config.yaml`.  

## Citation

If you use this toolkit in your work, please cite:  
> “오프라인 국방 RAG 시스템 구현 및 동형암호화 벡터 검색 성능 연구”

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
