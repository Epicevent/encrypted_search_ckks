```markdown
# encrypted_search_ckks  
Modular Python toolkit for benchmarking CKKS-encrypted vector similarity search with TenSEAL and SentenceTransformers.  

## Directory Structure  
```

vector\_search\_project/
├── config/
│   └── config.yaml               # All pipeline settings
├── data/
│   ├── keys/                     # Fernet & CKKS key files
│   ├── embeddings/               # Precomputed JSON embeddings
│   ├── encrypted\_dbs/            # Encrypted DB folders (HE)
│   └── plain\_dbs/                # Plain DB folders
├── miracl\_data\_prep/             # Precompute embeddings from MIRACL
├── plain\_db\_experiments/         # Plain-text vector search scripts
├── he\_db\_experiments/            # Homomorphic-encrypted DB scripts
├── evaluation/
│   ├── ndcg5\_with\_rels.py        # Compute NDCG\@5 from eval JSON
│   └── run\_eval.sh               # Shell script to run NDCG evaluation
├── requirements.txt
├── run\_all.sh                    # Orchestrates full pipeline
└── README.md                     # This file

````



## Usage

### installation

Install dependencies from the provided `requirements.txt`:
```bash
pip install -r requirements.txt
````

### configuration

1. `config/config.yaml` 파일을 열어 `experiment.model`에 Ollama에 존재하는 모델 이름을 지정합니다.
   예를 들어:

   ```yaml
   experiment:
     model: "snowflake-arctic-embed2:568m"
     ...
   ```

2. 지정한 모델을 로컬에 내려받으려면 터미널에서 다음을 실행합니다:

   ```bash
   ollama pull snowflake-arctic-embed2:568m
   ```

   (여기서 `"snowflake-arctic-embed2:568m"` 대신 `config.yaml`에 적어둔 모델 이름을 사용하세요.)

3. 나머지 설정은 이미 `config/config.yaml`에 정의되어 있으므로, 경로(path)나 파일명 등을 수정할 때는 해당 파일만 확인하면 됩니다.

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

* **Precompute embeddings**

  ```bash
  python miracl_data_prep/precompute_embeddings.py
  ```
* **Plain DB experiment**

  ```bash
  python plain_db_experiments/chroma.py
  python plain_db_experiments/chorma_eval.py
  ```
* **HE DB experiment**

  ```bash
  python he_db_experiments/makedb.py
  python he_db_experiments/eval.py
  ```
* **NDCG\@5 Evaluation**

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

```
```
