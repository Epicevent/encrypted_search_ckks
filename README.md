````markdown
# encrypted_search_ckks  
Modular Python toolkit for benchmarking CKKS-encrypted vector similarity search with TenSEAL and SentenceTransformers  

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/YourOrg/encrypted_search_ckks.git
   cd encrypted_search_ckks
````

2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # Linux / macOS
   .venv\Scripts\activate       # Windows
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
   python -c "import he_vector_db; print(he_vector_db.__version__)"
   ```

## Citation

If you use this toolkit, please cite:

“오프라인 국방 RAG 시스템 구현 및 동형암호화 벡터 검색 성능 연구”

```
```
