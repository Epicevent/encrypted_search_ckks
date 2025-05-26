# settings.py

# —— Database & key files ——
DB_PATH           = "./data/vectors.db"
FERNET_KEY_PATH    = "./data/fernet_symmetric.key"
CONTEXT_SECRET    = "./data/ckks_context.sk"
CONTEXT_PUBLIC    = "./data/ckks_context.pk"  # if you need a public‐only copy

# —— Embedding & sampling ——
MODEL             = "snowflake-arctic-embed2:568m"
SAMPLE_SIZE       = 50_000

# —— Parallel & retrieval ——
MAX_WORKERS       = 1  # None → os.cpu_count()
N_RESULTS         = 5
QUERY_NUM      = 50  # None means evaluate all queries
# —— CKKS params ——
POLY_MOD_DEGREE   = 8192
COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
GLOBAL_SCALE      = 2 ** 40

# —— Output files ——
DOC_EMBEDDINGS_FILE    = "./data/doc_embeddings.json"    # MIRACL document embeddings
QUERY_EMBEDDINGS_FILE  = "./data/query_embeddings.json"  # MIRACL query embeddings
DOCID_LIST_FILE   = "./data/docid_list.json"
EVAL_RESULTS_FILE = "./results/eval_results.json"
METRICS_PATH       = "./results/metrics.json"

RANDOM_SEED     = 42
