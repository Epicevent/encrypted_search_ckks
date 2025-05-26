# settings.py

import yaml
from pathlib import Path

# 1) Locate project root by finding config/config.yaml
HERE = Path(__file__).resolve()
root = HERE.parent
config_path = root / "config" / "config.yaml"
while not config_path.exists():
    if root.parent == root:
        raise FileNotFoundError(f"config/config.yaml not found. Last checked: {root}")
    root = root.parent
    config_path = root / "config" / "config.yaml"
PROJECT_ROOT = root

# 2) Load configuration
with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# 3) Common experiment settings
RANDOM_SEED = cfg.get("random_seed", 42)
MODEL = cfg.get("experiment", {}).get("model")
MAX_WORKERS = cfg.get("experiment", {}).get("max_workers")
N_RESULTS = cfg.get("experiment", {}).get("n_results")
QUERY_NUM = cfg.get("experiment", {}).get("query_num")

# 4) Key management settings
key_cfg = cfg.get("keys", {})
KEY_DIR = PROJECT_ROOT / key_cfg.get("base_dir", "data/keys")
FERNET_KEY_PATH = str(KEY_DIR / key_cfg.get("fernet_key", "fernet_symmetric.key"))
CONTEXT_SECRET = str(KEY_DIR / key_cfg.get("ckks", {}).get("secret", "ckks_context.sk"))
CONTEXT_PUBLIC = str(KEY_DIR / key_cfg.get("ckks", {}).get("public", "ckks_context.pk"))

# 5) Input embedding settings
input_cfg = cfg.get("input", {})
DATASET_NAME = input_cfg.get("dataset_name")
SAMPLE_SIZES = input_cfg.get("sample_sizes", [])
BASE_DATA_DIR = PROJECT_ROOT / input_cfg.get("base_dir", "data")
EMBEDDING_SUBDIR = input_cfg.get("embedding_dir", "embeddings")
EMB_DIR = BASE_DATA_DIR / EMBEDDING_SUBDIR / DATASET_NAME
EMB_FILE_PATTERN = input_cfg.get("embedding_file_pattern", "doc_embeddings_{size}.json")
DOCID_PATTERN = input_cfg.get("docid_pattern", "docid_list_{size}.json")

# 6) Vector DB settings
vdb_cfg = cfg.get("vector_db", {})
enc_cfg = vdb_cfg.get("encrypted", {})
ENC_BASE = PROJECT_ROOT / enc_cfg.get("base_dir", "data/encrypted_dbs")
ENC_PATTERN = enc_cfg.get("db_dir_pattern", "chroma_db_{size}")
plain_cfg = vdb_cfg.get("plain", {})
PLAIN_BASE = PROJECT_ROOT / plain_cfg.get("base_dir", "data/plain_dbs")
PLAIN_PATTERN = plain_cfg.get("db_dir_pattern", "chroma_db_{size}")
COLLECTION_NAME = vdb_cfg.get("collection_name", "docs")
TIME_LOG = PROJECT_ROOT / vdb_cfg.get("time_log", "logs/embedding_load_times.txt")
BATCH_SIZE = vdb_cfg.get("batch_size", 1000)

# 7) CKKS parameters
ckks_cfg = cfg.get("ckks_params", {})
POLY_MOD_DEGREE = ckks_cfg.get("poly_mod_degree")
COEFF_MOD_BIT_SIZES = ckks_cfg.get("coeff_mod_bit_sizes")
GLOBAL_SCALE = ckks_cfg.get("global_scale")

# 8) Output settings
out_cfg = cfg.get("output", {})
RESULTS_DIR = PROJECT_ROOT / out_cfg.get("results_dir", "results")
EVAL_PATTERN = out_cfg.get("eval_results_pattern", "eval_results_{size}.json")
METRICS_PATTERN = out_cfg.get("metrics_pattern", "metrics_{size}.json")

# 9) Default size and derived paths
_DEFAULT_SIZE = SAMPLE_SIZES[0] if SAMPLE_SIZES else None
DB_DIR_DEFAULT = ENC_BASE / ENC_PATTERN.format(size=_DEFAULT_SIZE) if _DEFAULT_SIZE else None
DB_PATH = str(DB_DIR_DEFAULT / "vectors.db") if DB_DIR_DEFAULT else None

# Input & output files for default size
QUERY_EMBEDDINGS_FILE = str(EMB_DIR / EMB_FILE_PATTERN.replace("doc_", "query_").format(size=_DEFAULT_SIZE))
DOCID_LIST_FILE = str(EMB_DIR / DOCID_PATTERN.format(size=_DEFAULT_SIZE))
EVAL_RESULTS_FILE = str(RESULTS_DIR / EVAL_PATTERN.format(size=_DEFAULT_SIZE))
METRICS_PATH = str(RESULTS_DIR / METRICS_PATTERN.format(size=_DEFAULT_SIZE))

# 10) Utility getters for dynamic paths

def get_embedding_path(size: int) -> str:
    return str(EMB_DIR / EMB_FILE_PATTERN.format(size=size))

def get_query_path(size: int) -> str:
    return str(EMB_DIR / EMB_FILE_PATTERN.replace("doc_", "query_").format(size=size))

def get_docid_path(size: int) -> str:
    return str(EMB_DIR / DOCID_PATTERN.format(size=size))

def get_enc_db_dir(size: int) -> str:
    db_dir = ENC_BASE / ENC_PATTERN.format(size=size)
    db_dir.mkdir(parents=True, exist_ok=True)
    return str(db_dir)

def get_plain_db_dir(size: int) -> str:
    db_dir = PLAIN_BASE / PLAIN_PATTERN.format(size=size)
    db_dir.mkdir(parents=True, exist_ok=True)
    return str(db_dir)

def get_eval_path(size: int) -> str:
    return str(RESULTS_DIR / EVAL_PATTERN.format(size=size))

def get_metrics_path(size: int) -> str:
    return str(RESULTS_DIR / METRICS_PATTERN.format(size=size))