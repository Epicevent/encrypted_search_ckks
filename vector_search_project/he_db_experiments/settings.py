# settings.py for hedb_experiments

import yaml
from pathlib import Path

# 1) Locate project root by finding config/config.yaml
HERE = Path(__file__).resolve()
root = HERE.parent
config_path = root / "config" / "config.yaml"
while not config_path.exists():
    if root.parent == root:
        raise FileNotFoundError(
            f"config/config.yaml not found. Last checked: {root}"
        )
    root = root.parent
    config_path = root / "config" / "config.yaml"

# PROJECT_ROOT is the directory containing config/
PROJECT_ROOT = root

# 2) Load YAML configuration
with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# 3) Experiment parameters
exp_cfg = cfg.get("experiment", {})
MODEL = exp_cfg.get("model")
MAX_WORKERS = exp_cfg.get("max_workers")
N_RESULTS = exp_cfg.get("n_results")
QUERY_NUM = exp_cfg.get("query_num")

input_cfg = cfg.get("input", {})
SAMPLE_SIZES = input_cfg.get("sample_sizes", [])

# 4) Input embedding file paths
BASE_EMB_DIR = PROJECT_ROOT / input_cfg.get("base_dir", "data")  / input_cfg.get("dataset_name", "") / input_cfg.get("embedding_dir", "embeddings")
EMB_FILE_PATTERN = input_cfg.get("embedding_file_pattern", "doc_embeddings_{size}.json")
DOCID_PATTERN = input_cfg.get("docid_pattern", "docid_list_{size}.json")

# 5) HE Vector DB settings
he_cfg = cfg.get("vector_db", {}).get("encrypted", {})
HE_DB_BASE = PROJECT_ROOT / input_cfg.get("dataset_name", "") /  he_cfg.get("base_dir", "data/he_dbs")
HE_DB_PATTERN = he_cfg.get("db_dir_pattern", "regulation_vectors_{size}.db")

# 6) CKKS context & key paths
key_cfg = cfg.get("keys", {})
KEY_BASE = PROJECT_ROOT / key_cfg.get("base_dir", "data/keys")

FERNET_KEY_PATH = str(PROJECT_ROOT / KEY_BASE/ key_cfg.get("fernet_key", "data/fernet_symmetric.key"))
context_cfg = key_cfg .get("ckks", {})
CONTEXT_SECRET = str(PROJECT_ROOT / KEY_BASE/ context_cfg.get("secret", "data/ckks_context.sk"))
CONTEXT_PUBLIC = str(PROJECT_ROOT / KEY_BASE/ context_cfg.get("public", "data/ckks_context.pk"))

# 7) CKKS parameters
ckks_cfg = cfg.get("ckks_params", {})
POLY_MOD_DEGREE = ckks_cfg.get("poly_mod_degree")
COEFF_MOD_BIT_SIZES = ckks_cfg.get("coeff_mod_bit_sizes")
GLOBAL_SCALE = ckks_cfg.get("global_scale")

# 8) Output files
out_cfg = cfg.get("output", {})
RESULTS_DIR = PROJECT_ROOT / out_cfg.get("results_dir", "results")
EVAL_FILE_PATTERN = out_cfg.get("eval_results_pattern", "eval_results_{size}.json")
METRICS_FILE_PATTERN = out_cfg.get("metrics_pattern", "metrics_{size}.json")

# 9) Random seed
RANDOM_SEED = cfg.get("random_seed")

# ── Utility functions ─────────────────────────

def get_he_db_path(size: int) -> str:
    """Return the HE-encrypted DB file path for given sample size."""
    return str(HE_DB_BASE / HE_DB_PATTERN.format(size=size))

def get_doc_embeddings_path(size: int) -> str:
    return str(BASE_EMB_DIR / EMB_FILE_PATTERN.format(size=size))

def get_query_embeddings_path(size: int) -> str:
    return str(BASE_EMB_DIR / EMB_FILE_PATTERN.replace("doc_", "query_").format(size=size))

def get_docid_list_path(size: int) -> str:
    return str(BASE_EMB_DIR / DOCID_PATTERN.format(size=size))

def get_eval_path(size: int) -> str:
    return str(RESULTS_DIR / EVAL_FILE_PATTERN.format(size=size))

def get_metrics_path(size: int) -> str:
    return str(RESULTS_DIR / METRICS_FILE_PATTERN.format(size=size))
