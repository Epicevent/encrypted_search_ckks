
# settings.py

import yaml
from pathlib import Path

# 1) Locate project root by finding config/config.yaml
HERE = Path(__file__).resolve()
root = HERE.parent
config_path = root / "config" / "config.yaml"
# Traverse upwards until config/config.yaml is found
while not config_path.exists():
    if root.parent == root:
        raise FileNotFoundError(
            f"config/config.yaml not found. Last checked: {root}"
        )
    root = root.parent
    config_path = root / "config" / "config.yaml"

# PROJECT_ROOT is directory containing config/
PROJECT_ROOT = root

# 2) Load YAML configuration
with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# 3) Common settings
RANDOM_SEED = cfg.get("random_seed")
MODEL = cfg.get("experiment", {}).get("model")
MAX_WORKERS = cfg.get("experiment", {}).get("max_workers")

# 4) Input embedding settings
input_cfg = cfg.get("input", {})
DATASET_NAME = input_cfg.get("dataset_name")
SAMPLE_SIZES = input_cfg.get("sample_sizes", [])
BASE_DATA_DIR = PROJECT_ROOT / input_cfg.get("base_dir", "data")
EMBEDDING_SUBDIR = input_cfg.get("embedding_dir", "embeddings")
EMB_DIR = BASE_DATA_DIR / EMBEDDING_SUBDIR / DATASET_NAME
EMB_FILE_PATTERN = input_cfg.get("embedding_file_pattern", "doc_embeddings_{size}.json")
DOCID_PATTERN = input_cfg.get("docid_pattern", "docid_list_{size}.json")

# 5) Vector DB settings
vdb_cfg = cfg.get("vector_db", {})
ENC_BASE = PROJECT_ROOT / vdb_cfg.get("encrypted", {}).get("base_dir", "data/encrypted_dbs")
ENC_PATTERN = vdb_cfg.get("encrypted", {}).get("db_dir_pattern", "chroma_db_{size}")
PLAIN_BASE = PROJECT_ROOT / vdb_cfg.get("plain", {}).get("base_dir", "data/plain_dbs")
PLAIN_PATTERN = vdb_cfg.get("plain", {}).get("db_dir_pattern", "chroma_db_{size}")
COLLECTION_NAME = vdb_cfg.get("collection_name", "docs")
TIME_LOG = PROJECT_ROOT / vdb_cfg.get("time_log", "logs/embedding_load_times.txt")
BATCH_SIZE = vdb_cfg.get("batch_size", 1000)

# 6) Output settings
out_cfg = cfg.get("output", {})
RESULTS_DIR = PROJECT_ROOT / out_cfg.get("results_dir", "results")
EVAL_PATTERN = out_cfg.get("eval_results_pattern", "eval_results_{size}.json")
METRICS_PATTERN = out_cfg.get("metrics_pattern", "metrics_{size}.json")

# 7) Utility getters

def get_embedding_path(size: int) -> str:
    return str(EMB_DIR / EMB_FILE_PATTERN.format(size=size))

def get_query_path(size: int) -> str:
    return str(EMB_DIR / EMB_FILE_PATTERN.replace("doc_", "query_").format(size=size))

def get_docid_path(size: int) -> str:
    return str(EMB_DIR / DOCID_PATTERN.format(size=size))

def get_enc_db_dir(size: int) -> str:
    return str(ENC_BASE / ENC_PATTERN.format(size=size))

def get_plain_db_dir(size: int) -> str:
    return str(PLAIN_BASE / PLAIN_PATTERN.format(size=size))

def get_eval_path(size: int) -> str:
    return str(RESULTS_DIR / EVAL_PATTERN.format(size=size))

def get_metrics_path(size: int) -> str:
    return str(RESULTS_DIR / METRICS_PATTERN.format(size=size))
