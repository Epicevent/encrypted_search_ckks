# settings.py for plain_db_experiments

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

# 4) Experiment parameters
exp_cfg = cfg.get("experiment", {})
MODEL = exp_cfg.get("model")
MAX_WORKERS = exp_cfg.get("max_workers")
N_RESULTS = exp_cfg.get("n_results")
QUERY_NUM = exp_cfg.get("query_num")

input_cfg = cfg.get("input", {})
SAMPLE_SIZES = input_cfg.get("sample_sizes", [])

# 6) Input embedding file patterns (if reused)
input_cfg = cfg.get("input", {})
BASE_EMB_DIR = PROJECT_ROOT / input_cfg.get("base_dir", "data") / input_cfg.get("embedding_dir", "embeddings") / input_cfg.get("dataset_name", "")
EMB_FILE_PATTERN = input_cfg.get("embedding_file_pattern", "doc_embeddings_{size}.json")
DOCID_PATTERN = input_cfg.get("docid_pattern", "docid_list_{size}.json")

# 7) Plain Vector DB settings
plain_cfg = cfg.get("vector_db", {}).get("plain", {})
PLAIN_DB_BASE = PROJECT_ROOT / plain_cfg.get("base_dir", "data/plain_dbs")
PLAIN_DB_PATTERN = plain_cfg.get("db_dir_pattern", "chroma_db_{size}")

# 8) Output files
out_cfg = cfg.get("output", {})
RESULTS_DIR = PROJECT_ROOT / out_cfg.get("results_dir", "results")
EVAL_FILE_PATTERN = out_cfg.get("eval_results_pattern", "eval_results_{size}.json")
METRICS_FILE_PATTERN = out_cfg.get("metrics_pattern", "metrics_{size}.json")

# 9) Random seed
RANDOM_SEED = cfg.get("random_seed")

# Utility functions for dynamic paths

def get_plain_db_dir(size: int) -> str:
    """Return the plain DB directory path for given sample size."""
    return str(PLAIN_DB_BASE / PLAIN_DB_PATTERN.format(size=size))


def get_eval_path(size: int) -> str:
    """Return evaluation results file path for given sample size."""
    return str(RESULTS_DIR / EVAL_FILE_PATTERN.format(size=size))


def get_metrics_path(size: int) -> str:
    """Return metrics file path for given sample size."""
    return str(RESULTS_DIR / METRICS_FILE_PATTERN.format(size=size))


def get_doc_embeddings_path(size: int) -> str:
    """Return document embeddings JSON path for given sample size."""
    return str(BASE_EMB_DIR / EMB_FILE_PATTERN.format(size=size))


def get_query_embeddings_path(size: int) -> str:
    """Return query embeddings JSON path for given sample size."""
    return str(BASE_EMB_DIR / EMB_FILE_PATTERN.replace("doc_", "query_").format(size=size))


def get_docid_list_path(size: int) -> str:
    """Return docid list JSON path for given sample size."""
    return str(BASE_EMB_DIR / DOCID_PATTERN.format(size=size))
