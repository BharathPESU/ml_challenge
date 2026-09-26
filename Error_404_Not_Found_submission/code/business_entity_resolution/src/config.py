"""
config.py — Centralized configuration for the Business Entity Resolution pipeline.

Supports:
  - Local development
  - Amazon SageMaker execution
  - S3 data access
  - Deterministic experiments

All paths and hyperparameters are configurable via dataclass fields or
environment variables — no hard-coded personal paths or credentials.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AWS / S3 Configuration
# ---------------------------------------------------------------------------

import glob

def _detect_kaggle_path() -> Optional[str]:
    """Auto-detect dataset directory on Kaggle."""
    base = "/kaggle/input"
    if os.path.exists(base):
        # Look for the innermost dataset folder containing train/test
        matches = glob.glob(f"{base}/**/dataset/train", recursive=True)
        if matches:
            return os.path.dirname(matches[0])  # Return the 'dataset' directory
    return None

def _get_default_data_path(subpath: str) -> str:
    """
    Return local path if it exists locally, otherwise auto-detect Kaggle path, 
    otherwise return public S3 URI.
    """
    use_s3 = os.environ.get("USE_S3_DATA", "false").lower() == "true"
    local_path = os.path.join("dataset", subpath)
    
    if os.path.exists(local_path) and not use_s3:
        return local_path
        
    kaggle_base = _detect_kaggle_path()
    if kaggle_base and not use_s3:
        return os.path.join(kaggle_base, subpath)

    s3_base = os.environ.get(
        "S3_DATASET_BASE",
        "s3://ml-challenge-bharath/ml_challenge/Error_404_Not_Found_submission/dataset"
    )
    return f"{s3_base}/{subpath}"


@dataclass
class AWSConfig:
    """AWS region and S3 bucket settings."""
    region: str = field(
        default_factory=lambda: os.environ.get("AWS_REGION", "us-east-1")
    )
    s3_bucket: str = field(
        default_factory=lambda: os.environ.get("S3_BUCKET", "ml-challenge-bharath")
    )
    s3_prefix: str = field(
        default_factory=lambda: os.environ.get("S3_PREFIX", "ml_challenge/Error_404_Not_Found_submission")
    )
    sagemaker_role: Optional[str] = field(
        default_factory=lambda: os.environ.get("SAGEMAKER_ROLE", None)
    )


# ---------------------------------------------------------------------------
# Data Path Configuration
# ---------------------------------------------------------------------------

@dataclass
class DataPathConfig:
    """
    Paths for training and test data.

    Paths may be local filesystem paths OR s3:// URIs.
    io_utils.py handles both transparently.
    """
    # Training
    train_source1_path: str = field(
        default_factory=lambda: os.environ.get(
            "TRAIN_S1_PATH",
            _get_default_data_path("train/train_source1.tsv")
        )
    )
    train_source2_path: str = field(
        default_factory=lambda: os.environ.get(
            "TRAIN_S2_PATH",
            _get_default_data_path("train/train_source2.tsv")
        )
    )
    train_source3_path: str = field(
        default_factory=lambda: os.environ.get(
            "TRAIN_S3_PATH",
            _get_default_data_path("train/train_source3.tsv")
        )
    )
    train_ground_truth_path: str = field(
        default_factory=lambda: os.environ.get(
            "TRAIN_GT_PATH",
            _get_default_data_path("train/train_ground_truth.tsv")
        )
    )

    # Test
    test_source1_path: str = field(
        default_factory=lambda: os.environ.get(
            "TEST_S1_PATH",
            _get_default_data_path("test/test_source1.tsv")
        )
    )
    test_source2_path: str = field(
        default_factory=lambda: os.environ.get(
            "TEST_S2_PATH",
            _get_default_data_path("test/test_source2.tsv")
        )
    )
    test_source3_path: str = field(
        default_factory=lambda: os.environ.get(
            "TEST_S3_PATH",
            _get_default_data_path("test/test_source3.tsv")
        )
    )

    # Output
    output_dir: str = field(
        default_factory=lambda: os.environ.get("OUTPUT_DIR", "output")
    )
    artifacts_s3_prefix: str = field(
        default_factory=lambda: os.environ.get(
            "ARTIFACTS_S3_PREFIX", "ml_challenge/Error_404_Not_Found_submission/artifacts"
        )
    )


# ---------------------------------------------------------------------------
# Experiment / Reproducibility Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Settings for deterministic, reproducible experiments."""
    random_seed: int = field(
        default_factory=lambda: int(os.environ.get("RANDOM_SEED", "42"))
    )
    validation_split_ratio: float = field(
        default_factory=lambda: float(os.environ.get("VAL_SPLIT_RATIO", "0.2"))
    )
    experiment_id: str = field(
        default_factory=lambda: os.environ.get("EXPERIMENT_ID", "exp_001")
    )
    config_hash: Optional[str] = None
    feature_version: str = "v1"
    blocking_version: str = "v1"
    model_version: str = "v1"


# ---------------------------------------------------------------------------
# TF-IDF / Blocking Configuration
# ---------------------------------------------------------------------------

@dataclass
class BlockingConfig:
    """Configuration for candidate blocking / retrieval stage."""
    # 8-Pass Blocking Toggles
    pass1_exact_norm: bool = True
    pass2_exact_core: bool = True
    pass3_address_token: bool = True
    pass4_numeric_address: bool = True
    pass5_rare_name_token: bool = True
    pass6_rare_address_token: bool = True
    pass7_fuzz_name: bool = True
    pass8_fuzz_address: bool = True

    rare_token_threshold: int = 1000

    # TF-IDF vectorizer settings (legacy/fallback if needed)
    tfidf_analyzer: str = "char"
    tfidf_ngram_range: Tuple[int, int] = (2, 4)
    tfidf_min_df: int = 1
    tfidf_sublinear_tf: bool = True
    tfidf_max_features: Optional[int] = None  # None = unlimited

    # Top-K candidates retrieved per S1 entity
    name_top_k: int = field(
        default_factory=lambda: int(os.environ.get("NAME_TOP_K", "50"))
    )
    address_top_k: int = field(
        default_factory=lambda: int(os.environ.get("ADDRESS_TOP_K", "50"))
    )

    # Configurable K values for blocking recall experiments
    name_top_k_candidates: Tuple[int, ...] = (20, 50, 100, 200)
    address_top_k_candidates: Tuple[int, ...] = (20, 50, 100, 200)

    # Minimum cosine similarity to retain a candidate (0.0 = keep all top-K)
    min_name_similarity: float = 0.0
    min_address_similarity: float = 0.0


# ---------------------------------------------------------------------------
# Feature & Memory Configuration
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """Toggle list for feature extraction and memory optimization."""
    use_char_features: bool = True
    use_token_features: bool = True
    use_numeric_features: bool = True
    use_interaction_features: bool = True
    use_float32: bool = True
    use_float16: bool = False  # Caution: evaluate stability before enabling

@dataclass
class MemoryConfig:
    """Settings for RAM and memory optimization."""
    chunk_size: int = 50000
    use_parquet_cache: bool = True
    cache_dir: str = "/kaggle/working/run"

# ---------------------------------------------------------------------------
# Negative Sampling Configuration
# ---------------------------------------------------------------------------

@dataclass
class NegativeSamplingConfig:
    """Controls how negatives are sampled for training."""
    # Number of negatives per positive
    neg_ratio: int = field(
        default_factory=lambda: int(os.environ.get("NEG_RATIO", "4"))
    )
    hard_neg_fraction: float = 0.5   # fraction of negatives that are hard negatives
    easy_neg_fraction: float = 0.5
    random_seed: int = 42


# ---------------------------------------------------------------------------
# XGBoost Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class XGBoostConfig:
    """Baseline XGBoost hyperparameters — treated as starting point only."""
    objective: str = "binary:logistic"
    eval_metric: str = "aucpr"
    
    # GPU / Tree method defaults
    tree_method: str = "hist"
    device: str = "cuda:0"

    max_depth: int = 6
    eta: float = 0.05
    min_child_weight: int = 3
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0.0
    reg_alpha: float = 0.1
    reg_lambda: float = 2.0
    num_round: int = 300
    early_stopping_rounds: int = 30

    # Optional class-imbalance weight — set to None to disable
    scale_pos_weight: Optional[float] = None

    # Hyperparameter search candidates (staged experiments)
    search_max_depth: Tuple[int, ...] = (4, 5, 6, 7, 8)
    search_eta: Tuple[float, ...] = (0.03, 0.05, 0.08, 0.10)
    search_min_child_weight: Tuple[int, ...] = (1, 3, 5, 10)
    search_subsample: Tuple[float, ...] = (0.7, 0.8, 0.9, 1.0)
    search_colsample_bytree: Tuple[float, ...] = (0.7, 0.8, 0.9, 1.0)
    search_gamma: Tuple[float, ...] = (0.0, 0.1, 0.3)
    search_reg_alpha: Tuple[float, ...] = (0.0, 0.1, 0.5)
    search_reg_lambda: Tuple[float, ...] = (1.0, 2.0, 5.0)
    search_num_round: Tuple[int, ...] = (200, 300, 500)


# ---------------------------------------------------------------------------
# Threshold Search Configuration
# ---------------------------------------------------------------------------

@dataclass
class ThresholdConfig:
    """Configuration for decision-threshold sweep on the validation set."""
    sweep_start: float = 0.10
    sweep_end: float = 0.99
    sweep_step: float = 0.01

    # Loaded at inference time from S3 / artifact location
    best_threshold: Optional[float] = None
    best_threshold_s2: Optional[float] = None
    best_threshold_s3: Optional[float] = None


# ---------------------------------------------------------------------------
# Runtime Options
# ---------------------------------------------------------------------------

@dataclass
class RuntimeConfig:
    """Toggle switches for pipeline stages."""
    run_on_sagemaker: bool = field(
        default_factory=lambda: os.environ.get("RUN_ON_SAGEMAKER", "false").lower() == "true"
    )
    use_s3_data: bool = field(
        default_factory=lambda: os.environ.get("USE_S3_DATA", "false").lower() == "true"
    )
    save_artifacts_to_s3: bool = field(
        default_factory=lambda: os.environ.get("SAVE_ARTIFACTS_S3", "false").lower() == "true"
    )
    log_level: str = field(
        default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO")
    )
    # Optional row limit for fast local debugging (None = full dataset)
    debug_sample_size: Optional[int] = field(
        default_factory=lambda: (
            int(os.environ.get("DEBUG_SAMPLE_SIZE"))
            if os.environ.get("DEBUG_SAMPLE_SIZE") else None
        )
    )


# ---------------------------------------------------------------------------
# Master Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """
    Master configuration dataclass.

    Usage:
        cfg = Config()                    # all defaults / env-vars
        cfg.blocking.name_top_k = 100    # override a specific field
    """
    aws: AWSConfig = field(default_factory=AWSConfig)
    data: DataPathConfig = field(default_factory=DataPathConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    negative_sampling: NegativeSamplingConfig = field(default_factory=NegativeSamplingConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def configure_logging(self) -> None:
        """Configure root logger based on runtime log level."""
        logging.basicConfig(
            level=getattr(logging, self.runtime.log_level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        )
        logger.info("Logging configured at level: %s", self.runtime.log_level)

    def to_dict(self) -> dict:
        """Serialize config to a plain dictionary for logging/saving."""
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Default global config instance
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = Config()
