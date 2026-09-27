"""
pipeline.py — Master end-to-end orchestration for Business Entity Resolution.

Each stage is independently callable to support:
  - Iterative development (run only blocking, or only threshold sweep)
  - SageMaker-managed execution (each stage as a separate job)
  - Reproducible experiment management

STAGE ORDER:
  1.  load_data
  2.  clean_data
  3.  split_training_validation
  4.  generate_candidates
  5.  evaluate_candidate_recall
  6.  build_training_pairs
  7.  build_features
  8.  train_xgboost
  9.  score_validation
  10. optimize_threshold
  11. perform_error_analysis
  12. generate_test_candidates
  13. build_test_features
  14. run_test_inference
  15. create_submission_files

DO NOT duplicate business logic across stages. Each stage calls
functions from the appropriate module.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State container for inter-stage data passing
# ---------------------------------------------------------------------------

class PipelineState:
    """Holds shared state across pipeline stages. LOW-RAM disk-backed version."""

    def __init__(self):
        # Raw data
        self.train_s1: Optional[pd.DataFrame] = None
        self.train_s2: Optional[pd.DataFrame] = None
        self.train_s3: Optional[pd.DataFrame] = None
        self.gt: Optional[Dict[str, Set[str]]] = None

        # Cleaned data
        self.train_s1_clean: Optional[pd.DataFrame] = None
        self.train_s2_clean: Optional[pd.DataFrame] = None
        self.train_s3_clean: Optional[pd.DataFrame] = None

        # Train / validation split
        self.train_s1_ids: Optional[List[str]] = None
        self.val_s1_ids: Optional[List[str]] = None
        self.train_gt: Optional[Dict[str, Set[str]]] = None
        self.val_gt: Optional[Dict[str, Set[str]]] = None

        # Candidates (Disk Paths)
        self.train_candidates_path: Optional[str] = None
        self.val_candidates_path: Optional[str] = None

        # Labelled pairs (Small enough for RAM)
        self.train_pairs: Optional[pd.DataFrame] = None
        self.val_pairs: Optional[pd.DataFrame] = None

        # Features
        self.train_features: Optional[pd.DataFrame] = None
        self.val_features: Optional[pd.DataFrame] = None
        self.feature_columns: Optional[List[str]] = None

        # Model
        self.booster: Optional[Any] = None
        self.model_path: Optional[str] = None

        # Validation scoring
        self.val_probabilities: Optional[np.ndarray] = None
        self.sweep_df: Optional[pd.DataFrame] = None
        self.best_threshold: Optional[float] = None
        self.threshold_path: Optional[str] = None

        # Test data
        self.test_s1_clean: Optional[pd.DataFrame] = None
        self.test_s2_clean: Optional[pd.DataFrame] = None
        self.test_s3_clean: Optional[pd.DataFrame] = None
        self.test_candidates_path: Optional[str] = None
        self.test_features: Optional[pd.DataFrame] = None
        self.test_metadata: Optional[pd.DataFrame] = None
        self.test_predictions: Optional[Dict[str, Set[str]]] = None


# ---------------------------------------------------------------------------
# Stage 1: Load training data
# ---------------------------------------------------------------------------

def load_training_data(cfg: Config, state: PipelineState) -> None:
    """
    Stage 1: Load all training source files and ground truth.
    """
    from io_utils import read_tsv, validate_source_file, validate_ground_truth_file
    from ground_truth import parse_ground_truth

    logger.info("=== Stage 1: Loading training data ===")

    nrows = cfg.runtime.debug_sample_size

    state.train_s1 = read_tsv(cfg.data.train_source1_path, nrows=nrows)
    state.train_s2 = read_tsv(cfg.data.train_source2_path)
    state.train_s3 = read_tsv(cfg.data.train_source3_path)

    validate_source_file(state.train_s1, "train_source1")
    validate_source_file(state.train_s2, "train_source2")
    validate_source_file(state.train_s3, "train_source3")

    gt_df = read_tsv(cfg.data.train_ground_truth_path)
    validate_ground_truth_file(gt_df)
    state.gt = parse_ground_truth(gt_df)

    logger.info(
        "Loaded: %d S1 | %d S2 | %d S3 | %d GT entries",
        len(state.train_s1), len(state.train_s2), len(state.train_s3), len(state.gt),
    )


# ---------------------------------------------------------------------------
# Stage 2: Clean training data
# ---------------------------------------------------------------------------

def clean_training_data(cfg: Config, state: PipelineState) -> None:
    """
    Stage 2: Apply text normalization to all training source DataFrames.
    """
    from cleaning import clean_source_dataframe

    logger.info("=== Stage 2: Cleaning training data ===")
    assert state.train_s1 is not None and state.train_s2 is not None and state.train_s3 is not None, "Run Stage 1 first!"

    state.train_s1_clean = clean_source_dataframe(state.train_s1, "train_S1")
    state.train_s2_clean = clean_source_dataframe(state.train_s2, "train_S2")
    state.train_s3_clean = clean_source_dataframe(state.train_s3, "train_S3")


# ---------------------------------------------------------------------------
# Stage 3: Train/Validation split
# ---------------------------------------------------------------------------

def split_training_validation(cfg: Config, state: PipelineState) -> None:
    """
    Stage 3: Split Source 1 entities into train and validation subsets.
    Split is by entity ID — never by candidate pairs (prevents leakage).
    """
    from splitting import prepare_train_val_split

    logger.info("=== Stage 3: Train/Validation split ===")
    assert state.train_s1_clean is not None and state.gt is not None, "Run Stages 1 & 2 first!"

    (
        state.train_s1_ids,
        state.val_s1_ids,
        _,
        _,
        state.train_gt,
        state.val_gt,
    ) = prepare_train_val_split(
        df_s1=state.train_s1_clean,
        gt=state.gt,
        val_ratio=cfg.experiment.validation_split_ratio,
        random_seed=cfg.experiment.random_seed,
    )


# ---------------------------------------------------------------------------
# Stage 4: Generate candidates
# ---------------------------------------------------------------------------


def generate_candidates(cfg: Config, state: PipelineState) -> None:
    from blocking import run_disk_backed_blocking
    logger.info("=== Stage 4: Disk-Backed Candidate Generation ===")
    
    state.train_candidates_path = run_disk_backed_blocking(
        s1_df=state.train_s1_clean,
        s2_df=state.train_s2_clean,
        s3_df=state.train_s3_clean,
        output_dir=cfg.data.output_dir,
        prefix="train",
        config=cfg,
        s1_id_set=set(state.train_s1_ids)
    )
    
    state.val_candidates_path = run_disk_backed_blocking(
        s1_df=state.train_s1_clean,
        s2_df=state.train_s2_clean,
        s3_df=state.train_s3_clean,
        output_dir=cfg.data.output_dir,
        prefix="val",
        config=cfg,
        s1_id_set=set(state.val_s1_ids)
    )


# ---------------------------------------------------------------------------
# Stage 5: Evaluate candidate recall
# ---------------------------------------------------------------------------

def evaluate_candidate_recall(cfg: Config, state: PipelineState) -> None:
    """
    Stage 5: Measure how many ground-truth matches are captured in candidates.
    """
    from evaluation import candidate_recall

    logger.info("=== Stage 5: Candidate recall evaluation ===")
    assert state.train_candidates_path is not None and state.val_candidates_path is not None, "Run Stage 4 first!"
    assert state.train_gt is not None and state.val_gt is not None, "Run Stage 3 first!"
    assert state.train_s1_ids is not None and state.val_s1_ids is not None, "Run Stage 3 first!"

    train_cands = pd.read_parquet(state.train_candidates_path)
    val_cands = pd.read_parquet(state.val_candidates_path)

    train_recall = candidate_recall(train_cands, state.train_gt, state.train_s1_ids)
    val_recall = candidate_recall(val_cands, state.val_gt, state.val_s1_ids)

    logger.info("Train blocking recall: %s", train_recall)
    logger.info("Val blocking recall: %s", val_recall)



# ---------------------------------------------------------------------------
# Stage 6: Build training pairs
# ---------------------------------------------------------------------------


def build_training_pairs(cfg: Config, state: PipelineState) -> None:
    from pair_dataset import build_disk_backed_training_pairs, build_disk_backed_validation_pairs
    logger.info("=== Stage 6: Building training and validation pairs ===")
    
    ns = cfg.negative_sampling
    combined_target = pd.concat([state.train_s2_clean, state.train_s3_clean], ignore_index=True)
    
    state.train_pairs = build_disk_backed_training_pairs(
        candidates_parquet_path=state.train_candidates_path,
        gt=state.train_gt,
        train_s1_ids=state.train_s1_ids,
        s1_df=state.train_s1_clean,
        target_df=combined_target,
        neg_ratio=ns.neg_ratio,
        hard_neg_fraction=ns.hard_neg_fraction,
        random_seed=cfg.experiment.random_seed,
    )
    
    state.val_pairs = build_disk_backed_validation_pairs(
        candidates_parquet_path=state.val_candidates_path,
        gt=state.val_gt,
        val_s1_ids=state.val_s1_ids,
    )


# ---------------------------------------------------------------------------
# Stage 7: Build features
# ---------------------------------------------------------------------------

def build_features(cfg: Config, state: PipelineState) -> None:
    """
    Stage 7: Extract pairwise features for training and validation pairs.
    """
    from features import extract_features, build_entity_lookup, get_feature_column_names

    logger.info("=== Stage 7: Feature extraction ===")
    assert state.train_s1_clean is not None and state.train_s2_clean is not None and state.train_s3_clean is not None, "Run Stage 2 first!"
    assert state.train_pairs is not None and state.val_pairs is not None, "Run Stage 6 first!"

    entity_lookup = build_entity_lookup(
        state.train_s1_clean, state.train_s2_clean, state.train_s3_clean
    )

    state.train_features, state.train_metadata = extract_features(
        pairs_df=state.train_pairs, entity_lookup=entity_lookup
    )
    state.val_features, state.val_metadata = extract_features(
        pairs_df=state.val_pairs, entity_lookup=entity_lookup
    )

    state.feature_columns = get_feature_column_names(state.train_features)
    logger.info("Feature columns (%d): %s", len(state.feature_columns), state.feature_columns[:5])


# ---------------------------------------------------------------------------
# Stage 8: Train XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(cfg: Config, state: PipelineState) -> None:
    """
    Stage 8: Train XGBoost model on training feature matrix.
    """
    from training import train_xgboost_local, get_feature_matrix, get_labels, BASELINE_XGBOOST_PARAMS, save_experiment_config

    if state.train_features is None and state.train_merged is not None:
        state.train_features = state.train_merged
    if state.val_features is None and state.val_merged is not None:
        state.val_features = state.val_merged

    assert state.train_features is not None and state.val_features is not None, "Run Stage 7 first!"
    assert state.train_pairs is not None and state.val_pairs is not None, "Run Stage 6 first!"
    assert state.feature_columns is not None, "Run Stage 7 first!"

    train_merged = state.train_features.copy()
    if "is_match" in state.train_pairs.columns and "is_match" not in train_merged.columns:
        train_merged["is_match"] = state.train_pairs["is_match"].values

    val_merged = state.val_features.copy()
    if "is_match" in state.val_pairs.columns and "is_match" not in val_merged.columns:
        val_merged["is_match"] = state.val_pairs["is_match"].values

    X_train = get_feature_matrix(train_merged, state.feature_columns)
    y_train = get_labels(train_merged)
    X_val = get_feature_matrix(val_merged, state.feature_columns)
    y_val = get_labels(val_merged)

    xgb_cfg = cfg.xgboost
    params = {
        "objective": xgb_cfg.objective,
        "eval_metric": xgb_cfg.eval_metric,
        "max_depth": xgb_cfg.max_depth,
        "eta": xgb_cfg.eta,
        "min_child_weight": xgb_cfg.min_child_weight,
        "subsample": xgb_cfg.subsample,
        "colsample_bytree": xgb_cfg.colsample_bytree,
        "gamma": xgb_cfg.gamma,
        "reg_alpha": xgb_cfg.reg_alpha,
        "reg_lambda": xgb_cfg.reg_lambda,
    }

    model_path = os.path.join(cfg.data.output_dir, "model.json")
    state.booster = train_xgboost_local(
        X_train, y_train, X_val, y_val,
        params=params,
        num_round=xgb_cfg.num_round,
        early_stopping_rounds=xgb_cfg.early_stopping_rounds,
        feature_columns=state.feature_columns,
        model_output_path=model_path,
        random_seed=cfg.experiment.random_seed,
    )
    state.model_path = model_path


# ---------------------------------------------------------------------------
# Stage 9: Score validation
# ---------------------------------------------------------------------------

def score_validation(cfg: Config, state: PipelineState) -> None:
    """
    Stage 9: Generate match probabilities on validation candidate pairs.
    """
    from training import predict_probabilities

    logger.info("=== Stage 9: Scoring validation candidates ===")
    assert state.booster is not None, "Run Stage 8 first!"
    assert state.val_features is not None and state.feature_columns is not None, "Run Stage 7 first!"

    state.val_probabilities = predict_probabilities(
        state.booster, state.val_features, state.feature_columns
    )


# ---------------------------------------------------------------------------
# Stage 10: Optimize threshold
# ---------------------------------------------------------------------------

def optimize_threshold(cfg: Config, state: PipelineState) -> None:
    """
    Stage 10: Sweep thresholds to find the best macro F0.5 threshold.
    """
    from threshold import sweep_thresholds, select_best_threshold, save_threshold

    logger.info("=== Stage 10: Threshold optimization ===")
    assert state.val_pairs is not None and state.val_probabilities is not None, "Run Stages 6 & 9 first!"
    assert state.val_gt is not None and state.val_s1_ids is not None, "Run Stage 3 first!"

    tc = cfg.threshold
    state.sweep_df = sweep_thresholds(
        val_pairs_df=state.val_pairs,
        probabilities=state.val_probabilities,
        gt=state.val_gt,
        s1_ids=state.val_s1_ids,
        sweep_start=tc.sweep_start,
        sweep_end=tc.sweep_end,
        sweep_step=tc.sweep_step,
    )

    state.best_threshold, _ = select_best_threshold(state.sweep_df)

    threshold_path = os.path.join(cfg.data.output_dir, "best_threshold.json")
    save_threshold(state.best_threshold, threshold_path)
    state.threshold_path = threshold_path


# ---------------------------------------------------------------------------
# Stage 11: Error analysis
# ---------------------------------------------------------------------------

def perform_error_analysis(cfg: Config, state: PipelineState) -> None:
    """
    Stage 11: Analyze false positives, false negatives, and singleton errors.
    """
    from threshold import analyze_errors

    logger.info("=== Stage 11: Error analysis ===")
    assert state.val_pairs is not None and state.val_probabilities is not None, "Run Stages 6 & 9 first!"
    assert state.val_gt is not None and state.best_threshold is not None and state.val_s1_ids is not None, "Run Stages 3 & 10 first!"

    errors = analyze_errors(
        val_pairs_df=state.val_pairs,
        probabilities=state.val_probabilities,
        gt=state.val_gt,
        best_threshold=state.best_threshold,
        s1_ids=state.val_s1_ids,
    )

    for error_type, df in errors.items():
        logger.info("Error type '%s': %d examples", error_type, len(df))


# ---------------------------------------------------------------------------
# Stage 12: Generate test candidates
# ---------------------------------------------------------------------------


def generate_test_candidates(cfg: Config, state: PipelineState) -> None:
    from inference import load_and_clean_test_data
    from blocking import run_disk_backed_blocking
    logger.info("=== Stage 12: Generating test candidates ===")

    state.test_s1_clean, state.test_s2_clean, state.test_s3_clean = \
        load_and_clean_test_data(
            cfg.data.test_source1_path,
            cfg.data.test_source2_path,
            cfg.data.test_source3_path,
            debug_sample_size=cfg.runtime.debug_sample_size,
        )

    state.test_candidates_path = run_disk_backed_blocking(
        s1_df=state.test_s1_clean,
        s2_df=state.test_s2_clean,
        s3_df=state.test_s3_clean,
        output_dir=cfg.data.output_dir,
        prefix="test",
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Stage 13: Build test features
# ---------------------------------------------------------------------------


def build_test_features(cfg: Config, state: PipelineState) -> None:
    from inference import extract_test_features_streaming
    logger.info("=== Stage 13: Extracting test features (Disk-Backed) ===")

    state.test_features, state.test_metadata = extract_test_features_streaming(
        test_candidates_path=state.test_candidates_path,
        test_s1_df=state.test_s1_clean,
        test_s2_df=state.test_s2_clean,
        test_s3_df=state.test_s3_clean,
        output_dir=cfg.data.output_dir
    )


# ---------------------------------------------------------------------------
# Stage 14: Run test inference
# ---------------------------------------------------------------------------

def run_test_inference(cfg: Config, state: PipelineState) -> None:
    """
    Stage 14: Score test candidates and apply threshold to produce predictions.
    """
    from inference import run_inference

    logger.info("=== Stage 14: Test inference ===")
    assert state.test_s1_clean is not None and state.test_s2_clean is not None and state.test_s3_clean is not None, "Run Stage 12 first!"
    assert state.test_candidates_path is not None and state.test_features is not None and state.test_metadata is not None, "Run Stages 12 & 13 first!"
    assert state.model_path is not None and state.threshold_path is not None and state.feature_columns is not None, "Run Stages 8 & 10 first!"

    state.test_predictions, _ = run_inference(
        test_s1_df=state.test_s1_clean,
        test_s2_df=state.test_s2_clean,
        test_s3_df=state.test_s3_clean,
        test_candidates_df=pd.read_parquet(state.test_candidates_path),
        test_features_df=state.test_features,
        test_metadata_df=state.test_metadata,
        model_path=state.model_path,
        threshold_path=state.threshold_path,
        feature_columns=state.feature_columns,
    )


# ---------------------------------------------------------------------------
# Stage 15: Create submission files
# ---------------------------------------------------------------------------

def create_submission_files(cfg: Config, state: PipelineState) -> None:
    """
    Stage 15: Write matching_results.tsv and candidate_pairs.tsv.
    Runs official challenge validator.
    """
    from submission import generate_submission_files

    logger.info("=== Stage 15: Generating submission files ===")
    assert state.test_predictions is not None, "Run Stage 14 first!"
    assert state.test_candidates_path is not None and state.test_s2_clean is not None and state.test_s3_clean is not None and state.test_s1_clean is not None, "Run Stage 12 first!"

    all_test_s1_ids = list(state.test_s1_clean["entity_id"])

    matching_path, candidates_path, passed = generate_submission_files(
        all_test_s1_ids=all_test_s1_ids,
        predictions=state.test_predictions,
        candidates_df=pd.read_parquet(state.test_candidates_path),
        test_s2_df=state.test_s2_clean,
        test_s3_df=state.test_s3_clean,
        output_dir=cfg.data.output_dir,
        test_dir=os.path.dirname(cfg.data.test_source1_path),
        run_validator=True,
    )

    logger.info("Submission files: %s | %s | Validator: %s", matching_path, candidates_path, "PASS" if passed else "FAIL")



# ---------------------------------------------------------------------------
# Load test data (convenience alias for orchestration)
# ---------------------------------------------------------------------------

def load_test_data(cfg: Config, state: PipelineState) -> None:
    """Load test data (convenience alias — same as start of Stage 12)."""
    generate_test_candidates(cfg, state)


def clean_test_data(cfg: Config, state: PipelineState) -> None:
    """Clean test data (already done as part of load_and_clean_test_data in Stage 12)."""
    logger.info("Test data cleaning is integrated into Stage 12 (load_and_clean_test_data).")


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(cfg: Optional[Config] = None) -> PipelineState:
    """
    Run the complete end-to-end entity resolution pipeline.

    Each stage is called in order. Individual stages can be skipped
    or replaced for iterative experimentation.

    Parameters
    ----------
    cfg : Config, optional
        Configuration object. Uses DEFAULT_CONFIG if not provided.

    Returns
    -------
    PipelineState
        Final pipeline state with all outputs.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG

    cfg.configure_logging()
    state = PipelineState()

    stages = [
        ("load_data",                  load_training_data),
        ("clean_data",                 clean_training_data),
        ("split_training_validation",  split_training_validation),
        ("generate_candidates",        generate_candidates),
        ("evaluate_candidate_recall",  evaluate_candidate_recall),
        ("build_training_pairs",       build_training_pairs),
        ("build_features",             build_features),
        ("train_xgboost",              train_xgboost),
        ("score_validation",           score_validation),
        ("optimize_threshold",         optimize_threshold),
        ("perform_error_analysis",     perform_error_analysis),
        ("generate_test_candidates",   generate_test_candidates),
        ("build_test_features",        build_test_features),
        ("run_test_inference",         run_test_inference),
        ("create_submission_files",    create_submission_files),
    ]

    import gc

    for stage_name, stage_fn in stages:
        logger.info("Running stage: %s", stage_name)
        stage_fn(cfg, state)
        gc.collect()

    logger.info("Pipeline complete.")
    return state


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument("--data-dir", default=None, help="Path or s3:// URI to dataset folder")
    parser.add_argument("--output-dir", default="output", help="Path to output folder")
    parser.add_argument("--debug-sample", type=int, default=None)
    parser.add_argument("--use-s3", action="store_true", help="Force using public S3 dataset")
    args = parser.parse_args()

    cfg = Config()
    if args.use_s3:
        s3_base = "s3://ml-challenge-bharath/ml_challenge/Error_404_Not_Found_submission/dataset"
        cfg.data.train_source1_path = f"{s3_base}/train/train_source1.tsv"
        cfg.data.train_source2_path = f"{s3_base}/train/train_source2.tsv"
        cfg.data.train_source3_path = f"{s3_base}/train/train_source3.tsv"
        cfg.data.train_ground_truth_path = f"{s3_base}/train/train_ground_truth.tsv"
        cfg.data.test_source1_path = f"{s3_base}/test/test_source1.tsv"
        cfg.data.test_source2_path = f"{s3_base}/test/test_source2.tsv"
        cfg.data.test_source3_path = f"{s3_base}/test/test_source3.tsv"
    elif args.data_dir:
        data_base = args.data_dir.rstrip("/")
        if data_base.startswith("s3://"):
            cfg.data.train_source1_path = f"{data_base}/train/train_source1.tsv"
            cfg.data.train_source2_path = f"{data_base}/train/train_source2.tsv"
            cfg.data.train_source3_path = f"{data_base}/train/train_source3.tsv"
            cfg.data.train_ground_truth_path = f"{data_base}/train/train_ground_truth.tsv"
            cfg.data.test_source1_path = f"{data_base}/test/test_source1.tsv"
            cfg.data.test_source2_path = f"{data_base}/test/test_source2.tsv"
            cfg.data.test_source3_path = f"{data_base}/test/test_source3.tsv"
        else:
            cfg.data.train_source1_path = os.path.join(args.data_dir, "train", "train_source1.tsv")
            cfg.data.train_source2_path = os.path.join(args.data_dir, "train", "train_source2.tsv")
            cfg.data.train_source3_path = os.path.join(args.data_dir, "train", "train_source3.tsv")
            cfg.data.train_ground_truth_path = os.path.join(args.data_dir, "train", "train_ground_truth.tsv")
            cfg.data.test_source1_path = os.path.join(args.data_dir, "test", "test_source1.tsv")
            cfg.data.test_source2_path = os.path.join(args.data_dir, "test", "test_source2.tsv")
            cfg.data.test_source3_path = os.path.join(args.data_dir, "test", "test_source3.tsv")

    cfg.data.output_dir = args.output_dir
    cfg.runtime.debug_sample_size = args.debug_sample

    run_full_pipeline(cfg)
