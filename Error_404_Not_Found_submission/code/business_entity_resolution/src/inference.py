"""
inference.py — Test-time inference for Business Entity Resolution.

Process:
  1. Load test_source1, test_source2, test_source3
  2. Apply IDENTICAL cleaning/normalization as training
  3. Apply the final validated blocking configuration
  4. Generate the exact candidate set passed to the ML model
  5. Extract IDENTICAL pairwise features as training
  6. Load trained XGBoost model artifact
  7. Predict match probabilities
  8. Load threshold selected on held-out validation
  9. Apply threshold → predict match/no-match per pair
  10. Aggregate per S1 entity → final match lists

CRITICAL RULES:
  - Do NOT train on test data.
  - Do NOT use test labels (none exist).
  - Do NOT change feature definitions from training.
  - Do NOT change preprocessing rules from training.
  - Do NOT use external data.
  - Feature columns MUST be identical to training (loaded from saved schema).
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from cleaning import clean_source_dataframe
try:
    from blocking import generate_candidates
except ImportError:
    import sys, importlib
    if 'blocking' in sys.modules:
        import blocking
        importlib.reload(blocking)
    from blocking import generate_candidates
from features import extract_features, build_entity_lookup, TFIDFSimilarityComputer
from training import predict_probabilities, load_xgboost_model_local, get_feature_matrix
from threshold import predictions_from_probabilities, load_threshold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test data loading and cleaning
# ---------------------------------------------------------------------------

def load_and_clean_test_data(
    test_s1_path: str,
    test_s2_path: str,
    test_s3_path: str,
    debug_sample_size: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and clean the three test source files.

    Applies EXACTLY the same cleaning logic as training data.
    This is the only way to prevent train/test feature distribution shift.

    Parameters
    ----------
    test_s1_path : str
        Path or S3 URI to test_source1.tsv.
    test_s2_path : str
        Path or S3 URI to test_source2.tsv.
    test_s3_path : str
        Path or S3 URI to test_source3.tsv.
    debug_sample_size : int, optional
        If provided, limit S1 to this many rows (for quick debugging).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (test_s1_clean, test_s2_clean, test_s3_clean)
    """
    from io_utils import read_tsv, validate_source_file
    from cleaning import clean_source_dataframe

    logger.info("Loading test data...")
    test_s1 = read_tsv(test_s1_path, nrows=debug_sample_size)
    test_s2 = read_tsv(test_s2_path)
    test_s3 = read_tsv(test_s3_path)

    validate_source_file(test_s1, "test_source1")
    validate_source_file(test_s2, "test_source2")
    validate_source_file(test_s3, "test_source3")

    logger.info(
        "Test data loaded: %d S1 | %d S2 | %d S3",
        len(test_s1), len(test_s2), len(test_s3),
    )

    # Apply IDENTICAL cleaning as training
    test_s1_clean = clean_source_dataframe(test_s1, source_label="test_S1")
    test_s2_clean = clean_source_dataframe(test_s2, source_label="test_S2")
    test_s3_clean = clean_source_dataframe(test_s3, source_label="test_S3")

    return test_s1_clean, test_s2_clean, test_s3_clean


# ---------------------------------------------------------------------------
# Test candidate generation
# ---------------------------------------------------------------------------

def generate_test_candidates(
    test_s1_df: pd.DataFrame,
    test_s2_df: pd.DataFrame,
    test_s3_df: pd.DataFrame,
    blocking_config: Optional[dict] = None,
    config: Any = None,
) -> pd.DataFrame:
    """
    Generate the FINAL candidate set for test data.

    This is the exact set passed to the ML model for scoring.
    This set becomes candidate_pairs.tsv.

    Parameters
    ----------
    test_s1_df : pd.DataFrame
        Cleaned test Source 1.
    test_s2_df : pd.DataFrame
        Cleaned test Source 2.
    test_s3_df : pd.DataFrame
        Cleaned test Source 3.
    blocking_config : dict, optional
        Blocking configuration dictionary.
    config : Config or Any, optional
        Master configuration object.

    Returns
    -------
    pd.DataFrame
        Candidate pairs with: source1_entity_id, candidate_entity_id, candidate_source.
    """
    logger.info("Generating test candidate pairs...")
    cfg = config if config is not None else blocking_config
    candidates_df = generate_candidates(
        s1_df=test_s1_df,
        s2_df=test_s2_df,
        s3_df=test_s3_df,
        config=cfg,
    )
    logger.info("Test candidates generated: %d pairs", len(candidates_df))
    return candidates_df

    return candidates_df


# ---------------------------------------------------------------------------
# Test feature extraction
# ---------------------------------------------------------------------------

def extract_test_features(
    test_candidates_df: pd.DataFrame,
    test_s1_df: pd.DataFrame,
    test_s2_df: pd.DataFrame,
    test_s3_df: pd.DataFrame,
    name_tfidf: Optional[TFIDFSimilarityComputer] = None,
    address_tfidf: Optional[TFIDFSimilarityComputer] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract pairwise features for test candidate pairs.

    Uses IDENTICAL feature computation as training.
    Feature columns are sorted deterministically (alphabetical).

    Parameters
    ----------
    test_candidates_df : pd.DataFrame
        Test candidate pairs.
    test_s1_df, test_s2_df, test_s3_df : pd.DataFrame
        Cleaned test source DataFrames.
    name_tfidf : TFIDFSimilarityComputer, optional
        TF-IDF computer fitted on training corpus (used for cosine features).
    address_tfidf : TFIDFSimilarityComputer, optional
        TF-IDF computer fitted on training corpus.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (test_features_df, test_metadata_df)
    """
    entity_lookup = build_entity_lookup(test_s1_df, test_s2_df, test_s3_df)
    test_features_df, test_metadata_df = extract_features(
        pairs_df=test_candidates_df,
        entity_lookup=entity_lookup,
        name_tfidf=name_tfidf,
        address_tfidf=address_tfidf,
    )
    return test_features_df, test_metadata_df


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def run_inference(
    test_s1_df: pd.DataFrame,
    test_s2_df: pd.DataFrame,
    test_s3_df: pd.DataFrame,
    test_candidates_df: pd.DataFrame,
    test_features_df: pd.DataFrame,
    test_metadata_df: pd.DataFrame,
    model_path: str,
    threshold_path: str,
    feature_columns: List[str],
) -> Tuple[Dict[str, Set[str]], pd.DataFrame]:
    """
    Run end-to-end inference: load model, score candidates, apply threshold.

    Parameters
    ----------
    test_s1_df : pd.DataFrame
        Cleaned test Source 1 (used to get all test S1 IDs).
    test_s2_df, test_s3_df : pd.DataFrame
        Cleaned test Source 2 and 3.
    test_candidates_df : pd.DataFrame
        Final candidate pairs (becomes candidate_pairs.tsv).
    test_features_df : pd.DataFrame
        Feature matrix for test candidates.
    test_metadata_df : pd.DataFrame
        Metadata (entity IDs, candidate_source) for test candidates.
    model_path : str
        Local path to trained XGBoost model (.json).
    threshold_path : str
        Local path to threshold JSON artifact.
    feature_columns : List[str]
        Feature column list from training schema (deterministic ordering).

    Returns
    -------
    Tuple[Dict[str, Set[str]], pd.DataFrame]
        (predictions, scored_candidates_df)
        predictions: s1_id → set of predicted matched IDs
        scored_candidates_df: test_candidates_df with probability column added
    """
    # Load model and threshold
    booster = load_xgboost_model_local(model_path)
    threshold = load_threshold(threshold_path)

    # Score candidate pairs
    probabilities = predict_probabilities(booster, test_features_df, feature_columns)

    # Attach probabilities to candidate df for candidate_pairs.tsv validation
    scored_df = test_candidates_df.copy()
    scored_df["probability"] = probabilities

    # Get all test S1 IDs (every one MUST appear in submission)
    all_test_s1_ids = list(test_s1_df["entity_id"])

    # Convert probabilities → per-entity predictions
    predictions = predictions_from_probabilities(
        val_pairs_df=test_candidates_df,
        probabilities=probabilities,
        threshold=threshold,
        s1_ids=all_test_s1_ids,
    )

    # Validate: every matched ID must be in candidates (pipeline bug check)
    candidate_set = set(
        zip(test_candidates_df["source1_entity_id"], test_candidates_df["candidate_entity_id"])
    )
    for s1_id, match_ids in predictions.items():
        for mid in match_ids:
            if (s1_id, mid) not in candidate_set:
                logger.error(
                    "PIPELINE BUG: predicted match (%s, %s) not in candidate set!",
                    s1_id, mid,
                )

    n_with_matches = sum(1 for v in predictions.values() if v)
    logger.info(
        "Inference complete: %d S1 entities | %d with matches | %d singletons | threshold=%.4f",
        len(all_test_s1_ids),
        n_with_matches,
        len(all_test_s1_ids) - n_with_matches,
        threshold,
    )
    return predictions, scored_df


import pyarrow.parquet as pq
import gc
import time

def extract_test_features_streaming(
    test_candidates_path: str,
    test_s1_df: pd.DataFrame,
    test_s2_df: pd.DataFrame,
    test_s3_df: pd.DataFrame,
    output_dir: str,
    chunk_size: int = 100000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract features out-of-core to prevent OOM."""
    logger.info("🦆 Extracting test features out-of-core...")
    
    pf = pq.ParquetFile(test_candidates_path)
    n_total = pf.metadata.num_rows
    n_chunks = math.ceil(n_total / chunk_size)
    
    feat_chunks = []
    meta_chunks = []
    
    for i, batch in enumerate(pf.iter_batches(batch_size=chunk_size)):
        t0 = time.time()
        chunk_df = batch.to_pandas()
        f_df, m_df = extract_test_features(chunk_df, test_s1_df, test_s2_df, test_s3_df)
        
        # Downcast floats
        for col in f_df.select_dtypes(include=['float64']).columns:
            f_df[col] = f_df[col].astype(np.float32)
            
        feat_chunks.append(f_df)
        meta_chunks.append(m_df)
        dt = time.time() - t0
        logger.info(f"   ✅ Test Chunk {i+1}/{n_chunks} done in {dt:.1f}s")
        del chunk_df, f_df, m_df
        gc.collect()
        
    final_feats = pd.concat(feat_chunks, ignore_index=True)
    final_meta = pd.concat(meta_chunks, ignore_index=True)
    return final_feats, final_meta
