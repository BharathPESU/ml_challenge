"""
blocking_eval.py — Evaluation and metrics for candidate generation.

Measures candidate recall, reduction ratio, candidate volume, and
logs detailed performance statistics for blocking passes.
"""

import logging
import time
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def candidate_recall(
    candidates_df: pd.DataFrame,
    gt: Dict[str, Set[str]],
    s1_ids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute the fraction of true ground-truth matches captured in the candidate set.
    """
    if s1_ids is not None:
        gt = {sid: gt.get(sid, set()) for sid in s1_ids}

    candidate_pairs: Set[Tuple[str, str]] = set(
        zip(candidates_df["source1_entity_id"], candidates_df["candidate_entity_id"])
    )

    total_tp = 0
    found_tp = 0
    found_s2 = 0
    total_s2 = 0
    found_s3 = 0
    total_s3 = 0

    for s1_id, matched_ids in gt.items():
        for mid in matched_ids:
            total_tp += 1
            if mid.startswith("S2-"):
                total_s2 += 1
            else:
                total_s3 += 1

            if (s1_id, mid) in candidate_pairs:
                found_tp += 1
                if mid.startswith("S2-"):
                    found_s2 += 1
                else:
                    found_s3 += 1

    overall = found_tp / total_tp if total_tp > 0 else 1.0
    s2_recall = found_s2 / total_s2 if total_s2 > 0 else 1.0
    s3_recall = found_s3 / total_s3 if total_s3 > 0 else 1.0

    logger.info(
        "Blocking Recall — Overall: %.4f | S2: %.4f | S3: %.4f (%d/%d true pairs found)",
        overall, s2_recall, s3_recall, found_tp, total_tp,
    )
    return {"overall": overall, "s2": s2_recall, "s3": s3_recall}


def candidate_count_statistics(candidates_df: pd.DataFrame) -> Dict[str, float]:
    """Compute per-S1 candidate count statistics."""
    if candidates_df.empty:
        return {"mean": 0.0, "median": 0.0, "max": 0.0, "min": 0.0, "p95": 0.0, "total_pairs": 0, "unique_s1": 0}
        
    counts = candidates_df.groupby("source1_entity_id")["candidate_entity_id"].count()
    return {
        "mean": float(counts.mean()),
        "median": float(counts.median()),
        "max": float(counts.max()),
        "min": float(counts.min()),
        "p95": float(counts.quantile(0.95)),
        "total_pairs": int(len(candidates_df)),
        "unique_s1": int(candidates_df["source1_entity_id"].nunique()),
    }


def reduction_ratio(
    candidates_df: pd.DataFrame,
    n_s1: int,
    n_s2: int,
    n_s3: int,
) -> float:
    """
    Reduction ratio: fraction of the full Cartesian space eliminated by blocking.
    """
    full_space = n_s1 * (n_s2 + n_s3)
    if full_space == 0:
        return 0.0
    rr = 1.0 - len(candidates_df) / full_space
    logger.info(
        "Reduction ratio: %.6f (%d candidates / %d full pairs)",
        rr, len(candidates_df), full_space,
    )
    return rr


def evaluate_blocking_quality(
    candidates_df: pd.DataFrame,
    gt: Dict[str, Set[str]],
    s1_ids: Optional[List[str]],
    n_s1: int,
    n_s2: int,
    n_s3: int,
    runtime_sec: float
) -> Dict[str, float]:
    """
    Evaluate blocking comprehensively: recall, candidate volume, and runtime.
    Logs detailed information.
    """
    logger.info("=== Blocking Quality Evaluation ===")
    
    recall_stats = candidate_recall(candidates_df, gt, s1_ids)
    count_stats = candidate_count_statistics(candidates_df)
    rr = reduction_ratio(candidates_df, n_s1, n_s2, n_s3)
    
    logger.info("Candidate Volume:")
    logger.info("  Total Pairs: %d", count_stats["total_pairs"])
    logger.info("  Unique S1: %d", count_stats["unique_s1"])
    logger.info("  Mean per S1: %.2f", count_stats["mean"])
    logger.info("  Median per S1: %.2f", count_stats["median"])
    logger.info("  P95 per S1: %.2f", count_stats["p95"])
    logger.info("  Max per S1: %d", count_stats["max"])
    logger.info("Runtime: %.2f seconds", runtime_sec)
    
    return {
        "recall_overall": recall_stats["overall"],
        "recall_s2": recall_stats["s2"],
        "recall_s3": recall_stats["s3"],
        "reduction_ratio": rr,
        "mean_candidates": count_stats["mean"],
        "p95_candidates": count_stats["p95"],
        "runtime_sec": runtime_sec
    }
