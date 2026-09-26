"""
evaluation.py — Official competition metric implementation for Entity Resolution.

Competition Metric:
  F_beta = (1 + beta^2) * Precision * Recall / (beta^2 * Precision + Recall)
  beta = 0.5 → precision-weighted (false merges penalised 2× over missed matches)

Evaluation is entity-level (set-based), then macro-averaged across all S1 entities.

SINGLETON RULE:
  If a Source 1 entity has no true matches:
    - Predicting empty list → F0.5 = 1.0 (correct singleton)
    - Predicting any match  → F0.5 = 0.0 (false merge on singleton)

DO NOT use:
  - accuracy
  - global F1
  - micro F0.5
  - ROC-AUC

The only official optimization target is: entity-level macro F0.5.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BETA = 0.5
BETA_SQ = BETA ** 2


# ---------------------------------------------------------------------------
# Per-entity metrics
# ---------------------------------------------------------------------------

def entity_level_precision(actual: Set[str], predicted: Set[str]) -> float:
    """
    Compute precision for a single S1 entity.

    precision = |actual ∩ predicted| / |predicted|

    Returns 1.0 if both actual and predicted are empty (correct singleton).
    Returns 0.0 if predicted is non-empty but actual is empty (false singleton merge).
    Returns 1.0 if predicted is empty (no false positives; recall handled separately).
    """
    if not actual and not predicted:
        return 1.0   # correct singleton
    if not predicted:
        return 1.0   # no predictions → perfect precision (recall may suffer)
    if not actual:
        return 0.0   # false singleton merge
    tp = len(actual & predicted)
    return tp / len(predicted)


def entity_level_recall(actual: Set[str], predicted: Set[str]) -> float:
    """
    Compute recall for a single S1 entity.

    recall = |actual ∩ predicted| / |actual|

    Returns 1.0 if both actual and predicted are empty (correct singleton).
    Returns 0.0 if actual is non-empty but predicted is empty.
    """
    if not actual and not predicted:
        return 1.0   # correct singleton
    if not actual:
        return 1.0   # no true matches; recall is undefined — treat as 1.0
    if not predicted:
        return 0.0
    tp = len(actual & predicted)
    return tp / len(actual)


def entity_level_fbeta(
    actual: Set[str],
    predicted: Set[str],
    beta: float = BETA,
) -> float:
    """
    Compute F_beta score for a single S1 entity using set-based evaluation.

    Singleton rule (when actual is empty):
      - predicted is empty → F_beta = 1.0
      - predicted is non-empty → F_beta = 0.0

    Parameters
    ----------
    actual : Set[str]
        Ground truth matched entity IDs.
    predicted : Set[str]
        Model-predicted matched entity IDs.
    beta : float
        Beta parameter (0.5 for this competition).

    Returns
    -------
    float
        F_beta score in [0, 1].
    """
    # Singleton case
    if not actual:
        return 1.0 if not predicted else 0.0

    prec = entity_level_precision(actual, predicted)
    rec = entity_level_recall(actual, predicted)

    beta_sq = beta ** 2
    denom = beta_sq * prec + rec
    if denom == 0.0:
        return 0.0

    return (1 + beta_sq) * prec * rec / denom


def macro_f05(
    gt: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    s1_ids: List[str],
) -> float:
    """
    Compute macro-averaged F0.5 across all Source 1 entities.

    Parameters
    ----------
    gt : Dict[str, Set[str]]
        Ground truth mapping: s1_id → set of true matched IDs.
    predictions : Dict[str, Set[str]]
        Model predictions: s1_id → set of predicted matched IDs.
    s1_ids : List[str]
        All S1 entity IDs to evaluate (must be the complete set for fair averaging).

    Returns
    -------
    float
        Macro-averaged F0.5 score.

    Notes
    -----
    Every s1_id in s1_ids is included in the average, including singletons.
    Missing prediction entries are treated as empty sets.
    """
    scores = []
    for s1_id in s1_ids:
        actual = gt.get(s1_id, set())
        predicted = predictions.get(s1_id, set())
        score = entity_level_fbeta(actual, predicted, beta=BETA)
        scores.append(score)

    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Full entity-level evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions_by_entity(
    gt: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    s1_ids: List[str],
) -> pd.DataFrame:
    """
    Evaluate predictions at the entity level and return a per-entity results DataFrame.

    Parameters
    ----------
    gt : Dict[str, Set[str]]
        Ground truth mapping.
    predictions : Dict[str, Set[str]]
        Predicted matches.
    s1_ids : List[str]
        All S1 entity IDs.

    Returns
    -------
    pd.DataFrame
        Per-entity evaluation results with columns:
        source1_entity_id, n_actual, n_predicted, tp, fp, fn,
        precision, recall, f05, is_singleton, singleton_correct.
    """
    rows = []
    for s1_id in s1_ids:
        actual = gt.get(s1_id, set())
        predicted = predictions.get(s1_id, set())

        tp = len(actual & predicted)
        fp = len(predicted - actual)
        fn = len(actual - predicted)

        prec = entity_level_precision(actual, predicted)
        rec = entity_level_recall(actual, predicted)
        f05 = entity_level_fbeta(actual, predicted, beta=BETA)

        is_singleton = int(len(actual) == 0)
        singleton_correct = int(is_singleton and len(predicted) == 0)

        rows.append({
            "source1_entity_id": s1_id,
            "n_actual": len(actual),
            "n_predicted": len(predicted),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "f05": f05,
            "is_singleton": is_singleton,
            "singleton_correct": singleton_correct,
        })

    return pd.DataFrame(rows)


def singleton_metrics(entity_results_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute singleton-specific evaluation metrics.

    Returns
    -------
    Dict with:
      n_singletons, n_singleton_correct, singleton_accuracy,
      n_singleton_false_merge (false positives on singletons)
    """
    singletons = entity_results_df[entity_results_df["is_singleton"] == 1]
    n_singletons = len(singletons)
    n_correct = int(singletons["singleton_correct"].sum())
    n_false_merge = n_singletons - n_correct

    return {
        "n_singletons": n_singletons,
        "n_singleton_correct": n_correct,
        "n_singleton_false_merge": n_false_merge,
        "singleton_accuracy": n_correct / n_singletons if n_singletons > 0 else 1.0,
    }


def candidate_recall(
    candidates_df: pd.DataFrame,
    gt: Dict[str, Set[str]],
    s1_ids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute blocking recall: fraction of true positives captured in the candidate set.

    candidate_recall = |true pairs ∩ candidate pairs| / |true pairs|

    Parameters
    ----------
    candidates_df : pd.DataFrame
        Candidate pairs with source1_entity_id and candidate_entity_id.
    gt : Dict[str, Set[str]]
        Ground truth mapping.
    s1_ids : List[str], optional
        Restrict to these S1 entities only.

    Returns
    -------
    Dict with keys: overall, s2, s3
    """
    if s1_ids is not None:
        gt = {sid: gt.get(sid, set()) for sid in s1_ids}

    candidate_set = set(
        zip(candidates_df["source1_entity_id"], candidates_df["candidate_entity_id"])
    )

    total = found = total_s2 = found_s2 = total_s3 = found_s3 = 0

    for s1_id, matched_ids in gt.items():
        for mid in matched_ids:
            total += 1
            if mid.startswith("S2-"):
                total_s2 += 1
            else:
                total_s3 += 1
            if (s1_id, mid) in candidate_set:
                found += 1
                if mid.startswith("S2-"):
                    found_s2 += 1
                else:
                    found_s3 += 1

    result = {
        "overall": found / total if total > 0 else 1.0,
        "s2": found_s2 / total_s2 if total_s2 > 0 else 1.0,
        "s3": found_s3 / total_s3 if total_s3 > 0 else 1.0,
        "found": found,
        "total": total,
    }
    logger.info(
        "Candidate recall — Overall: %.4f | S2: %.4f | S3: %.4f (%d/%d)",
        result["overall"], result["s2"], result["s3"], found, total,
    )
    return result


def reduction_ratio(
    candidates_df: pd.DataFrame,
    n_s1: int,
    n_s2: int,
    n_s3: int,
) -> float:
    """
    Fraction of Cartesian pairs eliminated by blocking.
    Higher is better (less work for the ML model).
    """
    full = n_s1 * (n_s2 + n_s3)
    if full == 0:
        return 0.0
    rr = 1.0 - len(candidates_df) / full
    logger.info("Reduction ratio: %.6f", rr)
    return rr
