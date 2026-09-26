"""
threshold.py — Precision-optimized threshold selection for F0.5 maximization.

Why threshold tuning is critical:
  F0.5 is a precision-heavy metric (precision weighted 2× over recall).
  The default probability threshold of 0.5 is almost never optimal.
  We sweep thresholds on the held-out validation set to find the threshold
  that maximizes entity-level macro F0.5.

CRITICAL RULE:
  The threshold is ALWAYS selected using VALIDATION data.
  Test data MUST NOT influence threshold selection.
  The threshold is then applied at inference time to test predictions.

Source-specific thresholds (threshold_S2, threshold_S3) may be explored,
but a global threshold remains the default unless validation shows improvement.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from evaluation import macro_f05, evaluate_predictions_by_entity, singleton_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prediction aggregation from probabilities
# ---------------------------------------------------------------------------

def predictions_from_probabilities(
    val_pairs_df: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    s1_ids: List[str],
) -> Dict[str, Set[str]]:
    """
    Convert per-pair match probabilities into per-entity prediction sets.

    Parameters
    ----------
    val_pairs_df : pd.DataFrame
        Candidate pairs with source1_entity_id and candidate_entity_id.
    probabilities : np.ndarray
        Match probabilities ∈ [0, 1] for each pair (same order as val_pairs_df).
    threshold : float
        Decision threshold: predict match if probability >= threshold.
    s1_ids : List[str]
        All S1 entity IDs to generate predictions for.
        Entities with no pairs above threshold get empty prediction sets.

    Returns
    -------
    Dict[str, Set[str]]
        Maps source1_entity_id → set of predicted matched IDs.
    """
    predictions: Dict[str, Set[str]] = {sid: set() for sid in s1_ids}

    for i, (_, row) in enumerate(val_pairs_df.iterrows()):
        if probabilities[i] >= threshold:
            s1_id = row["source1_entity_id"]
            cand_id = row["candidate_entity_id"]
            if s1_id in predictions:
                predictions[s1_id].add(cand_id)

    return predictions


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def sweep_thresholds(
    val_pairs_df: pd.DataFrame,
    probabilities: np.ndarray,
    gt: Dict[str, Set[str]],
    s1_ids: List[str],
    sweep_start: float = 0.10,
    sweep_end: float = 0.99,
    sweep_step: float = 0.01,
) -> pd.DataFrame:
    """
    Sweep decision thresholds and evaluate macro F0.5 at each threshold.

    Parameters
    ----------
    val_pairs_df : pd.DataFrame
        Validation candidate pairs.
    probabilities : np.ndarray
        Match probabilities from the trained model.
    gt : Dict[str, Set[str]]
        Validation ground truth.
    s1_ids : List[str]
        All validation S1 entity IDs (including singletons).
    sweep_start : float
        Starting threshold (inclusive).
    sweep_end : float
        Ending threshold (inclusive).
    sweep_step : float
        Step size for threshold grid.

    Returns
    -------
    pd.DataFrame
        Sweep results with columns:
        threshold, macro_f05, macro_precision, macro_recall,
        n_predicted_matches, n_total_predicted_ids,
        n_false_positives, n_false_negatives,
        singleton_accuracy, n_singleton_false_merges.

    Notes
    -----
    DO NOT hard-code threshold. The best threshold is selected empirically.
    """
    thresholds = np.arange(sweep_start, sweep_end + sweep_step / 2, sweep_step)
    thresholds = np.round(thresholds, 4)

    rows = []
    for t in thresholds:
        preds = predictions_from_probabilities(val_pairs_df, probabilities, t, s1_ids)

        entity_results = evaluate_predictions_by_entity(gt, preds, s1_ids)

        avg_f05 = float(entity_results["f05"].mean())
        avg_prec = float(entity_results["precision"].mean())
        avg_rec = float(entity_results["recall"].mean())

        total_fp = int(entity_results["fp"].sum())
        total_fn = int(entity_results["fn"].sum())
        n_with_preds = int((entity_results["n_predicted"] > 0).sum())
        total_pred_ids = int(entity_results["n_predicted"].sum())

        sing_metrics = singleton_metrics(entity_results)

        rows.append({
            "threshold":               round(float(t), 4),
            "macro_f05":               avg_f05,
            "macro_precision":         avg_prec,
            "macro_recall":            avg_rec,
            "n_s1_with_predictions":   n_with_preds,
            "n_total_predicted_ids":   total_pred_ids,
            "n_false_positives":       total_fp,
            "n_false_negatives":       total_fn,
            "singleton_accuracy":      sing_metrics["singleton_accuracy"],
            "n_singleton_false_merges": sing_metrics["n_singleton_false_merge"],
        })

    result_df = pd.DataFrame(rows)
    logger.info(
        "Threshold sweep complete: %d thresholds evaluated. "
        "Best F0.5=%.4f at threshold=%.2f",
        len(result_df),
        result_df["macro_f05"].max(),
        result_df.loc[result_df["macro_f05"].idxmax(), "threshold"],
    )
    return result_df


def select_best_threshold(sweep_df: pd.DataFrame) -> Tuple[float, float]:
    """
    Select the threshold that maximizes macro F0.5 on the validation set.

    Parameters
    ----------
    sweep_df : pd.DataFrame
        Output from sweep_thresholds().

    Returns
    -------
    Tuple[float, float]
        (best_threshold, best_f05_score)
    """
    best_row = sweep_df.loc[sweep_df["macro_f05"].idxmax()]
    best_t = float(best_row["threshold"])
    best_f05 = float(best_row["macro_f05"])

    logger.info(
        "Selected threshold: %.4f → macro F0.5 = %.4f | Precision = %.4f | Recall = %.4f",
        best_t,
        best_f05,
        float(best_row["macro_precision"]),
        float(best_row["macro_recall"]),
    )
    return best_t, best_f05


# ---------------------------------------------------------------------------
# Source-specific threshold analysis (optional)
# ---------------------------------------------------------------------------

def sweep_source_specific_thresholds(
    val_pairs_df: pd.DataFrame,
    probabilities: np.ndarray,
    gt: Dict[str, Set[str]],
    s1_ids: List[str],
    sweep_start: float = 0.10,
    sweep_end: float = 0.99,
    sweep_step: float = 0.01,
) -> Dict[str, pd.DataFrame]:
    """
    Optionally evaluate source-specific thresholds for S2 and S3 separately.

    A global threshold remains the default. Source-specific thresholds
    should only be used if validation data demonstrates clear improvement.

    Returns
    -------
    Dict with keys "s2" and "s3", each containing a threshold sweep DataFrame.
    """
    results = {}
    for source in ["S2", "S3"]:
        source_pairs = val_pairs_df[val_pairs_df["candidate_source"] == source].copy()
        source_indices = list(val_pairs_df.index.get_indexer(source_pairs.index))
        source_probs = probabilities[source_indices]

        sweep = sweep_thresholds(
            source_pairs, source_probs, gt, s1_ids,
            sweep_start, sweep_end, sweep_step,
        )
        results[source.lower()] = sweep
        best_t, best_f05 = select_best_threshold(sweep)
        logger.info(
            "[%s-specific] Best threshold: %.4f → F0.5 = %.4f", source, best_t, best_f05
        )

    return results


# ---------------------------------------------------------------------------
# Threshold persistence
# ---------------------------------------------------------------------------

def save_threshold(
    threshold: float,
    output_path: str,
    metadata: Optional[dict] = None,
) -> None:
    """
    Save the selected threshold to a JSON file.

    Parameters
    ----------
    threshold : float
        The selected decision threshold.
    output_path : str
        Local path (later uploaded to S3 for SageMaker inference).
    metadata : dict, optional
        Additional metadata to store (e.g., sweep summary, experiment_id).
    """
    record = {"threshold": threshold}
    if metadata:
        record.update(metadata)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(record, f, indent=2)

    logger.info("Threshold %.4f saved to: %s", threshold, output_path)


def load_threshold(input_path: str) -> float:
    """
    Load the selected threshold from a JSON file.

    Parameters
    ----------
    input_path : str
        Local path to the threshold JSON file.

    Returns
    -------
    float
        The decision threshold.
    """
    with open(input_path) as f:
        record = json.load(f)
    threshold = float(record["threshold"])
    logger.info("Loaded threshold: %.4f from: %s", threshold, input_path)
    return threshold


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def analyze_errors(
    val_pairs_df: pd.DataFrame,
    probabilities: np.ndarray,
    gt: Dict[str, Set[str]],
    best_threshold: float,
    s1_ids: List[str],
    features_df: Optional[pd.DataFrame] = None,
    top_n: int = 50,
) -> Dict[str, pd.DataFrame]:
    """
    Analyze high-confidence false positives, false negatives, and singleton errors.

    Parameters
    ----------
    val_pairs_df : pd.DataFrame
        Validation candidate pairs.
    probabilities : np.ndarray
        Match probabilities.
    gt : Dict[str, Set[str]]
        Validation ground truth.
    best_threshold : float
        Selected decision threshold.
    s1_ids : List[str]
        All validation S1 IDs.
    features_df : pd.DataFrame, optional
        If provided, attach feature values to error records for analysis.
    top_n : int
        Number of top errors to return in each category.

    Returns
    -------
    Dict with keys:
        "high_conf_fp" : high-confidence false positives
        "false_negatives" : missed true matches
        "singleton_fp"  : singletons with false merges
    """
    val_pairs_df = val_pairs_df.copy()
    val_pairs_df["probability"] = probabilities
    val_pairs_df["predicted_match"] = (probabilities >= best_threshold).astype(int)

    # True positive / false positive / false negative labelling
    val_pairs_df["is_true_match"] = val_pairs_df.apply(
        lambda row: int(row["candidate_entity_id"] in gt.get(row["source1_entity_id"], set())),
        axis=1,
    )

    # High-confidence false positives (predicted=1, actual=0, high probability)
    high_conf_fp = val_pairs_df[
        (val_pairs_df["predicted_match"] == 1) &
        (val_pairs_df["is_true_match"] == 0)
    ].sort_values("probability", ascending=False).head(top_n)

    # False negatives (predicted=0, actual=1)
    false_neg = val_pairs_df[
        (val_pairs_df["predicted_match"] == 0) &
        (val_pairs_df["is_true_match"] == 1)
    ].sort_values("probability", ascending=False).head(top_n)

    # Singleton false positives (S1 has no true matches but got a predicted match)
    singleton_s1_ids = {sid for sid, v in gt.items() if len(v) == 0}
    singleton_fp = val_pairs_df[
        (val_pairs_df["source1_entity_id"].isin(singleton_s1_ids)) &
        (val_pairs_df["predicted_match"] == 1)
    ].sort_values("probability", ascending=False).head(top_n)

    # Optionally attach feature values for inspection
    if features_df is not None:
        pass  # Feature join can be added here once feature index is stable

    logger.info(
        "Error analysis: %d high-conf FP | %d FN | %d singleton FP",
        len(high_conf_fp), len(false_neg), len(singleton_fp),
    )

    return {
        "high_conf_fp": high_conf_fp,
        "false_negatives": false_neg,
        "singleton_fp": singleton_fp,
    }


# ---------------------------------------------------------------------------
# Threshold sweep report
# ---------------------------------------------------------------------------

def print_threshold_report(sweep_df: pd.DataFrame, top_n: int = 10) -> None:
    """Print a readable summary of the top threshold sweep results."""
    top_rows = sweep_df.nlargest(top_n, "macro_f05")[
        ["threshold", "macro_f05", "macro_precision", "macro_recall",
         "n_total_predicted_ids", "singleton_accuracy"]
    ]
    logger.info("Top %d threshold results:\n%s", top_n, top_rows.to_string(index=False))
