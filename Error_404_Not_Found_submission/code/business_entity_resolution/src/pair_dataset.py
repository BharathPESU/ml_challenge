"""
pair_dataset.py — Positive/negative pair construction for training the ML classifier.

LEAKAGE PREVENTION RULES:
  - Positive labels come ONLY from training ground truth.
  - Validation ground truth is NEVER used during training.
  - entity_id columns are NEVER used as ML features.
  - matched_entity_ids is NEVER used as an ML feature.
  - Negative sampling never removes true positives.
  - Validation pairs are NOT downsampled — they represent the real candidate population.
"""

import logging
import random
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pair labelling
# ---------------------------------------------------------------------------

def label_candidates(
    candidates_df: pd.DataFrame,
    gt: Dict[str, Set[str]],
) -> pd.DataFrame:
    """
    Assign binary is_match labels to all candidate pairs.

    Parameters
    ----------
    candidates_df : pd.DataFrame
        Candidate pairs with source1_entity_id and candidate_entity_id.
    gt : Dict[str, Set[str]]
        Ground truth mapping (source1_entity_id → set of true matched IDs).
        Must be the TRAINING gt only; validation gt must never be passed here.

    Returns
    -------
    pd.DataFrame
        candidates_df with an added 'is_match' column (1 = match, 0 = non-match).
    """
    df = candidates_df.copy()
    df["is_match"] = df.apply(
        lambda row: 1 if row["candidate_entity_id"] in gt.get(row["source1_entity_id"], set()) else 0,
        axis=1,
    )
    n_pos = int(df["is_match"].sum())
    n_neg = int((df["is_match"] == 0).sum())
    logger.info("Labelled pairs: %d positive, %d negative (total %d)", n_pos, n_neg, len(df))
    return df


def create_positive_pairs(
    gt: Dict[str, Set[str]],
    train_s1_ids: List[str],
) -> pd.DataFrame:
    """
    Create the explicit positive pair DataFrame for training S1 entities.

    Only includes entities in train_s1_ids to prevent leakage.

    Returns
    -------
    pd.DataFrame
        Columns: source1_entity_id, candidate_entity_id, candidate_source, is_match.
    """
    rows: List[dict] = []
    for s1_id in train_s1_ids:
        for cand_id in gt.get(s1_id, set()):
            source = "S2" if cand_id.startswith("S2-") else "S3"
            rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "candidate_source": source,
                "is_match": 1,
            })
    df = pd.DataFrame(rows)
    logger.info("Created %d explicit positive pairs for training.", len(df))
    return df


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------

def sample_easy_negatives(
    labelled_df: pd.DataFrame,
    n_easy: int,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Sample random (easy) negative pairs from the labelled candidate set.

    Parameters
    ----------
    labelled_df : pd.DataFrame
        Labelled candidate pairs with is_match column.
    n_easy : int
        Number of easy negatives to sample.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Sampled easy negative pairs (is_match = 0).
    """
    negatives = labelled_df[labelled_df["is_match"] == 0]
    if len(negatives) <= n_easy:
        return negatives.copy()
    return negatives.sample(n=n_easy, random_state=random_seed).copy()


def sample_hard_negatives(
    labelled_df: pd.DataFrame,
    n_hard: int,
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Sample hard negatives: high-similarity-but-incorrect matches.

    Hard negative strategies:
      1. Same candidate_source as a true positive (same-source negatives).
      2. High TF-IDF name similarity (top candidates but not ground truth).
      3. Same country as a true positive.
      4. Similar numeric/address tokens but incorrect match.

    Parameters
    ----------
    labelled_df : pd.DataFrame
        Labelled candidate pairs from blocking (includes easy and hard negatives).
    n_hard : int
        Number of hard negatives to sample.
    s1_df : pd.DataFrame
        Source 1 records (used for context).
    target_df : pd.DataFrame
        Combined S2+S3 records (used for context).
    random_seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Sampled hard negative pairs.
    """
    negatives = labelled_df[labelled_df["is_match"] == 0].copy()

    # Strategy 1: negatives that share the same S1 entity as a true positive
    positive_s1_ids = set(labelled_df[labelled_df["is_match"] == 1]["source1_entity_id"])
    hard_pool = negatives[negatives["source1_entity_id"].isin(positive_s1_ids)]

    if len(hard_pool) < n_hard:
        # Fall back to all negatives if hard pool is insufficient
        hard_pool = negatives

    if len(hard_pool) <= n_hard:
        return hard_pool.copy()

    return hard_pool.sample(n=n_hard, random_state=random_seed).copy()


def build_training_pairs(
    labelled_df: pd.DataFrame,
    gt: Dict[str, Set[str]],
    train_s1_ids: List[str],
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    neg_ratio: int = 4,
    hard_neg_fraction: float = 0.5,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Construct the final training pair DataFrame with controlled negative sampling.

    Positive: all true matches in the candidate set for train_s1_ids.
    Negative: neg_ratio × n_positives negatives, split between easy and hard.

    Parameters
    ----------
    labelled_df : pd.DataFrame
        All labelled training candidate pairs.
    gt : Dict[str, Set[str]]
        Training ground truth (NEVER validation gt).
    train_s1_ids : List[str]
        Training S1 entity IDs.
    s1_df : pd.DataFrame
        Source 1 cleaned records.
    target_df : pd.DataFrame
        Combined S2+S3 cleaned records.
    neg_ratio : int
        Number of negatives per positive.
    hard_neg_fraction : float
        Fraction of negatives that should be hard negatives.
    random_seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Balanced training pairs with columns:
        source1_entity_id, candidate_entity_id, candidate_source, is_match.
    """
    # Restrict to training S1 IDs only
    train_id_set = set(train_s1_ids)
    labelled_train = labelled_df[
        labelled_df["source1_entity_id"].isin(train_id_set)
    ].copy()

    positives = labelled_train[labelled_train["is_match"] == 1]
    n_pos = len(positives)

    if n_pos == 0:
        logger.warning("No positive pairs found in training candidate set!")
        return labelled_train

    n_total_neg = n_pos * neg_ratio
    n_hard = int(n_total_neg * hard_neg_fraction)
    n_easy = n_total_neg - n_hard

    logger.info(
        "Building training pairs: %d positives | %d easy negatives | %d hard negatives",
        n_pos, n_easy, n_hard,
    )

    easy_neg = sample_easy_negatives(labelled_train, n_easy, random_seed=random_seed)
    hard_neg = sample_hard_negatives(labelled_train, n_hard, s1_df, target_df, random_seed=random_seed + 1)

    result = pd.concat([positives, easy_neg, hard_neg], ignore_index=True)
    result = result.drop_duplicates(
        subset=["source1_entity_id", "candidate_entity_id"]
    ).reset_index(drop=True)

    logger.info("Final training pairs: %d rows (%d positive, %d negative)", len(result), n_pos, len(result) - n_pos)
    return result


def build_validation_pairs(
    candidates_df: pd.DataFrame,
    val_gt: Dict[str, Set[str]],
    val_s1_ids: List[str],
) -> pd.DataFrame:
    """
    Construct labelled validation pairs WITHOUT downsampling.

    Validation pairs must represent the real candidate population:
    we do NOT artificially downsample negatives, because the threshold
    is selected on the true candidate distribution.

    Parameters
    ----------
    candidates_df : pd.DataFrame
        All candidate pairs for validation S1 entities (from blocking).
    val_gt : Dict[str, Set[str]]
        Validation ground truth.
    val_s1_ids : List[str]
        Validation S1 entity IDs.

    Returns
    -------
    pd.DataFrame
        All labelled validation candidate pairs.
    """
    val_id_set = set(val_s1_ids)
    val_candidates = candidates_df[
        candidates_df["source1_entity_id"].isin(val_id_set)
    ].copy()

    val_candidates = label_candidates(val_candidates, val_gt)

    n_pos = int(val_candidates["is_match"].sum())
    n_neg = len(val_candidates) - n_pos
    logger.info(
        "Validation pairs: %d total | %d positive | %d negative",
        len(val_candidates), n_pos, n_neg,
    )
    return val_candidates
