"""
splitting.py — Entity-level train/validation splitting for Entity Resolution.

WHY SPLIT BY SOURCE 1 ENTITY IDs (NOT by rows)?
================================================
In entity resolution, the evaluation metric (macro F0.5) is computed
PER SOURCE 1 ENTITY. If we split by rows (candidate pairs), the same
S1 entity might appear in both training and validation sets — this would
cause data leakage: the model would have seen some of an entity's candidate
context during training, inflating validation scores.

Correct approach: pick a random 80% of Source 1 entity IDs for training
and the remaining 20% for validation. All candidate pairs for a given
S1 entity stay entirely within one split.

LEAKAGE PREVENTION:
  - Validation S1 IDs are hidden from training.
  - Ground truth labels for validation IDs are NOT used during training.
  - The split is deterministic (fixed random seed).
  - Test data is never mixed with either split.
"""

import logging
import random
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core split function
# ---------------------------------------------------------------------------

def split_source1_entities(
    s1_entity_ids: List[str],
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Randomly split Source 1 entity IDs into training and validation subsets.

    Parameters
    ----------
    s1_entity_ids : List[str]
        All Source 1 entity IDs to split (e.g., from train_source1.tsv).
    val_ratio : float
        Fraction of entities reserved for validation. Default = 0.2 (20%).
    random_seed : int
        Random seed for deterministic, reproducible splits.

    Returns
    -------
    Tuple[List[str], List[str]]
        (train_s1_ids, validation_s1_ids)
        The same entity ID will never appear in both lists.

    Example
    -------
        train_ids, val_ids = split_source1_entities(s1_ids, val_ratio=0.2)
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got: {val_ratio}")

    ids = list(s1_entity_ids)
    rng = random.Random(random_seed)
    rng.shuffle(ids)

    n_val = max(1, int(len(ids) * val_ratio))
    val_ids = ids[:n_val]
    train_ids = ids[n_val:]

    logger.info(
        "Split: %d train S1 entities | %d validation S1 entities (val_ratio=%.2f, seed=%d)",
        len(train_ids), len(val_ids), val_ratio, random_seed,
    )
    return train_ids, val_ids


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def filter_source_by_s1_ids(
    df: pd.DataFrame,
    s1_ids: List[str],
    id_column: str = "entity_id",
) -> pd.DataFrame:
    """
    Filter a Source 1 DataFrame to keep only rows with entity_id in s1_ids.

    Parameters
    ----------
    df : pd.DataFrame
        Source 1 DataFrame (must contain id_column).
    s1_ids : List[str]
        The subset of S1 entity IDs to keep.
    id_column : str
        Column name containing the entity ID. Default = "entity_id".

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame. Original index is reset.
    """
    s1_id_set = set(s1_ids)
    filtered = df[df[id_column].isin(s1_id_set)].reset_index(drop=True)
    logger.debug(
        "Filtered source from %d → %d rows by %d S1 IDs.",
        len(df), len(filtered), len(s1_id_set),
    )
    return filtered


def filter_ground_truth_by_s1_ids(
    gt: Dict[str, Set[str]],
    s1_ids: List[str],
) -> Dict[str, Set[str]]:
    """
    Filter a ground truth dictionary to keep only entries for the given S1 IDs.

    This ensures validation labels are completely separate from training labels.

    Parameters
    ----------
    gt : Dict[str, Set[str]]
        Full ground truth mapping.
    s1_ids : List[str]
        The subset of S1 entity IDs to keep.

    Returns
    -------
    Dict[str, Set[str]]
        Filtered ground truth. Includes singletons (empty sets) for IDs
        present in s1_ids but missing from gt.
    """
    s1_id_set = set(s1_ids)
    filtered_gt = {sid: gt.get(sid, set()) for sid in s1_id_set}

    n_with_matches = sum(1 for v in filtered_gt.values() if v)
    n_singletons = len(filtered_gt) - n_with_matches

    logger.debug(
        "Filtered GT: %d entities | %d with matches | %d singletons.",
        len(filtered_gt), n_with_matches, n_singletons,
    )
    return filtered_gt


# ---------------------------------------------------------------------------
# Candidate-level split helpers
# ---------------------------------------------------------------------------

def filter_candidates_by_s1_ids(
    candidates_df: pd.DataFrame,
    s1_ids: List[str],
    s1_col: str = "source1_entity_id",
) -> pd.DataFrame:
    """
    Filter a candidate pairs DataFrame to rows whose source1_entity_id is in s1_ids.

    Used after blocking to create training/validation candidate subsets.

    Parameters
    ----------
    candidates_df : pd.DataFrame
        Candidate pairs DataFrame with a source1_entity_id column.
    s1_ids : List[str]
        The subset of S1 entity IDs to keep.
    s1_col : str
        Column name for Source 1 entity IDs.

    Returns
    -------
    pd.DataFrame
        Filtered candidates. Original index is reset.
    """
    s1_id_set = set(s1_ids)
    filtered = candidates_df[candidates_df[s1_col].isin(s1_id_set)].reset_index(drop=True)
    logger.debug(
        "Filtered candidates: %d → %d pairs by %d S1 IDs.",
        len(candidates_df), len(filtered), len(s1_id_set),
    )
    return filtered


# ---------------------------------------------------------------------------
# Convenience: full split pack
# ---------------------------------------------------------------------------

def prepare_train_val_split(
    df_s1: pd.DataFrame,
    gt: Dict[str, Set[str]],
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> Tuple[
    List[str],                  # train_s1_ids
    List[str],                  # val_s1_ids
    pd.DataFrame,               # train_s1_df
    pd.DataFrame,               # val_s1_df
    Dict[str, Set[str]],        # train_gt
    Dict[str, Set[str]],        # val_gt
]:
    """
    Complete train/validation split returning both ID lists and filtered DataFrames.

    Parameters
    ----------
    df_s1 : pd.DataFrame
        Full training Source 1 DataFrame.
    gt : Dict[str, Set[str]]
        Full ground truth mapping from parse_ground_truth().
    val_ratio : float
        Fraction for validation.
    random_seed : int
        Random seed.

    Returns
    -------
    Tuple of:
        train_s1_ids   : List[str]
        val_s1_ids     : List[str]
        train_s1_df    : pd.DataFrame
        val_s1_df      : pd.DataFrame
        train_gt       : Dict[str, Set[str]]
        val_gt         : Dict[str, Set[str]]
    """
    all_s1_ids = list(df_s1["entity_id"])
    train_s1_ids, val_s1_ids = split_source1_entities(
        all_s1_ids, val_ratio=val_ratio, random_seed=random_seed
    )

    train_s1_df = filter_source_by_s1_ids(df_s1, train_s1_ids)
    val_s1_df = filter_source_by_s1_ids(df_s1, val_s1_ids)

    train_gt = filter_ground_truth_by_s1_ids(gt, train_s1_ids)
    val_gt = filter_ground_truth_by_s1_ids(gt, val_s1_ids)

    return train_s1_ids, val_s1_ids, train_s1_df, val_s1_df, train_gt, val_gt
