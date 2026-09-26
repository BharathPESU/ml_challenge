"""
ground_truth.py — Ground-truth parsing and validation utilities.

The challenge ground truth file (train_ground_truth.tsv) contains:
  - source1_entity_id : ID of a Source 1 entity
  - matched_entity_ids: comma-separated list of S2/S3 IDs, or empty for singletons

LEAKAGE PREVENTION:
  Ground truth is used ONLY for:
    1. Labelling training candidate pairs (is_match = 1)
    2. Labelling validation candidate pairs for metric computation
    3. Computing candidate recall (to measure blocking quality)
  
  Ground truth is NEVER used to:
    - Generate candidate pairs
    - Create model features
    - Influence blocking decisions
    - Touch test data in any way
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_ground_truth(df_gt: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Parse the ground truth DataFrame into a lookup dictionary.

    Parameters
    ----------
    df_gt : pd.DataFrame
        DataFrame with columns: source1_entity_id, matched_entity_ids.

    Returns
    -------
    Dict[str, Set[str]]
        Mapping: source1_entity_id → set of matched entity IDs.
        For singletons, the set is empty.

    Example:
        {
            "S1-00001": {"S2-00047", "S3-00812"},
            "S1-00002": {"S3-00004"},
            "S1-00003": set(),   # singleton
        }
    """
    gt: Dict[str, Set[str]] = {}

    for _, row in df_gt.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        raw_matched = str(row["matched_entity_ids"]).strip()

        if raw_matched:
            matched_ids = {m.strip() for m in raw_matched.split(",") if m.strip()}
        else:
            matched_ids = set()

        gt[s1_id] = matched_ids

    n_singletons = sum(1 for v in gt.values() if len(v) == 0)
    n_with_matches = len(gt) - n_singletons
    total_matched = sum(len(v) for v in gt.values())

    logger.info(
        "Ground truth parsed: %d S1 entities | %d with matches | %d singletons | %d total matched IDs",
        len(gt), n_with_matches, n_singletons, total_matched,
    )
    return gt


def expand_ground_truth_pairs(gt: Dict[str, Set[str]]) -> pd.DataFrame:
    """
    Expand the ground truth dictionary into a flat DataFrame of positive pairs.

    Each row represents one true (S1, candidate) match.
    Singleton S1 entities (with empty match sets) are NOT included.

    Parameters
    ----------
    gt : Dict[str, Set[str]]
        Ground truth mapping from parse_ground_truth().

    Returns
    -------
    pd.DataFrame
        Columns: source1_entity_id, matched_entity_id, is_match (always 1).

    Example output:
        source1_entity_id | matched_entity_id | is_match
        S1-00001          | S2-00047          | 1
        S1-00001          | S3-00812          | 1
        S1-00002          | S3-00004          | 1
    """
    rows: List[dict] = []
    for s1_id, matched_ids in gt.items():
        for cand_id in matched_ids:
            rows.append({
                "source1_entity_id": s1_id,
                "matched_entity_id": cand_id,
                "is_match": 1,
            })

    df = pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_id", "is_match"])
    logger.info("Expanded %d positive ground truth pairs.", len(df))
    return df


def build_ground_truth_sets(
    gt: Dict[str, Set[str]],
    s1_ids: Optional[List[str]] = None,
) -> Dict[str, Set[str]]:
    """
    Build a filtered ground truth set for a subset of S1 entity IDs.

    Parameters
    ----------
    gt : Dict[str, Set[str]]
        Full ground truth mapping.
    s1_ids : list of str, optional
        If provided, restrict to these S1 entity IDs.

    Returns
    -------
    Dict[str, Set[str]]
        Filtered ground truth mapping. Includes singletons (empty sets).
    """
    if s1_ids is None:
        return dict(gt)

    filtered = {sid: gt.get(sid, set()) for sid in s1_ids}
    logger.debug(
        "Filtered ground truth to %d S1 entities (from full set of %d).",
        len(filtered), len(gt),
    )
    return filtered


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_ground_truth_ids(
    gt: Dict[str, Set[str]],
    valid_s2_ids: Set[str],
    valid_s3_ids: Set[str],
    s1_ids: Optional[Set[str]] = None,
) -> Tuple[int, List[str]]:
    """
    Validate that all matched entity IDs in ground truth are valid S2 or S3 IDs.

    Parameters
    ----------
    gt : Dict[str, Set[str]]
        Ground truth mapping to validate.
    valid_s2_ids : Set[str]
        All entity_ids present in Source 2.
    valid_s3_ids : Set[str]
        All entity_ids present in Source 3.
    s1_ids : Set[str], optional
        If provided, also validate that source1_entity_ids belong to Source 1.

    Returns
    -------
    Tuple[int, List[str]]
        (n_invalid, list_of_invalid_ids)
        n_invalid = 0 means fully valid ground truth.
    """
    valid_target_ids = valid_s2_ids | valid_s3_ids
    invalid_ids: List[str] = []

    for s1_id, matched_ids in gt.items():
        # Validate that S1 IDs don't appear in matched_entity_ids (no self-match)
        for cand_id in matched_ids:
            if cand_id.startswith("S1-"):
                logger.warning(
                    "Ground truth leakage: S1 ID '%s' appears as a match for '%s'.",
                    cand_id, s1_id,
                )
                invalid_ids.append(cand_id)
            elif cand_id not in valid_target_ids:
                logger.warning(
                    "Ground truth ID '%s' not found in S2 or S3 datasets.", cand_id
                )
                invalid_ids.append(cand_id)

        if s1_ids is not None and s1_id not in s1_ids:
            logger.warning("Ground truth S1 ID '%s' not found in Source 1 dataset.", s1_id)

    n_invalid = len(invalid_ids)
    if n_invalid == 0:
        logger.info("Ground truth validation passed — all matched IDs are valid S2/S3 IDs.")
    else:
        logger.warning("Ground truth validation found %d invalid matched IDs.", n_invalid)

    return n_invalid, invalid_ids


def get_gt_matched_sources(gt: Dict[str, Set[str]]) -> Dict[str, Dict[str, Set[str]]]:
    """
    Split ground truth matches by source (S2 vs S3) for per-source analysis.

    Returns
    -------
    Dict with keys "s2" and "s3", each mapping s1_id → set of matched IDs.
    """
    s2_gt: Dict[str, Set[str]] = {}
    s3_gt: Dict[str, Set[str]] = {}

    for s1_id, matched_ids in gt.items():
        s2_matches = {mid for mid in matched_ids if mid.startswith("S2-")}
        s3_matches = {mid for mid in matched_ids if mid.startswith("S3-")}
        if s2_matches:
            s2_gt[s1_id] = s2_matches
        if s3_matches:
            s3_gt[s1_id] = s3_matches

    logger.debug(
        "GT by source: %d S1 entities have S2 matches, %d have S3 matches.",
        len(s2_gt), len(s3_gt),
    )
    return {"s2": s2_gt, "s3": s3_gt}
