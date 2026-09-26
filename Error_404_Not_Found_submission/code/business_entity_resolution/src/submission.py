"""
submission.py — Generate challenge-compliant output files.

Generates:
  1. output/matching_results.tsv  — final entity matches (scored on leaderboard)
  2. output/candidate_pairs.tsv   — final candidate set fed to the ML model

SUBMISSION RULES:
  - Every Source 1 test entity must appear exactly once in both files.
  - matched_entity_ids: empty string for singletons/no-match predictions.
  - Only Source 2 and Source 3 IDs may appear in matched_entity_ids.
  - No Source 1 self-matches.
  - No duplicate entity IDs within a single ID list.
  - Comma-separated matched IDs (no spaces), tab-separated columns.
  - candidate_entity_ids must be a superset of matched_entity_ids.
  - All IDs must exist in the test S2/S3 datasets.
"""

import logging
import os
import subprocess
import sys
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core output generation
# ---------------------------------------------------------------------------

def build_matching_results_df(
    all_test_s1_ids: List[str],
    predictions: Dict[str, Set[str]],
    valid_s2_ids: Set[str],
    valid_s3_ids: Set[str],
) -> pd.DataFrame:
    """
    Build the matching_results.tsv DataFrame.

    Parameters
    ----------
    all_test_s1_ids : List[str]
        Every Source 1 entity ID in the test set (order is stable).
    predictions : Dict[str, Set[str]]
        Model predictions: s1_id → set of predicted matched IDs.
    valid_s2_ids : Set[str]
        All entity_ids present in test Source 2.
    valid_s3_ids : Set[str]
        All entity_ids present in test Source 3.

    Returns
    -------
    pd.DataFrame
        Columns: source1_entity_id, matched_entity_ids.
    """
    valid_target_ids = valid_s2_ids | valid_s3_ids
    rows = []

    for s1_id in all_test_s1_ids:
        matched = predictions.get(s1_id, set())

        # Safety: remove any S1 self-matches or invalid IDs
        matched = {
            mid for mid in matched
            if mid in valid_target_ids and not mid.startswith("S1-")
        }

        # Remove duplicates and sort for deterministic ordering
        matched_sorted = sorted(matched)
        matched_str = ",".join(matched_sorted)

        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": matched_str,
        })

    df = pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])
    logger.info(
        "matching_results built: %d rows | %d with matches | %d singletons",
        len(df),
        int((df["matched_entity_ids"] != "").sum()),
        int((df["matched_entity_ids"] == "").sum()),
    )
    return df


def build_candidate_pairs_df(
    all_test_s1_ids: List[str],
    candidates_df: pd.DataFrame,
    valid_s2_ids: Set[str],
    valid_s3_ids: Set[str],
) -> pd.DataFrame:
    """
    Build the candidate_pairs.tsv DataFrame.

    This MUST be the exact LAST candidate set passed to the ML model.
    Not an earlier pass; not a larger intermediate set.

    Parameters
    ----------
    all_test_s1_ids : List[str]
        Every Source 1 entity ID in the test set.
    candidates_df : pd.DataFrame
        Final test candidates with source1_entity_id and candidate_entity_id.
    valid_s2_ids : Set[str]
        All entity_ids present in test Source 2.
    valid_s3_ids : Set[str]
        All entity_ids present in test Source 3.

    Returns
    -------
    pd.DataFrame
        Columns: source1_entity_id, candidate_entity_ids.
    """
    valid_target_ids = valid_s2_ids | valid_s3_ids

    # Group candidates by S1 entity
    grouped: Dict[str, Set[str]] = {}
    for _, row in candidates_df.iterrows():
        s1_id = row["source1_entity_id"]
        cand_id = row["candidate_entity_id"]
        if cand_id in valid_target_ids and not cand_id.startswith("S1-"):
            grouped.setdefault(s1_id, set()).add(cand_id)

    rows = []
    for s1_id in all_test_s1_ids:
        cands = grouped.get(s1_id, set())
        cands_sorted = sorted(cands)
        cands_str = ",".join(cands_sorted)
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": cands_str,
        })

    df = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])
    logger.info(
        "candidate_pairs built: %d rows | %d with candidates",
        len(df),
        int((df["candidate_entity_ids"] != "").sum()),
    )
    return df


# ---------------------------------------------------------------------------
# File writing
# ---------------------------------------------------------------------------

def write_matching_results(df: pd.DataFrame, output_path: str) -> None:
    """
    Write matching_results.tsv to disk.

    Format:
      source1_entity_id<TAB>matched_entity_ids
      S1-00001<TAB>S2-00047,S3-00812
      S1-00002<TAB>
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info("matching_results.tsv written: %s (%d rows)", output_path, len(df))


def write_candidate_pairs(df: pd.DataFrame, output_path: str) -> None:
    """
    Write candidate_pairs.tsv to disk.

    Format:
      source1_entity_id<TAB>candidate_entity_ids
      S1-00001<TAB>S2-00047,S3-00812,S3-00999
      S1-00002<TAB>S3-00004
      S1-00003<TAB>
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info("candidate_pairs.tsv written: %s (%d rows)", output_path, len(df))


# ---------------------------------------------------------------------------
# Self-consistency validation
# ---------------------------------------------------------------------------

def validate_matches_subset_of_candidates(
    matching_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
) -> bool:
    """
    Verify that every matched ID in matching_results also appears in candidate_pairs.

    The invariant: final_matches ⊆ final_candidates.
    A violation signals a pipeline bug.

    Returns
    -------
    bool
        True if all matched IDs are present in candidates; False otherwise.
    """
    # Build candidate lookup
    cand_lookup: Dict[str, Set[str]] = {}
    for _, row in candidates_df.iterrows():
        s1_id = row["source1_entity_id"]
        cids = row["candidate_entity_ids"]
        if cids:
            cand_lookup[s1_id] = set(cids.split(","))
        else:
            cand_lookup[s1_id] = set()

    violations = 0
    for _, row in matching_df.iterrows():
        s1_id = row["source1_entity_id"]
        mids = row["matched_entity_ids"]
        if not mids:
            continue
        for mid in mids.split(","):
            if mid and mid not in cand_lookup.get(s1_id, set()):
                logger.error(
                    "PIPELINE BUG: Matched ID '%s' for '%s' not in candidate_pairs!",
                    mid, s1_id,
                )
                violations += 1

    if violations == 0:
        logger.info("Subset check passed: all matched IDs are in candidate_pairs.")
        return True
    else:
        logger.error("Subset check FAILED: %d violations found.", violations)
        return False


# ---------------------------------------------------------------------------
# Official challenge validator runner
# ---------------------------------------------------------------------------

def run_official_validator(
    matching_path: str,
    candidate_path: str,
    test_dir: str,
    validator_script: Optional[str] = None,
) -> bool:
    """
    Run the official challenge validation script (utils/validate_submission.py).

    This does NOT create a fake validator — it invokes the official script
    provided by the challenge organizers.

    Parameters
    ----------
    matching_path : str
        Path to output/matching_results.tsv.
    candidate_path : str
        Path to output/candidate_pairs.tsv.
    test_dir : str
        Path to the dataset/test/ directory.
    validator_script : str, optional
        Path to utils/validate_submission.py.
        If None, searches relative to the project root.

    Returns
    -------
    bool
        True if validation passes (exit code 0), False otherwise.

    Usage from student_resource/:
        python3 utils/validate_submission.py \\
            --matching output/matching_results.tsv \\
            --candidate output/candidate_pairs.tsv \\
            --test-dir dataset/test
    """
    if validator_script is None:
        # Search common locations relative to this file
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, "../../../utils/validate_submission.py"),
            os.path.join(here, "../../../../utils/validate_submission.py"),
            "utils/validate_submission.py",
        ]
        validator_script = next(
            (p for p in candidates if os.path.exists(p)), None
        )

    if validator_script is None or not os.path.exists(validator_script):
        logger.warning(
            "Official validator script not found. Skipping official validation. "
            "Run manually: python3 utils/validate_submission.py "
            "--matching %s --candidate %s --test-dir %s",
            matching_path, candidate_path, test_dir,
        )
        return False

    cmd = [
        sys.executable,
        validator_script,
        "--matching", matching_path,
        "--candidate", candidate_path,
        "--test-dir", test_dir,
    ]
    logger.info("Running official validator: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    passed = result.returncode == 0
    if passed:
        logger.info("Official validator: PASS")
    else:
        logger.error("Official validator: FAIL (exit code %d)", result.returncode)

    return passed


# ---------------------------------------------------------------------------
# Convenience: full submission generation
# ---------------------------------------------------------------------------

def generate_submission_files(
    all_test_s1_ids: List[str],
    predictions: Dict[str, Set[str]],
    candidates_df: pd.DataFrame,
    test_s2_df: pd.DataFrame,
    test_s3_df: pd.DataFrame,
    output_dir: str,
    test_dir: str,
    run_validator: bool = True,
) -> Tuple[str, str, bool]:
    """
    Full submission file generation pipeline.

    Parameters
    ----------
    all_test_s1_ids : List[str]
        All test S1 entity IDs.
    predictions : Dict[str, Set[str]]
        Model predictions.
    candidates_df : pd.DataFrame
        Final test candidates.
    test_s2_df : pd.DataFrame
        Test Source 2 (for valid ID checking).
    test_s3_df : pd.DataFrame
        Test Source 3 (for valid ID checking).
    output_dir : str
        Directory for output files.
    test_dir : str
        Path to dataset/test/ for official validator.
    run_validator : bool
        Whether to run the official challenge validator.

    Returns
    -------
    Tuple[str, str, bool]
        (matching_path, candidates_path, validator_passed)
    """
    valid_s2_ids = set(test_s2_df["entity_id"])
    valid_s3_ids = set(test_s3_df["entity_id"])

    matching_df = build_matching_results_df(
        all_test_s1_ids, predictions, valid_s2_ids, valid_s3_ids
    )
    cand_df = build_candidate_pairs_df(
        all_test_s1_ids, candidates_df, valid_s2_ids, valid_s3_ids
    )

    # Self-consistency check
    validate_matches_subset_of_candidates(matching_df, cand_df)

    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidates_path = os.path.join(output_dir, "candidate_pairs.tsv")

    write_matching_results(matching_df, matching_path)
    write_candidate_pairs(cand_df, candidates_path)

    # Run official validator
    validator_passed = False
    if run_validator:
        validator_passed = run_official_validator(
            matching_path, candidates_path, test_dir
        )

    return matching_path, candidates_path, validator_passed
