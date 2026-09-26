"""
blocking.py — Multi-pass candidate generation (blocking) for Entity Resolution.

Goal
----
Reduce the full Cartesian space (S1 × (S2 ∪ S3)) from O(N²) to a tractable
candidate set of ~20–100 plausible matches per S1 entity while maintaining
very high blocking recall (≥ 98% of ground-truth matches must appear in candidates).

Pipeline
--------
PASS 1: Exact match on business_name_norm
PASS 2: Exact match on business_name_core
PASS 3: Exact match on business_address_norm
PASS 4: TF-IDF character n-gram approximate nearest neighbors on business_name_norm
PASS 5: TF-IDF character n-gram approximate nearest neighbors on business_address_norm
PASS 6: Informative first-token blocking (optional)

All passes are unioned; duplicates removed; S1 IDs never appear as candidates.

LEAKAGE PREVENTION:
  Ground truth is NEVER used to generate candidates.
  Candidate recall is measured AFTER candidate generation.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: build candidate record
# ---------------------------------------------------------------------------

def _make_candidate_rows(
    s1_id: str,
    cand_ids: List[str],
    candidate_source: str,
) -> List[dict]:
    """Build candidate row dicts for a single S1 entity."""
    return [
        {
            "source1_entity_id": s1_id,
            "candidate_entity_id": cid,
            "candidate_source": candidate_source,
        }
        for cid in cand_ids
    ]


# ---------------------------------------------------------------------------
# PASS 1: Exact normalized name blocking
# ---------------------------------------------------------------------------

def generate_exact_name_candidates(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    candidate_source: str,
) -> pd.DataFrame:
    """
    Generate candidates by exact match on business_name_norm.

    Parameters
    ----------
    s1_df : pd.DataFrame
        Source 1 records (must have entity_id, business_name_norm).
    target_df : pd.DataFrame
        Source 2 or 3 records (must have entity_id, business_name_norm).
    candidate_source : str
        "S2" or "S3".

    Returns
    -------
    pd.DataFrame
        Candidate pairs with columns:
        source1_entity_id, candidate_entity_id, candidate_source.
    """
    logger.debug("[PASS 1] Exact name blocking against %s", candidate_source)

    # Build lookup: name_norm → list of target entity_ids
    name_to_targets: Dict[str, List[str]] = {}
    for _, row in target_df.iterrows():
        name = row["business_name_norm"]
        if name:
            name_to_targets.setdefault(name, []).append(row["entity_id"])

    rows: List[dict] = []
    for _, row in s1_df.iterrows():
        name = row["business_name_norm"]
        if name and name in name_to_targets:
            rows.extend(
                _make_candidate_rows(row["entity_id"], name_to_targets[name], candidate_source)
            )

    df = pd.DataFrame(rows) if rows else _empty_candidates()
    logger.debug("[PASS 1] Exact name — %d candidates from %s", len(df), candidate_source)
    return df


# ---------------------------------------------------------------------------
# PASS 2: Exact core name blocking
# ---------------------------------------------------------------------------

def generate_exact_core_name_candidates(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    candidate_source: str,
) -> pd.DataFrame:
    """
    Generate candidates by exact match on business_name_core (legal suffix stripped).
    """
    logger.debug("[PASS 2] Exact core-name blocking against %s", candidate_source)

    name_to_targets: Dict[str, List[str]] = {}
    for _, row in target_df.iterrows():
        core = row.get("business_name_core", "")
        if core and len(core) > 2:   # skip very short cores (noise)
            name_to_targets.setdefault(core, []).append(row["entity_id"])

    rows: List[dict] = []
    for _, row in s1_df.iterrows():
        core = row.get("business_name_core", "")
        if core and len(core) > 2 and core in name_to_targets:
            rows.extend(
                _make_candidate_rows(row["entity_id"], name_to_targets[core], candidate_source)
            )

    df = pd.DataFrame(rows) if rows else _empty_candidates()
    logger.debug("[PASS 2] Exact core-name — %d candidates from %s", len(df), candidate_source)
    return df


# ---------------------------------------------------------------------------
# PASS 3: Exact normalized address blocking
# ---------------------------------------------------------------------------

def generate_exact_address_candidates(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    candidate_source: str,
) -> pd.DataFrame:
    """
    Generate candidates by exact match on business_address_norm.
    Only considered when address is non-empty.
    """
    logger.debug("[PASS 3] Exact address blocking against %s", candidate_source)

    addr_to_targets: Dict[str, List[str]] = {}
    for _, row in target_df.iterrows():
        addr = row.get("business_address_norm", "")
        if addr and len(addr) > 5:
            addr_to_targets.setdefault(addr, []).append(row["entity_id"])

    rows: List[dict] = []
    for _, row in s1_df.iterrows():
        addr = row.get("business_address_norm", "")
        if addr and len(addr) > 5 and addr in addr_to_targets:
            rows.extend(
                _make_candidate_rows(row["entity_id"], addr_to_targets[addr], candidate_source)
            )

    df = pd.DataFrame(rows) if rows else _empty_candidates()
    logger.debug("[PASS 3] Exact address — %d candidates from %s", len(df), candidate_source)
    return df


# ---------------------------------------------------------------------------
# Shared TF-IDF helper
# ---------------------------------------------------------------------------

def _build_tfidf_vectors(
    query_texts: List[str],
    corpus_texts: List[str],
    analyzer: str = "char",
    ngram_range: Tuple[int, int] = (2, 4),
    min_df: int = 1,
    sublinear_tf: bool = True,
    max_features: Optional[int] = None,
) -> Tuple[csr_matrix, csr_matrix]:
    """
    Fit a TF-IDF vectorizer on corpus_texts and transform both sets.

    Returns
    -------
    Tuple[csr_matrix, csr_matrix]
        (query_matrix, corpus_matrix)
    """
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        max_features=max_features,
    )
    corpus_matrix = vectorizer.fit_transform(corpus_texts)
    query_matrix = vectorizer.transform(query_texts)
    return query_matrix, corpus_matrix


def _tfidf_top_k_candidates(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    text_column: str,
    candidate_source: str,
    top_k: int,
    min_similarity: float,
    analyzer: str,
    ngram_range: Tuple[int, int],
    min_df: int,
    sublinear_tf: bool,
    max_features: Optional[int],
    batch_size: int = 1000,
) -> pd.DataFrame:
    """
    Shared TF-IDF nearest-neighbor blocking for any text column.

    Processes queries in batches to avoid OOM on large datasets.
    """
    query_texts = s1_df[text_column].fillna("").tolist()
    corpus_texts = target_df[text_column].fillna("").tolist()
    target_ids = target_df["entity_id"].tolist()
    s1_ids = s1_df["entity_id"].tolist()

    # Replace empty strings with a single space so TF-IDF doesn't crash
    query_texts = [t if t.strip() else " " for t in query_texts]
    corpus_texts = [t if t.strip() else " " for t in corpus_texts]

    # Fit on corpus only (target S2/S3), transform queries
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        max_features=max_features,
    )
    corpus_matrix = vectorizer.fit_transform(corpus_texts)

    rows: List[dict] = []
    n_queries = len(query_texts)

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        batch_texts = query_texts[start:end]
        batch_s1_ids = s1_ids[start:end]

        q_matrix = vectorizer.transform(batch_texts)
        sims = cosine_similarity(q_matrix, corpus_matrix)  # shape: (batch, n_target)

        for i, s1_id in enumerate(batch_s1_ids):
            sim_row = sims[i]
            # Get top-K indices by cosine similarity
            if top_k < len(sim_row):
                top_indices = np.argpartition(sim_row, -top_k)[-top_k:]
            else:
                top_indices = np.arange(len(sim_row))

            # Filter by minimum similarity
            top_indices = [
                idx for idx in top_indices
                if sim_row[idx] >= min_similarity
            ]

            for idx in top_indices:
                rows.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": target_ids[idx],
                    "candidate_source": candidate_source,
                })

    df = pd.DataFrame(rows) if rows else _empty_candidates()
    return df


# ---------------------------------------------------------------------------
# PASS 4: TF-IDF name blocking
# ---------------------------------------------------------------------------

def generate_tfidf_name_candidates(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    candidate_source: str,
    top_k: int = 50,
    min_similarity: float = 0.0,
    analyzer: str = "char",
    ngram_range: Tuple[int, int] = (2, 4),
    min_df: int = 1,
    sublinear_tf: bool = True,
    max_features: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate candidates via TF-IDF character n-gram nearest neighbors on business_name_norm.
    """
    logger.debug("[PASS 4] TF-IDF name blocking against %s (top_k=%d)", candidate_source, top_k)
    df = _tfidf_top_k_candidates(
        s1_df=s1_df,
        target_df=target_df,
        text_column="business_name_norm",
        candidate_source=candidate_source,
        top_k=top_k,
        min_similarity=min_similarity,
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        max_features=max_features,
    )
    logger.debug("[PASS 4] TF-IDF name — %d candidates from %s", len(df), candidate_source)
    return df


# ---------------------------------------------------------------------------
# PASS 5: TF-IDF address blocking
# ---------------------------------------------------------------------------

def generate_tfidf_address_candidates(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    candidate_source: str,
    top_k: int = 50,
    min_similarity: float = 0.0,
    analyzer: str = "char",
    ngram_range: Tuple[int, int] = (2, 4),
    min_df: int = 1,
    sublinear_tf: bool = True,
    max_features: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate candidates via TF-IDF character n-gram nearest neighbors on business_address_norm.
    Only useful when address is non-empty.
    """
    logger.debug("[PASS 5] TF-IDF address blocking against %s (top_k=%d)", candidate_source, top_k)

    # Filter to records with non-empty addresses before vectorizing
    s1_with_addr = s1_df[s1_df["business_address_norm"].str.strip() != ""].copy()
    target_with_addr = target_df[target_df["business_address_norm"].str.strip() != ""].copy()

    if s1_with_addr.empty or target_with_addr.empty:
        logger.debug("[PASS 5] Skipped — no non-empty addresses.")
        return _empty_candidates()

    df = _tfidf_top_k_candidates(
        s1_df=s1_with_addr,
        target_df=target_with_addr,
        text_column="business_address_norm",
        candidate_source=candidate_source,
        top_k=top_k,
        min_similarity=min_similarity,
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        max_features=max_features,
    )
    logger.debug("[PASS 5] TF-IDF address — %d candidates from %s", len(df), candidate_source)
    return df


# ---------------------------------------------------------------------------
# PASS 6: Informative token blocking (optional)
# ---------------------------------------------------------------------------

def generate_token_candidates(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    candidate_source: str,
    min_token_length: int = 4,
) -> pd.DataFrame:
    """
    Generate candidates by shared informative first tokens of business_name_norm.

    Tokens shorter than min_token_length are excluded (stop-word-like).
    This pass catches cases where TF-IDF misses phonetically similar but
    orthographically different names.
    """
    logger.debug("[PASS 6] Token blocking against %s", candidate_source)

    token_to_targets: Dict[str, List[str]] = {}
    for _, row in target_df.iterrows():
        tokens = row.get("business_name_norm", "").split()
        for tok in tokens:
            if len(tok) >= min_token_length:
                token_to_targets.setdefault(tok, []).append(row["entity_id"])

    rows: List[dict] = []
    for _, row in s1_df.iterrows():
        tokens = row.get("business_name_norm", "").split()
        seen_cands: Set[str] = set()
        for tok in tokens:
            if len(tok) >= min_token_length and tok in token_to_targets:
                for cand_id in token_to_targets[tok]:
                    if cand_id not in seen_cands:
                        seen_cands.add(cand_id)
                        rows.append({
                            "source1_entity_id": row["entity_id"],
                            "candidate_entity_id": cand_id,
                            "candidate_source": candidate_source,
                        })

    df = pd.DataFrame(rows) if rows else _empty_candidates()
    logger.debug("[PASS 6] Token — %d candidates from %s", len(df), candidate_source)
    return df


# ---------------------------------------------------------------------------
# Union all passes
# ---------------------------------------------------------------------------

def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["source1_entity_id", "candidate_entity_id", "candidate_source"]
    )


def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    name_top_k: int = 50,
    address_top_k: int = 50,
    min_name_similarity: float = 0.0,
    min_address_similarity: float = 0.0,
    tfidf_analyzer: str = "char",
    tfidf_ngram_range: Tuple[int, int] = (2, 4),
    tfidf_min_df: int = 1,
    tfidf_sublinear_tf: bool = True,
    tfidf_max_features: Optional[int] = None,
    use_token_blocking: bool = True,
    s1_id_set: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Master blocking function: union of all passes for S1 → S2 and S1 → S3.

    Parameters
    ----------
    s1_df : pd.DataFrame
        Cleaned Source 1 records.
    s2_df : pd.DataFrame
        Cleaned Source 2 records.
    s3_df : pd.DataFrame
        Cleaned Source 3 records.
    name_top_k : int
        Top-K TF-IDF name candidates per S1 entity.
    address_top_k : int
        Top-K TF-IDF address candidates per S1 entity.
    min_name_similarity : float
        Minimum cosine similarity to retain a name candidate.
    min_address_similarity : float
        Minimum cosine similarity to retain an address candidate.
    tfidf_* : TF-IDF vectorizer settings (all configurable).
    use_token_blocking : bool
        Whether to run PASS 6 (token blocking).
    s1_id_set : Set[str], optional
        If provided, restrict to these S1 entity IDs (for train/val splits).

    Returns
    -------
    pd.DataFrame
        Deduplicated candidate pairs:
        source1_entity_id, candidate_entity_id, candidate_source.
    """
    if s1_id_set is not None:
        s1_df = s1_df[s1_df["entity_id"].isin(s1_id_set)].reset_index(drop=True)

    logger.info(
        "Generating candidates: %d S1 | %d S2 | %d S3",
        len(s1_df), len(s2_df), len(s3_df),
    )

    all_frames: List[pd.DataFrame] = []

    for target_df, source_label in [(s2_df, "S2"), (s3_df, "S3")]:
        # PASS 1
        all_frames.append(generate_exact_name_candidates(s1_df, target_df, source_label))
        # PASS 2
        all_frames.append(generate_exact_core_name_candidates(s1_df, target_df, source_label))
        # PASS 3
        all_frames.append(generate_exact_address_candidates(s1_df, target_df, source_label))
        # PASS 4
        all_frames.append(generate_tfidf_name_candidates(
            s1_df, target_df, source_label,
            top_k=name_top_k,
            min_similarity=min_name_similarity,
            analyzer=tfidf_analyzer,
            ngram_range=tfidf_ngram_range,
            min_df=tfidf_min_df,
            sublinear_tf=tfidf_sublinear_tf,
            max_features=tfidf_max_features,
        ))
        # PASS 5
        all_frames.append(generate_tfidf_address_candidates(
            s1_df, target_df, source_label,
            top_k=address_top_k,
            min_similarity=min_address_similarity,
            analyzer=tfidf_analyzer,
            ngram_range=tfidf_ngram_range,
            min_df=tfidf_min_df,
            sublinear_tf=tfidf_sublinear_tf,
            max_features=tfidf_max_features,
        ))
        # PASS 6 (optional)
        if use_token_blocking:
            all_frames.append(generate_token_candidates(s1_df, target_df, source_label))

    combined = pd.concat(all_frames, ignore_index=True)

    # Remove duplicates
    combined = combined.drop_duplicates(
        subset=["source1_entity_id", "candidate_entity_id"]
    ).reset_index(drop=True)

    # Safety: remove any S1 self-matches (should never happen, but guard)
    combined = combined[~combined["candidate_entity_id"].str.startswith("S1-")].reset_index(drop=True)

    logger.info(
        "Blocking complete: %d unique candidate pairs for %d S1 entities.",
        len(combined), combined["source1_entity_id"].nunique(),
    )
    return combined


# ---------------------------------------------------------------------------
# Candidate statistics
# ---------------------------------------------------------------------------

def candidate_recall(
    candidates_df: pd.DataFrame,
    gt: Dict[str, Set[str]],
    s1_ids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute the fraction of true ground-truth matches captured in the candidate set.

    candidate_recall = |true_pairs ∩ candidate_pairs| / |true_pairs|

    Parameters
    ----------
    candidates_df : pd.DataFrame
        Candidate pairs DataFrame with source1_entity_id and candidate_entity_id.
    gt : Dict[str, Set[str]]
        Ground truth mapping.
    s1_ids : List[str], optional
        If provided, restrict recall computation to these S1 IDs.

    Returns
    -------
    Dict[str, float]
        {"overall": float, "s2": float, "s3": float}
    """
    if s1_ids is not None:
        gt = {sid: gt.get(sid, set()) for sid in s1_ids}

    # Build set of candidate pairs
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
    counts = candidates_df.groupby("source1_entity_id")["candidate_entity_id"].count()
    return {
        "mean": float(counts.mean()),
        "median": float(counts.median()),
        "max": float(counts.max()),
        "min": float(counts.min()),
        "total_pairs": int(len(candidates_df)),
        "unique_s1": int(candidates_df["source1_entity_id"].nunique()),
    }


def average_candidates_per_entity(candidates_df: pd.DataFrame) -> float:
    """Average number of candidates per S1 entity."""
    return candidate_count_statistics(candidates_df)["mean"]


def median_candidates_per_entity(candidates_df: pd.DataFrame) -> float:
    """Median number of candidates per S1 entity."""
    return candidate_count_statistics(candidates_df)["median"]


def max_candidates_per_entity(candidates_df: pd.DataFrame) -> int:
    """Maximum number of candidates for any single S1 entity."""
    return int(candidate_count_statistics(candidates_df)["max"])


def reduction_ratio(
    candidates_df: pd.DataFrame,
    n_s1: int,
    n_s2: int,
    n_s3: int,
) -> float:
    """
    Reduction ratio: fraction of the full Cartesian space eliminated by blocking.

    reduction_ratio = 1 - (|candidates| / (n_s1 * (n_s2 + n_s3)))
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
