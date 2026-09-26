"""
features.py — Pairwise feature engineering for Business Entity Resolution.

For every (S1, candidate) pair, this module generates a fixed-length numerical
feature vector used to train and score the XGBoost matching classifier.

FEATURE GROUPS:
  1. Name features (exact, TF-IDF, edit distance, token set, Jaccard, length)
  2. Address features (same set of computations as name features)
  3. Numeric address overlap features (street numbers, PINs, ZIP codes)
  4. Country match feature (open-set string comparison — no one-hot encoding)
  5. Structural features (source indicator, first character match)
  6. Interaction features (name × address cross terms)

LEAKAGE PREVENTION:
  - entity_id values are stored as METADATA columns — never used as ML features.
  - matched_entity_ids is never used.
  - Feature column ordering is deterministic (alphabetically sorted by group).
  - Identical feature computation is applied at training AND inference time.

DEPENDENCIES:
  - rapidfuzz: all string similarity calculations
  - sklearn: TF-IDF pairwise similarity
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cleaning import tokenize_text, extract_numeric_tokens, extract_digit_sequences

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Metadata columns — stored alongside features but NEVER fed to the ML model
METADATA_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "is_match",          # present in training/validation, absent in test
]

# Deterministic feature column order (grouped, then sorted within group)
FEATURE_COLUMNS: Optional[List[str]] = None  # set at first build


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def _safe_str(val) -> str:
    """Convert any value to a clean string (empty string on None/NaN)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return str(val).strip()


def _token_set(text: str) -> Set[str]:
    tokens = tokenize_text(text)
    return set(tokens)


def _jaccard(s1: Set[str], s2: Set[str]) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union > 0 else 0.0


def _containment(s1: Set[str], s2: Set[str]) -> float:
    """Fraction of s1 tokens contained in s2."""
    if not s1:
        return 1.0
    return len(s1 & s2) / len(s1)


def _token_overlap_ratio(s1: Set[str], s2: Set[str]) -> float:
    """Overlap coefficient: |intersection| / min(|s1|, |s2|)."""
    if not s1 or not s2:
        return 0.0
    inter = len(s1 & s2)
    return inter / min(len(s1), len(s2))


def _common_token_count(s1: Set[str], s2: Set[str]) -> int:
    return len(s1 & s2)


def _length_diff_norm(t1: str, t2: str) -> float:
    """Normalized absolute length difference."""
    l1, l2 = len(t1), len(t2)
    mx = max(l1, l2, 1)
    return abs(l1 - l2) / mx


def _length_ratio(t1: str, t2: str) -> float:
    """Length ratio min/max."""
    l1, l2 = len(t1), len(t2)
    mx = max(l1, l2, 1)
    mn = min(l1, l2)
    return mn / mx


# ---------------------------------------------------------------------------
# String similarity helpers (RapidFuzz)
# ---------------------------------------------------------------------------

def _levenshtein_sim(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity in [0, 1]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return fuzz.ratio(s1, s2) / 100.0


def _jaro_winkler(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return fuzz.WRatio(s1, s2) / 100.0   # use WRatio as a proxy for JW


def _partial_ratio(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    return fuzz.partial_ratio(s1, s2) / 100.0


def _token_sort_ratio(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    return fuzz.token_sort_ratio(s1, s2) / 100.0


def _token_set_ratio(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    return fuzz.token_set_ratio(s1, s2) / 100.0


# ---------------------------------------------------------------------------
# Per-pair feature computation
# ---------------------------------------------------------------------------

def _compute_name_features(n1: str, n2: str, n1_core: str, n2_core: str) -> dict:
    """Compute all name-related features for a pair."""
    t1 = _token_set(n1)
    t2 = _token_set(n2)

    return {
        "name_exact":               int(n1 == n2 and bool(n1)),
        "name_core_exact":          int(n1_core == n2_core and bool(n1_core)),
        "name_levenshtein_sim":     _levenshtein_sim(n1, n2),
        "name_jaro_winkler":        _jaro_winkler(n1, n2),
        "name_partial_ratio":       _partial_ratio(n1, n2),
        "name_token_sort_ratio":    _token_sort_ratio(n1, n2),
        "name_token_set_ratio":     _token_set_ratio(n1, n2),
        "name_jaccard":             _jaccard(t1, t2),
        "name_containment":         _containment(t1, t2),
        "name_token_overlap_ratio": _token_overlap_ratio(t1, t2),
        "name_common_token_count":  _common_token_count(t1, t2),
        "name_length_diff_norm":    _length_diff_norm(n1, n2),
        "name_length_ratio":        _length_ratio(n1, n2),
        "name_missing":             int(not n1 or not n2),
    }


def _compute_address_features(a1: str, a2: str) -> dict:
    """Compute all address-related features for a pair."""
    t1 = _token_set(a1)
    t2 = _token_set(a2)

    return {
        "address_levenshtein_sim":     _levenshtein_sim(a1, a2),
        "address_jaro_winkler":        _jaro_winkler(a1, a2),
        "address_partial_ratio":       _partial_ratio(a1, a2),
        "address_token_sort_ratio":    _token_sort_ratio(a1, a2),
        "address_token_set_ratio":     _token_set_ratio(a1, a2),
        "address_jaccard":             _jaccard(t1, t2),
        "address_containment":         _containment(t1, t2),
        "address_token_overlap_ratio": _token_overlap_ratio(t1, t2),
        "address_common_token_count":  _common_token_count(t1, t2),
        "address_length_diff_norm":    _length_diff_norm(a1, a2),
        "address_length_ratio":        _length_ratio(a1, a2),
        "address_missing":             int(not a1 or not a2),
    }


def _compute_numeric_features(a1: str, a2: str) -> dict:
    """Compute numeric overlap features for address strings."""
    nums1 = set(extract_numeric_tokens(a1))
    nums2 = set(extract_numeric_tokens(a2))
    digits1 = set(extract_digit_sequences(a1))
    digits2 = set(extract_digit_sequences(a2))

    # Numeric token Jaccard
    num_jaccard = _jaccard(nums1, nums2)
    # Numeric token overlap
    num_overlap = _token_overlap_ratio(nums1, nums2)
    # Long digit sequence overlap (PINs/ZIPs)
    digit_jaccard = _jaccard(digits1, digits2)
    # Exact match of at least one long digit sequence
    has_exact_digit_match = int(bool(digits1 & digits2)) if digits1 and digits2 else 0
    # Numeric count difference
    num_count_diff = abs(len(nums1) - len(nums2))
    # Conflicting numbers: both have numbers but none overlap
    has_numeric_conflict = int(
        bool(nums1) and bool(nums2) and len(nums1 & nums2) == 0
    )

    return {
        "numeric_token_jaccard":      num_jaccard,
        "numeric_token_overlap":      num_overlap,
        "digit_seq_jaccard":          digit_jaccard,
        "has_exact_digit_match":      has_exact_digit_match,
        "numeric_count_diff":         float(num_count_diff),
        "has_numeric_conflict":       has_numeric_conflict,
    }


def _compute_country_features(c1: str, c2: str) -> dict:
    """
    Country match feature.

    IMPORTANT: country is treated as an open-set string.
    We never one-hot encode or hard-code US/India.
    The test set contains France, which is unseen during training.
    """
    return {
        "country_match":   int(c1 == c2 and bool(c1)),
        "country_missing": int(not c1 or not c2),
    }


def _compute_structural_features(
    s1_id: str,
    cand_id: str,
    candidate_source: str,
    n1: str,
    n2: str,
) -> dict:
    """Compute structural / indicator features."""
    return {
        "source_is_s2":        int(candidate_source == "S2"),
        "source_is_s3":        int(candidate_source == "S3"),
        "same_first_char":     int(bool(n1) and bool(n2) and n1[0] == n2[0]),
    }


def _compute_interaction_features(name_feats: dict, addr_feats: dict) -> dict:
    """Compute cross-field interaction features."""
    name_sim = name_feats.get("name_token_set_ratio", 0.0)
    addr_sim = addr_feats.get("address_token_set_ratio", 0.0)
    return {
        "name_x_address_sim":    name_sim * addr_sim,
        "min_name_address_sim":  min(name_sim, addr_sim),
        "max_name_address_sim":  max(name_sim, addr_sim),
    }


# ---------------------------------------------------------------------------
# TF-IDF pairwise feature helpers
# ---------------------------------------------------------------------------

class TFIDFSimilarityComputer:
    """
    Pre-fitted TF-IDF vectorizer for computing pairwise cosine similarities.

    Fit once on the corpus (S2 ∪ S3 or training set), then score pairs.
    """

    def __init__(
        self,
        analyzer: str = "char",
        ngram_range: Tuple[int, int] = (2, 4),
        min_df: int = 1,
        sublinear_tf: bool = True,
        max_features: Optional[int] = None,
    ):
        self._char_vec = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
            max_features=max_features,
        )
        self._word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=min_df,
            sublinear_tf=sublinear_tf,
        )
        self._fitted = False

    def fit(self, corpus: List[str]) -> None:
        clean_corpus = [t if t.strip() else " " for t in corpus]
        self._char_vec.fit(clean_corpus)
        self._word_vec.fit(clean_corpus)
        self._fitted = True

    def cosine_similarity_char(self, texts_a: List[str], texts_b: List[str]) -> np.ndarray:
        """Compute diagonal cosine similarities between paired texts (char n-gram)."""
        assert self._fitted, "Call fit() first."
        a = self._char_vec.transform([t or " " for t in texts_a])
        b = self._char_vec.transform([t or " " for t in texts_b])
        return np.array([cosine_similarity(a[i], b[i])[0, 0] for i in range(a.shape[0])])

    def cosine_similarity_word(self, texts_a: List[str], texts_b: List[str]) -> np.ndarray:
        """Compute diagonal cosine similarities between paired texts (word n-gram)."""
        assert self._fitted, "Call fit() first."
        a = self._word_vec.transform([t or " " for t in texts_a])
        b = self._word_vec.transform([t or " " for t in texts_b])
        return np.array([cosine_similarity(a[i], b[i])[0, 0] for i in range(a.shape[0])])


# ---------------------------------------------------------------------------
# Main feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    pairs_df: pd.DataFrame,
    entity_lookup: Dict[str, dict],
    name_tfidf: Optional[TFIDFSimilarityComputer] = None,
    address_tfidf: Optional[TFIDFSimilarityComputer] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract all pairwise features for a set of candidate pairs.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        Candidate pairs with at minimum:
        source1_entity_id, candidate_entity_id, candidate_source.
        May also contain is_match for training/validation.
    entity_lookup : Dict[str, dict]
        Maps entity_id → dict of cleaned field values.
        Built from s1_df, s2_df, s3_df combined.
    name_tfidf : TFIDFSimilarityComputer, optional
        Pre-fitted TF-IDF computer for name similarity.
    address_tfidf : TFIDFSimilarityComputer, optional
        Pre-fitted TF-IDF computer for address similarity.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (features_df, metadata_df)
        features_df : numerical feature columns only (no IDs)
        metadata_df : source1_entity_id, candidate_entity_id, candidate_source, [is_match]
    """
    logger.info("Extracting features for %d candidate pairs...", len(pairs_df))

    feature_rows: List[dict] = []
    meta_rows: List[dict] = []

    # Pre-extract TF-IDF scores if computers are provided
    use_name_tfidf = name_tfidf is not None and name_tfidf._fitted
    use_addr_tfidf = address_tfidf is not None and address_tfidf._fitted

    if use_name_tfidf:
        s1_names = [
            _safe_str(entity_lookup.get(r["source1_entity_id"], {}).get("business_name_norm", ""))
            for _, r in pairs_df.iterrows()
        ]
        cand_names = [
            _safe_str(entity_lookup.get(r["candidate_entity_id"], {}).get("business_name_norm", ""))
            for _, r in pairs_df.iterrows()
        ]
        name_char_sims = name_tfidf.cosine_similarity_char(s1_names, cand_names)
        name_word_sims = name_tfidf.cosine_similarity_word(s1_names, cand_names)
    else:
        name_char_sims = np.zeros(len(pairs_df))
        name_word_sims = np.zeros(len(pairs_df))

    if use_addr_tfidf:
        s1_addrs = [
            _safe_str(entity_lookup.get(r["source1_entity_id"], {}).get("business_address_norm", ""))
            for _, r in pairs_df.iterrows()
        ]
        cand_addrs = [
            _safe_str(entity_lookup.get(r["candidate_entity_id"], {}).get("business_address_norm", ""))
            for _, r in pairs_df.iterrows()
        ]
        addr_char_sims = address_tfidf.cosine_similarity_char(s1_addrs, cand_addrs)
        addr_word_sims = address_tfidf.cosine_similarity_word(s1_addrs, cand_addrs)
    else:
        addr_char_sims = np.zeros(len(pairs_df))
        addr_word_sims = np.zeros(len(pairs_df))

    for i, (_, row) in enumerate(pairs_df.iterrows()):
        s1_id = row["source1_entity_id"]
        cand_id = row["candidate_entity_id"]
        cand_src = row.get("candidate_source", "")

        s1_rec = entity_lookup.get(s1_id, {})
        cand_rec = entity_lookup.get(cand_id, {})

        # Field extraction
        n1      = _safe_str(s1_rec.get("business_name_norm", ""))
        n2      = _safe_str(cand_rec.get("business_name_norm", ""))
        n1_core = _safe_str(s1_rec.get("business_name_core", ""))
        n2_core = _safe_str(cand_rec.get("business_name_core", ""))
        a1      = _safe_str(s1_rec.get("business_address_norm", ""))
        a2      = _safe_str(cand_rec.get("business_address_norm", ""))
        c1      = _safe_str(s1_rec.get("country_norm", ""))
        c2      = _safe_str(cand_rec.get("country_norm", ""))

        # Compute feature groups
        name_feats = _compute_name_features(n1, n2, n1_core, n2_core)
        addr_feats = _compute_address_features(a1, a2)
        num_feats  = _compute_numeric_features(a1, a2)
        cntry_feats = _compute_country_features(c1, c2)
        struct_feats = _compute_structural_features(s1_id, cand_id, cand_src, n1, n2)
        inter_feats = _compute_interaction_features(name_feats, addr_feats)

        # TF-IDF features
        tfidf_feats = {
            "name_char_tfidf":    float(name_char_sims[i]),
            "name_word_tfidf":    float(name_word_sims[i]),
            "address_char_tfidf": float(addr_char_sims[i]),
            "address_word_tfidf": float(addr_word_sims[i]),
        }

        all_feats = {
            **name_feats,
            **addr_feats,
            **num_feats,
            **cntry_feats,
            **struct_feats,
            **inter_feats,
            **tfidf_feats,
        }
        feature_rows.append(all_feats)

        meta = {
            "source1_entity_id": s1_id,
            "candidate_entity_id": cand_id,
            "candidate_source": cand_src,
        }
        if "is_match" in row.index:
            meta["is_match"] = int(row["is_match"])
        meta_rows.append(meta)

    features_df = pd.DataFrame(feature_rows)
    metadata_df = pd.DataFrame(meta_rows)

    # Ensure deterministic feature column order (alphabetical)
    features_df = features_df[sorted(features_df.columns)].reset_index(drop=True)
    metadata_df = metadata_df.reset_index(drop=True)

    logger.info("Feature extraction complete: %d rows × %d features", len(features_df), len(features_df.columns))
    return features_df, metadata_df


def build_entity_lookup(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
) -> Dict[str, dict]:
    """
    Build a combined entity_id → field dict lookup for fast per-pair access.

    Parameters
    ----------
    s1_df, s2_df, s3_df : pd.DataFrame
        Cleaned source DataFrames.

    Returns
    -------
    Dict[str, dict]
        Maps entity_id → dict of all cleaned fields.
    """
    combined = pd.concat([s1_df, s2_df, s3_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["entity_id"])
    lookup = combined.set_index("entity_id").to_dict("index")
    logger.info("Entity lookup built: %d records.", len(lookup))
    return lookup


def get_feature_column_names(features_df: pd.DataFrame) -> List[str]:
    """Return the sorted feature column list (deterministic ordering)."""
    return sorted(features_df.columns.tolist())
