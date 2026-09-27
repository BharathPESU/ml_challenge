"""
blocking.py — Multi-pass scalable candidate generation (blocking) for Entity Resolution.

Vectorized and optimized for high-performance low-memory execution on large datasets.
Passes:
1. Exact name_norm
2. Exact name_core
3. Exact address_norm
4. Numeric-address blocking (exact numeric extraction overlap)
5. Rare name-token blocking (vectorized explode)
6. Rare address-token blocking (vectorized explode)
7. Fuzzy Name blocking (via RapidFuzz bounded retrieval)
8. Fuzzy Address blocking (via RapidFuzz bounded retrieval)
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz

logger = logging.getLogger(__name__)

def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "candidate_source"])

# ---------------------------------------------------------------------------
# Passes 1-3: Exact Matches (Fast Vectorized pd.merge)
# ---------------------------------------------------------------------------

def generate_exact_match_candidates(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    col_name: str,
    candidate_source: str,
    min_len: int = 1
) -> pd.DataFrame:
    """Generic exact match generator using fast vectorized pd.merge."""
    logger.debug(f"[Exact Blocking] {col_name} against {candidate_source}")
    if col_name not in s1_df.columns or col_name not in target_df.columns:
        return _empty_candidates()

    s1_sub = s1_df[["entity_id", col_name]].dropna()
    s1_sub = s1_sub[s1_sub[col_name].astype(str).str.len() >= min_len]

    tgt_sub = target_df[["entity_id", col_name]].dropna()
    tgt_sub = tgt_sub[tgt_sub[col_name].astype(str).str.len() >= min_len]

    if s1_sub.empty or tgt_sub.empty:
        return _empty_candidates()

    merged = pd.merge(s1_sub, tgt_sub, on=col_name, suffixes=("_s1", "_cand"))
    if merged.empty:
        return _empty_candidates()

    res = pd.DataFrame({
        "source1_entity_id": merged["entity_id_s1"],
        "candidate_entity_id": merged["entity_id_cand"],
        "candidate_source": candidate_source,
    }).drop_duplicates()

    logger.debug(f"[Exact Blocking] {col_name} — {len(res)} candidates from {candidate_source}")
    return res

# ---------------------------------------------------------------------------
# Pass 4: Numeric Address Blocking
# ---------------------------------------------------------------------------

def generate_numeric_address_candidates(
    s1_df: pd.DataFrame, target_df: pd.DataFrame, candidate_source: str
) -> pd.DataFrame:
    """Block on identical extracted number strings (length >= 3) to capture matching postcodes/buildings."""
    import re
    logger.debug(f"[Numeric Address] against {candidate_source}")
    
    def extract_nums(addr: str) -> str:
        if not addr or not isinstance(addr, str): return ""
        nums = [n for n in re.findall(r"\d+", addr) if len(n) >= 3]
        return "-".join(sorted(nums))
        
    target_df = target_df.copy()
    target_df["_num_sig"] = target_df["business_address_norm"].apply(extract_nums)
    s1_df = s1_df.copy()
    s1_df["_num_sig"] = s1_df["business_address_norm"].apply(extract_nums)
    
    df = generate_exact_match_candidates(s1_df, target_df, "_num_sig", candidate_source, min_len=3)
    logger.debug(f"[Numeric Address] — {len(df)} candidates from {candidate_source}")
    return df

# ---------------------------------------------------------------------------
# Pass 5 & 6: Rare Token Blocking (Fast Vectorized Explode)
# ---------------------------------------------------------------------------

def generate_rare_token_candidates(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    col_name: str,
    candidate_source: str,
    rare_threshold: int = 1000
) -> pd.DataFrame:
    """Block using tokens that appear fewer than `rare_threshold` times in the target corpus (Vectorized)."""
    logger.debug(f"[Rare Token] {col_name} against {candidate_source}")
    if col_name not in s1_df.columns or col_name not in target_df.columns:
        return _empty_candidates()

    s1_series = s1_df[["entity_id", col_name]].dropna()
    tgt_series = target_df[["entity_id", col_name]].dropna()

    if s1_series.empty or tgt_series.empty:
        return _empty_candidates()

    # Explode tokens in target
    tgt_toks = (
        tgt_series.assign(token=tgt_series[col_name].astype(str).str.split())
        .explode("token")
        .dropna()
    )
    tgt_toks = tgt_toks[tgt_toks["token"].str.len() >= 4]

    if tgt_toks.empty:
        return _empty_candidates()

    # Calculate token frequencies
    counts = tgt_toks["token"].value_counts()
    # Filter rare tokens (appear <= rare_threshold and >= 2 times)
    rare_set = set(counts[(counts <= rare_threshold) & (counts >= 2)].index)

    if not rare_set:
        return _empty_candidates()

    tgt_toks = tgt_toks[tgt_toks["token"].isin(rare_set)]

    # Explode S1 tokens
    s1_toks = (
        s1_series.assign(token=s1_series[col_name].astype(str).str.split())
        .explode("token")
        .dropna()
    )
    s1_toks = s1_toks[s1_toks["token"].isin(rare_set)]

    if s1_toks.empty:
        return _empty_candidates()

    # Fast merged join on token
    merged = pd.merge(
        s1_toks[["entity_id", "token"]],
        tgt_toks[["entity_id", "token"]],
        on="token",
        suffixes=("_s1", "_cand")
    )

    if merged.empty:
        return _empty_candidates()

    res = pd.DataFrame({
        "source1_entity_id": merged["entity_id_s1"],
        "candidate_entity_id": merged["entity_id_cand"],
        "candidate_source": candidate_source,
    }).drop_duplicates()

    logger.debug(f"[Rare Token] {col_name} — {len(res)} candidates from {candidate_source}")
    return res

# ---------------------------------------------------------------------------
# Pass 7 & 8: Bounded RapidFuzz Search (Batch Safe)
# ---------------------------------------------------------------------------

def generate_fuzzy_candidates(
    s1_df: pd.DataFrame, target_df: pd.DataFrame, col_name: str, candidate_source: str, limit: int = 10
) -> pd.DataFrame:
    """Fuzzy retrieval via RapidFuzz. We restrict to the same country to bound the search space and prevent OOM."""
    logger.debug(f"[Fuzzy] {col_name} against {candidate_source}")
    
    rows = []
    if 'country_norm' not in s1_df.columns or 'country_norm' not in target_df.columns:
        return _empty_candidates()

    countries = s1_df['country_norm'].dropna().unique()
    
    for c in countries:
        s1_c = s1_df[s1_df['country_norm'] == c]
        tgt_c = target_df[target_df['country_norm'] == c]
        if tgt_c.empty or s1_c.empty:
            continue
            
        choices = tgt_c[col_name].fillna("").to_dict()
        tgt_ids = tgt_c["entity_id"].to_dict()
        
        choices_vals = list(choices.values())
        choices_keys = list(choices.keys())
        
        # Limit per country to prevent infinite loops / OOM on massive countries
        for _, row in s1_c.head(50000).iterrows():
            query = row.get(col_name, "")
            if not query or not isinstance(query, str) or len(query) < 4:
                continue
                
            results = process.extract(query, choices_vals, scorer=fuzz.token_sort_ratio, limit=limit, score_cutoff=80.0)
            seen = set()
            for (match_str, score, match_idx) in results:
                real_idx = choices_keys[match_idx]
                cand_id = tgt_ids[real_idx]
                if cand_id not in seen:
                    seen.add(cand_id)
                    rows.append({
                        "source1_entity_id": row["entity_id"],
                        "candidate_entity_id": cand_id,
                        "candidate_source": candidate_source
                    })
                    
    df = pd.DataFrame(rows) if rows else _empty_candidates()
    logger.debug(f"[Fuzzy] {col_name} — {len(df)} candidates from {candidate_source}")
    return df


# ---------------------------------------------------------------------------
# Master Entrypoint
# ---------------------------------------------------------------------------

def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    config=None,
    s1_id_set: Optional[Set[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Master multi-pass blocking function."""
    from typing import Any
    
    if s1_id_set is not None:
        s1_df = s1_df[s1_df["entity_id"].isin(s1_id_set)].reset_index(drop=True)

    logger.info("Generating candidates: %d S1 | %d S2 | %d S3", len(s1_df), len(s2_df), len(s3_df))
    all_frames = []

    bcfg = None
    if config is not None:
        if hasattr(config, "blocking"):
            bcfg = config.blocking
        elif isinstance(config, dict):
            bcfg = config
    
    if bcfg is None and "blocking_config" in kwargs:
        bcfg = kwargs["blocking_config"]

    def _get_val(key: str, default: Any) -> Any:
        if bcfg is not None:
            if hasattr(bcfg, key):
                return getattr(bcfg, key)
            elif isinstance(bcfg, dict) and key in bcfg:
                return bcfg[key]
        if key in kwargs:
            return kwargs[key]
        return default

    pass1 = _get_val("pass1_exact_norm", True)
    pass2 = _get_val("pass2_exact_core", True)
    pass3 = _get_val("pass3_address_token", True)
    pass4 = _get_val("pass4_numeric_address", True)
    pass5 = _get_val("pass5_rare_name_token", True)
    pass6 = _get_val("pass6_rare_address_token", True)
    pass7 = _get_val("pass7_fuzz_name", True)
    pass8 = _get_val("pass8_fuzz_address", True)
    rare_threshold = _get_val("rare_token_threshold", 1000)

    for target_df, source_label in [(s2_df, "S2"), (s3_df, "S3")]:
        if pass1:
            all_frames.append(generate_exact_match_candidates(s1_df, target_df, "business_name_norm", source_label, 2))
        if pass2:
            all_frames.append(generate_exact_match_candidates(s1_df, target_df, "business_name_core", source_label, 3))
        if pass3:
            all_frames.append(generate_exact_match_candidates(s1_df, target_df, "business_address_norm", source_label, 5))
        if pass4:
            all_frames.append(generate_numeric_address_candidates(s1_df, target_df, source_label))
        if pass5:
            all_frames.append(generate_rare_token_candidates(s1_df, target_df, "business_name_norm", source_label, rare_threshold))
        if pass6:
            all_frames.append(generate_rare_token_candidates(s1_df, target_df, "business_address_norm", source_label, rare_threshold))
        if pass7:
            all_frames.append(generate_fuzzy_candidates(s1_df, target_df, "business_name_norm", source_label, limit=10))
        if pass8:
            all_frames.append(generate_fuzzy_candidates(s1_df, target_df, "business_address_norm", source_label, limit=10))

    if not all_frames:
        return _empty_candidates()

    res = pd.concat(all_frames, ignore_index=True)
    res = res.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id", "candidate_source"])
    logger.info("Total unique candidates generated: %d", len(res))
    return res
