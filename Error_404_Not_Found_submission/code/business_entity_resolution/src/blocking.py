"""
blocking.py — Multi-pass scalable candidate generation (blocking) for Entity Resolution.

Replaces the single-pass TF-IDF matrix with an 8-Pass scalable framework.
Passes:
1. Exact name_norm
2. Exact name_core
3. Exact address_norm
4. Numeric-address blocking (exact numeric extraction overlap)
5. Rare name-token blocking
6. Rare address-token blocking
7. Fuzzy Name blocking (via RapidFuzz bounded retrieval)
8. Fuzzy Address blocking (via RapidFuzz bounded retrieval)

Chunked execution per country / batch is used to prevent RAM overflow.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
from rapidfuzz import process, fuzz

logger = logging.getLogger(__name__)

def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "candidate_source"])

def _make_candidate_rows(s1_id: str, cand_ids: List[str], candidate_source: str) -> List[dict]:
    return [
        {
            "source1_entity_id": s1_id,
            "candidate_entity_id": cid,
            "candidate_source": candidate_source,
        }
        for cid in cand_ids
    ]

# ---------------------------------------------------------------------------
# Passes 1-3: Exact Matches
# ---------------------------------------------------------------------------

def generate_exact_match_candidates(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    col_name: str,
    candidate_source: str,
    min_len: int = 1
) -> pd.DataFrame:
    """Generic exact match generator for a given column."""
    logger.debug(f"[Exact Blocking] {col_name} against {candidate_source}")
    val_to_targets: Dict[str, List[str]] = {}
    
    for _, row in target_df.iterrows():
        val = row.get(col_name, "")
        if pd.notna(val) and len(str(val)) >= min_len:
            val_to_targets.setdefault(str(val), []).append(row["entity_id"])

    rows: List[dict] = []
    for _, row in s1_df.iterrows():
        val = row.get(col_name, "")
        if pd.notna(val) and len(str(val)) >= min_len and str(val) in val_to_targets:
            rows.extend(_make_candidate_rows(row["entity_id"], val_to_targets[str(val)], candidate_source))

    df = pd.DataFrame(rows) if rows else _empty_candidates()
    logger.debug(f"[Exact Blocking] {col_name} — {len(df)} candidates from {candidate_source}")
    return df

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
# Pass 5 & 6: Rare Token Blocking
# ---------------------------------------------------------------------------

def generate_rare_token_candidates(
    s1_df: pd.DataFrame, target_df: pd.DataFrame, col_name: str, candidate_source: str, rare_threshold: int = 1000
) -> pd.DataFrame:
    """Block using tokens that appear fewer than `rare_threshold` times in the target corpus."""
    logger.debug(f"[Rare Token] {col_name} against {candidate_source}")
    
    from collections import Counter
    token_freq = Counter()
    
    target_tokens_map = {}
    for _, row in target_df.iterrows():
        val = str(row.get(col_name, ""))
        tokens = set(val.split()) if val else set()
        target_tokens_map[row["entity_id"]] = tokens
        for t in tokens:
            if len(t) >= 4:
                token_freq[t] += 1
                
    rare_tokens = {t for t, freq in token_freq.items() if freq <= rare_threshold}
    
    token_to_targets = {}
    for eid, tokens in target_tokens_map.items():
        for t in tokens:
            if t in rare_tokens:
                token_to_targets.setdefault(t, []).append(eid)
                
    rows = []
    for _, row in s1_df.iterrows():
        val = str(row.get(col_name, ""))
        tokens = set(val.split()) if val else set()
        seen = set()
        for t in tokens:
            if t in rare_tokens and t in token_to_targets:
                for cand_id in token_to_targets[t]:
                    if cand_id not in seen:
                        seen.add(cand_id)
                        rows.append({
                            "source1_entity_id": row["entity_id"],
                            "candidate_entity_id": cand_id,
                            "candidate_source": candidate_source
                        })
                        
    df = pd.DataFrame(rows) if rows else _empty_candidates()
    logger.debug(f"[Rare Token] {col_name} — {len(df)} candidates from {candidate_source}")
    return df

# ---------------------------------------------------------------------------
# Pass 7 & 8: Bounded RapidFuzz Search
# ---------------------------------------------------------------------------

def generate_fuzzy_candidates(
    s1_df: pd.DataFrame, target_df: pd.DataFrame, col_name: str, candidate_source: str, limit: int = 10
) -> pd.DataFrame:
    """Fuzzy retrieval via RapidFuzz. We restrict to the same country to bound the search space and prevent OOM."""
    logger.debug(f"[Fuzzy] {col_name} against {candidate_source}")
    
    rows = []
    countries = s1_df['country_norm'].unique()
    
    for c in countries:
        s1_c = s1_df[s1_df['country_norm'] == c]
        tgt_c = target_df[target_df['country_norm'] == c]
        if tgt_c.empty or s1_c.empty:
            continue
            
        choices = tgt_c[col_name].fillna("").to_dict() # index to string
        tgt_ids = tgt_c["entity_id"].to_dict()
        
        # choices_list allows passing dict to RapidFuzz process.extract
        # However, it's faster to pass a sequence of strings and map back.
        choices_vals = list(choices.values())
        choices_keys = list(choices.keys())
        
        for _, row in s1_c.iterrows():
            query = row.get(col_name, "")
            if not query or not isinstance(query, str) or len(query) < 4:
                continue
                
            results = process.extract(query, choices_vals, scorer=fuzz.token_sort_ratio, limit=limit, score_cutoff=70.0)
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
    config,
    s1_id_set: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Master multi-pass blocking function."""
    
    if s1_id_set is not None:
        s1_df = s1_df[s1_df["entity_id"].isin(s1_id_set)].reset_index(drop=True)

    logger.info("Generating candidates: %d S1 | %d S2 | %d S3", len(s1_df), len(s2_df), len(s3_df))
    all_frames = []

    for target_df, source_label in [(s2_df, "S2"), (s3_df, "S3")]:
        bcfg = config.blocking
        
        if bcfg.pass1_exact_norm:
            all_frames.append(generate_exact_match_candidates(s1_df, target_df, "business_name_norm", source_label, 2))
        
        if bcfg.pass2_exact_core:
            all_frames.append(generate_exact_match_candidates(s1_df, target_df, "business_name_core", source_label, 3))
            
        if bcfg.pass3_address_token:
            all_frames.append(generate_exact_match_candidates(s1_df, target_df, "business_address_norm", source_label, 5))
            
        if bcfg.pass4_numeric_address:
            all_frames.append(generate_numeric_address_candidates(s1_df, target_df, source_label))
            
        if bcfg.pass5_rare_name_token:
            all_frames.append(generate_rare_token_candidates(s1_df, target_df, "business_name_norm", source_label, bcfg.rare_token_threshold))
            
        if bcfg.pass6_rare_address_token:
            all_frames.append(generate_rare_token_candidates(s1_df, target_df, "business_address_norm", source_label, bcfg.rare_token_threshold))
            
        if bcfg.pass7_fuzz_name:
            all_frames.append(generate_fuzzy_candidates(s1_df, target_df, "business_name_norm", source_label, limit=10))
            
        if bcfg.pass8_fuzz_address:
            all_frames.append(generate_fuzzy_candidates(s1_df, target_df, "business_address_norm", source_label, limit=10))

    if not all_frames:
        return _empty_candidates()
        
    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"]).reset_index(drop=True)
    combined = combined[~combined["candidate_entity_id"].str.startswith("S1-")].reset_index(drop=True)

    logger.info("Blocking complete: %d unique candidate pairs for %d S1 entities.", len(combined), combined["source1_entity_id"].nunique())
    return combined
