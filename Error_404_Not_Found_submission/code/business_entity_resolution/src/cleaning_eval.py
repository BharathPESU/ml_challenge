"""
cleaning_eval.py — Quality gates and evaluation for the cleaning stage.

Ensures that:
  - No rows or IDs are lost during cleaning.
  - Normalization doesn't result in excessive collision (many entities mapped to one generic name).
  - Empty-after-cleaning count is tracked.
  - Before/after samples are logged for manual review.
"""

import logging
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger(__name__)


def evaluate_cleaning_quality(df_raw: pd.DataFrame, df_clean: pd.DataFrame, source_label: str) -> Dict[str, Any]:
    """
    Evaluate the quality of the cleaning process with strict gates.
    Raises ValueError if critical quality gates fail.
    
    Parameters
    ----------
    df_raw : pd.DataFrame
        Original uncleaned dataframe.
    df_clean : pd.DataFrame
        Cleaned dataframe.
    source_label : str
        Label for logging.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing cleaning metrics.
    """
    logger.info("=== Cleaning Quality Evaluation for %s ===", source_label)
    
    # 1. Row & ID Preservation Gate
    if len(df_raw) != len(df_clean):
        raise ValueError(f"[{source_label}] Row count mismatch! Raw: {len(df_raw)}, Clean: {len(df_clean)}")
        
    raw_ids = set(df_raw['entity_id'].dropna())
    clean_ids = set(df_clean['entity_id'].dropna())
    if raw_ids != clean_ids:
        raise ValueError(f"[{source_label}] Entity IDs lost or altered during cleaning!")
        
    # 2. Empty-After-Cleaning Rate
    # A high rate of names becoming empty indicates over-aggressive cleaning
    raw_empty_names = df_raw['business_name'].isna().sum() + (df_raw['business_name'].str.strip() == '').sum()
    clean_empty_names = (df_clean['business_name_norm'].str.strip() == '').sum()
    empty_diff = clean_empty_names - raw_empty_names
    
    empty_diff_pct = (empty_diff / len(df_clean)) * 100 if len(df_clean) > 0 else 0
    logger.info("[%s] Names becoming empty after cleaning: %d (%.2f%%)", source_label, empty_diff, empty_diff_pct)
    
    if empty_diff_pct > 2.0:  # Arbitrary strict gate: max 2% names destroyed
        raise ValueError(f"[{source_label}] Too many names destroyed by cleaning: {empty_diff_pct:.2f}% > 2.0%")
        
    # 3. Collision Rate (Over-normalization check)
    # If many unique raw names collapse into the exact same normalized name, we might be destroying signal
    raw_unique_names = df_raw['business_name'].nunique()
    clean_unique_names = df_clean['business_name_norm'].nunique()
    
    collision_ratio = (raw_unique_names - clean_unique_names) / raw_unique_names if raw_unique_names > 0 else 0
    logger.info("[%s] Name collision (unique reduction): %.2f%% (from %d to %d)", 
                source_label, collision_ratio * 100, raw_unique_names, clean_unique_names)
    
    if collision_ratio > 0.50:  # Arbitrary strict gate: max 50% reduction in uniqueness
        logger.warning("[%s] HIGH NAME COLLISION DETECTED (%.2f%% reduction)", source_label, collision_ratio * 100)
    
    # Check top collisions
    top_collisions = df_clean['business_name_norm'].value_counts().head(5)
    logger.info("[%s] Top normalized names:\n%s", source_label, top_collisions.to_string())

    # 4. Display Before/After Samples
    sample_size = min(5, len(df_clean))
    if sample_size > 0:
        sample_df = df_clean.sample(sample_size, random_state=42)
        logger.info("[%s] Before/After Samples:", source_label)
        for _, row in sample_df.iterrows():
            logger.info("  RAW  : %s", row.get('business_name', ''))
            logger.info("  NORM : %s", row.get('business_name_norm', ''))
            logger.info("  CORE : %s", row.get('business_name_core', ''))
            logger.info("  RAW ADDR : %s", row.get('business_address', ''))
            logger.info("  NORM ADDR: %s", row.get('business_address_norm', ''))
            logger.info("  ---")

    logger.info("[%s] Cleaning Quality Gates PASSED.", source_label)
    
    return {
        "row_count": len(df_clean),
        "empty_diff_pct": empty_diff_pct,
        "collision_ratio": collision_ratio
    }
