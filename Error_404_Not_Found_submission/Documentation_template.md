# Documentation Template — Business Entity Resolution
## Amazon ML Challenge 2026

**Team Name:** Error_404_Not_Found

> ⚠️ NOTE: Metric values, thresholds, and blocking recall numbers in this template
> are placeholders. Actual values will be filled in after running the pipeline.

---

## 1. Methodology Overview

This solution frames business entity resolution as a **pairwise binary classification** problem.

**Pipeline stages:**
1. Text normalization and cleaning (both train and test)
2. Multi-pass candidate blocking (exact + TF-IDF nearest neighbor)
3. Pairwise feature extraction (~40 numerical features)
4. XGBoost binary classifier training (local or SageMaker-managed)
5. Precision-optimized F0.5 threshold selection on held-out validation
6. Test inference and submission generation

**Key design decisions:**
- Blocking recall is maximized before optimizing classification precision.
- F0.5 (β=0.5) is directly optimized via threshold sweep — never assumed at 0.5.
- Country is treated as an open-set string feature — no hard-coding of US/India.
- Identical preprocessing is applied to training and test data.
- Entity IDs are never used as ML features (leakage prevention).

---

## 2. Candidate Generation / Blocking Strategy

### Objective
Reduce the full Cartesian pair space (Source 1 × (Source 2 ∪ Source 3)) from O(N²) 
to a tractable set of ~20–100 candidates per Source 1 entity while maintaining 
blocking recall ≥ 98%.

### Blocking Passes
| Pass | Method | Field |
|------|--------|-------|
| 1 | Exact string match | `business_name_norm` |
| 2 | Exact string match | `business_name_core` (legal suffixes stripped) |
| 3 | Exact string match | `business_address_norm` |
| 4 | TF-IDF char n-gram ANN (top-K) | `business_name_norm` |
| 5 | TF-IDF char n-gram ANN (top-K) | `business_address_norm` |
| 6 | Shared informative token blocking | `business_name_norm` |

All 6 passes are independently run for S1→S2 and S1→S3, then unioned and deduplicated.

### Hyperparameter Experiments
- `name_top_k`: tested at {20, 50, 100, 200}
- `address_top_k`: tested at {20, 50, 100, 200}
- TF-IDF `ngram_range`: (2,4) character n-grams
- Final K selected based on validation blocking recall vs. reduction ratio trade-off.

### Results
*(Fill in after running experiments)*
- Blocking recall (overall): **[TBD]**
- Blocking recall (S2): **[TBD]**
- Blocking recall (S3): **[TBD]**
- Reduction ratio: **[TBD]**
- Average candidates per S1 entity: **[TBD]**

---

## 3. Model Architecture

**Algorithm:** XGBoost Binary Classifier (`binary:logistic`)

**Input:** Fixed-length numerical feature vector per (S1, candidate) pair  
**Output:** Match probability ∈ [0, 1]  
**Training signal:** Binary label (1 = ground-truth match, 0 = non-match)

**Baseline hyperparameters:**
```
objective        = binary:logistic
eval_metric      = aucpr
max_depth        = 6
eta              = 0.05
min_child_weight = 3
subsample        = 0.8
colsample_bytree = 0.8
gamma            = 0.0
reg_alpha        = 0.1
reg_lambda       = 2.0
num_round        = 300 (with early stopping, patience=30)
```

**Final hyperparameters (after staged experiments):**
*(Fill in after running experiments)*
- max_depth: **[TBD]**
- eta: **[TBD]**
- num_round: **[TBD]**

---

## 4. Feature Engineering

All features are computed from cleaned/normalized field values.
Raw entity IDs and ground-truth columns are **never** used as features.

### Name Features (14 features)
- Exact match (normalized and core)
- Levenshtein similarity, Jaro-Winkler, partial ratio
- Token sort ratio, token set ratio (RapidFuzz)
- Token Jaccard, containment, overlap ratio
- Common token count, length difference, length ratio
- TF-IDF char and word cosine similarity

### Address Features (12 features)
- Same computations as name features applied to `business_address_norm`
- Missing address indicator

### Numeric Address Features (6 features)
- Numeric token Jaccard and overlap ratio
- Digit sequence Jaccard (long sequences ≥ 3 digits for PINs/ZIPs)
- Exact digit sequence match indicator
- Numeric count difference
- Numeric conflict indicator (both have numbers, none overlap)

### Country Feature (2 features)
- `country_match`: exact string equality of normalized country
- `country_missing`: either record has missing country

### Structural Features (3 features)
- `source_is_s2`, `source_is_s3`: candidate source indicator
- `same_first_char`: first character of normalized names matches

### Interaction Features (3 features)
- `name_x_address_sim`: product of name and address token_set_ratio
- `min_name_address_sim`, `max_name_address_sim`

**Total features: ~40**

---

## 5. Validation Methodology

**Split strategy:** Random 80/20 split of **Source 1 entity IDs**.

The same Source 1 entity never appears in both training and validation.
Splitting by entity ID (not by candidate pair rows) prevents leakage:
validation F0.5 would otherwise be inflated if some of an entity's
candidate pairs appeared in training.

**Negative sampling:**
- Training: 1 positive : 4 negatives (50% hard, 50% easy)
- Validation: NO downsampling — full candidate population is preserved

**Validation metric:** Entity-level macro F0.5 (official competition formula)

---

## 6. Threshold Selection Methodology

1. Score all validation candidate pairs with the trained XGBoost model.
2. Sweep thresholds t ∈ [0.10, 0.99] in steps of 0.01.
3. For each threshold:
   - Mark pairs with probability ≥ t as predicted matches
   - Aggregate by Source 1 entity → per-entity prediction sets
   - Compute entity-level macro F0.5, precision, recall
   - Count singleton correctness
4. Select threshold that maximizes validation macro F0.5.
5. Save threshold to JSON artifact for test inference.

**Default threshold of 0.5 is NOT used.**  
F0.5 is precision-heavy (false merges penalized 2× over missed matches),
so optimal threshold is typically higher than 0.5.

**Selected threshold:** *(Fill in after experiments)*  
**Validation macro F0.5:** *(Fill in after experiments)*

---

## 7. Error Analysis

After threshold selection, `src/threshold.py::analyze_errors()` identifies:

**High-confidence false positives:** Pairs the model is very confident are matches
but are actually different businesses. These often indicate:
- Name collisions (common business names like "City Traders")
- Addresses with shared landmarks but different locations

**False negatives:** True matches the model missed. These often indicate:
- Heavy transliteration variations
- Completely different address formats for the same location
- Legal name vs. DBA/trade name

**Singleton false positives:** Source 1 entities with no true matches that
received at least one predicted match. Singleton correctness is especially
important because it is included in the macro average.

---

## 8. Other Relevant Implementation Details

### Country Handling
The training set contains only US and India. The test set additionally contains France.
Country is treated as an open-set string feature — it is lowercased and normalized
but never one-hot encoded or filtered. The `country_match` feature is simply 1 if
both records have the same normalized country string, 0 otherwise. This generalizes
cleanly to France without any code changes.

### Unicode / Accent Normalization
French business names contain accented characters (é, è, ç, œ, etc.).
`cleaning.py::strip_accents()` uses NFKD normalization to convert these to
ASCII equivalents before all string comparisons and TF-IDF vectorization.

### Missing Address Handling
Approximately 3.3% of Source 2 and Source 3 records have empty addresses.
Records with empty addresses are NOT dropped — they remain in the candidate pool.
The `address_missing` feature flags these pairs, allowing the model to learn
that address similarity is unreliable for these records.

### SageMaker Compatibility
All file paths are configurable via `src/config.py` and environment variables.
No personal paths or AWS credentials are hard-coded.
Data can be read directly from S3 URIs via `io_utils.read_tsv("s3://...")`.

### Reproducibility
- Random seed: 42 (configurable)
- Feature columns: sorted alphabetically — identical ordering across train/inference
- Candidate ordering: sorted by entity_id
- XGBoost: seed parameter set to random seed
