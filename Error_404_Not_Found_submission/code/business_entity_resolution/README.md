# Business Entity Resolution
## Amazon ML Challenge 2026 — Team: Error_404_Not_Found

---

## 1. Challenge Objective

Given business records from 3 independent, noisy data sources (Source 1, Source 2, Source 3),
determine which records across sources refer to the same real-world business entity.

- Source 1 is the deduplicated reference source.
- For every Source 1 entity, find all matching records in Source 2 and Source 3.
- Evaluation metric: **Entity-level macro F0.5** (precision-weighted, 2× penalty on false merges).

---

## 2. Exact Project Structure

```
Error_404_Not_Found_submission/
│
├── output/
│   ├── matching_results.tsv        # final match predictions (leaderboard submission)
│   └── candidate_pairs.tsv         # blocking candidate set (pipeline audit)
│
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py
│       │   ├── config.py           # centralized configuration
│       │   ├── io_utils.py         # local + S3 I/O, column validation
│       │   ├── cleaning.py         # text normalization, legal suffix stripping
│       │   ├── ground_truth.py     # GT parsing, expansion, validation
│       │   ├── splitting.py        # entity-level train/val split
│       │   ├── blocking.py         # multi-pass candidate generation
│       │   ├── pair_dataset.py     # positive/negative pair construction
│       │   ├── features.py         # pairwise feature extraction
│       │   ├── training.py         # XGBoost local + SageMaker training
│       │   ├── evaluation.py       # macro F0.5 implementation
│       │   ├── threshold.py        # threshold sweep + error analysis
│       │   ├── inference.py        # test-time inference pipeline
│       │   ├── submission.py       # output file generation + validator
│       │   └── pipeline.py         # 15-stage master orchestration
│       ├── README.md
│       └── requirements.txt
│
└── Documentation_template.md
```

---

## 3. AWS / SageMaker Execution Architecture

```
Local Machine
    ↓ aws s3 sync
S3 Bucket: s3://<bucket>/student_resource/
    ├── dataset/train/
    ├── dataset/test/
    └── code/business_entity_resolution/
    
SageMaker Studio / Notebook Instance
    ↓ aws s3 sync (download)
    ~/student_resource/
    
    1. pip install -r requirements.txt
    2. python3 src/pipeline.py --data-dir ../../dataset --output-dir ../../output
    
    ↓ aws s3 sync (upload results)
S3 Bucket: s3://<bucket>/output/
    ├── matching_results.tsv
    └── candidate_pairs.tsv
```

---

## 4. S3 Storage Architecture

| S3 Path | Content |
|---------|---------|
| `s3://<bucket>/student_resource/dataset/train/` | Training TSV files |
| `s3://<bucket>/student_resource/dataset/test/` | Test TSV files |
| `s3://<bucket>/student_resource/code/` | Source code |
| `s3://<bucket>/artifacts/<experiment_id>/model.json` | Trained XGBoost model |
| `s3://<bucket>/artifacts/<experiment_id>/best_threshold.json` | Selected threshold |
| `s3://<bucket>/output/` | Final submission files |

No model, data, or artifact folders are committed to the repository.
All runtime artifacts are stored in S3 or temporary SageMaker storage.

---

## 5. Required Dependencies

```
pip install -r requirements.txt
```

Key packages:
- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `scipy` — sparse matrix operations for TF-IDF
- `scikit-learn` — TF-IDF vectorization
- `rapidfuzz` — fast string similarity (Levenshtein, Jaro-Winkler, token ratios)
- `xgboost` — gradient boosted tree binary classifier
- `boto3` — AWS S3 access
- `sagemaker` — SageMaker SDK for managed training jobs

---

## 6. Configuration

All settings are in `src/config.py` via the `Config` dataclass.
Override any setting using environment variables:

```bash
export S3_BUCKET=your-bucket-name
export AWS_REGION=us-east-1
export NAME_TOP_K=100
export VAL_SPLIT_RATIO=0.2
export RANDOM_SEED=42
export LOG_LEVEL=INFO
```

---

## 7. Data Loading

```python
from io_utils import read_tsv
df = read_tsv("dataset/train/train_source1.tsv")           # local
df = read_tsv("s3://my-bucket/dataset/train_source1.tsv")  # S3
```

All TSV files are read with `sep="\t"`, `dtype=str`, `keep_default_na=False`.
Empty strings are never silently converted to NaN.

---

## 8. Data Cleaning

`src/cleaning.py` applies these transformations to **all** source files (train AND test):

- Unicode normalization (NFKD) → ASCII — critical for French test entities
- Lowercase + punctuation normalization (`&` → `and`, `/` → space)
- Legal suffix removal for core name: `ltd`, `pvt`, `corp`, `llc`, `s.a.`, etc.
- Street abbreviation expansion: `St` → `street`, `Rd` → `road`
- Numeric token preservation (house numbers, PIN/ZIP codes)
- Country treated as **open-set string** — never one-hot encoded or hard-coded to US/India

New columns added: `business_name_norm`, `business_name_core`, `business_address_norm`,
`country_norm`, `name_missing`, `address_missing`, `name_char_length`, etc.
Raw original columns are preserved.

---

## 9. Ground-Truth Parsing

```python
from ground_truth import parse_ground_truth
gt = parse_ground_truth(gt_df)
# → Dict[str, Set[str]]: s1_id → set of matched IDs (empty set = singleton)
```

Ground truth is used **only** to:
- Label training/validation candidate pairs
- Compute validation metrics and blocking recall

Ground truth **never** influences candidate generation, feature values, or test data.

---

## 10. Validation Split

Split is by **Source 1 entity IDs** (not by candidate pair rows).

```python
from splitting import prepare_train_val_split
train_ids, val_ids, train_s1_df, val_s1_df, train_gt, val_gt = prepare_train_val_split(
    df_s1, gt, val_ratio=0.2, random_seed=42
)
```

The same S1 entity never appears in both training and validation.
This prevents leakage and ensures fair F0.5 estimation.

---

## 11. Blocking (Candidate Generation)

`src/blocking.py` implements 6 blocking passes:

| Pass | Method |
|------|--------|
| 1 | Exact match on `business_name_norm` |
| 2 | Exact match on `business_name_core` (legal suffixes stripped) |
| 3 | Exact match on `business_address_norm` |
| 4 | TF-IDF character n-gram nearest neighbors on `business_name_norm` |
| 5 | TF-IDF character n-gram nearest neighbors on `business_address_norm` |
| 6 | Shared informative token blocking (optional) |

All passes are unioned, deduplicated, and S1 self-matches are removed.
Configurable: `name_top_k`, `address_top_k`, `tfidf_ngram_range`, `tfidf_min_df`.

---

## 12. Candidate Recall

After blocking, evaluate:
```
candidate_recall = |true ground-truth pairs ∩ candidate pairs| / |total true pairs|
```

Target: ≥ 98% overall recall (measured separately for S2 and S3).
Ground truth is **never** used to generate candidates — only to measure recall afterward.

---

## 13. Hard-Negative Construction

Training pairs include:
- All true positives from ground truth
- Easy negatives: random non-matching candidates
- Hard negatives: same-entity-context negatives (share an S1 entity with a true positive),
  similar names/addresses, same country, conflicting numeric tokens

Default ratio: 1 positive : 4 negatives (50% hard, 50% easy). Configurable.

---

## 14. Feature Engineering

`src/features.py` computes ~40 numerical features per candidate pair:

**Name features**: exact match, core exact match, Levenshtein similarity, Jaro-Winkler,
partial ratio, token sort ratio, token set ratio, Jaccard, containment, overlap ratio,
common token count, length difference, TF-IDF char cosine, TF-IDF word cosine.

**Address features**: same set as name features applied to normalized address.

**Numeric features**: numeric token Jaccard, digit sequence Jaccard, exact digit match,
numeric count difference, numeric conflict indicator.

**Country feature**: country_match (open-set string equality — no one-hot encoding).

**Structural features**: source_is_s2, source_is_s3, same_first_char.

**Interaction features**: name × address, min/max of name and address similarities.

Feature columns are sorted alphabetically for deterministic ordering across train/inference.

---

## 15. XGBoost Training

```python
from training import train_xgboost_local, BASELINE_XGBOOST_PARAMS
booster = train_xgboost_local(X_train, y_train, X_val, y_val, BASELINE_XGBOOST_PARAMS)
```

Or use `launch_sagemaker_training_job()` for managed SageMaker training.

Baseline hyperparameters:
- `objective`: binary:logistic
- `eval_metric`: aucpr
- `max_depth`: 6, `eta`: 0.05, `min_child_weight`: 3
- `subsample`: 0.8, `colsample_bytree`: 0.8

---

## 16. Hyperparameter Experiments

Staged experiments (not Cartesian grid):

```
search_max_depth        : [4, 5, 6, 7, 8]
search_eta              : [0.03, 0.05, 0.08, 0.10]
search_min_child_weight : [1, 3, 5, 10]
search_subsample        : [0.7, 0.8, 0.9, 1.0]
search_colsample_bytree : [0.7, 0.8, 0.9, 1.0]
search_gamma            : [0.0, 0.1, 0.3]
search_reg_alpha        : [0.0, 0.1, 0.5]
search_reg_lambda       : [1.0, 2.0, 5.0]
search_num_round        : [200, 300, 500]
```

Model selection criterion: **entity-level macro F0.5** (not AUC-PR).

---

## 17. F0.5 Calculation

```python
from evaluation import macro_f05
score = macro_f05(gt, predictions, s1_ids)
```

Formula: F0.5 = (1.25 × P × R) / (0.25 × P + R)

Computed per S1 entity using set-based TP/FP/FN, then macro-averaged.
Singleton rule: empty prediction on singleton entity = 1.0; any prediction = 0.0.

---

## 18. Threshold Optimization

```python
from threshold import sweep_thresholds, select_best_threshold
sweep_df = sweep_thresholds(val_pairs, val_probs, val_gt, val_s1_ids)
best_t, best_f05 = select_best_threshold(sweep_df)
```

Threshold is swept from 0.10 to 0.99 in steps of 0.01.
Best threshold is selected to maximize **validation macro F0.5**.
Default threshold of 0.5 is NOT assumed. Threshold is saved to a JSON artifact.

---

## 19. Error Analysis

`src/threshold.py` provides `analyze_errors()` which identifies:
- High-confidence false positives (predicted match, wrong)
- False negatives (missed true matches)
- Singleton false positives (false merges on entities with no true matches)

---

## 20. Test Inference

```python
from inference import load_and_clean_test_data, generate_test_candidates, run_inference
```

Identical cleaning, blocking, and feature computation as training.
Feature column ordering is loaded from the saved schema artifact.
Model and threshold are loaded from saved artifacts.

---

## 21. Output File Generation

```python
from submission import generate_submission_files
generate_submission_files(s1_ids, predictions, candidates_df, s2_df, s3_df, output_dir, test_dir)
```

Produces:
- `output/matching_results.tsv`: source1_entity_id TAB matched_entity_ids
- `output/candidate_pairs.tsv`: source1_entity_id TAB candidate_entity_ids

---

## 22. Official Submission Validation

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Or via Python:
```python
from submission import run_official_validator
passed = run_official_validator(matching_path, candidate_path, test_dir)
```

---

## 23. Reproducibility

- Fixed random seed: `cfg.experiment.random_seed = 42`
- Deterministic train/val split: split by shuffled S1 entity IDs with fixed seed
- Deterministic feature column ordering: alphabetically sorted
- Deterministic candidate ordering: sorted by entity_id
- XGBoost `seed` parameter set to random seed

---

## 24. How to Reproduce End-to-End

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up data directory
# Place train/*.tsv and test/*.tsv under dataset/

# 3. Run full pipeline
cd code/business_entity_resolution
python3 src/pipeline.py \
    --data-dir ../../dataset \
    --output-dir ../../output

# 4. Validate output
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

# 5. Upload matching_results.tsv to the challenge leaderboard portal
```

For SageMaker:
```bash
# Upload data to S3
aws s3 sync dataset/ s3://your-bucket/student_resource/dataset/
aws s3 sync code/ s3://your-bucket/student_resource/code/

# In SageMaker Studio terminal:
aws s3 sync s3://your-bucket/student_resource/ ~/student_resource/
cd ~/student_resource/code/business_entity_resolution
pip install -r requirements.txt
python3 src/pipeline.py --data-dir ../../dataset --output-dir ../../output

# Download results
aws s3 sync ~/student_resource/output/ s3://your-bucket/output/
```
