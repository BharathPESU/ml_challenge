# Code Restructuring & Kaggle 2×T4 Optimization Plan (plan2.md)

This document maps out the exact code restructuring, modular changes, new files, memory management, GPU utilization, and notebook updates required to transition the current codebase into the production-grade entity resolution system specified in [`docs/plan.md`](file:///home/bharath/Desktop/projects/ml_challenge/docs/plan.md).

---

## 1. Executive Summary & Restructuring Goals

| Metric / Aspect | Previous Codebase | New Architecture (`plan.md` + `plan2.md`) |
| :--- | :--- | :--- |
| **Candidate Blocking** | Single-pass TF-IDF top-K | **8-Pass Multi-Pass Blocking** (Exact, Core Name, Address Token, Numeric, Rare Tokens, Bounded RapidFuzz) |
| **Candidate Recall Target** | ~80-85% (single pass) | **98%–99%+ Candidate Recall** with minimal reduction ratio |
| **Negative Sampling** | Random candidate sampling | **Hard Negative Construction** (high similarity, wrong entity, numeric overlap) |
| **Feature Set** | Basic string distances | **30+ Comprehensive Pairwise Features** (Name, Core Name, Address, Numeric Extraction, Interaction terms) |
| **Memory Strategy** | Whole DataFrame in memory | **Chunked Batch Processing**, `float32`/`category` dtype optimizations, `gc.collect()` after each phase |
| **GPU Utilization** | CPU-only XGBoost / default | **XGBoost CUDA GPU acceleration (`tree_method='hist'`, `device='cuda:0'`)** + Dual T4 usage |
| **Evaluation Metric** | Standard Precision/Recall/F1 | **Macro-averaged Entity-level F0.5** with full singleton handling & false-merge protection |
| **Thresholding** | Fixed 0.5 default | **Grid Sweep Optimization (0.10 to 0.99)** per entity & optional source-specific thresholds |

---

## 2. Module-by-Module Code Changes

Below is the detailed comparison of **what exists in the previous code**, **what needs to be changed**, and **which extra files need to be created**.

### Summary of File Changes

| File Path | Status | Primary Purpose / Changes Needed |
| :--- | :--- | :--- |
| [`src/config.py`](file:///home/bharath/Desktop/projects/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution/src/config.py) | **Modify** | Add multi-pass blocking keys, hard negative ratios, GPU config (`device`), memory batch size, threshold sweep range. |
| [`src/io_utils.py`](file:///home/bharath/Desktop/projects/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution/src/io_utils.py) | **Modify** | Memory-optimized TSV reader with explicit string/category types, disk-backed intermediate parquet/feather streaming. |
| [`src/cleaning.py`](file:///home/bharath/Desktop/projects/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution/src/cleaning.py) | **Modify** | Add `business_name_core` (legal suffix removal), `business_address_norm`, `address_numbers` extraction, missingness indicators. |
| `src/cleaning_eval.py` | **NEW FILE** | Dedicated Cleaning Quality Evaluator (structural validation, collision analysis, token drop stats, sanity samples). |
| [`src/ground_truth.py`](file:///home/bharath/Desktop/projects/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution/src/ground_truth.py) | **Modify** | Parse singleton vs multi-match clusters, provide fast indexing for recall calculation. |
| [`src/splitting.py`](file:///home/bharath/Desktop/projects/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution/src/splitting.py) | **Modify** | Enforce zero-leakage S1 entity split, stratify by country and ground-truth match count. |
| [`src/blocking.py`](file:///home/bharath/Desktop/projects/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution/src/blocking.py) | **Major Rewrite** | Replace single-pass TF-IDF with 8-pass multi-pass scalable blocking framework with country-wise chunking. |
| `src/blocking_eval.py` | **NEW FILE** | Compute recall by source (S2/S3), by country, by match-count bucket, candidate count percentiles (p95/max). |
| [`src/pair_dataset.py`](file:///home/bharath/Desktop/projects/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution/src/pair_dataset.py) | **Modify** | Construct training pairs with hard negative sampling (1:3 to 1:8 ratio) based on fuzzy similarity & numeric overlap. |
| [`src/features.py`](file:///home/bharath/Desktop/projects/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution/src/features.py) | **Major Rewrite** | Implement 30+ pairwise feature extraction pipelines with vectorized C-RapidFuzz and float32/float16 downcasting. |
| [`src/training.py`](file:///home/bharath/Desktop/projects/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution/src/training.py) | **Modify** | Enable XGBoost GPU hist mode (`tree_method='hist'`, `device='cuda:0'`), hyperparameter trial execution, early stopping. |
| [`src/evaluation.py`](file:///home/bharath/Desktop/projects/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution/src/evaluation.py) | **Modify** | Calculate entity-level macro F0.5, singleton correctness, FP/FN count metrics. |
| [`src/threshold.py`](file:///home/bharath/Desktop/projects/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution/src/threshold.py) | **Modify** | Complete grid sweep (0.10 - 0.99, step 0.01), global vs source-specific threshold optimizer. |
| [`src/error_analysis.py`](file:///home/bharath/Desktop/projects/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution/src/error_analysis.py) | **NEW FILE** | Detailed FP/FN inspection logger, top error categories breakdown for iterative improvement loops. |
| [`src/inference.py`](file:///home/bharath/Desktop/projects/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution/src/inference.py) | **Modify** | Chunked memory-safe test set candidate scoring & batch prediction. |
| [`src/submission.py`](file:///home/bharath/Desktop/projects/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution/src/submission.py) | **Modify** | Export `matching_results.tsv` and `candidate_pairs.tsv` adhering strictly to official validator formats. |
| [`src/pipeline.py`](file:///home/bharath/Desktop/projects/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution/src/pipeline.py) | **Modify** | Orchestrate 15-stage modular pipeline, explicit memory recycling after each phase (`gc.collect()`). |
| [`ml_challenge.ipynb`](file:///home/bharath/Desktop/projects/ml_challenge/ml_challenge.ipynb) | **Modify** | Restructure into 19 discrete, re-runnable notebook cells matching the plan step-by-step. |

---

## 3. Detailed Component Comparison & Code Modifications

### A. Configuration (`src/config.py`)
- **Previous Code**: Basic data paths and sample size parameters.
- **Code Changes Needed**:
  - Add `BlockingConfig`: multi-pass flags, block top-k limits per pass, rare token threshold.
  - Add `FeatureConfig`: toggle list for char, token, numeric, interaction features.
  - Add `TrainingConfig`: XGBoost hyperparams (`max_depth`, `eta`, `subsample`, `colsample_bytree`, `tree_method="hist"`, `device="cuda:0"`).
  - Add `MemoryConfig`: `chunk_size` (e.g. 50,000 S1 records), `use_parquet_cache=True`.
  - **Dynamic Paths**: Do not hard-code the Kaggle path (`/kaggle/input/datasets/...`); detect the dataset directory or keep it fully configurable via environment variables or auto-discovery.
  - **Reproducibility Tracking**: Add strict random seed initialization, and capture config hashes, feature version, blocking version, and model version.

### B. Cleaning & Cleaning Audit (`src/cleaning.py` & `src/cleaning_eval.py`)
- **Previous Code**: Simple lowercase and basic regex character removal.
- **Code Changes Needed**:
  - `business_name_norm`: Lowercase, unicode NFKD normalization, remove special punctuation, normalize `&` to `and`.
  - `business_name_core`: Strip trailing legal entity terms (`ltd`, `limited`, `pvt`, `inc`, `corp`, `llp`, `co`, `gmbh`, `sa`, etc.) while keeping `business_name_norm` intact.
  - `business_address_norm`: Address normalization while strictly preserving numbers and alphanumeric unit identifiers (`b12`, `101`).
  - `address_numbers`: Regex extraction of numeric tokens into sorted token sets.
  - Diagnostic columns: `name_missing`, `address_missing`, `country_missing`, `name_len`, `address_len`.
  - **`src/cleaning_eval.py`**: Check zero row/ID loss, measure % names/addresses modified, calculate high-frequency name collisions.
  - **Cleaning Quality Gates**: Ensure strict checks on row/ID preservation, normalization collision rate, empty-after-cleaning count, and display before/after samples. Break pipeline if gates fail.

### C. 8-Pass Scalable Candidate Blocking (`src/blocking.py`)
- **Previous Code**: Single TF-IDF vectorizer calculated globally on concatenated S1/S2/S3 text, creating a large sparse matrix susceptible to RAM OOM.
- **Code Changes Needed**:
  Implement an **8-Pass Scalable Multi-Pass Blocker**:
  1. `Pass 1`: Exact match on `(country, business_name_norm)`
  2. `Pass 2`: Exact match on `(country, business_name_core)`
  3. `Pass 3`: Address-token blocking (shared informative address tokens within same country)
  4. `Pass 4`: Numeric-address blocking (matching house/building number + postal code within country)
  5. `Pass 5`: Rare name-token inverted index blocking (token frequency < 1000)
  6. `Pass 6`: Rare address-token inverted index blocking
  7. `Pass 7`: RapidFuzz character/token similarity candidate retrieval within coarse blocks
  8. `Pass 8`: RapidFuzz address similarity candidate retrieval within coarse blocks
  - Execution Strategy: Run blocking **country-by-country** (or in chunks of S1 entities) to guarantee RAM usage remains under 4 GB.
  - **Detailed Blocking Logging**: Log not only recall, but also candidate volume (total/avg/p95 per S1) vs runtime/memory for every blocking pass.

### D. Feature Engineering (`src/features.py`)
- **Previous Code**: Simple Levenshtein distance and token ratio.
- **Code Changes Needed**:
  Extract **30+ pairwise features** in parallel using C-optimized `rapidfuzz` and `numpy`:
  - **Name Features**: exact match, core exact match, char similarity ratio, Levenshtein distance, token sort ratio, token set ratio, Jaccard similarity, containment ratio, length difference, length ratio, common token count.
  - **Address Features**: exact match, char similarity, Levenshtein distance, token sort/set ratio, Jaccard similarity, containment ratio, length difference, length ratio.
  - **Numeric Address Features**: numeric set overlap count, numeric Jaccard similarity, shared number count, number count difference.
  - **Meta & Categorical Features**: `country_match`, `name_missing_flag`, `address_missing_flag`, `source_is_s2`, `source_is_s3`.
  - **Interaction Features**: `name_sim * address_sim`, `min(name_sim, address_sim)`, `max(name_sim, address_sim)`.
  - **Memory Optimization**: Use `np.float32` by default for all XGBoost inputs/features to save RAM. Only benchmark/use `np.float16` if stability is strictly confirmed (XGBoost often prefers float32).

### E. Hard Negative Sampling & Training Pair Construction (`src/pair_dataset.py`)
- **Previous Code**: Basic positive/negative pair extraction from candidates.
- **Code Changes Needed**:
  - Positive pairs: Ground truth matching S1-S2 / S1-S3 candidates (`label = 1`).
  - **Positive-Pair Capture Rate Check**: Before applying any hard-negative sampling, explicitly calculate and log the positive-pair capture rate. Every true pair recovered from blocking must survive into the dataset.
  - Negative pairs: Subsample non-ground-truth candidates with a focus on **hard negatives** (candidates having high TF-IDF / name similarity > 0.6 or shared numeric tokens, but distinct entity IDs).
  - Configurable ratio: Test negative ratios of `1:3`, `1:5`, and `1:8`.

### F. GPU-Accelerated Model Training (`src/training.py`)
- **Previous Code**: CPU-bound XGBoost classifier.
- **Code Changes Needed**:
  - Configure XGBoost for GPU execution: `tree_method='hist'`, `device='cuda:0'`.
  - Support hyperparameter sweep trials: tuning `max_depth` (4-8), `eta` (0.03-0.10), `min_child_weight` (1-10), `subsample` (0.7-1.0), `colsample_bytree` (0.7-1.0), and regularization (`reg_alpha`, `reg_lambda`).
  - **Probability Calibration Check**: Validate reliability of model output probabilities before threshold sweeping.
  - Evaluate models based on **downstream entity-level macro F0.5**, not just binary accuracy or ROC-AUC.

### G. Threshold Sweep & Entity Macro F0.5 (`src/threshold.py` & `src/evaluation.py`)
- **Previous Code**: Fixed 0.5 probability cutoff.
- **Code Changes Needed**:
  - Sweep decision thresholds globally from `0.10` to `0.99` with step `0.01`.
  - Threshold must be derived globally (or at source-level for S2 vs S3) through trial-and-error, *never* tuned "per entity" (which causes overfitting/leakage).
  - Implement official metric: **Macro-Averaged Entity-Level F0.5**:
    $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
  - Strictly support singletons (empty match lists `[]`) and multi-matches (`[S2-xxx, S3-yyy]`).
  - **Error Analysis Module**: Add top-K match-count/error analysis specifically focusing on high-confidence false positives and singleton false positives.

---

## 4. Kaggle 2×T4 Memory & Hardware Efficiency Strategy

To prevent Kaggle environment crashes (Draft Session Timeout, CPU RAM Out-Of-Memory, or CUDA OOM):

```text
[Kaggle Hardware Budget]
├── CPU RAM : 13.0 GB Limit
├── GPU 0   : NVIDIA T4 (16 GB VRAM)
└── GPU 1   : NVIDIA T4 (16 GB VRAM)
```

### Critical Guidelines for Memory & GPU Management

1. **Explicit Data Type Downcasting**:
   - Store candidate entity IDs as `category` or string views.
   - Downcast feature matrices to `float32` (or `float16` for trees).
   - Downcast integer counts to `uint16` / `uint32`.

2. **Chunked & Batched Candidate Processing**:
   - Process blocking and feature extraction in batches of S1 entities (e.g. 50,000 S1 records per chunk).
   - Never create a single combined $N \times M$ dense pairwise DataFrame in memory for millions of rows.

3. **Active Garbage Collection**:
   - Insert explicit `del df` and `gc.collect()` at the end of every pipeline stage (loading, cleaning, blocking, feature extraction, scoring).

4. **Disk Caching for Intermediate Artifacts**:
   - Cache intermediate candidates and features to disk using compressed Parquet format (`/kaggle/working/run/candidates/candidates_pass.parquet`) with `snappy` or `zstd` compression.

5. **Dual T4 GPU Allocation Strategy**:
   - **GPU 0 (`device='cuda:0'`)**: Used for main XGBoost training, DMatrix operations, and test set inference scoring.
   - **GPU 1 (`device='cuda:1'`)**: Reserved for parallel hyperparameter search trials or rapid threshold sweep scoring.

---

## 5. Notebook Restructuring Plan (`ml_challenge.ipynb`)

`ml_challenge.ipynb` will be updated to feature 19 modular cells matching the plan execution order:

1. **Cell 1**: Environment setup (Git pull, Kaggle environment detection, GPU T4 check, package install).
2. **Cell 2**: Config initialization & paths setup (`/kaggle/input/datasets/cdbharath234/dataset3/student_resource/dataset/`).
3. **Cell 3**: Stage 1 — Load Training Data (Chunked TSV reader).
4. **Cell 4**: Stage 2 — Data Cleaning & Normalization (Name, Core Name, Address, Numbers).
5. **Cell 5**: Stage 2.1 — Cleaning Quality Audit & Collision Analysis.
6. **Cell 6**: Stage 3 — Zero-Leakage Train/Validation Split by S1 Entity.
7. **Cell 7**: Stage 4 — Multi-Pass Candidate Blocking (8 Passes, Country-chunked).
8. **Cell 8**: Stage 5 — Candidate Recall & Reduction Ratio Evaluation.
9. **Cell 9**: Stage 6 — Hard-Negative Training Pair Construction.
10. **Cell 10**: Stage 7 — Pairwise Feature Extraction (30+ features, `float32`).
11. **Cell 11**: Stage 8 — XGBoost GPU Training & Hyperparameter Tuning.
12. **Cell 12**: Stage 9 — Validation Set Scoring (Probability generation).
13. **Cell 13**: Stage 10 — Threshold Grid Sweep (0.10 - 0.99) & Macro F0.5 Optimization.
14. **Cell 14**: Stage 11 — Comprehensive Error Analysis (FP/FN Inspection).
15. **Cell 15**: Stage 12 — Test Set Candidate Generation & Feature Extraction.
16. **Cell 16**: Stage 13 — GPU Test Inference & Scored Predictions.
17. **Cell 17**: Stage 14 — Generate Submission Files (`matching_results.tsv` & `candidate_pairs.tsv`).
18. **Cell 18**: Stage 15 — Submission Integrity Validation (Row counts, singleton check, format check).
19. **Cell 19**: Full Dataset Run Switcher (Resets sample limit to `None` for full Kaggle run).

---

## 6. Execution & Verification Checklist

- [ ] Create `plan2.md` in `docs/` repository path.
- [ ] Implement new & modified Python source files in `src/`.
- [ ] Verify zero module import errors in Kaggle kernel.
- [ ] **Smoke Test**: Run a small-sample smoke test first (e.g., 500 records) to confirm end-to-end memory stability and pipeline flow.
- [ ] Execute full pipeline on Kaggle 2×T4 environment.
- [ ] Validate final `matching_results.tsv` and `candidate_pairs.tsv` against `utils/validate_submission.py`.
