# Amazon ML Challenge 2026 — Business Entity Resolution

## Kaggle 2×T4 End-to-End ML Plan

## 1. Objective

Build a reproducible entity-resolution system that maps every Source 1 business in the test set to zero, one, or many matching Source 2 and Source 3 records.

This is a **candidate-pair binary classification problem**:

```text
S1 record + S2/S3 candidate
            ↓
        target 0/1
```

* `1` = same real-world business
* `0` = different business

`train_source1.tsv` has no target column. The target is created after candidate generation using `train_ground_truth.tsv`.

The competition metric is **macro-averaged entity-level F0.5**. F0.5 is precision-heavy, so false merges must be controlled carefully. Singletons are important.

---

# 2. Execution Environment

Run the entire workflow in:

```text
Kaggle
```

using:

```text
2 × NVIDIA T4 GPU
```

Do NOT use:

* AWS SageMaker
* AWS S3
* SageMaker Processing
* SageMaker Training Jobs
* SageMaker Batch Transform
* external APIs
* external entity-resolution services
* geocoding
* external business databases
* internet-based business lookup
* external data enrichment

The challenge explicitly prohibits external business identity lookup and external data augmentation.

## Kaggle Input

The challenge files are mounted read-only under:

```text
/kaggle/input/<dataset-folder>/dataset/train/
```

and:

```text
/kaggle/input/<dataset-folder>/dataset/test/
```

Training:

```text
train_source1.tsv
train_source2.tsv
train_source3.tsv
train_ground_truth.tsv
```

Test:

```text
test_source1.tsv
test_source2.tsv
test_source3.tsv
```

## Kaggle Working Storage

Use:

```text
/kaggle/working/
```

for:

* cleaned data
* candidates
* features
* models
* predictions
* experiments
* reports
* final outputs

Never modify:

```text
/kaggle/input/
```

---

# 3. Final Submission Structure

The final submission MUST remain:

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
│
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
│
└── Documentation_template.md
```

Do not add additional root-level folders to the final package.

Runtime files can exist under `/kaggle/working/`, but the final package must preserve the structure above.

---

# 4. Overall ML Architecture

```text
Kaggle Input
      ↓
TASK 1 — Data Understanding
      ↓
TASK 2 — Data Cleaning / Normalization
      ↓
Cleaning Quality Evaluation
      ↓
TASK 3 — Train/Validation Split
      ↓
TASK 4 — Candidate Generation / Blocking
      ↓
TASK 5 — Candidate Recall Evaluation
      ↓
TASK 6 — Feature Engineering
      ↓
TASK 7 — Training Pair Construction
      ↓
TASK 8 — XGBoost Training
      ↓
Validation Predictions
      ↓
TASK 9 — Threshold Optimization
      ↓
Error Analysis
      ↓
Parameter / Feature / Blocking Improvement
      ↓
Re-train + Re-evaluate
      ↓
TASK 10 — Test Inference + Submission
      ↓
matching_results.tsv
candidate_pairs.tsv
      ↓
Official Validator
```

---

# 5. TASK 1 — Data Understanding

## Goal

Understand the actual dataset before changing anything.

## Load all seven files

Always use:

```python
pd.read_csv(path, sep="\t", dtype=str)
```

## Check every dataset

Measure:

* row count
* column count
* schema
* data types
* missing values
* empty strings
* duplicate `entity_id`
* unique names
* duplicate names
* unique addresses
* duplicate addresses
* country distribution
* name length distribution
* address length distribution

## Validate IDs

Source 1:

```text
S1-
```

Source 2:

```text
S2-
```

Source 3:

```text
S3-
```

Check that `entity_id` is unique within each source.

## Ground-truth analysis

Calculate:

* total Source 1 entities
* singleton count
* one-match count
* multi-match count
* maximum matches per Source 1
* total S2 matches
* total S3 matches
* match-count distribution
* match-count distribution by country

## Important

Do NOT remove duplicate business names.

Do NOT remove duplicate addresses.

Different businesses may legitimately share the same name or address.

---

# 6. TASK 2 — Data Cleaning / Normalization

The challenge data may be structurally clean but contains semantic noise.

The goal is:

```text
raw record
    ↓
normalized representations
```

not:

```text
raw record
    ↓
delete rows
```

## Preserve raw columns

Keep:

```text
business_name
business_address
country
```

Create:

```text
business_name_norm
business_name_core
business_address_norm
address_numbers
```

## Name normalization

Apply:

* lowercase
* Unicode normalization
* punctuation normalization
* whitespace normalization
* safe alphanumeric preservation
* useful `&` normalization

Example:

```text
ABC, Pvt. Ltd.
→
abc pvt ltd
```

## Core-name normalization

Create:

```text
business_name_core
```

Remove only trailing legal suffixes such as:

```text
ltd
limited
pvt
private
private limited
inc
incorporated
corp
corporation
llp
co
company
```

Keep:

```text
business_name_norm
business_name_core
```

separately.

## Address normalization

Apply:

* lowercase
* Unicode normalization
* punctuation normalization
* whitespace normalization
* preserve numeric information
* preserve useful alphanumeric identifiers

Example:

```text
12, M.G. Road, Bengaluru
→
12 mg road bengaluru
```

## Numeric extraction

Extract useful numeric tokens:

```text
12
560001
101
B12
```

These will later become pairwise similarity features.

## Diagnostic columns

Create:

```text
name_missing
address_missing
country_missing

name_length
address_length

name_token_count
address_token_count

numeric_token_count
```

## Country

Treat country as an **open-set string**.

Do not assume only:

```text
US
India
```

Test data can contain additional countries such as France.

Use pairwise comparison:

```text
country_match
```

rather than a fixed country list.

---

# 7. Cleaning Quality Evaluation

This is a dedicated stage and must happen **before blocking**.

Do not assume cleaning is correct just because it runs.

## A. Structural checks

Compare before vs after:

```text
row count
entity count
duplicate entity IDs
empty values
```

Expected:

```text
no accidental row loss
no accidental ID loss
```

## B. Normalization statistics

Measure:

```text
% names changed
% addresses changed
% names becoming empty
% addresses becoming empty
average token-count change
```

## C. Collision analysis

Calculate:

```text
raw name
   ↓
normalized name
   ↓
number of raw values collapsed into it
```

Investigate highly frequent normalized names.

Do the same for addresses.

## D. Sanity samples

Display:

```text
raw name → normalized name → core name
raw address → normalized address
```

Inspect:

* punctuation
* abbreviations
* legal suffixes
* spelling
* word order
* transliteration
* numbers

## E. Cleaning acceptance criteria

Continue only if:

1. rows are preserved
2. IDs are preserved
3. numbers are preserved
4. normalization reduces irrelevant formatting noise
5. meaningful tokens are not destroyed
6. high-frequency collisions are understood
7. no obvious over-normalization exists

---

# 8. TASK 3 — Train / Validation Split

Split by:

```text
Source 1 entity_id
```

NOT by candidate rows.

Default:

```text
80% train
20% validation
random_state = 42
```

Where practical, stratify using:

```text
country
+
match-count bucket
```

## Leakage validation

Verify:

```text
train_S1_ids ∩ validation_S1_ids = empty
```

Never allow:

* same S1 in train and validation
* validation labels in training
* test data in training
* ground truth to influence candidate generation

Ground truth can only be used to:

* label generated training candidates
* evaluate candidate recall
* calculate validation metrics

---

# 9. TASK 4 — Candidate Generation / Blocking

This is a critical stage because the model cannot recover a true match that never enters the candidate set.

Do not perform:

```text
S1 × S2
S1 × S3
```

full Cartesian comparisons.

## Multi-pass blocking

### Pass 1 — Exact normalized name

```text
country + business_name_norm
```

### Pass 2 — Exact core name

```text
country + business_name_core
```

### Pass 3 — Address-token blocking

Use informative address tokens.

### Pass 4 — Numeric-address blocking

Use shared useful numeric tokens.

### Pass 5 — Rare name-token blocking

Avoid giant generic blocks.

### Pass 6 — Rare address-token blocking

Use informative tokens.

### Pass 7 — Fuzzy name retrieval

Run RapidFuzz only inside manageable candidate blocks.

Use:

```text
character similarity
token-set similarity
token-sort similarity
```

### Pass 8 — Fuzzy address retrieval

Run fuzzy address comparison only inside manageable blocks.

## Scalability

Use:

```text
deterministic block
      ↓
small block
      ↓
fuzzy ranking
```

Avoid a global dense all-to-all similarity computation over millions of rows.

Generate candidates separately:

```text
S1 → S2
S1 → S3
```

Then union them.

Every candidate retains:

```text
source1_entity_id
candidate_entity_id
candidate_source
```

---

# 10. TASK 5 — Candidate Recall Evaluation

Candidate recall is the first optimization target.

## Formula

```text
candidate recall =
true matches captured by candidate generation
/
all true matches
```

## Measure

* overall candidate recall
* S2 candidate recall
* S3 candidate recall
* recall by country
* recall by match-count bucket
* singleton candidate behavior
* average candidates/S1
* median candidates/S1
* p95 candidates/S1
* maximum candidates/S1
* reduction ratio
* runtime
* memory

## Trial-and-error blocking experiments

Try:

```text
Name K:
20
50
100
200
```

and:

```text
Address/Fuzzy K:
20
50
100
200
```

Also compare different combinations of blocking keys.

Create an experiment table:

```text
experiment
blocking configuration
candidate recall
avg candidates
p95 candidates
max candidates
reduction ratio
runtime
memory
```

## Selection rule

Choose the **smallest practical candidate set** that preserves very high candidate recall.

Target approximately:

```text
98–99%+ candidate recall
```

but select using actual validation results.

---

# 11. TASK 6 — Feature Engineering

After blocking:

```text
S1
+
candidate S2/S3
```

becomes one pair.

## Name features

```text
name_exact
name_core_exact
name_char_similarity
name_levenshtein
name_token_sort_ratio
name_token_set_ratio
name_jaccard
name_containment
name_length_difference
name_length_ratio
common_name_tokens
```

## Address features

```text
address_exact
address_char_similarity
address_levenshtein
address_token_sort_ratio
address_token_set_ratio
address_jaccard
address_containment
address_length_difference
address_length_ratio
```

## Numeric address features

```text
numeric_overlap
numeric_jaccard
shared_number_count
number_count_difference
```

## Other features

```text
country_match
name_missing
address_missing
country_missing
source_is_s2
source_is_s3
```

## Interaction features

```text
name_similarity * address_similarity
min(name_similarity, address_similarity)
max(name_similarity, address_similarity)
```

Never use:

```text
entity_id
matched_entity_ids
```

as model features.

---

# 12. TASK 7 — Training Pair Construction

Use:

```text
candidate pairs
+
ground truth
```

to construct the supervised dataset.

## Positive pairs

Candidate is present in ground truth:

```text
label = 1
```

## Negative pairs

Candidate is not present:

```text
label = 0
```

## Negative categories

Use:

### Easy negatives

Random/unrelated candidates.

### Hard negatives

* high name similarity but wrong entity
* high address similarity but wrong entity
* same country + similar name
* same country + similar address
* high TF-IDF similarity but wrong entity
* shared numeric/address information but wrong entity

Start with:

```text
1 positive : 3–5 negatives
```

Then test:

```text
1:3
1:5
1:8
```

Do not downsample validation candidates purely for convenience.

---

# 13. TASK 8 — XGBoost Training

Use:

```text
XGBoost
```

as the primary pairwise binary classifier.

Input:

```text
pairwise features
```

Output:

```text
P(same business)
```

## GPU strategy

Use one T4 for the first reliable model.

The second T4 can later be used for:

* parallel parameter experiments
* separate model variants
* independent experiments

Do not assume a single XGBoost process automatically uses both GPUs.

## Baseline parameters

```text
objective = binary:logistic
eval_metric = aucpr

max_depth = 6
eta = 0.05
min_child_weight = 3

subsample = 0.8
colsample_bytree = 0.8

gamma = 0.1
reg_alpha = 0.1
reg_lambda = 2

num_boost_round = 300
```

These are only baseline values.

## Parameter trial-and-error

Tune systematically.

### Experiment A — max_depth

```text
4
5
6
7
8
```

### Experiment B — eta

```text
0.03
0.05
0.08
0.10
```

### Experiment C — min_child_weight

```text
1
3
5
10
```

### Experiment D — subsample

```text
0.7
0.8
0.9
1.0
```

### Experiment E — colsample_bytree

```text
0.7
0.8
0.9
1.0
```

### Experiment F — regularization

Experiment with:

```text
gamma
reg_alpha
reg_lambda
```

### Experiment G — boosting rounds

```text
200
300
500
```

Use early stopping where practical.

## Model selection

Do not select the model only by:

```text
accuracy
AUC
training loss
```

For each candidate model:

```text
model
 ↓
validation probabilities
 ↓
threshold sweep
 ↓
entity-level macro F0.5
```

Use downstream validation F0.5 as the competition-oriented model-selection criterion.

---

# 14. TASK 9 — Threshold Optimization

There is no universal correct threshold.

Do not assume:

```text
0.5
```

or:

```text
0.8
```

Find the threshold experimentally.

## Sweep

Start:

```text
0.10
0.11
0.12
...
0.99
```

## For each threshold

1. Select candidates where:

```text
model_probability >= threshold
```

2. Group by Source 1.
3. Allow zero matches.
4. Allow one match.
5. Allow many matches.
6. Calculate entity precision.
7. Calculate entity recall.
8. Calculate entity F0.5.
9. Macro-average F0.5.
10. Calculate singleton correctness.
11. Count false positives.
12. Count false negatives.
13. Count predicted matches.

## Metric

```text
F0.5 =
(1.25 × Precision × Recall)
/
(0.25 × Precision + Recall)
```

Calculate per Source 1 entity and then macro-average.

## Threshold experiment table

Record:

```text
threshold
macro_f05
macro_precision
macro_recall
singleton_correctness
false_positive_count
false_negative_count
predicted_match_count
```

## Selection

```text
best_threshold =
threshold with highest validation macro F0.5
```

If two thresholds are effectively tied, prefer the higher/more conservative threshold.

## Source-specific experiment

After global threshold optimization, optionally test:

```text
threshold_S2
threshold_S3
```

Keep them only if validation F0.5 improves.

---

# 15. Singleton Handling

Never force one match per Source 1.

Valid predictions:

```text
S1-001 → []
```

```text
S1-002 → [S2-100]
```

```text
S1-003 → [S2-101,S2-102,S3-200]
```

For a singleton:

```text
actual = []
predicted = []
```

is correct.

A false match on a true singleton is especially harmful because of the competition metric.

---

# 16. Error Analysis

After every important model run:

## False positives

Find:

```text
prediction = match
truth = non-match
```

Sort by highest confidence.

Inspect:

* same generic name
* same street/address
* shared building number
* same city
* high name similarity but different address

## False negatives

Find:

```text
prediction = non-match
truth = match
```

Inspect:

* spelling variation
* abbreviations
* transliteration
* partial address
* missing numbers
* DBA/trade names

## Improvement loop

```text
error pattern
      ↓
feature/blocking change
      ↓
retrain
      ↓
threshold sweep
      ↓
compare F0.5
```

Change one major component at a time.

---

# 17. Experiment Tracking

Every experiment should record:

```text
experiment_id
blocking_configuration
feature_configuration
negative_ratio
model_parameters
threshold
candidate_recall
candidate_count
macro_precision
macro_recall
macro_F0.5
singleton_correctness
false_positive_count
false_negative_count
runtime
memory
```

This becomes the competition experiment history.

---

# 18. Optimization Priority

Use this order.

## Priority 1

Candidate recall.

```text
missing candidate
      ↓
impossible to recover
```

## Priority 2

Hard-negative quality.

```text
easy negatives
      ↓
easy classifier
      ↓
weak protection against false merges
```

## Priority 3

Feature quality.

Focus on:

```text
name
address
numbers
country
interactions
```

## Priority 4

XGBoost parameter tuning.

## Priority 5

Threshold optimization.

Re-run it after every meaningful model/feature change.

## Priority 6

Source-specific threshold/model experiments.

Only keep them when validation proves improvement.

---

# 19. TASK 10 — Test Inference + Submission

Load:

```text
test_source1.tsv
test_source2.tsv
test_source3.tsv
```

Apply exactly the same:

```text
normalization
blocking
feature engineering
```

used during training.

Then:

```text
test candidates
      ↓
XGBoost
      ↓
probabilities
      ↓
final validated threshold
      ↓
final matches
```

Do not retrain using test data.

Do not use test labels.

Do not force one match per Source 1.

---

# 20. Generate matching_results.tsv

Required columns:

```text
source1_entity_id
matched_entity_ids
```

Rules:

* exactly one row for every test Source 1 entity
* zero, one, or many matches allowed
* empty value when there is no match
* no duplicate matched IDs
* only S2/S3 IDs
* IDs must exist in test data
* tab-separated
* no extra columns

---

# 21. Generate candidate_pairs.tsv

Required columns:

```text
source1_entity_id
candidate_entity_ids
```

This must contain the **exact final candidate set passed to the matching model**.

Do not output an earlier intermediate blocker.

For every Source 1 entity:

```text
final_matches ⊆ final_candidates
```

must hold.

---

# 22. Final Structural Validation

Validate:

## matching_results.tsv

* every test S1 exists
* one row per S1
* no duplicate S1 rows
* valid S2/S3 IDs
* no duplicate matched IDs
* no S1 self-matches

## candidate_pairs.tsv

* every test S1 exists
* one row per S1
* valid S2/S3 IDs
* no duplicate candidate IDs
* exact final candidate set used by model

## Official validator

Run:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Do not replace the official validator.

---

# 23. Kaggle Runtime Layout

Use:

```text
/kaggle/working/
├── run/
│   ├── cleaned/
│   ├── candidates/
│   ├── features/
│   ├── models/
│   ├── predictions/
│   ├── experiments/
│   └── reports/
│
└── final_submission/
    ├── output/
    │   ├── matching_results.tsv
    │   └── candidate_pairs.tsv
    │
    ├── code/
    │   └── business_entity_resolution/
    │       ├── src/
    │       ├── README.md
    │       └── requirements.txt
    │
    └── Documentation_template.md
```

---

# 24. Recommended Source Modules

Keep reusable code under:

```text
code/business_entity_resolution/src/
```

Recommended:

```text
__init__.py
config.py
io_utils.py
cleaning.py
ground_truth.py
splitting.py
blocking.py
pair_dataset.py
features.py
evaluation.py
threshold.py
training.py
inference.py
submission.py
pipeline.py
```

The Kaggle notebook should orchestrate these modules instead of duplicating the logic.

---

# 25. Kaggle Notebook Execution Order

```text
01. Configuration
02. Detect Kaggle environment
03. Detect T4 GPUs
04. Locate Kaggle Input
05. Load train data
06. Load test data

07. TASK 1 — Data understanding
08. TASK 2 — Cleaning
09. Cleaning quality evaluation

10. TASK 3 — Train/validation split

11. TASK 4 — Candidate generation
12. TASK 5 — Candidate recall

13. Blocking experiments
14. Select blocking configuration

15. TASK 6 — Feature engineering
16. TASK 7 — Training pair construction
17. Hard-negative analysis

18. TASK 8 — XGBoost baseline
19. Parameter experiments
20. Compare models

21. Validation predictions
22. TASK 9 — Threshold sweep
23. Compare thresholds

24. Error analysis
25. Improvement experiments

26. Final model selection
27. Final threshold selection
28. Final blocking selection

29. TASK 10 — Test candidate generation
30. Test features
31. Test inference
32. Apply selected threshold
33. Generate outputs
34. Structural validation
35. Official validator
36. Create final submission package
```

---

# 26. Full Trial-and-Error Strategy

Do not optimize everything simultaneously.

Use this controlled sequence:

```text
BASELINE
   ↓
Cleaning Quality
   ↓
Blocking Experiment
   ↓
Candidate Recall
   ↓
Select Blocking
   ↓
Feature Baseline
   ↓
Negative Sampling Experiment
   ↓
XGBoost Baseline
   ↓
Parameter Experiment
   ↓
Validation Probabilities
   ↓
Threshold Sweep
   ↓
Macro F0.5
   ↓
Error Analysis
   ↓
ONE Improvement
   ↓
Retrain
   ↓
Threshold Sweep Again
   ↓
Compare F0.5
```

Every experiment must have an experiment ID.

---

# 27. What NOT to Optimize Against

Do not select the final solution using only:

```text
accuracy
```

```text
ROC-AUC
```

```text
global F1
```

```text
maximum recall
```

```text
largest candidate set
```

The real target is:

```text
macro entity-level F0.5
```

while preserving sufficiently high candidate recall.

---

# 28. Final Model Selection

The final configuration consists of four independently selected components:

```text
1. Blocking configuration
2. Feature configuration
3. XGBoost configuration
4. Decision threshold
```

Select them using actual validation experiments.

Conceptually:

```text
               BLOCKING
                   ↓
            candidate recall
                   ↓
              FEATURES
                   ↓
            HARD NEGATIVES
                   ↓
               XGBOOST
                   ↓
        validation probabilities
                   ↓
          threshold trial-and-error
                   ↓
           entity-level F0.5
                   ↓
             ERROR ANALYSIS
                   ↓
             final selection
```

---

# 29. Final Engineering Principles

1. `train_source1.tsv` has no target column.
2. The target is created at candidate-pair level.
3. This is binary classification, not regression.
4. Do not remove duplicate business names or addresses.
5. Normalize instead of blindly deleting records.
6. Evaluate cleaning quality before blocking.
7. Split by Source 1 entity.
8. Prevent train/validation leakage.
9. Candidate recall is a hard ceiling on recoverable recall.
10. Use multi-pass scalable blocking.
11. Use hard negatives.
12. Use XGBoost as the primary pairwise classifier.
13. Use the T4 GPUs for training and controlled experiments.
14. Optimize model configurations using downstream entity-level F0.5.
15. Never assume threshold = 0.5.
16. Determine threshold through validation trial-and-error.
17. Re-run threshold search after meaningful model changes.
18. Never force exactly one match per Source 1.
19. Treat country as an open-set feature.
20. Final matches must be a subset of final candidates.
21. Validate outputs before submission.
22. Keep experiments reproducible and traceable.

---

# 30. Final Goal

Create a competition-ready Kaggle pipeline that starts from:

```text
/kaggle/input/<dataset-folder>/dataset/
```

and produces:

```text
output/matching_results.tsv
output/candidate_pairs.tsv
```

through:

```text
Data Understanding
→
Data Cleaning
→
Cleaning Quality Evaluation
→
Train/Validation Split
→
Candidate Generation
→
Candidate Recall Evaluation
→
Blocking Experiments
→
Feature Engineering
→
Training Pair Construction
→
Hard-Negative Sampling
→
XGBoost Training
→
XGBoost Parameter Trial-and-Error
→
Validation Prediction
→
Threshold Trial-and-Error
→
Macro F0.5 Optimization
→
Error Analysis
→
Iterative Improvement
→
Test Inference
→
Submission Validation
```

The final blocker, features, XGBoost parameters, negative-sampling strategy, and threshold must be chosen from **actual validation experiments** rather than assumptions.
