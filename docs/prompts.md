You are a senior machine learning engineer building a competition-ready Business Entity Resolution system for Amazon ML Challenge 2026.

IMPORTANT EXECUTION RULE:
DO NOT execute anything.

Do NOT:

* run Python
* run shell commands
* run tests
* install packages
* access AWS
* access S3
* access the internet
* execute notebooks
* execute SageMaker jobs
* generate fake results
* invent dataset statistics
* create fake predictions

Your job in this prompt is ONLY to create the project structure and initial source files.

The final submission MUST maintain this exact structure:

team_name_submission/
│
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

CRITICAL:

* Do NOT create additional root-level directories.
* Do NOT create notebooks outside src/.
* Do NOT create separate config/ folders.
* Do NOT create models/ folders.
* Do NOT create data/ folders.
* Do NOT create artifacts/ folders.
* All reusable Python source code must live inside:
  code/business_entity_resolution/src/
* Runtime data, models, intermediate artifacts, logs, and experiments should later be stored in S3 or temporary SageMaker storage, NOT added as new root folders.
* output/ must contain only the two required submission files.
* Documentation_template.md must remain at the submission root.

Create:

code/business_entity_resolution/src/

with these initial Python modules:

**init**.py
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

Also create:

code/business_entity_resolution/README.md
code/business_entity_resolution/requirements.txt
Documentation_template.md

Create empty placeholder files for:

output/matching_results.tsv
output/candidate_pairs.tsv

Do not put fake headers/data into the final output files.

Design all modules so they can later be executed in Amazon SageMaker.

The project must support:

* local development
* S3 data access
* SageMaker execution
* deterministic experiments
* reproducible inference
* final competition packaging

Use clean modular architecture.

Do not implement the full ML logic yet.

DO NOT EXECUTE ANYTHING.





prompt 2:
Continue the existing Amazon ML Challenge Business Entity Resolution project.

IMPORTANT:
DO NOT execute anything.
Only create or edit code files.

Do NOT:

* run Python
* run shell commands
* access AWS
* access S3
* install packages
* run tests
* generate fake data
* invent statistics

All code MUST remain inside:

code/business_entity_resolution/src/

The final root structure must remain unchanged.

IMPLEMENT:

1. src/config.py

Create a centralized configuration system.

It must support:

* AWS region
* S3 bucket
* S3 project prefix
* training S1 path
* training S2 path
* training S3 path
* training ground-truth path
* test S1 path
* test S2 path
* test S3 path
* random seed
* validation split ratio
* TF-IDF settings
* candidate top-K settings
* negative sampling settings
* XGBoost hyperparameters
* threshold search settings
* runtime options

Do not hard-code the user's actual bucket name.

Support environment variables and/or a configuration dataclass so the same source code can run in SageMaker without modification.

2. src/io_utils.py

Implement:

* read TSV from local path
* read TSV from s3:// path
* write TSV locally
* write TSV to S3
* validate expected columns
* helper functions for creating runtime directories
* safe handling of NaN and empty strings

All TSV files MUST use:

sep="\t"

3. src/cleaning.py

Implement reusable cleaning/normalization functions.

Create:

normalize_text()
normalize_business_name()
normalize_business_address()
tokenize_text()
extract_numeric_tokens()
extract_digit_sequences()

For names create:

* business_name_norm
* business_name_core

For addresses create:

* business_address_norm

Preserve original raw columns.

Name normalization should handle:

* lowercase
* Unicode normalization
* punctuation
* whitespace
* common formatting noise

Name core normalization should remove common legal suffixes such as:

* ltd
* limited
* pvt
* private
* private limited
* inc
* incorporated
* corp
* corporation
* llp
* co
* company

Do NOT aggressively remove meaningful words.

Address normalization should:

* preserve numbers
* normalize punctuation
* normalize whitespace
* preserve useful alphanumeric information

Create missingness indicators:

* name_missing
* address_missing
* country_missing

Create basic length/token-count information where useful.

IMPORTANT COUNTRY RULE:
Treat country as an OPEN-SET STRING FEATURE.

Never:

* hard-code US/India
* filter to US/India
* assume test countries are known from training
* one-hot encode only US and India

The test set contains France according to the challenge statement.

4. src/pipeline.py

Create a lightweight orchestration interface but do not yet implement the full pipeline.

Provide clearly named callable functions for:

* load_training_data()
* clean_training_data()
* load_test_data()
* clean_test_data()

Do not execute them.

5. requirements.txt

Add only required packages for this pipeline, including appropriate versions where practical.

Expected core packages:

* pandas
* numpy
* scipy
* scikit-learn
* rapidfuzz
* xgboost
* boto3
* sagemaker
* pyyaml

Do not add unnecessary ML frameworks.

6. README.md

Write a concise initial project description and explain:

* exact folder structure
* SageMaker execution model
* S3 storage model
* local code authoring model

Do not claim any metrics.

DO NOT EXECUTE ANYTHING.







prompt 3:
Continue the existing project.

IMPORTANT:
DO NOT execute anything.
Only create/edit source code.

All code must remain under:

code/business_entity_resolution/src/

Do not create new root-level folders.

IMPLEMENT:

1. src/ground_truth.py

The challenge ground truth contains:

source1_entity_id
matched_entity_ids

matched_entity_ids is a comma-separated list and may be empty.

Implement:

parse_ground_truth()
expand_ground_truth_pairs()
build_ground_truth_sets()
validate_ground_truth_ids()

Convert:

S1-A → S2-X,S3-Y,S2-Z

into:

S1-A,S2-X,1
S1-A,S3-Y,1
S1-A,S2-Z,1

Correctly handle empty matched_entity_ids.

Validate that ground-truth matched IDs belong only to Source 2 or Source 3.

2. src/splitting.py

Implement an entity-level validation split.

CRITICAL:
Split Source 1 entity IDs, NOT individual pair rows.

The same Source 1 entity must never occur in both training and validation.

Use:

* configurable validation fraction
* deterministic random seed

Return:
train_source1_ids
validation_source1_ids

Also create helper functions to filter source data and ground truth by Source 1 IDs.

3. Prevent leakage.

Do not allow:

* ground-truth labels to influence candidate generation
* validation labels to influence training
* entity IDs to become predictive features
* matched_entity_ids to become a model feature
* test data to influence training
* test data to influence threshold selection

Ground truth may ONLY be used to:

* label training candidates
* label validation candidates
* calculate validation metrics
* measure candidate recall

4. Add validation documentation/comments explaining why the split is by Source 1 entities.

5. Update README.md with the validation methodology.

Do not calculate or claim any actual metric yet.

DO NOT EXECUTE ANYTHING.









prompt 4:
Continue the existing Amazon ML Challenge entity-resolution project.

IMPORTANT:
DO NOT execute anything.
Only create/edit source code.

All code must remain under:
code/business_entity_resolution/src/

Do not create new root-level directories.

IMPLEMENT src/blocking.py.

Goal:
For every Source 1 entity, efficiently generate plausible candidates from Source 2 and Source 3 without performing full Cartesian pair comparison.

Implement a MULTI-PASS BLOCKING system.

Blocking passes:

PASS 1:
Exact normalized business name.

PASS 2:
Exact core business name.

PASS 3:
Exact normalized business address.

PASS 4:
Character-level TF-IDF nearest neighbors on business_name_norm.

PASS 5:
Character-level TF-IDF nearest neighbors on business_address_norm.

PASS 6:
Optional informative-token blocking where useful.

TF-IDF defaults must be configurable:

* analyzer = char
* configurable ngram_range
* configurable min_df
* sublinear_tf where appropriate

Initial candidate retrieval values must be configurable:
name_top_k = 50
address_top_k = 50

Do NOT hard-code them in functions.

The blocker must run independently for:
S1 → S2
S1 → S3

Each candidate must retain:

* source1_entity_id
* candidate_entity_id
* candidate_source

The candidate system must:

* union candidates from all blocking passes
* remove duplicates
* never include Source 1 IDs as candidates
* never generate invalid IDs
* remain deterministic
* maintain stable ordering
* avoid O(N²) full comparison for large datasets

IMPORTANT:
The final candidate set is the set that will actually be passed to the ML model.

Implement functions for:

generate_exact_name_candidates()
generate_exact_core_name_candidates()
generate_exact_address_candidates()
generate_tfidf_name_candidates()
generate_tfidf_address_candidates()
generate_token_candidates()
generate_candidates()

Also implement candidate statistics:

candidate_recall()
candidate_count_statistics()
average_candidates_per_entity()
median_candidates_per_entity()
max_candidates_per_entity()
reduction_ratio()

Candidate recall must be calculated AFTER candidate generation using training ground truth.

Ground truth MUST NOT be used to create candidates.

Candidate recall is:

true positive ground-truth pairs present in candidates
/
all true positive ground-truth pairs

Also calculate recall separately for:

* S2
* S3

And analyze singleton Source 1 entities separately where useful.

Implement configuration-driven experiments for:
name_top_k = 20, 50, 100, 200
address_top_k = 20, 50, 100, 200

Do not choose a final K automatically based on invented values.

The actual best blocking configuration will later be selected using real validation experiments.

Update README.md with blocking methodology.

DO NOT EXECUTE ANYTHING.



prompt 5:
Continue the project.

IMPORTANT:
DO NOT execute anything.
Only create/edit files under:

code/business_entity_resolution/src/

Do not create additional root directories.

IMPLEMENT:

A. src/pair_dataset.py
B. src/features.py

PAIR DATASET:

Implement:

* create_positive_pairs()
* label_candidates()
* sample_easy_negatives()
* sample_hard_negatives()
* build_training_pairs()
* build_validation_pairs()

Positive labels come strictly from training ground truth.

Negative examples must include:

1. Random/easy negatives
2. High name similarity but incorrect match
3. High address similarity but incorrect match
4. Same country but incorrect match
5. TF-IDF top-K candidate but not ground truth
6. Similar numeric/address information but incorrect match

Initial configurable negative ratio:
1 positive : 3–5 negatives

Do NOT use every possible negative pair.

Do not remove true positives during negative sampling.

Validation candidate pairs should NOT be artificially downsampled if it would distort the real candidate population.

PAIRWISE FEATURES:

For every S1-candidate pair, create numerical features.

NAME FEATURES:

* name_exact
* name_core_exact
* name_char_tfidf
* name_token_tfidf
* name_levenshtein_similarity
* name_ratio
* name_token_sort_ratio
* name_token_set_ratio
* name_jaccard
* name_length_difference
* name_length_ratio
* name_common_token_count
* name_containment
* name_missing

ADDRESS FEATURES:

* address_char_tfidf
* address_token_tfidf
* address_levenshtein_similarity
* address_ratio
* address_token_sort_ratio
* address_token_set_ratio
* address_jaccard
* address_length_difference
* address_length_ratio
* address_common_token_count
* address_containment
* address_missing

NUMERIC ADDRESS FEATURES:

* numeric_token_overlap
* numeric_token_jaccard
* numeric_token_count_difference
* useful exact numeric sequence indicators

COUNTRY:

* country_match
* country_missing

STRUCTURAL:

* source_is_s2
* source_is_s3
* same_first_character
* common_token_count where appropriate

INTERACTION FEATURES:

* name_similarity * address_similarity
* minimum(name_similarity, address_similarity)
* maximum(name_similarity, address_similarity)

Use RapidFuzz for string similarities.

Use sparse TF-IDF features efficiently.

Do not use:

* raw entity IDs as features
* matched_entity_ids as features
* external data
* external embeddings
* geocoding
* internet lookup

Preserve metadata separately:

* source1_entity_id
* candidate_entity_id
* candidate_source
* label

Implement deterministic feature-column ordering.

Store feature names/schema metadata in a runtime artifact location configured through src/config.py.

Do not create a new root directory for feature artifacts.

The code should support:

* training feature creation
* validation feature creation
* test feature creation

The feature definitions MUST be identical across training and inference.

Update README.md with feature-engineering methodology.

DO NOT EXECUTE ANYTHING.



prompt 6:
Continue the project.

IMPORTANT:
DO NOT execute anything.
Only write/edit code.

All source code must remain inside:

code/business_entity_resolution/src/

No new root-level directories.

IMPLEMENT src/training.py.

Use XGBoost as the main pairwise binary classifier.

Model task:

Input:
S1 record + S2/S3 candidate record

Output:
probability that both records refer to the same real-world business

Training labels:
1 = true match
0 = non-match

CRITICAL DATA-LEAKAGE RULES:

* never use entity IDs as ML features
* never use matched_entity_ids as an ML feature
* never use validation labels during training
* never use test data during training
* never generate negatives using validation ground truth in a way that leaks validation information
* do not train on test candidates
* keep metadata separate from numerical features

Implement SageMaker-managed XGBoost training using the SageMaker Python SDK.

The source code must support:

1. Preparing training feature data
2. Preparing validation feature data
3. Uploading feature datasets to S3
4. Creating SageMaker XGBoost Estimator
5. Configuring training/validation channels
6. Setting model output path
7. Launching training jobs
8. Saving experiment configuration
9. Returning model artifact information

Do not hard-code AWS credentials.

Use SageMaker execution role/IAM role resolution.

DO NOT create a permanent real-time endpoint unless necessary.
The competition can use batch/offline inference.

Initial configurable hyperparameters:

objective = binary:logistic
eval_metric = aucpr

max_depth = 6
eta = 0.05
min_child_weight = 3
subsample = 0.8
colsample_bytree = 0.8
gamma = 0
reg_alpha = 0.1
reg_lambda = 2
num_round = 300

These are ONLY baseline values.

Implement staged hyperparameter experiments, rather than a giant Cartesian grid.

Search candidates for:
max_depth: 4,5,6,7,8
eta: 0.03,0.05,0.08,0.10
min_child_weight: 1,3,5,10
subsample: 0.7,0.8,0.9,1.0
colsample_bytree: 0.7,0.8,0.9,1.0
gamma: 0,0.1,0.3
reg_alpha: 0,0.1,0.5
reg_lambda: 1,2,5
num_round: 200,300,500

Do not automatically run every combination.

The model-selection system should record:

* experiment ID
* parameters
* model artifact
* training job name
* validation artifact location

AUC-PR may be used as a model diagnostic.

BUT:
The ultimate model selection metric for the competition is entity-level macro F0.5, calculated separately by the evaluation module.

Add support for class imbalance handling through configurable negative sampling and optional scale_pos_weight.

Do not assume a fixed imbalance ratio before seeing the actual data.

Add reproducibility:

* random seed
* deterministic feature order
* deterministic preprocessing

Update README.md.

DO NOT EXECUTE ANYTHING.


prompt 7:
Continue the existing project.

IMPORTANT:
DO NOT execute anything.
Only create/edit source code.

All source code must remain under:

code/business_entity_resolution/src/

No new root-level directories.

IMPLEMENT:

src/evaluation.py
src/threshold.py

COMPETITION METRIC:

F0.5 =
(1.25 * precision * recall) /
(0.25 * precision + recall)

The competition score is calculated:

1. separately for each Source 1 entity
2. then macro-averaged across all Source 1 entities

Implement exact set-based entity-level evaluation.

For each Source 1 entity:

actual IDs = ground-truth matched IDs
predicted IDs = model-selected candidate IDs

Calculate:

* TP
* FP
* FN
* precision
* recall
* F0.5

Singleton rule:
If actual matches are empty:

* empty prediction = score 1.0
* any predicted match = score 0.0

Implement:
entity_level_precision()
entity_level_recall()
entity_level_fbeta()
macro_f05()
evaluate_predictions_by_entity()
singleton_metrics()
candidate_recall()
reduction_ratio()

DO NOT use only:

* accuracy
* global F1
* micro F0.5
* ROC-AUC

The official optimization target is macro entity-level F0.5.

THRESHOLD OPTIMIZATION:

The XGBoost model outputs a probability for each candidate pair.

Do NOT assume threshold = 0.5.

Implement threshold sweep:

0.10 to 0.99

with configurable step, initially:
0.01

For EVERY threshold:

1. mark probability >= threshold as predicted match
2. group predictions by Source 1 entity
3. permit multiple matches
4. permit zero matches
5. calculate macro F0.5
6. calculate macro precision
7. calculate macro recall
8. calculate singleton correctness
9. count predicted matches
10. count false positives
11. count false negatives

Select:

best_threshold =
threshold producing maximum validation macro F0.5

Do NOT hard-code the final threshold.

Also implement optional source-specific threshold analysis:

threshold_S2
threshold_S3

But global threshold must remain the baseline.

Only choose source-specific thresholds if validation demonstrates improvement.

Generate runtime reports for:

* threshold
* macro F0.5
* macro precision
* macro recall
* singleton metric
* number of predicted matches
* false positives
* false negatives

Implement error analysis for:

* high-confidence false positives
* false negatives
* singleton false positives

Preserve useful pairwise feature values in the error analysis.

Do not fabricate results.

The system must save the selected threshold to a runtime S3/artifact location controlled by configuration.

The code should allow a future SageMaker job to load the threshold and use it for test inference.

Update README.md with the threshold-selection methodology.

DO NOT EXECUTE ANYTHING.








prompt 8:
Continue the Amazon ML Challenge Business Entity Resolution project.

IMPORTANT:
DO NOT execute anything.
DO NOT run tests.
DO NOT run SageMaker jobs.
DO NOT access AWS.
DO NOT access S3.
DO NOT install packages.
DO NOT generate fake predictions.
DO NOT create fake metrics.
ONLY create/edit source code and documentation.

All source code must remain under:

code/business_entity_resolution/src/

Do not create additional root-level folders.

FINAL REQUIRED STRUCTURE:

team_name_submission/
│
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

IMPLEMENT FINAL INFERENCE.

src/inference.py

Test-time process:

1. Load test_source1.tsv
2. Load test_source2.tsv
3. Load test_source3.tsv
4. Apply exactly the same cleaning/normalization logic as training
5. Apply the final validated blocking configuration
6. Generate the LAST candidate set that will be passed into the model
7. Generate exactly the same pairwise features used during training
8. Load the trained SageMaker XGBoost model artifact
9. Predict candidate-match probabilities
10. Load the threshold selected on held-out validation
11. Apply threshold
12. Allow zero, one, or many matches per Source 1 entity
13. Create final match lists

CRITICAL:
Do NOT train on test data.
Do NOT use any test labels.
Do NOT change feature definitions.
Do NOT change preprocessing rules.
Do NOT use external data.

IMPLEMENT src/submission.py.

Generate:

output/matching_results.tsv

EXACT columns:
source1_entity_id
matched_entity_ids

Rules:

* exactly one row for every Source 1 test entity
* preserve all test Source 1 IDs
* only Source 2 and Source 3 IDs can appear in matched_entity_ids
* no Source 1 self-matches
* no duplicate matched IDs
* comma-separated matched IDs
* empty string for singleton/no-match predictions
* tab-separated file
* no extra columns

Generate:

output/candidate_pairs.tsv

EXACT columns:
source1_entity_id
candidate_entity_ids

This MUST be the exact LAST candidate set passed to the ML model during inference.

Do NOT output:

* an earlier blocking set
* raw TF-IDF candidates
* a larger intermediate set

The following must always hold:

final_matches ⊆ final_candidates

Ensure:

* every final matched ID exists in the test S2/S3 datasets
* every candidate ID exists in the test S2/S3 datasets
* no duplicates
* deterministic ordering
* every S1 test entity appears exactly once

IMPLEMENT src/pipeline.py.

Create a master end-to-end callable pipeline with stages:

load_data
clean_data
split_training_validation
generate_candidates
evaluate_candidate_recall
build_training_pairs
build_features
train_xgboost
score_validation
optimize_threshold
perform_error_analysis
generate_test_candidates
build_test_features
run_test_inference
create_submission_files

Each stage should be callable independently.

The master pipeline should support configuration-driven execution in SageMaker.

IMPORTANT:
Do not duplicate business logic between stages.
Call functions from the appropriate modules.

VALIDATION:

Create code that can invoke the official challenge validator:

utils/validate_submission.py

using:

python3 utils/validate_submission.py 
--matching output/matching_results.tsv 
--candidate output/candidate_pairs.tsv 
--test-dir dataset/test

Do NOT create a fake validator.

Do NOT bundle a modified copy of the official validator.

The code should simply make it easy for the user to invoke the official validator from the challenge's student_resource environment.

README.md:

Complete the README and include:

1. Challenge objective
2. Exact project structure
3. AWS/SageMaker execution architecture
4. S3 storage architecture
5. Required dependencies
6. Configuration
7. Data loading
8. Data cleaning
9. Ground-truth parsing
10. Validation split
11. Blocking
12. Candidate recall
13. Hard-negative construction
14. Feature engineering
15. XGBoost training
16. Hyperparameter experiments
17. F0.5 calculation
18. Threshold optimization
19. Error analysis
20. Test inference
21. Output file generation
22. Official submission validation
23. Reproducibility
24. How to reproduce end-to-end

Do not claim metrics or thresholds that have not actually been measured.

requirements.txt:

Pin versions where practical and keep dependencies limited to what the code actually imports.

Documentation_template.md:

Fill it with a structured methodology template covering:

* Methodology used
* Candidate generation/blocking strategy
* Model architecture
* Feature engineering
* Validation methodology
* Threshold methodology
* Error analysis
* Other relevant implementation details

Do not invent final results.

FINAL CODE QUALITY REQUIREMENTS:

* Python 3 compatible
* type hints
* docstrings
* logging
* deterministic random seeds
* robust error handling
* no hard-coded personal paths
* no hard-coded AWS credentials
* no external APIs
* no external business lookup
* no external geocoding
* no external enrichment
* no external internet data
* no LLM web lookup
* no raw entity IDs as ML features
* no ground-truth leakage
* stable feature-column ordering
* stable output ordering

FINAL SUBMISSION CONSTRAINT:

Do not create additional root-level directories.

The only intended runtime output directory is:

output/

The only code directory is:

code/business_entity_resolution/src/

Everything else needed for runtime should be stored in S3 or temporary SageMaker storage.

DO NOT EXECUTE ANYTHING.
