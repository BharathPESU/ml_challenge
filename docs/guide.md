The challenge is an entity-resolution problem. Source 1 is the reference source, and one Source 1 entity can match zero, one, or many records from Source 2 and Source 3. The final prediction is therefore not a normal single-label row classification task; we create candidate pairs and classify each candidate pair as match/non-match. The supplied challenge also requires matching_results.tsv and candidate_pairs.tsv, and final matches must be a subset of the candidate set.

1. Understand the challenge before touching AWS
1.1 What are the input files?

Training:

train_source1.tsv
train_source2.tsv
train_source3.tsv
train_ground_truth.tsv

Test:

test_source1.tsv
test_source2.tsv
test_source3.tsv

All source files are tab-separated. Always use:

pd.read_csv(path, sep="\t")

Do not use the default CSV comma separator.

The source files contain:

entity_id
business_name
business_address
country

The ground truth contains:

source1_entity_id
matched_entity_ids

The challenge explicitly states that country is an open-set string field: training has US and India, while test also contains France. Do not build a pipeline that only supports US and India.

2. What is the ML problem?
2.1 It is not regression

You are not predicting a continuous number.

2.2 It is not ordinary multiclass classification

You do not predict:

restaurant
hospital
shop
...
2.3 It is a pairwise binary classification problem inside an entity-resolution pipeline

For a pair:

Source 1 record
        +
Source 2 or Source 3 candidate

predict:

0 = not the same business
1 = the same business

The complete system is:

S1
 |
 | candidate generation
 v
S2/S3 candidates
 |
 | pairwise feature engineering
 v
XGBoost
 |
 | probability
 v
P(match)
 |
 | threshold selected on validation data
 v
final matches

There is no target column inside train_source1.tsv.

The target is created from train_ground_truth.tsv after candidate pairs are generated.

3. Recommended final folder structure

The challenge requires a specific final submission structure. Keep that structure from the beginning.

team_name_submission/
│
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
│
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── io_utils.py
│       │   ├── cleaning.py
│       │   ├── ground_truth.py
│       │   ├── splitting.py
│       │   ├── blocking.py
│       │   ├── pair_dataset.py
│       │   ├── features.py
│       │   ├── evaluation.py
│       │   ├── threshold.py
│       │   ├── training.py
│       │   ├── inference.py
│       │   ├── submission.py
│       │   └── pipeline.py
│       │
│       ├── README.md
│       └── requirements.txt
│
└── Documentation_template.md
Important

Do not create:

data/
models/
features/
notebooks/
artifacts/

at the root of the final submission package if you want to preserve the required structure.

These are runtime concerns, and their large artifacts should live in S3.

Your actual Python source code belongs under:

code/business_entity_resolution/src/

Your final competition files belong under:

output/
4. Recommended S3 structure

S3 "folders" are actually object-key prefixes. They look like directories in the AWS console, but S3 stores objects under keys such as:

dataset/train/train_source1.tsv

The AWS documentation describes these as prefixes rather than real directories.

Recommended bucket structure:

your-ml-challenge-bucket/
│
├── dataset/
│   │
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   │
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
│
├── code/
│   └── business_entity_resolution/
│       └── src/...
│
└── runtime/
    └── <run-id>/
        ├── cleaned/
        ├── candidates/
        ├── features/
        ├── models/
        ├── predictions/
        ├── reports/
        └── output/
Recommended rule

Treat:

dataset/

as read-only challenge input.

Do not overwrite the original challenge files.

Write processed versions somewhere like:

runtime/<run-id>/cleaned/

or, if you prefer a stable structure:

processed/train/
processed/test/

For experiments, a run ID is safer because you can compare runs without overwriting previous results.

5. Step 0 — Choose one AWS Region

Pick one region and keep the entire workflow there:

S3
SageMaker Studio
SageMaker Processing
SageMaker Training
Batch Transform

Use the same region wherever possible.

This is especially important for Data Wrangler integrations and SageMaker workflows that consume S3 data.

Example:

ap-south-1

or another region you intentionally choose.

Do not create the bucket in one region and the SageMaker resources in another unless you have a specific reason.

6. Step 1 — Create the S3 bucket
AWS Console navigation

Go to:

AWS Console
→ S3
→ General purpose buckets
→ Create bucket

Give it a unique name, for example:

amazon-ml-challenge-2026-yourname

Bucket names are globally unique, so your actual name will need to be different.

Keep:

Block all public access = ON

Do not make your challenge data public.

Create the prefixes

After opening the bucket:

Objects
→ Create folder

Create:

dataset/

Then:

dataset/train/
dataset/test/

S3 folders are implemented as common object-key prefixes.

7. Step 2 — Upload the challenge datasets

Inside:

dataset/train/

upload:

train_source1.tsv
train_source2.tsv
train_source3.tsv
train_ground_truth.tsv

Inside:

dataset/test/

upload:

test_source1.tsv
test_source2.tsv
test_source3.tsv

The S3 URI will look like:

s3://YOUR_BUCKET/dataset/train/train_source1.tsv

and:

s3://YOUR_BUCKET/dataset/test/test_source1.tsv
Important

In Python you use:

s3://bucket-name/prefix/file.tsv

not:

https://...

The S3 URI is the correct way to point SageMaker/Data Wrangler at an object.

8. Step 3 — Set up IAM permissions

This is the part that often confuses beginners.

Think of IAM like this:

YOU
 |
 | sign in to AWS
 v
AWS account
 |
 v
SageMaker
 |
 | assumes
 v
SageMaker Execution Role
 |
 +----> S3
 +----> SageMaker Training
 +----> SageMaker Processing
 +----> Model artifacts

Your SageMaker code normally accesses AWS resources using the SageMaker execution role, not by putting an AWS access key and secret key inside Python.

Never hard-code:

AWS_ACCESS_KEY_ID = "..."
AWS_SECRET_ACCESS_KEY = "..."

inside your project.

9. Create a SageMaker execution role

AWS Console:

AWS Console
→ IAM
→ Roles
→ Create role

Under trusted entity:

AWS service

Choose the SageMaker execution use case.

AWS currently documents creating a SageMaker AI execution role from IAM and associating it with a SageMaker domain, user profile, or notebook environment.

For a first working setup, AWS documentation describes AmazonSageMakerFullAccess as a possible managed policy. However, that managed policy does not automatically grant unrestricted S3 object access to every arbitrary bucket name. For a custom bucket such as your competition bucket, you normally need additional S3 permissions.

10. Give the role access to your challenge bucket
Beginner-friendly approach

Create a custom S3 policy restricted to your competition bucket.

Replace:

YOUR_BUCKET

with your actual bucket.

Example policy:

{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BucketAccess",
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket",
        "s3:GetBucketLocation"
      ],
      "Resource": "arn:aws:s3:::YOUR_BUCKET"
    },
    {
      "Sid": "ObjectAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:AbortMultipartUpload"
      ],
      "Resource": "arn:aws:s3:::YOUR_BUCKET/*"
    }
  ]
}

This permits SageMaker to:

List bucket contents
Read objects
Write processed/model/output objects

If your workflow specifically needs deletion, add:

s3:DeleteObject

only when necessary.

Avoid giving delete permission simply because it is convenient.

11. How to attach the policy

AWS Console:

IAM
→ Roles
→ select your SageMaker execution role
→ Permissions
→ Add permissions
→ Create inline policy

Choose:

JSON

paste the policy, replace YOUR_BUCKET, review and save.

12. Find the SageMaker execution role later

In SageMaker, your role may appear as:

arn:aws:iam::123456789012:role/AmazonSageMaker-ExecutionRole-...

You can print the active role from SageMaker:

from sagemaker import get_execution_role

role = get_execution_role()

print(role)

This only works when the notebook is running in an appropriate SageMaker environment.

Do not copy AWS credentials into the notebook.

13. SageMaker Studio setup

For a beginner, use the current SageMaker Studio/JupyterLab experience for your code.

AWS Console:

SageMaker AI
→ Studio

If you already have a domain:

select your domain

Then create or open a user profile/space.

Current SageMaker documentation describes JupyterLab spaces as the environment where your code, environment and notebooks run. A JupyterLab space uses SageMaker compute and persistent EBS-backed storage.

Create a JupyterLab space

Inside Studio:

JupyterLab
→ Create JupyterLab space

Choose:

Private

unless you explicitly need a shared space.

Choose a reasonable instance.

For your challenge:

ml.m5.xlarge

is a reasonable starting point for lightweight work.

For larger preprocessing/feature generation:

ml.m5.2xlarge

or larger may be more appropriate.

Do not immediately choose an expensive GPU instance.

This problem is primarily tabular/string similarity work and XGBoost; a GPU is not the first thing I would spend money on.

14. Connect your local Antigravity project to AWS

You have two separate concepts:

Antigravity
= code authoring on your laptop

SageMaker
= cloud execution

That separation is good.

The simplest workflow is:

Antigravity
    ↓
edit Python files
    ↓
upload project/code to S3
    ↓
SageMaker
    ↓
run the code

You can also keep the code in GitHub and use S3 only for data/artifacts. That is generally better for version control, but it is not required for the competition.

15. How to upload your project to S3

Example final project:

team_name_submission/

Upload:

code/
Documentation_template.md

to:

s3://YOUR_BUCKET/code/

You do not have to upload the final empty output/ files before inference.

Those will be generated after the pipeline runs.

16. Use S3 URIs in your Python code

Example:

BUCKET = "YOUR_BUCKET"

TRAIN_S1 = (
    f"s3://{BUCKET}/dataset/train/train_source1.tsv"
)

TRAIN_S2 = (
    f"s3://{BUCKET}/dataset/train/train_source2.tsv"
)

TRAIN_S3 = (
    f"s3://{BUCKET}/dataset/train/train_source3.tsv"
)

GROUND_TRUTH = (
    f"s3://{BUCKET}/dataset/train/train_ground_truth.tsv"
)

TEST_S1 = (
    f"s3://{BUCKET}/dataset/test/test_source1.tsv"
)

TEST_S2 = (
    f"s3://{BUCKET}/dataset/test/test_source2.tsv"
)

TEST_S3 = (
    f"s3://{BUCKET}/dataset/test/test_source3.tsv"
)

Then:

df = pd.read_csv(
    TRAIN_S1,
    sep="\t",
    dtype=str
)
17. TASK 1 — Data understanding in SageMaker

Before cleaning, inspect the data.

Start with:

import pandas as pd

df = pd.read_csv(
    "s3://YOUR_BUCKET/dataset/train/train_source1.tsv",
    sep="\t",
    dtype=str
)

print(df.shape)
print(df.columns.tolist())
display(df.head())

Do the same for Source 2 and Source 3.

18. What to check

For each source:

row count
column count
column names
data types
missing values
empty strings
duplicate entity IDs
duplicate names
duplicate addresses
country distribution
name length
address length

Example:

print(df.isna().sum())
print(df["entity_id"].duplicated().sum())
print(df["country"].value_counts())
19. Check duplicate names correctly

This is an entity-resolution challenge.

Do NOT do this:

df.drop_duplicates("business_name")

A repeated business name does not necessarily mean the same entity.

Two different businesses can both be:

Primary Care Group

Your job is to determine whether the full records refer to the same real-world entity.

20. TASK 2 — Data cleaning

The challenge data may be structurally usable while still containing semantic noise.

Cleaning for this task means:

raw business name
        ↓
normalized business name

raw address
        ↓
normalized address

Do not destroy the original columns.

Keep:

business_name
business_address

and add:

business_name_norm
business_name_core
business_address_norm
21. Name normalization

Typical steps:

lowercase
Unicode normalization
punctuation normalization
whitespace normalization
conjunction normalization

Example:

"ABC, Pvt. Ltd."

becomes approximately:

"abc pvt ltd"

Then create a core-name representation:

"abc"

after removing trailing legal suffixes such as:

ltd
limited
pvt
private limited
corp
corporation
inc
incorporated
llp
company
co

Only remove known legal suffixes from the end of the name. Do not aggressively delete tokens everywhere.

22. Address normalization

Typical steps:

lowercase
Unicode normalization
punctuation normalization
whitespace normalization

Preserve numbers.

For example:

"12, M.G. Road, Bengaluru"

can become:

"12 mg road bengaluru"

Do not remove:

12
560001
B12
101

Address numbers can be strong matching signals.

23. Create numeric-address features

Extract numbers from the address:

import re

def extract_numeric_tokens(text):
    return re.findall(r"\d+", str(text))

Example:

"12 MG Road Bangalore 560001"

becomes:

["12", "560001"]

Later calculate:

number_overlap
numeric_jaccard
shared_number_count
24. Data Wrangler — what it is

Amazon SageMaker Data Wrangler is a visual data-preparation tool for importing, analyzing, transforming and exporting data.

AWS currently documents Data Wrangler with S3 import support, sampling options, transformations, analyses and S3 export. Data Wrangler is also integrated into SageMaker Canvas in the newer experience; the exact UI can depend on the SageMaker experience enabled in your account.

Use Data Wrangler when you want to visually inspect and perform understandable transformations.

For this challenge, do not assume the entire ML pipeline should be implemented visually.

A good division is:

Data Wrangler
    ↓
visual inspection + basic data preparation

SageMaker Processing / Python
    ↓
large-scale normalization + blocking + feature engineering

SageMaker XGBoost
    ↓
model training
25. Import your TSV into Data Wrangler

The challenge files are TSV.

AWS Data Wrangler supports tab as a delimiter for CSV-style tabular import.

The general flow is:

SageMaker
→ Studio / Data Wrangler experience
→ Import data
→ Amazon S3

Then provide:

s3://YOUR_BUCKET/dataset/train/train_source1.tsv

or browse the S3 bucket and prefix.

When configuring the file:

File type = CSV-style tabular
Delimiter = Tab
First row = header

The exact labels can differ slightly in the current UI.

AWS documents S3 import, sampling and configurable delimiters including Tab.

26. Data Wrangler sampling

Data Wrangler lets you:

None
First K
Randomized
Stratified

For huge datasets, sampling is useful for exploration.

For example:

100,000 rows

can be enough to inspect patterns before running expensive full-data transformations.

But understand this:

A sample is not the complete dataset.

If a transformation learns parameters from data, fitting it only on a sample can produce a different result from fitting it on the complete dataset.

AWS documents that some transforms based on learned parameters can be refit on the entire dataset during export.

27. Very important for your challenge: do not accidentally train from a Data Wrangler sample

Suppose you inspect 50,000 rows in Data Wrangler and create:

TF-IDF
categorical encoding
text features

using that sample.

That does not automatically mean you have a correct full-dataset preprocessing pipeline.

For the final pipeline:

training preprocessing

must be fitted using the intended training data, with validation/test held out appropriately.

Do not leak validation/test information into fitting.

28. Data Wrangler cleaning flow

A beginner-friendly first flow can be:

S3 source
   ↓
Data Wrangler
   ↓
Data Types
   ↓
Missing value analysis
   ↓
String/text normalization
   ↓
Remove obvious malformed rows only if justified
   ↓
Create derived columns
   ↓
Export

For this challenge, however, I would keep advanced entity-resolution logic in Python so that the process is reproducible and easy to submit.

29. What cleaning should you actually do?

For these business records, focus on:

Business name
lowercase
Unicode normalization
punctuation normalization
whitespace normalization
legal suffix normalization
Address
lowercase
Unicode normalization
punctuation normalization
whitespace normalization
number extraction
Country
trim whitespace
case normalization

Do not replace France with an unknown token simply because it was not present in training.

Treat country as an open-set value.

30. Export cleaned data from Data Wrangler to S3

After building the flow:

Data Wrangler flow
    ↓
add destination
    ↓
Amazon S3

or use:

Export to
→ Amazon S3 via Jupyter Notebook

AWS documents both destination-node processing and export via a generated Jupyter notebook.

For a destination node, Data Wrangler can create a SageMaker Processing job that runs the transformation and writes the result to S3.

This is the important mental model:

S3 input
   ↓
SageMaker Processing
   ↓
cleaned data
   ↓
S3 output
31. Recommended S3 path for cleaned data

Do not overwrite:

dataset/train/train_source1.tsv

Instead use:

runtime/<run-id>/cleaned/train_source1.tsv
runtime/<run-id>/cleaned/train_source2.tsv
runtime/<run-id>/cleaned/train_source3.tsv
runtime/<run-id>/cleaned/test_source1.tsv
runtime/<run-id>/cleaned/test_source2.tsv
runtime/<run-id>/cleaned/test_source3.tsv

Keep the original dataset untouched.

32. Full-data processing for large files

Your files are hundreds of MB each.

For those sizes, I would not build the final pipeline around dragging full datasets through an interactive Data Wrangler session.

Use Data Wrangler mainly for:

EDA
sampling
visual transformation design
validation of transformations

Then run the full transformation using:

SageMaker Processing

AWS describes SageMaker Processing as a managed environment in which input data is downloaded from S3, processed, and outputs are uploaded back to S3.

Conceptually:

S3:
dataset/train/*
       ↓
SageMaker Processing
       ↓
/opt/ml/processing/input
       ↓
Python transformation
       ↓
/opt/ml/processing/output
       ↓
S3:
runtime/.../cleaned/

This is much safer for full-dataset processing.

33. Do not load everything into memory unnecessarily

For large TSV files:

pd.read_csv(...)

can consume much more RAM than the file's size on disk.

For basic inspection, use:

pd.read_csv(
    path,
    sep="\t",
    dtype=str,
    nrows=100000
)

For chunked processing:

for chunk in pd.read_csv(
    path,
    sep="\t",
    dtype=str,
    chunksize=100000
):
    # process chunk
    pass

This allows you to process large files without holding the entire raw dataset in memory.

34. TASK 3 — Train / validation split

Do not split arbitrary candidate rows.

Split by Source 1 entity:

Source 1 entities
       ↓
80%
training entities

20%
validation entities

The same Source 1 ID must not appear in both.

Why?

Because evaluation is defined per Source 1 entity.

If a Source 1 entity appears in both training and validation pairs, your evaluation can become artificially optimistic.

35. TASK 4 — Candidate generation / blocking

This is one of the most important stages.

If there are:

millions of Source 1 records
millions of Source 2/3 records

you cannot compare every possible pair.

Instead:

S1 record
   ↓
blocking
   ↓
small candidate set
   ↓
ML scoring
36. Multi-pass blocking

Start with several candidate generators.

Pass 1

Exact normalized name.

Pass 2

Exact core name.

Pass 3

Exact normalized address.

Pass 4

Shared informative name token.

Pass 5

Shared address token.

Pass 6

Numeric-address agreement.

Pass 7

Character similarity inside manageable blocks.

Union the candidate sets.

Name candidates
      ∪
Address candidates
      ∪
Numeric candidates
      ∪
Token candidates
      ↓
final candidate set
37. Why candidate recall matters

Suppose the true match is:

S1-A → S2-X

but your blocker produces:

S1-A → S2-Y, S2-Z

Then XGBoost will never see X.

Therefore:

candidate generation recall

is the upper bound on final recall.

The challenge explicitly emphasizes this.

For validation:

candidate recall =
true matches present in candidate set
/
all true matches

Aim for very high candidate recall while keeping candidate counts manageable.

38. TASK 5 — Candidate recall experiments

Run experiments such as:

name K = 20
name K = 50
name K = 100
name K = 200

and:

address K = 20
address K = 50
address K = 100
address K = 200

Measure:

candidate recall
average candidates per S1
median candidates per S1
95th percentile candidates per S1
maximum candidates per S1
runtime

Do not blindly choose K=200.

The correct configuration is the one that gives excellent recall without making the feature/model stage unnecessarily huge.

39. TASK 6 — Feature engineering

After blocking, create a row per:

S1 + candidate

Example:

S1-001
S2-100

becomes one ML observation.

40. Name features

Useful features:

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
common_token_count
41. Address features

Useful features:

address_exact
address_char_similarity
address_levenshtein
address_token_sort_ratio
address_token_set_ratio
address_jaccard
address_containment
address_length_difference
address_length_ratio
42. Numeric-address features

Useful features:

numeric_overlap
numeric_jaccard
shared_number_count
number_count_difference

Example:

12 MG Road Bangalore
12 MG Road Bengaluru

shares an important number.

43. Country features

Use:

country_match

For example:

country_match = (
    left_country.strip().lower()
    ==
    right_country.strip().lower()
)

Do not one-hot only:

US
India

because France occurs in the test set.

44. Interaction features

Useful combinations include:

name_similarity * address_similarity
min(name_similarity, address_similarity)
max(name_similarity, address_similarity)

A candidate with:

name = 0.96
address = 0.91

is much stronger evidence than:

name = 0.96
address = 0.20
45. TASK 7 — Training pair construction

From ground truth:

S1-A → S2-X,S3-Y

create:

S1-A,S2-X,1
S1-A,S3-Y,1

Any generated candidate that is not in the ground truth becomes:

label = 0
46. Use hard negatives

Random negatives are often too easy.

A better negative set includes:

same country
similar name
wrong address
very similar address
wrong name
high TF-IDF similarity
but not a true match
similar numeric address
but different business

This teaches the model to avoid false merges.

The official metric is precision-heavy, so hard negatives are especially valuable.

47. TASK 8 — Train XGBoost

The main model should be:

XGBoost binary classifier

Input:

pairwise features

Output:

P(match)

A reasonable baseline is:

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

These are starting values, not guaranteed optimal values.

48. Do not tune the model using accuracy

Your competition metric is:

F0.5

not:

accuracy

and not simply:

ROC-AUC

Use AUC-PR as a model diagnostic, but select the final system using the competition-style entity-level F0.5.

49. TASK 9 — Threshold optimization

This is extremely important.

XGBoost may produce:

S1-A → S2-X = 0.96
S1-A → S2-Y = 0.84
S1-A → S3-Z = 0.42

Do not automatically say:

probability >= 0.5

Use validation data to search for the threshold.

For example:

0.10
0.11
0.12
...
0.99

For every threshold:

probability >= threshold
        ↓
predicted matches
        ↓
entity-level macro F0.5

The best threshold is:

threshold with maximum validation macro F0.5

There is no universal "correct" threshold such as 0.5 or 0.8.

50. Understand the F0.5 metric

The challenge uses:

F0.5 = (1.25 × Precision × Recall)
       /
       (0.25 × Precision + Recall)

and computes the score separately for each Source 1 entity before taking the macro average.

This means precision matters more than recall.

In practical terms:

false merge

is especially dangerous.

51. Singleton handling

Suppose:

S1-123

has no real matches.

Correct prediction:

S1-123

with an empty matched_entity_ids.

Do not force one candidate to be selected.

Correct empty predictions are valuable under this competition's metric.

52. TASK 10 — Test inference

After:

blocking
+
features
+
trained XGBoost
+
validated threshold

run the test data.

The flow is:

test_source1
      +
test_source2/test_source3
      ↓
blocking
      ↓
candidate_pairs
      ↓
pairwise features
      ↓
XGBoost
      ↓
probabilities
      ↓
validated threshold
      ↓
final matches

Do not use test labels because they are not provided.

Do not retrain the model on test data.

53. Generate candidate_pairs.tsv

The file must have:

source1_entity_id
candidate_entity_ids

Example:

S1-00001    S2-00047,S2-00193,S3-00812
S1-00002    S3-00004
S1-00003

Crucially:

It must represent the final candidate set that was actually passed to the ML model.

Do not save an earlier blocking set and call it the final candidate set.

54. Generate matching_results.tsv

The file must have:

source1_entity_id
matched_entity_ids

Example:

S1-00001    S2-00047,S3-00812
S1-00002    S3-00004
S1-00003

Rules:

one row per test Source 1 entity
empty string if no match
no duplicate IDs
only S2/S3 IDs
only IDs that actually exist in the test data
55. Store results in S3

Store a copy of the final outputs under:

runtime/<run-id>/output/

For example:

s3://YOUR_BUCKET/runtime/20260926_120000/output/matching_results.tsv
s3://YOUR_BUCKET/runtime/20260926_120000/output/candidate_pairs.tsv

Keep your challenge submission copy under the required local:

team_name_submission/output/
56. SageMaker Processing vs. notebook code

As a beginner, it helps to separate these.

Notebook

Use it to:

orchestrate
inspect
launch jobs
inspect results
SageMaker Processing

Use it for:

large-scale cleaning
large-scale candidate generation
feature generation

SageMaker Training

Use it for:

XGBoost training
Batch Transform

Use it for:

large-scale model inference

This gives:

Notebook
   |
   +--> Processing
   |
   +--> Training
   |
   +--> Batch Transform

instead of forcing your notebook kernel to do everything.

57. SageMaker Processing input/output mental model

SageMaker Processing works like:

S3 input
   ↓
Processing container
   ↓
/opt/ml/processing/input/
   ↓
your Python program
   ↓
/opt/ml/processing/output/
   ↓
S3 output

You define:

ProcessingInput
ProcessingOutput

in the SageMaker Python SDK.

This is particularly useful for your large TSV files.

58. Which instance should I use?

Start conservatively.

SageMaker Studio
ml.m5.xlarge

for:

notebook control
small experiments
EDA
SageMaker Processing

For full-data cleaning/blocking/features, use a larger memory instance if required:

ml.m5.2xlarge

or larger.

XGBoost

Start with:

ml.m5.xlarge

and scale if training is slow or memory is insufficient.

The correct choice depends on the actual size of the generated candidate/feature dataset.

Do not assume the raw TSV size directly equals the memory requirement.

59. Do not run expensive resources continuously

AWS costs are primarily driven by resource usage.

Good habit:

Start resource
    ↓
do work
    ↓
save outputs to S3
    ↓
stop/release resource

S3 remains available even after compute stops.

For a beginner:

S3 = durable storage
SageMaker = compute

Remember this distinction.

60. Recommended S3 runtime layout

Use a unique run ID:

runtime/
└── 20260926_120000_ab12cd/
    │
    ├── cleaned/
    │
    ├── split/
    │
    ├── candidates/
    │
    ├── features/
    │
    ├── models/
    │
    ├── predictions/
    │
    ├── reports/
    │
    └── output/

The runtime output might contain:

cleaned_source1
candidate_recall_report
feature_schema
train_features
validation_features
model.tar.gz
validation_predictions
threshold_report
test_predictions
matching_results.tsv
candidate_pairs.tsv
61. Use a configuration file in your code

Inside:

code/business_entity_resolution/src/config.py

define the configuration.

For example:

S3_BUCKET = "YOUR_BUCKET"

TRAIN_PREFIX = "dataset/train"
TEST_PREFIX = "dataset/test"

RUNTIME_PREFIX = "runtime"

RANDOM_SEED = 42

VALIDATION_SIZE = 0.20

NAME_TOP_K = 50
ADDRESS_TOP_K = 50

THRESHOLD_MIN = 0.10
THRESHOLD_MAX = 0.99
THRESHOLD_STEP = 0.01

Do not hard-code these values throughout multiple files.

62. Recommended source code responsibilities
config.py

Configuration.

io_utils.py

S3/local read/write.

cleaning.py

Normalization.

ground_truth.py

Ground-truth parsing.

splitting.py

Entity-level train/validation split.

blocking.py

Candidate generation.

pair_dataset.py

Positive and negative pair construction.

features.py

Pairwise similarity features.

evaluation.py

Candidate recall and entity-level F0.5.

threshold.py

Threshold sweep and selection.

training.py

SageMaker XGBoost training.

inference.py

Test inference.

submission.py

Final output generation.

pipeline.py

End-to-end orchestration.

63. Keep the notebook simple

Your:

ml_challenge.ipynb

should act as the controller.

Conceptually:

1. configure
2. load data
3. run quality checks
4. preprocess
5. generate candidates
6. evaluate candidate recall
7. build features
8. launch XGBoost
9. evaluate validation
10. find threshold
11. run test inference
12. generate outputs
13. validate submission

Do not put 2,000 lines of repeated ML logic into the notebook.

The reusable logic belongs in src/.

64. Training data format for XGBoost

Your final training matrix should conceptually be:

label
name_exact
name_core_exact
name_similarity
address_similarity
numeric_overlap
country_match
...

The entity IDs are metadata, not model features.

Do not train on:

entity_id

because IDs are identifiers, not meaningful business attributes.

Do not train on:

matched_entity_ids

because that would be direct ground-truth leakage.

65. Validation data must remain untouched

The validation set should be used to answer:

How well does my pipeline generalize?

Do not use validation labels to fit normalization parameters or the model.

Use validation labels only for:

candidate recall evaluation
model evaluation
threshold selection
error analysis
66. Model improvement loop

Your first working system is not expected to be perfect.

Use this loop:

Baseline
   ↓
Validation F0.5
   ↓
Inspect false positives
   ↓
Inspect false negatives
   ↓
Improve blocker/features
   ↓
Retrain
   ↓
Threshold sweep
   ↓
Validation F0.5

Keep an experiment table:

experiment_id
blocking_config
feature_version
negative_ratio
xgboost_params
threshold
candidate_recall
F0.5
precision
recall
runtime

This prevents losing track of which experiment worked.

67. Error analysis: what to inspect
False positives

Model says:

match

but ground truth says:

not match

These are especially important because F0.5 penalizes false merges.

False negatives

Model says:

not match

but ground truth says:

match

Look for patterns such as:

strong name similarity but weak address
strong address similarity but weak name
transliteration
word order change
typo
abbreviation
missing address component

Then add useful features/blocking logic based on real observed failures.

68. How to choose the final threshold

Do not choose:

0.5

just because it is the default classification threshold.

Do not choose:

0.8

because someone told you it is good for entity resolution.

Instead:

Validation scores
       ↓
threshold = 0.10
       ↓
F0.5
threshold = 0.11
       ↓
F0.5
...
threshold = 0.99
       ↓
F0.5
       ↓
choose maximum

That is your data-driven threshold.

69. Optional: separate S2 and S3 thresholds

After the global threshold works, test:

threshold_S2
threshold_S3

because different sources may have different noise characteristics.

But do not introduce complexity unless held-out validation shows a real improvement.

Start with one global threshold.

70. Data Wrangler or Python?

For your challenge, I recommend:

Task	Tool
S3 storage	S3
Visual EDA	Data Wrangler
Basic transformation design	Data Wrangler
Full large-file preprocessing	SageMaker Processing + Python
Blocking	Python/SageMaker Processing
Feature engineering	Python/SageMaker Processing
Model training	SageMaker XGBoost
Validation	Python
Threshold optimization	Python
Test inference	SageMaker Batch Transform or managed inference
Submission generation	Python
Final storage	S3
Reproducible code	src/

This avoids using a visual tool for logic that needs to be reproducible and competition-ready.

71. If Data Wrangler gives an AccessDenied error

Check in this order:

1. Which IAM role is SageMaker using?
2. Does that role have s3:ListBucket?
3. Does it have s3:GetObject?
4. Does it have s3:PutObject?
5. Is the bucket in the expected AWS Region?
6. Does the bucket policy deny the role?
7. Is the object encrypted using a customer-managed KMS key?
8. If yes, does the role have the required KMS permissions?

AWS documents that Data Wrangler uses an IAM execution role for access to data sources.

Do not randomly add AdministratorAccess just because one operation failed.

72. If SageMaker cannot see your S3 bucket

Test from the SageMaker notebook:

import boto3

s3 = boto3.client("s3")

response = s3.list_objects_v2(
    Bucket="YOUR_BUCKET",
    Prefix="dataset/train/",
    MaxKeys=10
)

print(response.get("Contents", []))

If this fails with:

AccessDenied

the problem is usually IAM or a bucket/KMS policy, not pandas.

73. If pandas fails to read the TSV

Check:

pd.read_csv(
    "s3://YOUR_BUCKET/dataset/train/train_source1.tsv",
    sep="\t",
    dtype=str
)

and verify:

sep="\t"

not:

sep=","

Also inspect:

df.columns

If you get one giant column, the delimiter is probably wrong.

74. If the SageMaker notebook runs out of memory

Do not immediately assume the algorithm is wrong.

Reduce memory pressure:

chunked reading
process S2 and S3 separately
save intermediate results to S3
use sparse TF-IDF matrices
avoid building the full Cartesian pair table
use SageMaker Processing with a larger instance

For this challenge, avoiding all-pairs comparison is critical.

75. If training becomes too slow

Check the size of your candidate set first.

The sequence is:

too many candidates
      ↓
too many feature rows
      ↓
too much training time

Do not immediately jump to a larger training instance.

First ask:

Can blocking reduce unnecessary candidates?

Candidate generation is one of the highest-impact engineering decisions in this challenge.

76. If your F0.5 is low

Debug in this order:

1. Candidate recall
2. Hard-negative quality
3. Feature quality
4. XGBoost model
5. Threshold

Do not start with hyperparameter tuning immediately.

If candidate recall is 85%, a perfect classifier still cannot recover the missing 15%.

77. Final validation

Before submitting:

output/matching_results.tsv
output/candidate_pairs.tsv

must satisfy the challenge rules.

Check:

one row per Source 1 test entity
no duplicate Source 1 IDs
no duplicate candidate IDs
no duplicate match IDs
only S2/S3 IDs
all IDs exist in test data
final matches ⊆ candidates
TSV format

Then run the official validator supplied by the challenge:

python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

Do not replace this validator with your own approximation.

Your own checks are useful, but the official challenge validator is the final formatting check.

78. Final submission package

Keep exactly:

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

Zip it:

<team_name>_submission.zip

Do not add huge datasets to the final ZIP unless the challenge specifically asks for them.

The challenge wants the code to be reproducible from the supplied data, not a giant copy of the raw data.

79. Recommended beginner workflow

Do not try to learn the whole AWS platform first.

Follow this sequence.

Day/Phase 1 — AWS basics

Learn only:

S3
IAM role
SageMaker Studio
JupyterLab

Goal:

SageMaker notebook can read one TSV from S3
Phase 2 — Data

Goal:

all seven challenge files
        ↓
SageMaker
        ↓
schema + quality report
Phase 3 — Cleaning

Goal:

raw fields
   ↓
normalized fields

Store the cleaned results in S3.

Phase 4 — Validation split

Goal:

training S1 entities
validation S1 entities

Verify there is no overlap.

Phase 5 — Blocking

Goal:

candidate pairs

Then measure:

candidate recall
Phase 6 — Features

Goal:

candidate pair
      ↓
numeric feature vector
Phase 7 — XGBoost

Goal:

features
   ↓
XGBoost
   ↓
probability
Phase 8 — F0.5

Goal:

validation probabilities
        ↓
threshold sweep
        ↓
best validation F0.5
Phase 9 — Test

Goal:

test candidates
      ↓
model
      ↓
threshold
      ↓
matching_results.tsv
candidate_pairs.tsv
Phase 10 — Submit

Goal:

official validator
       ↓
PASS
       ↓
final ZIP
       ↓
leaderboard
80. What you should NOT do

Do not:

❌ use external business databases
❌ call geocoding APIs
❌ search businesses online
❌ use external entity-resolution APIs
❌ leak ground truth into candidate generation
❌ split individual pair rows randomly
❌ drop duplicate names blindly
❌ force exactly one match per Source 1
❌ force every Source 1 entity to have a match
❌ use entity_id as a model feature
❌ hard-code country to US/India
❌ assume threshold = 0.5
❌ compare every S1 with every S2/S3
❌ keep expensive SageMaker instances running when not needed
❌ put AWS secret keys in Python

The challenge explicitly prohibits external entity lookup and data enrichment and requires the final model to meet its licensing/parameter constraints.

81. Your complete architecture
                         AWS
                         │
                ┌────────┴─────────┐
                │                  │
                ▼                  ▼
               S3              SageMaker
                │                  │
        ┌───────┴───────┐     ┌────┴─────┐
        │               │     │          │
      dataset/        runtime/ JupyterLab Data Wrangler
        │               │     │          │
        │               │     └────┬─────┘
        │               │          │
        │               │          ▼
        │               │      data prep
        │               │
        └───────────────┴──────────┐
                                   ▼
                            SageMaker Processing
                                   │
                          ┌────────┼────────┐
                          │        │        │
                     cleaning  blocking  features
                          │        │        │
                          └────────┼────────┘
                                   ▼
                                   S3
                                   │
                                   ▼
                           SageMaker XGBoost
                                   │
                                   ▼
                                model
                                   │
                                   ▼
                            validation score
                                   │
                                   ▼
                            threshold sweep
                                   │
                                   ▼
                              best F0.5
                                   │
                                   ▼
                              test data
                                   │
                                   ▼
                           Batch inference
                                   │
                           ┌───────┴───────┐
                           ▼               ▼
                  matching_results    candidate_pairs
                           │               │
                           └───────┬───────┘
                                   ▼
                              S3 / output/
                                   │
                                   ▼
                           Official validator
                                   │
                                   ▼
                           Final submission
82. The three layers to remember

When you get confused, return to these three layers.

Layer 1 — Storage
S3

Stores:

raw data
cleaned data
features
models
predictions
outputs
Layer 2 — Compute
SageMaker

Runs:

preprocessing
blocking
feature engineering
training
inference
Layer 3 — Code
Antigravity

Used to write:

Python modules
configuration
pipeline
README
documentation

This separation will make AWS much easier to understand.

83. Your first practical milestone

Do not start with XGBoost.

First achieve:

[ ] S3 bucket created
[ ] dataset/train uploaded
[ ] dataset/test uploaded
[ ] SageMaker execution role configured
[ ] SageMaker can access S3
[ ] SageMaker JupyterLab opens
[ ] train_source1.tsv loads correctly
[ ] train_source2.tsv loads correctly
[ ] train_source3.tsv loads correctly
[ ] train_ground_truth.tsv loads correctly
[ ] basic data-quality report works

Only after this should you move to:

cleaning

Then:

blocking

Then:

features

Then:

XGBoost

That order will save you a lot of confusion.

84. Official AWS references

Use the AWS documentation as the source of truth when the console UI changes.

Amazon S3

AWS S3 user guide:

Creating and organizing S3 folders/prefixes
Uploading objects

https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-folders.html

SageMaker execution roles

https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html

SageMaker JupyterLab

https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-jl-user-guide.html

Creating a JupyterLab space

https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-jl-user-guide-create-space.html

SageMaker Data Wrangler — import

https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler-import.html

SageMaker Data Wrangler — transformations

https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler-transform.html

SageMaker Data Wrangler — export to S3

https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler-data-export.html

SageMaker Processing

https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html

SageMaker XGBoost

https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-how-to-use.html

85. Final checklist

Before you start ML:

AWS SETUP
[ ] One AWS Region selected
[ ] S3 bucket created
[ ] Block Public Access enabled
[ ] Train data uploaded
[ ] Test data uploaded
[ ] SageMaker domain ready
[ ] JupyterLab space ready
[ ] SageMaker execution role identified
[ ] S3 access policy attached

DATA:

[ ] TSV delimiter verified
[ ] Schema verified
[ ] Missing values checked
[ ] Duplicate IDs checked
[ ] Duplicate names analyzed
[ ] Country values analyzed
[ ] Name/address normalization created

ML:

[ ] Entity-level validation split
[ ] Multi-pass blocking
[ ] Candidate recall measured
[ ] Hard negatives generated
[ ] Similarity features generated
[ ] XGBoost trained
[ ] Validation probabilities generated
[ ] F0.5 calculated per Source 1 entity
[ ] Threshold optimized
[ ] Singleton behavior checked

FINAL:

[ ] Test candidates generated
[ ] Test features generated
[ ] Test predictions generated
[ ] Threshold applied
[ ] matching_results.tsv created
[ ] candidate_pairs.tsv created
[ ] final matches ⊆ candidates
[ ] official validator = PASS
[ ] final ZIP structure correct
86. The one-sentence mental model

Remember this:

S3 stores the data, SageMaker provides the compute, Data Wrangler helps visually prepare/inspect data, Python implements the reproducible entity-resolution logic, XGBoost learns whether candidate pairs are matches, validation determines the threshold, and S3 stores the resulting artifacts.

For your particular challenge, the most important technical chain is:

clean
→ block
→ measure candidate recall
→ engineer pair features
→ create hard negatives
→ train XGBoost
→ optimize threshold for entity-level macro F0.5
→ infer on test
→ generate required TSV files
→ validate
→ submit

'''

path = Path("/mnt/data/guide.md")
path.write_text(guide, encoding="utf-8")

print(f"Created: {path}")
print(f"Size: {path.stat().st_size:,} bytes")

Analysis errored
from pathlib import Path
Path("/mnt/data/guide.md").write_text("""# Amazon ML Challenge 2026 — Beginner AWS + SageMaker Guide

This guide covers the complete workflow for the Business Entity Resolution challenge.

## 1. ML problem

This is an **entity-resolution problem implemented as pairwise binary classification**.

You compare:

```text
Source 1 record + Source 2/3 candidate

and predict:

1 = same real-world business
0 = different business

There is no target column inside train_source1.tsv. The target relationship is created from train_ground_truth.tsv.

The challenge allows each Source 1 entity to match zero, one, or many Source 2/3 records.

2. Required final submission structure
team_name_submission/
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── config.py
│       │   ├── io_utils.py
│       │   ├── cleaning.py
│       │   ├── ground_truth.py
│       │   ├── splitting.py
│       │   ├── blocking.py
│       │   ├── pair_dataset.py
│       │   ├── features.py
│       │   ├── evaluation.py
│       │   ├── threshold.py
│       │   ├── training.py
│       │   ├── inference.py
│       │   ├── submission.py
│       │   └── pipeline.py
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md

Keep this exact structure. Large runtime files should go to S3, not into new root folders.

3. Recommended S3 structure
YOUR_BUCKET/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── code/
└── runtime/
    └── <run-id>/
        ├── cleaned/
        ├── candidates/
        ├── features/
        ├── models/
        ├── predictions/
        ├── reports/
        └── output/

Use the same AWS Region for S3 and SageMaker.

4. Create S3 bucket

AWS Console:

AWS Console
→ S3
→ General purpose buckets
→ Create bucket

Keep Block all public access enabled.

Create:

dataset/
dataset/train/
dataset/test/

Upload the challenge files into the correct prefixes.

Your Python path will look like:

s3://YOUR_BUCKET/dataset/train/train_source1.tsv

S3 folders are prefixes rather than normal filesystem directories.

5. IAM permissions

SageMaker should access S3 using an IAM execution role, not access keys embedded in code.

Never place:

AWS_ACCESS_KEY_ID = "..."
AWS_SECRET_ACCESS_KEY = "..."

in your project.

Create/find the role:

AWS Console
→ IAM
→ Roles
→ Create role

Choose the SageMaker execution use case.

For a beginner setup, AWS's AmazonSageMakerFullAccess managed policy can be a starting point, but its S3 access does not automatically mean unrestricted access to every custom bucket. Add a bucket-scoped policy for your challenge bucket.

Example:

{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BucketAccess",
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket",
        "s3:GetBucketLocation"
      ],
      "Resource": "arn:aws:s3:::YOUR_BUCKET"
    },
    {
      "Sid": "ObjectReadWrite",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:AbortMultipartUpload"
      ],
      "Resource": "arn:aws:s3:::YOUR_BUCKET/*"
    }
  ]
}

Attach it to the SageMaker execution role.

Inside SageMaker you can check the role with:

from sagemaker import get_execution_role

role = get_execution_role()
print(role)
6. Create SageMaker Studio environment

AWS Console:

SageMaker AI
→ Studio

If your domain already exists, open it.

For JupyterLab:

Studio
→ JupyterLab
→ Create JupyterLab space

Use a private space for individual work.

Start with something like:

ml.m5.xlarge

for the notebook environment.

For heavy full-data processing, use SageMaker Processing with more memory instead of keeping a very large notebook instance running.

7. Antigravity vs SageMaker

Keep this simple:

Antigravity = write code

SageMaker = execute code

S3 = store data/results

Your local source code lives in:

code/business_entity_resolution/src/

Your SageMaker notebook can orchestrate that code.

8. Read TSV from S3

The challenge files are TSV.

Always use:

import pandas as pd

df = pd.read_csv(
    "s3://YOUR_BUCKET/dataset/train/train_source1.tsv",
    sep="\\t",
    dtype=str
)

print(df.shape)
display(df.head())

Do not use the default comma separator.

9. Large-file rule

These challenge files are large enough that you should avoid loading all sources into memory blindly.

For inspection:

df = pd.read_csv(
    path,
    sep="\\t",
    dtype=str,
    nrows=100000
)

For chunk processing:

for chunk in pd.read_csv(
    path,
    sep="\\t",
    dtype=str,
    chunksize=100000
):
    # process chunk
    pass

For full large-data work, prefer SageMaker Processing.

TASK 1 — Data understanding

For all three training and three test source files, inspect:

row count
column names
data types
missing values
empty strings
duplicate entity_id
unique business names
duplicate business names
unique addresses
duplicate addresses
country distribution
name length
address length

Expected source columns:

entity_id
business_name
business_address
country

Verify prefixes:

assert s1["entity_id"].str.startswith("S1-").all()
assert s2["entity_id"].str.startswith("S2-").all()
assert s3["entity_id"].str.startswith("S3-").all()

Do not drop duplicate business names. Multiple businesses can legitimately have the same name.

TASK 2 — Data cleaning

For this challenge, cleaning mostly means normalization, not deleting rows.

Keep the raw columns and create:

business_name_norm
business_name_core
business_address_norm

Also useful:

name_missing
address_missing
country_missing
name_length
address_length
name_token_count
address_token_count
Business-name normalization

Apply:

lowercase
Unicode normalization
punctuation normalization
whitespace normalization

For example:

ABC, Pvt. Ltd.

→

abc pvt ltd

Create business_name_core by carefully removing trailing legal suffixes such as:

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

Do not remove meaningful words globally.

Address normalization

Normalize:

lowercase
Unicode normalization
punctuation
whitespace

Preserve digits such as:

12
560001
101
B12

Extract numeric tokens:

import re

def extract_numeric_tokens(text):
    return re.findall(r"\\d+", str(text))
Country

Do not hard-code country to US/India.

The test set contains France.

Use country as an open-set string value and later derive a pairwise feature such as:

country_match = (
    left_country.strip().lower()
    == right_country.strip().lower()
)
Data Wrangler

Data Wrangler can be used for:

visual inspection
sampling
basic transformations
data-quality exploration

In the newer SageMaker experience, Data Wrangler is integrated into SageMaker Canvas; older documentation also references the Studio Classic experience.

General path:

SageMaker
→ Data Wrangler / Data preparation
→ Import data
→ Amazon S3

Use:

s3://YOUR_BUCKET/dataset/train/train_source1.tsv

For the challenge:

first row = header
delimiter = Tab

AWS Data Wrangler supports Tab as a CSV-style delimiter.

Data Wrangler sampling

For large files, sampling is useful for exploration:

None
First K
Randomized
Stratified

Use a sample to understand the data, but remember:

A sample is not the same as a full-data transformation.

Some transformations learn parameters from data. If you designed a flow using a sample, use Data Wrangler's full-data/refit/export workflow when needed.

Export Data Wrangler output to S3

Preferred flow:

Data Wrangler
→ destination node
→ Amazon S3
→ configure job
→ Run

Data Wrangler can launch a SageMaker Processing job to execute the flow and write the transformed output to S3.

Another option:

Data Wrangler
→ + on node
→ Export to
→ Amazon S3 (via Jupyter Notebook)
Where to store cleaned data

Do not overwrite the original:

dataset/train/
dataset/test/

Use:

runtime/<run-id>/cleaned/

For example:

runtime/run001/cleaned/train_source1.tsv
runtime/run001/cleaned/train_source2.tsv
runtime/run001/cleaned/train_source3.tsv
When to use Data Wrangler vs Python
Task	Tool
Visual EDA	Data Wrangler
Sampling	Data Wrangler
Basic transformation design	Data Wrangler
Full large-file processing	SageMaker Processing
Custom normalization	Python
Blocking	Python / Processing
Feature engineering	Python / Processing
Model training	SageMaker XGBoost
Threshold optimization	Python
Final outputs	Python + S3

For this challenge, keep the real competition logic in src/ so it is reproducible.

TASK 3 — Train/validation split

Split by Source 1 entity, not by candidate rows.

Recommended:

80% Source 1 entities → training
20% Source 1 entities → validation

Use a fixed seed such as:

42

The same Source 1 ID must never occur in both sets.

TASK 4 — Candidate generation / blocking

Do not compare every Source 1 row with every Source 2/3 row.

Generate candidates using multiple passes:

1. exact normalized name
2. exact core name
3. exact normalized address
4. shared informative name token
5. shared informative address token
6. shared numeric/address token
7. fuzzy name retrieval within manageable blocks
8. fuzzy address retrieval within manageable blocks

Union the candidates:

name candidates
      ∪
address candidates
      ∪
numeric candidates
      ∪
token candidates

The challenge specifically emphasizes blocking because it determines the maximum achievable recall.

TASK 5 — Candidate recall

Calculate:

candidate recall =
true matches contained in candidate set
/
all true matches

Also measure:

average candidates/S1
median candidates/S1
95th percentile
maximum
reduction ratio

Try:

K = 20
K = 50
K = 100
K = 200

for fuzzy name/address retrieval.

Aim for very high candidate recall while keeping the candidate set computationally manageable. 98–99%+ is a useful engineering target, not a guaranteed result.

TASK 6 — Feature engineering

For every:

S1 + candidate S2/S3

create features.

Name
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
common_token_count
Address
address_exact
address_char_similarity
address_levenshtein
address_token_sort_ratio
address_token_set_ratio
address_jaccard
address_containment
address_length_difference
address_length_ratio
Numbers
numeric_overlap
numeric_jaccard
shared_number_count
Other
country_match
name_missing
address_missing
source_is_s2
source_is_s3
Interactions
name_similarity * address_similarity
min(name_similarity, address_similarity)
max(name_similarity, address_similarity)

Use RapidFuzz and scikit-learn TF-IDF.

Never use:

entity_id
matched_entity_ids

as model features.

TASK 7 — Training pair construction

For every generated training candidate:

candidate appears in ground truth → label 1
otherwise → label 0

Do not generate the full Cartesian product.

Create hard negatives, such as:

same country + similar name + wrong address
similar address + wrong name
high TF-IDF name score but wrong entity
shared address number + wrong business

Start around:

1 positive : 3–5 negatives
TASK 8 — XGBoost training

Use:

XGBoost binary classifier

Input:

pairwise feature vector

Output:

P(match)

Baseline:

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

Use SageMaker-managed XGBoost training.

Do not select the final system using accuracy alone.

TASK 9 — Threshold optimization

Never assume:

threshold = 0.5

Search:

0.10
0.11
0.12
...
0.99

For each threshold:

probability >= threshold
        ↓
predicted matches
        ↓
group by Source 1
        ↓
calculate macro entity-level F0.5

Choose:

threshold with highest validation macro F0.5

The challenge uses:

F0.5 =
(1.25 × Precision × Recall)
/
(0.25 × Precision + Recall)

and calculates it per Source 1 entity before taking the macro average.

Singletons matter:

actual = []
prediction = []

is the correct result for a singleton.

Do not force exactly one prediction for every Source 1 record.

After the global threshold works, optionally test separate S2/S3 thresholds, but keep them only if validation improves.

TASK 10 — Test inference and submission

Final pipeline:

test S1/S2/S3
      ↓
same normalization
      ↓
same blocking
      ↓
same features
      ↓
trained XGBoost
      ↓
validated threshold
      ↓
final matches

Do not retrain using test labels.

candidate_pairs.tsv
source1_entity_id
candidate_entity_ids

This must be the exact candidate set passed to the final model.

matching_results.tsv
source1_entity_id
matched_entity_ids

Every test Source 1 entity must have exactly one row.

Empty match lists are valid.

Always enforce:

final_matches ⊆ final_candidates
SageMaker Processing

For large cleaning/blocking/feature jobs:

S3 input
   ↓
SageMaker Processing
   ↓
/opt/ml/processing/input
   ↓
your Python code
   ↓
/opt/ml/processing/output
   ↓
S3 output

This is preferable to keeping large datasets entirely in the interactive notebook memory.

SageMaker XGBoost

The training flow is:

S3 feature data
      ↓
SageMaker XGBoost training job
      ↓
model artifact
      ↓
S3

For the built-in tabular XGBoost interface, put the label in the first column of the training CSV and do not include a header in that training CSV.

Recommended S3 runtime layout
s3://YOUR_BUCKET/runtime/<run-id>/

cleaned/
candidates/
features/
models/
predictions/
reports/
output/

This gives you reproducible experiment runs.

Cost control

Remember:

S3 = storage
SageMaker = compute

Good practice:

start compute
→ run task
→ save results to S3
→ stop/release compute

Do not keep expensive notebook instances or endpoints running when not needed.

For this competition, batch/offline inference is usually more appropriate than keeping a permanent real-time endpoint.

Debugging
AccessDenied

Check:

SageMaker execution role
→ s3:ListBucket
→ s3:GetObject
→ s3:PutObject

Then check:

bucket policy
KMS permissions, if applicable
AWS Region
One giant column

Use:

sep="\\t"

The Data Wrangler delimiter should also be Tab.

Out of memory

Use:

chunksize
sparse TF-IDF
SageMaker Processing
larger memory instance

and avoid all-pairs comparison.

Low F0.5

Debug in this order:

1. candidate recall
2. hard negatives
3. feature quality
4. model
5. threshold

A classifier cannot recover a true match that was never generated as a candidate.

Final mental model

When confused, remember:

S3
= stores data

SageMaker Studio
= notebook/work environment

Data Wrangler
= visual data preparation

SageMaker Processing
= large-scale preprocessing

SageMaker XGBoost
= model training

Python evaluation
= F0.5 + threshold selection

S3
= stores final artifacts

And the challenge pipeline is simply:

DATA
→ CLEAN
→ SPLIT
→ BLOCK
→ MEASURE CANDIDATE RECALL
→ BUILD FEATURES
→ BUILD PAIRS
→ TRAIN XGBOOST
→ OPTIMIZE F0.5 THRESHOLD
→ TEST INFERENCE
→ CREATE TSV OUTPUTS
→ VALIDATE
→ SUBMIT
Official AWS documentation
S3 folders/prefixes: https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-folders.html
SageMaker execution roles: https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html
SageMaker JupyterLab: https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-jl-user-guide.html
JupyterLab spaces: https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-jl-user-guide-create-space.html
Data Wrangler import: https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler-import.html
Data Wrangler transformations: https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler-transform.html
Data Wrangler export to S3: https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler-data-export.html
SageMaker Processing: https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html
SageMaker XGBoost: https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-how-to-use.html
""", encoding="utf-8")
print("/mnt/data/guide.md")

I put together the full beginner runbook. One limitation: the file-creation service is currently failing in this session, so I can't attach the .md file itself right now. Below is the complete guide.md content, ready to save as guide.md.

The AWS navigation below is based on the current AWS documentation: SageMaker uses an IAM execution role for access to S3/resources, JupyterLab is launched through a SageMaker Studio space, Data Wrangler can import from S3 and use Tab as a delimiter, and Data Wrangler can export transformations through a SageMaker Processing job.

guide.md — Amazon ML Challenge AWS + SageMaker Beginner Guide