# 🚀 AWS SageMaker Execution Guide
## Business Entity Resolution — Error_404_Not_Found

This guide walks you through running the full pipeline on **Amazon SageMaker**
by cloning this repository. Follow every step in the exact order listed.

---

## 📋 Prerequisites

Before starting, make sure you have:

| Requirement | Details |
|-------------|---------|
| AWS Account | With SageMaker, S3, and IAM access |
| GitHub Repository | This project pushed to GitHub (e.g., `https://github.com/<your-username>/ml_challenge`) |
| S3 Bucket | Create one to store data and artifacts (e.g., `ml-challenge-2026`) |
| SageMaker Studio or Notebook Instance | Launched with `ml.m5.2xlarge` or larger |

---

## STEP 1 — Create an S3 Bucket

Open the **AWS Console → S3 → Create bucket**.

```
Bucket name : ml-challenge-2026        ← choose a unique name
Region      : us-east-1               ← match your SageMaker region
Block public access : ON (keep default)
```

Click **Create bucket**.

---

## STEP 2 — Launch SageMaker Studio or Notebook Instance

### Option A: SageMaker Studio (Recommended)
1. Go to **AWS Console → Amazon SageMaker → Studio**.
2. Click **Open Studio**.
3. Select **JupyterLab** and open a **Terminal**.

### Option B: SageMaker Notebook Instance
1. Go to **AWS Console → Amazon SageMaker → Notebook Instances**.
2. Click **Create notebook instance**.
3. Instance type: `ml.m5.2xlarge` (32 GB RAM — handles large TSV files).
4. Click **Create** → wait until status is **InService** → click **Open JupyterLab**.
5. Open a **Terminal** from JupyterLab.

---

## STEP 3 — Clone the Repository in SageMaker Terminal

In the SageMaker terminal, run:

```bash
# Navigate to home directory
cd ~

# Clone your GitHub repository
git clone https://github.com/<your-username>/ml_challenge.git

# Enter the project root
cd ml_challenge/Error_404_Not_Found_submission
```

> ⚠️ Replace `<your-username>` with your actual GitHub username.
> If your repo is private, you will need a GitHub Personal Access Token (PAT):
> ```bash
> git clone https://oauth2:<YOUR_PAT>@github.com/<your-username>/ml_challenge.git
> ```

---

## STEP 4 — Verify the Folder Structure

After cloning, confirm the structure is correct:

```bash
ls -la ~/ml_challenge/Error_404_Not_Found_submission/
```

**Expected output:**
```
Documentation_template.md
GUIDE.md
README.md
code/
dataset/
output/
utils/
```

```bash
ls ~/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution/src/
```

**Expected output:**
```
__init__.py   blocking.py   cleaning.py   config.py    evaluation.py
features.py   ground_truth.py   inference.py   io_utils.py   pair_dataset.py
pipeline.py   splitting.py   submission.py   threshold.py   training.py
```

---

## STEP 5 — Install Python Dependencies

```bash
# Navigate to the code directory
cd ~/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution

# Install all required packages
pip install -r requirements.txt
```

This installs: `pandas`, `numpy`, `scipy`, `scikit-learn`, `rapidfuzz`,
`xgboost`, `boto3`, `sagemaker`, `pyyaml`.

**Expected time: ~2–3 minutes**

Verify installation:
```bash
python3 -c "import pandas, numpy, xgboost, rapidfuzz, sklearn; print('All packages OK')"
```

---

## STEP 6 — Verify Dataset Files Are Present

The dataset TSV files are large and should be in the cloned repo (they were committed).
Confirm they exist:

```bash
ls -lh ~/ml_challenge/Error_404_Not_Found_submission/dataset/train/
ls -lh ~/ml_challenge/Error_404_Not_Found_submission/dataset/test/
```

**Expected files:**
```
dataset/train/
  train_source1.tsv       ~200 MB
  train_source2.tsv       ~467 MB
  train_source3.tsv       ~480 MB
  train_ground_truth.tsv  ~121 MB

dataset/test/
  test_source1.tsv        ~167 MB
  test_source2.tsv        ~486 MB
  test_source3.tsv        ~483 MB
```

> ⚠️ If the TSV files are missing (e.g., they were excluded from git via `.gitignore`),
> upload them from your local machine using the AWS CLI:
> ```bash
> # On your LOCAL machine (not SageMaker):
> aws s3 sync /home/bharath/Desktop/projects/ml_challenge/Error_404_Not_Found_submission/dataset \
>     s3://ml-challenge-2026/dataset/
>
> # Then in SageMaker terminal:
> aws s3 sync s3://ml-challenge-2026/dataset/ \
>     ~/ml_challenge/Error_404_Not_Found_submission/dataset/
> ```

---

## STEP 7 — Quick Sanity Check (Optional but Recommended)

Before running the full pipeline, test with a small sample to verify
all modules load and the pipeline runs end-to-end without errors:

```bash
cd ~/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution

python3 src/pipeline.py \
    --data-dir   ../../dataset \
    --output-dir ../../output \
    --debug-sample 500
```

**What `--debug-sample 500` does:**
- Loads only the first 500 Source 1 entities from training
- Runs the full pipeline end-to-end (cleaning → blocking → features → train → threshold → inference → submission)
- Takes ~5–10 minutes instead of hours
- Validates the output files automatically

**Expected output (last few lines):**
```
=== Stage 15: Generating submission files ===
matching_results.tsv written: ../../output/matching_results.tsv (500 rows)
candidate_pairs.tsv written:  ../../output/candidate_pairs.tsv (500 rows)
Official validator: PASS
Pipeline complete.
```

---

## STEP 8 — Run the Full Pipeline

Once the sanity check passes, run on the **complete dataset**:

```bash
cd ~/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution

python3 src/pipeline.py \
    --data-dir   ../../dataset \
    --output-dir ../../output
```

> ⏱️ **Expected runtime:** 3–8 hours depending on instance type.
> Use `ml.m5.4xlarge` (64 GB RAM) for faster blocking on the full dataset.

### Pipeline Stages — What Happens When You Run

| # | Stage | Description | Approx. Time |
|---|-------|-------------|-------------|
| 1 | Load data | Read all 7 TSV files | 5–15 min |
| 2 | Clean data | Normalize names, addresses, country | 10–20 min |
| 3 | Split train/val | 80/20 split by S1 entity ID | < 1 min |
| 4 | Generate candidates | Multi-pass blocking (TF-IDF + exact match) | 60–120 min |
| 5 | Evaluate blocking recall | Count ground-truth pairs captured | 5 min |
| 6 | Build training pairs | Label positives + sample negatives | 5–10 min |
| 7 | Build features | Extract ~40 pairwise features | 30–60 min |
| 8 | Train XGBoost | Binary classifier training | 10–30 min |
| 9 | Score validation | Predict probabilities on val set | 5 min |
| 10 | Optimize threshold | Sweep t ∈ [0.10, 0.99] → best F0.5 | 5–10 min |
| 11 | Error analysis | FP/FN/singleton analysis | < 1 min |
| 12 | Generate test candidates | Blocking on test data | 60–120 min |
| 13 | Build test features | Feature extraction for test pairs | 30–60 min |
| 14 | Test inference | Predict + apply threshold | 5 min |
| 15 | Create submission files | Write TSVs + run validator | < 1 min |

---

## STEP 9 — Verify Submission Output

After the pipeline completes:

```bash
# Check output files exist and are non-empty
ls -lh ~/ml_challenge/Error_404_Not_Found_submission/output/

# Preview matching_results.tsv (first 10 rows)
head -10 ~/ml_challenge/Error_404_Not_Found_submission/output/matching_results.tsv

# Preview candidate_pairs.tsv (first 10 rows)
head -10 ~/ml_challenge/Error_404_Not_Found_submission/output/candidate_pairs.tsv
```

**Expected preview of matching_results.tsv:**
```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S3-00812
S1-00002	S3-00004
S1-00003	
```

---

## STEP 10 — Run the Official Challenge Validator

```bash
cd ~/ml_challenge/Error_404_Not_Found_submission

python3 utils/validate_submission.py \
    --matching  output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir  dataset/test
```

**Expected output:**
```
PASS
```

If you see any issues, fix them before uploading to the leaderboard.

---

## STEP 11 — Save Outputs to S3 (Backup)

Save your results to S3 so they are not lost if the SageMaker instance stops:

```bash
aws s3 sync \
    ~/ml_challenge/Error_404_Not_Found_submission/output/ \
    s3://ml-challenge-2026/output/

echo "Outputs backed up to S3 ✓"
```

---

## STEP 12 — Download matching_results.tsv for Leaderboard Submission

### Option A: Download from SageMaker Studio
In JupyterLab file browser, navigate to:
```
ml_challenge/Error_404_Not_Found_submission/output/matching_results.tsv
```
Right-click → **Download**.

### Option B: Download from S3 Console
1. Go to **AWS Console → S3 → ml-challenge-2026 → output/**.
2. Click `matching_results.tsv` → **Download**.

### Option C: Download via AWS CLI (on your local machine)
```bash
# On your LOCAL machine:
aws s3 cp s3://ml-challenge-2026/output/matching_results.tsv ./matching_results.tsv
```

---

## ⚙️ Optional — Environment Variables for Configuration

You can override any pipeline setting via environment variables **before** running:

```bash
# Use more TF-IDF candidates for higher blocking recall (slower)
export NAME_TOP_K=100
export ADDRESS_TOP_K=100

# Change validation split ratio
export VAL_SPLIT_RATIO=0.15

# Change negative sampling ratio
export NEG_RATIO=5

# Set log level
export LOG_LEVEL=DEBUG

# Then run the pipeline
python3 src/pipeline.py --data-dir ../../dataset --output-dir ../../output
```

---

## ⚠️ Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'rapidfuzz'` | Run `pip install -r requirements.txt` again |
| `FileNotFoundError: File not found: dataset/train/train_source1.tsv` | Dataset not cloned — sync from S3 (see Step 6) |
| `MemoryError` or kernel crashes | Upgrade instance to `ml.m5.4xlarge` or `ml.r5.2xlarge` (64 GB RAM) |
| `Official validator: FAIL` | Check the error message — usually a missing S1 entity or invalid ID |
| Pipeline hangs at blocking stage | Normal for full dataset — blocking takes 1–2 hours on 2xlarge |
| `git clone` fails (private repo) | Use a GitHub PAT token (see Step 3) |

---

## 📁 File Execution Order Summary

```
STEP 3  → git clone (one-time setup)
STEP 5  → pip install -r requirements.txt (one-time setup)
STEP 7  → python3 src/pipeline.py --debug-sample 500  (sanity check)
STEP 8  → python3 src/pipeline.py  (full run)
STEP 10 → python3 utils/validate_submission.py  (validation)
```

**The only file you need to run is: `src/pipeline.py`**  
It orchestrates all 15 stages automatically in the correct order.
