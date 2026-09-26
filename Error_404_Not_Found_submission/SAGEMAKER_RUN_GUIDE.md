# 🚀 How to Run This Project on AWS SageMaker
## Business Entity Resolution — Step-by-Step Guide

---

## 🗺️ Overview: What You Will Do

```
Your Local Machine
    │
    ├── 1. Push repository to GitHub  (already done ✓)
    │
    ↓
AWS Console
    ├── 2. Create S3 bucket
    ├── 3. Launch SageMaker Notebook Instance
    │
    ↓
SageMaker Terminal
    ├── 4. Clone the GitHub repository
    ├── 5. Install dependencies
    ├── 6. Run sanity check  (500 rows, ~10 min)
    ├── 7. Run full pipeline (~4–8 hours)
    ├── 8. Validate output
    └── 9. Download matching_results.tsv → submit to leaderboard
```

---

## PART 1 — AWS Setup (Do This Once)

### Step 1.1 — Create an S3 Bucket

You need S3 only to backup your outputs. All actual computation happens on SageMaker.

1. Go to: **AWS Console → S3 → Create bucket**
2. Fill in:
   ```
   Bucket name  :  ml-challenge-2026        ← pick any unique name
   AWS Region   :  us-east-1
   Block all public access: ON (keep default)
   ```
3. Click **Create bucket**

---

### Step 1.2 — Launch a SageMaker Notebook Instance

1. Go to: **AWS Console → Amazon SageMaker → Notebook instances**
2. Click **Create notebook instance**
3. Fill in:
   ```
   Notebook instance name : ml-challenge-notebook
   Notebook instance type : ml.m5.2xlarge    ← minimum recommended (32 GB RAM)
   Platform identifier    : notebook-al2-v2
   ```
4. Under **IAM role**, select an existing role OR click **Create a new role**
   - Allow S3 access → **Any S3 bucket** → Create role
5. Click **Create notebook instance**
6. Wait 3–5 minutes until the status turns **InService**
7. Click **Open JupyterLab**

> 💡 **Tip:** If the full dataset is too slow on `ml.m5.2xlarge`, upgrade to
> `ml.m5.4xlarge` (64 GB RAM). You can resize the instance without losing data.

---

## PART 2 — Inside SageMaker (Run Everything Here)

### Step 2.1 — Open a Terminal

In JupyterLab:
- Click **File → New → Terminal**

A black terminal window opens. All commands below go into this terminal.

---

### Step 2.2 — Clone the GitHub Repository

```bash
cd ~
git clone https://github.com/<YOUR_GITHUB_USERNAME>/ml_challenge.git
```

> ⚠️ Replace `<YOUR_GITHUB_USERNAME>` with your actual GitHub username.
>
> If your repository is **private**, use a Personal Access Token (PAT):
> ```bash
> git clone https://oauth2:<YOUR_PAT>@github.com/<YOUR_USERNAME>/ml_challenge.git
> ```
> To create a PAT: GitHub → Settings → Developer Settings → Personal access tokens → Generate new token (check `repo` scope)

---

### Step 2.3 — Navigate to the Project

```bash
cd ~/ml_challenge/Error_404_Not_Found_submission
```

Verify the structure looks correct:
```bash
ls
```
**You should see:**
```
code/   dataset/   Documentation_template.md   GUIDE.md   output/   README.md   utils/
```

```bash
ls dataset/train/
```
**You should see:**
```
train_ground_truth.tsv  train_source1.tsv  train_source2.tsv  train_source3.tsv
```

```bash
ls dataset/test/
```
**You should see:**
```
test_source1.tsv  test_source2.tsv  test_source3.tsv
```

> ⚠️ **If the TSV files are missing** (git LFS or gitignore excluded them):
> Upload them manually from your local machine using:
> ```bash
> # On your LOCAL machine terminal:
> aws s3 sync ~/Desktop/projects/ml_challenge/Error_404_Not_Found_submission/dataset \
>     s3://ml-challenge-2026/dataset/
>
> # Then in SageMaker terminal:
> aws s3 sync s3://ml-challenge-2026/dataset/ \
>     ~/ml_challenge/Error_404_Not_Found_submission/dataset/
> ```

---

### Step 2.4 — Install Dependencies

```bash
cd ~/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution
pip install -r requirements.txt
```

**What gets installed:**
```
pandas         → data loading and manipulation
numpy          → numerical operations
scipy          → sparse matrices for TF-IDF
scikit-learn   → TF-IDF vectorizer
rapidfuzz      → string similarity (Levenshtein, Jaro-Winkler, etc.)
xgboost        → machine learning classifier
boto3          → AWS SDK for S3 access
sagemaker      → SageMaker SDK
pyyaml         → config file parsing
```

**Expected time: 2–3 minutes**

Verify everything installed correctly:
```bash
python3 -c "import pandas, numpy, xgboost, rapidfuzz, sklearn; print('✓ All packages installed')"
```

---

### Step 2.5 — ✅ Run a Sanity Check First (Strongly Recommended)

Before running the full pipeline (which takes hours), run a quick test with
only 500 training entities to make sure nothing crashes.

**Option A: Loading data directly from S3 (`s3://ml-challenge-bharath/`)**
```bash
cd ~/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution

python3 src/pipeline.py \
    --use-s3 \
    --output-dir ../../output \
    --debug-sample 500
```

**Option B: Loading data from local folder**
```bash
cd ~/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution

python3 src/pipeline.py \
    --data-dir   ../../dataset \
    --output-dir ../../output \
    --debug-sample 500
```

**What this does:**
- Loads only the first 500 Source 1 entities
- Runs ALL 15 pipeline stages end-to-end
- Verifies output files are created correctly
- Takes ~5–10 minutes

**Expected output at the end:**
```
=== Stage 14: Test inference ===
=== Stage 15: Generating submission files ===
matching_results.tsv written: ../../output/matching_results.tsv
candidate_pairs.tsv written: ../../output/candidate_pairs.tsv
Official validator: PASS
Pipeline complete.
```

If you see `PASS` → proceed to the full run.  
If you see errors → check the error message and fix before running full pipeline.

---

### Step 2.6 — 🚀 Run the Full Pipeline

Once the sanity check passes, run the complete pipeline:

**Using S3 dataset:**
```bash
cd ~/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution

python3 src/pipeline.py \
    --use-s3 \
    --output-dir ../../output
```

**Using local dataset:**
```bash
cd ~/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution

python3 src/pipeline.py \
    --data-dir   ../../dataset \
    --output-dir ../../output
```

python3 src/pipeline.py \
    --data-dir   ../../dataset \
    --output-dir ../../output
```

> ⏱️ **Expected total time: 4–8 hours** on `ml.m5.2xlarge`

---

## What Happens at Each Stage (All Automatic)

The pipeline runs 15 stages in order automatically. Here is what each one does:

| Stage | Name | What It Does | Approx. Time |
|-------|------|-------------|-------------|
| **1** | Load data | Reads all 7 TSV files into memory | 5–15 min |
| **2** | Clean data | Normalizes names, addresses, removes legal suffixes | 10–20 min |
| **3** | Split train/val | Splits Source 1 entities: 80% train, 20% validation | < 1 min |
| **4** | Generate candidates | Runs 6-pass blocking (TF-IDF + exact match) to find similar pairs | 60–120 min |
| **5** | Evaluate recall | Checks what % of ground-truth matches are in candidate set | 5 min |
| **6** | Build training pairs | Labels candidates as match/non-match, samples negatives | 5–10 min |
| **7** | Build features | Computes ~40 similarity features per candidate pair | 30–60 min |
| **8** | Train XGBoost | Trains binary classifier (match vs. no-match) | 10–30 min |
| **9** | Score validation | Predicts match probabilities on validation set | 5 min |
| **10** | Optimize threshold | Finds best decision threshold to maximize F0.5 | 5–10 min |
| **11** | Error analysis | Finds false positives, false negatives, singleton errors | < 1 min |
| **12** | Test candidates | Runs blocking on test data to generate final candidate set | 60–120 min |
| **13** | Test features | Extracts features for test candidate pairs | 30–60 min |
| **14** | Test inference | Applies trained model + threshold to produce predictions | 5 min |
| **15** | Create submission | Writes matching_results.tsv, candidate_pairs.tsv, validates | < 1 min |

**You only run ONE command.** All 15 stages run automatically in order.

---

### Step 2.7 — ✅ Validate the Output

After the pipeline completes, run the official challenge validator:

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

---

### Step 2.8 — Preview the Output Files

```bash
# Check files were created
ls -lh ~/ml_challenge/Error_404_Not_Found_submission/output/

# Preview matching_results.tsv (this is what you submit)
head -5 ~/ml_challenge/Error_404_Not_Found_submission/output/matching_results.tsv
```

**Expected preview:**
```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S3-00812
S1-00002	S3-00004
S1-00003	
S1-00004	S2-01093
```

Empty `matched_entity_ids` = the model predicts that entity has no match (singleton).

---

### Step 2.9 — Backup to S3

The SageMaker notebook instance stops after inactivity and files may be lost.
Back up your output immediately:

```bash
aws s3 sync \
    ~/ml_challenge/Error_404_Not_Found_submission/output/ \
    s3://ml-challenge-2026/output/

echo "✓ Backed up to S3"
```

---

### Step 2.10 — Download matching_results.tsv for Submission

**Method A: JupyterLab file browser**
- In JupyterLab left panel, navigate to:
  `ml_challenge/Error_404_Not_Found_submission/output/`
- Right-click `matching_results.tsv` → **Download**

**Method B: From S3 Console**
1. Go to **AWS Console → S3 → ml-challenge-2026 → output/**
2. Click `matching_results.tsv` → **Download**

**Method C: AWS CLI on your local machine**
```bash
aws s3 cp s3://ml-challenge-2026/output/matching_results.tsv ./
```

---

## ⚙️ Advanced: Run Only Specific Stages

The pipeline is designed so you can re-run individual stages without
rerunning the full pipeline. Open a Python interactive session:

```bash
cd ~/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution
python3
```

```python
import os, sys
sys.path.insert(0, "src")

from config import Config
from pipeline import PipelineState, optimize_threshold

cfg = Config()
cfg.data.output_dir = "../../output"

# Load a previously saved state and re-run only threshold optimization
# (useful when experimenting with different thresholds without retraining)
```

---

## 🔧 Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'rapidfuzz'` | Dependencies not installed | Run `pip install -r requirements.txt` |
| `FileNotFoundError: ../../dataset/train/train_source1.tsv` | Dataset files not present | Sync from S3 (see Step 2.3 above) |
| `MemoryError` or Jupyter kernel crashes | Insufficient RAM | Upgrade instance to `ml.m5.4xlarge` (64 GB) |
| `ImportError: cannot import name 'Config'` | Wrong working directory | Make sure you're in `code/business_entity_resolution/` |
| Pipeline hangs at Stage 4 for >3 hours | Normal for large data | Wait — blocking is the slowest stage |
| `Official validator: FAIL` | Output format error | Check the validator error message for the specific issue |
| `git clone: Repository not found` | Wrong URL or private repo | Double-check URL; use PAT token for private repos |
| `aws: command not found` | AWS CLI not in PATH | Run `pip install awscli` then retry |

---

## 📋 Quick Command Reference

```bash
# --- SETUP (run once) ---
cd ~
git clone https://github.com/<USERNAME>/ml_challenge.git
cd ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution
pip install -r requirements.txt

# --- SANITY CHECK (run before full pipeline) ---
python3 src/pipeline.py \
    --data-dir ../../dataset \
    --output-dir ../../output \
    --debug-sample 500

# --- FULL PIPELINE (the main run) ---
python3 src/pipeline.py \
    --data-dir ../../dataset \
    --output-dir ../../output

# --- VALIDATE OUTPUT ---
cd ~/ml_challenge/Error_404_Not_Found_submission
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

# --- BACKUP TO S3 ---
aws s3 sync output/ s3://ml-challenge-2026/output/
```

---

## 📁 Files You Should Know About

| File | What It Is |
|------|-----------|
| `src/pipeline.py` | **The only file you run.** Master orchestrator — runs all 15 stages |
| `src/config.py` | All configuration (paths, hyperparameters, S3 settings) |
| `src/cleaning.py` | Text normalization (Unicode, legal suffixes, addresses) |
| `src/blocking.py` | Candidate pair generation (reduces search space) |
| `src/features.py` | 40 pairwise similarity features per candidate pair |
| `src/training.py` | XGBoost training + SageMaker job launcher |
| `src/evaluation.py` | Official F0.5 metric calculation |
| `src/threshold.py` | Decision threshold sweep (0.10 → 0.99) |
| `src/inference.py` | Test-time prediction on new data |
| `src/submission.py` | Writes final TSV output files |
| `output/matching_results.tsv` | **YOUR FINAL SUBMISSION FILE** ← upload this to leaderboard |
| `output/candidate_pairs.tsv` | Required alongside matching_results.tsv |

---

> **Summary: You only need to run `pipeline.py` once.**
> It handles everything — data loading, training, prediction, and submission file generation.
