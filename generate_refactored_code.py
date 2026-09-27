import os
import json

base_dir = "/home/bharath/Desktop/projects/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution"
src_dir = os.path.join(base_dir, "src")

# 1. memory_utils.py is already created.

# 2. blocking.py
with open(os.path.join(src_dir, "blocking.py"), "r") as f:
    blocking_code = f.read()

# Replace generate_candidates with disk-backed logic
new_blocking_code = blocking_code.replace(
    "def generate_candidates(", 
    "def generate_candidates_deprecated("
)

disk_backed_blocking = """
import os
import gc
import time
import math
import duckdb
from memory_utils import report_memory

def run_disk_backed_blocking(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    output_dir: str,
    prefix: str,
    chunk_size: int = 10000,
    config=None,
    s1_id_set=None,
) -> str:
    \"\"\"Runs multi-pass blocking out-of-core, chunking S1 to prevent OOM.\"\"\"
    
    if s1_id_set is not None:
        s1_df = s1_df[s1_df["entity_id"].isin(s1_id_set)].reset_index(drop=True)

    final_output = os.path.join(output_dir, f"{prefix}_candidates.parquet")
    if os.path.exists(final_output):
        logger.info(f"⚡ [RESUME HIT] {final_output} exists. Skipping generation.")
        return final_output

    chunk_dir = os.path.join(output_dir, f"{prefix}_chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    
    # Read passes config
    bcfg = config.blocking if config and hasattr(config, 'blocking') else config
    
    targets = [("S2", s2_df), ("S3", s3_df)]
    n_chunks = math.ceil(len(s1_df) / chunk_size)
    
    logger.info(f"🚀 Starting Disk-Backed Blocking ({prefix}) | {n_chunks} chunks of {chunk_size} rows")
    
    for tgt_label, tgt_df in targets:
        logger.info(f"   => Processing Target: {tgt_label}")
        for chunk_idx in range(n_chunks):
            chunk_file = os.path.join(chunk_dir, f"{tgt_label}_chunk_{chunk_idx}.parquet")
            if os.path.exists(chunk_file):
                logger.info(f"      ⏭️  Skipping completed chunk {chunk_idx}")
                continue
                
            t0 = time.time()
            start_i = chunk_idx * chunk_size
            end_i = min((chunk_idx + 1) * chunk_size, len(s1_df))
            s1_chunk = s1_df.iloc[start_i:end_i]
            
            frames = []
            
            # Pass 1-4: Exact
            frames.append(generate_exact_match_candidates(s1_chunk, tgt_df, "business_name_norm", tgt_label, 2, 500))
            frames.append(generate_exact_match_candidates(s1_chunk, tgt_df, "business_name_core", tgt_label, 3, 500))
            frames.append(generate_exact_match_candidates(s1_chunk, tgt_df, "business_address_norm", tgt_label, 5, 500))
            frames.append(generate_numeric_address_candidates(s1_chunk, tgt_df, tgt_label))
            
            # Pass 5-6: Rare
            frames.append(generate_rare_token_candidates(s1_chunk, tgt_df, "business_name_norm", tgt_label, 1000))
            frames.append(generate_rare_token_candidates(s1_chunk, tgt_df, "business_address_norm", tgt_label, 1000))
            
            # Pass 7-8: Fuzzy
            frames.append(generate_fuzzy_candidates(s1_chunk, tgt_df, "business_name_norm", tgt_label, 10))
            frames.append(generate_fuzzy_candidates(s1_chunk, tgt_df, "business_address_norm", tgt_label, 10))
            
            frames = [f for f in frames if not f.empty]
            if frames:
                chunk_res = pd.concat(frames, ignore_index=True)
                chunk_res.drop_duplicates(inplace=True)
                # Downcast to save RAM
                chunk_res["source1_entity_id"] = chunk_res["source1_entity_id"].astype("string[pyarrow]")
                chunk_res["candidate_entity_id"] = chunk_res["candidate_entity_id"].astype("string[pyarrow]")
                chunk_res["candidate_source"] = chunk_res["candidate_source"].astype("category")
                chunk_res.to_parquet(chunk_file, index=False)
                c_len = len(chunk_res)
            else:
                c_len = 0
                pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "candidate_source"]).to_parquet(chunk_file, index=False)
                
            dt = time.time() - t0
            logger.info(f"      ✅ Chunk {chunk_idx+1}/{n_chunks} ({tgt_label}) done in {dt:.1f}s — {c_len} candidates")
            
        report_memory(f"After {tgt_label}")
        # Drop tgt_df if we can? We need it for both S1 chunks, but S2 is done now.
        gc.collect()

    logger.info("🦆 Running DuckDB Deduplication on Disk...")
    os.makedirs("/kaggle/working/duckdb_temp", exist_ok=True)
    duckdb.execute("PRAGMA temp_directory='/kaggle/working/duckdb_temp';")
    duckdb.execute("PRAGMA memory_limit='15GB';")
    
    q = f\"\"\"
    COPY (
        SELECT DISTINCT source1_entity_id, candidate_entity_id, candidate_source
        FROM read_parquet('{chunk_dir}/*.parquet')
    ) TO '{final_output}' (FORMAT PARQUET);
    \"\"\"
    duckdb.execute(q)
    logger.info(f"✅ Saved deduplicated candidates to {final_output}")
    
    return final_output
"""

with open(os.path.join(src_dir, "blocking.py"), "w") as f:
    f.write(new_blocking_code + "\n" + disk_backed_blocking)

# 3. pair_dataset.py
with open(os.path.join(src_dir, "pair_dataset.py"), "r") as f:
    pair_code = f.read()

disk_backed_pairs = """
import duckdb
from memory_utils import report_memory

def build_disk_backed_training_pairs(
    candidates_parquet_path: str,
    gt: Dict[str, Set[str]],
    train_s1_ids: List[str],
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    neg_ratio: int = 4,
    hard_neg_fraction: float = 0.5,
    random_seed: int = 42,
) -> pd.DataFrame:
    \"\"\"
    Construct the final training pair DataFrame using DuckDB out-of-core SQL.
    \"\"\"
    logger.info("🦆 Building pairs via DuckDB out-of-core...")
    
    # 1. Build GT Dataframe
    gt_rows = []
    train_id_set = set(train_s1_ids)
    for s1_id in train_s1_ids:
        for cand_id in gt.get(s1_id, set()):
            gt_rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "is_match": 1
            })
    gt_df = pd.DataFrame(gt_rows)
    if gt_df.empty:
        gt_df = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "is_match"])
        
    n_pos = len(gt_df)
    n_total_neg = n_pos * neg_ratio
    n_hard = int(n_total_neg * hard_neg_fraction)
    n_easy = n_total_neg - n_hard
    
    logger.info(f"Targets: {n_pos} pos, {n_easy} easy neg, {n_hard} hard neg")

    duckdb.execute("PRAGMA temp_directory='/kaggle/working/duckdb_temp';")
    duckdb.execute("PRAGMA memory_limit='15GB';")
    
    query = f\"\"\"
    WITH candidates AS (
        SELECT * FROM read_parquet('{candidates_parquet_path}')
    ),
    labelled AS (
        SELECT c.source1_entity_id, c.candidate_entity_id, c.candidate_source,
               COALESCE(g.is_match, 0) AS is_match
        FROM candidates c
        LEFT JOIN gt_df g ON c.source1_entity_id = g.source1_entity_id 
                         AND c.candidate_entity_id = g.candidate_entity_id
    ),
    positives AS (
        SELECT * FROM labelled WHERE is_match = 1
    ),
    negatives AS (
        SELECT * FROM labelled WHERE is_match = 0
    ),
    easy_neg AS (
        SELECT * FROM negatives ORDER BY random() LIMIT {n_easy}
    ),
    hard_pool AS (
        SELECT n.* FROM negatives n
        JOIN positives p ON n.source1_entity_id = p.source1_entity_id
    ),
    hard_neg AS (
        SELECT * FROM hard_pool ORDER BY random() LIMIT {n_hard}
    )
    SELECT * FROM positives
    UNION ALL
    SELECT * FROM easy_neg
    UNION ALL
    SELECT * FROM hard_neg
    \"\"\"
    
    res = duckdb.query(query).df()
    logger.info(f"✅ Final training pairs extracted to RAM: {len(res)} rows")
    return res

def build_disk_backed_validation_pairs(
    candidates_parquet_path: str,
    gt: Dict[str, Set[str]],
    val_s1_ids: List[str]
) -> pd.DataFrame:
    \"\"\"Validation pairs are not downsampled.\"\"\"
    
    gt_rows = []
    for s1_id in val_s1_ids:
        for cand_id in gt.get(s1_id, set()):
            gt_rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "is_match": 1
            })
    gt_df = pd.DataFrame(gt_rows)
    if gt_df.empty:
        gt_df = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "is_match"])

    duckdb.execute("PRAGMA temp_directory='/kaggle/working/duckdb_temp';")
    duckdb.execute("PRAGMA memory_limit='15GB';")
    
    query = f\"\"\"
    WITH candidates AS (
        SELECT * FROM read_parquet('{candidates_parquet_path}')
    ),
    labelled AS (
        SELECT c.source1_entity_id, c.candidate_entity_id, c.candidate_source,
               COALESCE(g.is_match, 0) AS is_match
        FROM candidates c
        LEFT JOIN gt_df g ON c.source1_entity_id = g.source1_entity_id 
                         AND c.candidate_entity_id = g.candidate_entity_id
    )
    SELECT * FROM labelled
    \"\"\"
    res = duckdb.query(query).df()
    logger.info(f"✅ Final validation pairs extracted to RAM: {len(res)} rows")
    return res
"""

with open(os.path.join(src_dir, "pair_dataset.py"), "a") as f:
    f.write("\n" + disk_backed_pairs)


# 4. Pipeline state update
with open(os.path.join(src_dir, "pipeline.py"), "r") as f:
    pipeline_code = f.read()

# Replace PipelineState 
new_state = """class PipelineState:
    \"\"\"Holds shared state across pipeline stages. LOW-RAM disk-backed version.\"\"\"

    def __init__(self):
        # Raw data
        self.train_s1: Optional[pd.DataFrame] = None
        self.train_s2: Optional[pd.DataFrame] = None
        self.train_s3: Optional[pd.DataFrame] = None
        self.gt: Optional[Dict[str, Set[str]]] = None

        # Cleaned data
        self.train_s1_clean: Optional[pd.DataFrame] = None
        self.train_s2_clean: Optional[pd.DataFrame] = None
        self.train_s3_clean: Optional[pd.DataFrame] = None

        # Train / validation split
        self.train_s1_ids: Optional[List[str]] = None
        self.val_s1_ids: Optional[List[str]] = None
        self.train_gt: Optional[Dict[str, Set[str]]] = None
        self.val_gt: Optional[Dict[str, Set[str]]] = None

        # Candidates (Disk Paths)
        self.train_candidates_path: Optional[str] = None
        self.val_candidates_path: Optional[str] = None

        # Labelled pairs (Small enough for RAM)
        self.train_pairs: Optional[pd.DataFrame] = None
        self.val_pairs: Optional[pd.DataFrame] = None

        # Features
        self.train_features: Optional[pd.DataFrame] = None
        self.val_features: Optional[pd.DataFrame] = None
        self.feature_columns: Optional[List[str]] = None

        # Model
        self.booster: Optional[Any] = None
        self.model_path: Optional[str] = None

        # Validation scoring
        self.val_probabilities: Optional[np.ndarray] = None
        self.sweep_df: Optional[pd.DataFrame] = None
        self.best_threshold: Optional[float] = None
        self.threshold_path: Optional[str] = None

        # Test data
        self.test_s1_clean: Optional[pd.DataFrame] = None
        self.test_s2_clean: Optional[pd.DataFrame] = None
        self.test_s3_clean: Optional[pd.DataFrame] = None
        self.test_candidates_path: Optional[str] = None
        self.test_features: Optional[pd.DataFrame] = None
        self.test_metadata: Optional[pd.DataFrame] = None
        self.test_predictions: Optional[Dict[str, Set[str]]] = None
"""

import re
pipeline_code = re.sub(r'class PipelineState:.*?def load_training_data', new_state + '\n\n# ---------------------------------------------------------------------------\n# Stage 1: Load training data\n# ---------------------------------------------------------------------------\n\ndef load_training_data', pipeline_code, flags=re.DOTALL)

# Update stage 4
stage4_new = """
def generate_candidates(cfg: Config, state: PipelineState) -> None:
    from blocking import run_disk_backed_blocking
    logger.info("=== Stage 4: Disk-Backed Candidate Generation ===")
    
    state.train_candidates_path = run_disk_backed_blocking(
        s1_df=state.train_s1_clean,
        s2_df=state.train_s2_clean,
        s3_df=state.train_s3_clean,
        output_dir=cfg.data.output_dir,
        prefix="train",
        config=cfg,
        s1_id_set=set(state.train_s1_ids)
    )
    
    state.val_candidates_path = run_disk_backed_blocking(
        s1_df=state.train_s1_clean,
        s2_df=state.train_s2_clean,
        s3_df=state.train_s3_clean,
        output_dir=cfg.data.output_dir,
        prefix="val",
        config=cfg,
        s1_id_set=set(state.val_s1_ids)
    )
"""
pipeline_code = re.sub(r'def generate_candidates\(cfg: Config, state: PipelineState\) -> None:.*?def evaluate_candidate_recall', stage4_new + '\n\n# ---------------------------------------------------------------------------\n# Stage 5: Evaluate candidate recall\n# ---------------------------------------------------------------------------\n\ndef evaluate_candidate_recall', pipeline_code, flags=re.DOTALL)

# Update stage 6
stage6_new = """
def build_training_pairs(cfg: Config, state: PipelineState) -> None:
    from pair_dataset import build_disk_backed_training_pairs, build_disk_backed_validation_pairs
    logger.info("=== Stage 6: Building training and validation pairs ===")
    
    ns = cfg.negative_sampling
    combined_target = pd.concat([state.train_s2_clean, state.train_s3_clean], ignore_index=True)
    
    state.train_pairs = build_disk_backed_training_pairs(
        candidates_parquet_path=state.train_candidates_path,
        gt=state.train_gt,
        train_s1_ids=state.train_s1_ids,
        s1_df=state.train_s1_clean,
        target_df=combined_target,
        neg_ratio=ns.neg_ratio,
        hard_neg_fraction=ns.hard_neg_fraction,
        random_seed=cfg.experiment.random_seed,
    )
    
    state.val_pairs = build_disk_backed_validation_pairs(
        candidates_parquet_path=state.val_candidates_path,
        gt=state.val_gt,
        val_s1_ids=state.val_s1_ids,
    )
"""
pipeline_code = re.sub(r'def build_training_pairs\(cfg: Config, state: PipelineState\) -> None:.*?def build_features', stage6_new + '\n\n# ---------------------------------------------------------------------------\n# Stage 7: Build features\n# ---------------------------------------------------------------------------\n\ndef build_features', pipeline_code, flags=re.DOTALL)

# Update stage 12
stage12_new = """
def generate_test_candidates(cfg: Config, state: PipelineState) -> None:
    from inference import load_and_clean_test_data
    from blocking import run_disk_backed_blocking
    logger.info("=== Stage 12: Generating test candidates ===")

    state.test_s1_clean, state.test_s2_clean, state.test_s3_clean = \\
        load_and_clean_test_data(
            cfg.data.test_source1_path,
            cfg.data.test_source2_path,
            cfg.data.test_source3_path,
            debug_sample_size=cfg.runtime.debug_sample_size,
        )

    state.test_candidates_path = run_disk_backed_blocking(
        s1_df=state.test_s1_clean,
        s2_df=state.test_s2_clean,
        s3_df=state.test_s3_clean,
        output_dir=cfg.data.output_dir,
        prefix="test",
        config=cfg,
    )
"""
pipeline_code = re.sub(r'def generate_test_candidates\(cfg: Config, state: PipelineState\) -> None:.*?def build_test_features', stage12_new + '\n\n# ---------------------------------------------------------------------------\n# Stage 13: Build test features\n# ---------------------------------------------------------------------------\n\ndef build_test_features', pipeline_code, flags=re.DOTALL)

# Update evaluation (stage 5)
pipeline_code = pipeline_code.replace("evaluate_candidate_recall(state.train_candidates", "evaluate_candidate_recall(pd.read_parquet(state.train_candidates_path)")
pipeline_code = pipeline_code.replace("evaluate_candidate_recall(state.val_candidates", "evaluate_candidate_recall(pd.read_parquet(state.val_candidates_path)")
pipeline_code = pipeline_code.replace("candidates_df=state.test_candidates", "candidates_df=pd.read_parquet(state.test_candidates_path)")

with open(os.path.join(src_dir, "pipeline.py"), "w") as f:
    f.write(pipeline_code)

# 5. Notebook Cell 6 Rewrite
nb_path = "/home/bharath/Desktop/projects/ml_challenge/kaggle_training.ipynb"
with open(nb_path, "r") as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'def run_pass_incremental_resumable' in source:
            new_source = """from pipeline import generate_candidates
import time
print("⏳ Stage 4: Disk-Backed DuckDB Candidate Generation starting...")
t0 = time.time()
generate_candidates(cfg, state)
print(f"✅ Stage 4 Complete in {(time.time()-t0)/60:.1f} minutes!")
_download_btn(state.train_candidates_path, state.val_candidates_path)
"""
            cell['source'] = [line + '\\n' for line in new_source.split('\\n')[:-1]] + [new_source.split('\\n')[-1]]

        if 'def extract_features_streaming' in source:
            # We don't strictly need to rewrite cell 9 if we just call build_features
            new_source = """from pipeline import build_features
import time
print("⏳ Stage 7: Feature Extraction starting...")
t0 = time.time()
build_features(cfg, state)
print(f"✅ Stage 7 Complete in {(time.time()-t0)/60:.1f} minutes!")
train_feat_ckpt = os.path.join(CKPT_DIR, "train_features.parquet")
val_feat_ckpt = os.path.join(CKPT_DIR, "val_features.parquet")
state.train_features.to_parquet(train_feat_ckpt, index=False)
state.val_features.to_parquet(val_feat_ckpt, index=False)
_download_btn(train_feat_ckpt, val_feat_ckpt)
"""
            cell['source'] = [line + '\\n' for line in new_source.split('\\n')[:-1]] + [new_source.split('\\n')[-1]]

with open(nb_path, "w") as f:
    json.dump(nb, f, indent=1)

print("Refactoring complete.")
