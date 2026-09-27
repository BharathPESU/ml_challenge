import json
import os
import re

src_dir = "/home/bharath/Desktop/projects/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution/src"

# 1. Update pipeline.py to not load full parquet
with open(os.path.join(src_dir, "pipeline.py"), "r") as f:
    pipe_code = f.read()

stage13_new = """
def build_test_features(cfg: Config, state: PipelineState) -> None:
    from inference import extract_test_features_streaming
    logger.info("=== Stage 13: Extracting test features (Disk-Backed) ===")

    state.test_features, state.test_metadata = extract_test_features_streaming(
        test_candidates_path=state.test_candidates_path,
        test_s1_df=state.test_s1_clean,
        test_s2_df=state.test_s2_clean,
        test_s3_df=state.test_s3_clean,
        output_dir=cfg.data.output_dir
    )
"""
pipe_code = re.sub(r'def build_test_features\(cfg: Config, state: PipelineState\) -> None:.*?def run_test_inference', stage13_new + '\n\n# ---------------------------------------------------------------------------\n# Stage 14: Run test inference\n# ---------------------------------------------------------------------------\n\ndef run_test_inference', pipe_code, flags=re.DOTALL)

with open(os.path.join(src_dir, "pipeline.py"), "w") as f:
    f.write(pipe_code)


# 2. Update inference.py to add extract_test_features_streaming
with open(os.path.join(src_dir, "inference.py"), "r") as f:
    inf_code = f.read()

streaming_func = """
import pyarrow.parquet as pq
import gc
import time

def extract_test_features_streaming(
    test_candidates_path: str,
    test_s1_df: pd.DataFrame,
    test_s2_df: pd.DataFrame,
    test_s3_df: pd.DataFrame,
    output_dir: str,
    chunk_size: int = 100000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    \"\"\"Extract features out-of-core to prevent OOM.\"\"\"
    logger.info("🦆 Extracting test features out-of-core...")
    
    pf = pq.ParquetFile(test_candidates_path)
    n_total = pf.metadata.num_rows
    n_chunks = math.ceil(n_total / chunk_size)
    
    feat_chunks = []
    meta_chunks = []
    
    for i, batch in enumerate(pf.iter_batches(batch_size=chunk_size)):
        t0 = time.time()
        chunk_df = batch.to_pandas()
        f_df, m_df = extract_test_features(chunk_df, test_s1_df, test_s2_df, test_s3_df)
        
        # Downcast floats
        for col in f_df.select_dtypes(include=['float64']).columns:
            f_df[col] = f_df[col].astype(np.float32)
            
        feat_chunks.append(f_df)
        meta_chunks.append(m_df)
        dt = time.time() - t0
        logger.info(f"   ✅ Test Chunk {i+1}/{n_chunks} done in {dt:.1f}s")
        del chunk_df, f_df, m_df
        gc.collect()
        
    final_feats = pd.concat(feat_chunks, ignore_index=True)
    final_meta = pd.concat(meta_chunks, ignore_index=True)
    return final_feats, final_meta
"""

with open(os.path.join(src_dir, "inference.py"), "a") as f:
    f.write("\n" + streaming_func)


# 3. Notebook Cell 15 Rewrite
nb_path = "/home/bharath/Desktop/projects/ml_challenge/kaggle_training.ipynb"
with open(nb_path, "r") as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'def extract_test_features(' in source or 'CHUNK_SIZE = 100_000' in source:
            new_source = """from pipeline import build_test_features
import time
print("⏳ Stage 13: Test Feature Extraction starting...")
t0 = time.time()
build_test_features(cfg, state)
print(f"✅ Stage 13 Complete in {(time.time()-t0)/60:.1f} minutes!")
test_feat_ckpt = os.path.join(CKPT_DIR, "test_features.parquet")
state.test_features.to_parquet(test_feat_ckpt, index=False)
_download_btn(test_feat_ckpt)
"""
            # Check if it was stage 13 cell
            if "extract_test_features" in source:
                cell['source'] = [line + '\\n' for line in new_source.split('\\n')[:-1]] + [new_source.split('\\n')[-1]]

with open(nb_path, "w") as f:
    json.dump(nb, f, indent=1)

print("Patch 13 complete.")
