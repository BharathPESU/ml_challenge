import json

with open('/home/bharath/Desktop/projects/ml_challenge/kaggle_training.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'def run_pass_incremental_resumable(' in source:
            old_code = """                if frames:
                    pass_result = pd.concat(frames, ignore_index=True)
                    before_cnt = len(accumulated_df)
                    accumulated_df = pd.concat([accumulated_df, pass_result], ignore_index=True)
                    accumulated_df.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id", "candidate_source"], inplace=True)"""
            new_code = """                if frames:
                    pass_result = pd.concat(frames, ignore_index=True)
                    # 🛡️ Downcast to pyarrow strings and categories to save 80% RAM!
                    pass_result["source1_entity_id"] = pass_result["source1_entity_id"].astype("string[pyarrow]")
                    pass_result["candidate_entity_id"] = pass_result["candidate_entity_id"].astype("string[pyarrow]")
                    pass_result["candidate_source"] = pass_result["candidate_source"].astype("category")
                    
                    before_cnt = len(accumulated_df)
                    if accumulated_df.empty:
                        accumulated_df = pass_result
                    else:
                        accumulated_df = pd.concat([accumulated_df, pass_result], ignore_index=True)
                        accumulated_df.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id", "candidate_source"], inplace=True)"""
            
            if old_code in source:
                new_source = source.replace(old_code, new_code)
                # Reconstruct list of lines with newlines
                lines = new_source.split('\n')
                new_source_list = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
                cell['source'] = new_source_list
                print("Patched Cell 6 successfully!")

with open('/home/bharath/Desktop/projects/ml_challenge/kaggle_training.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

