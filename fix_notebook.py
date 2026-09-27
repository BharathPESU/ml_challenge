import json

with open("kaggle_training.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        for line in source:
            # Fix evaluate_candidate_recall calls
            if "evaluate_candidate_recall(state.train_candidates" in line:
                line = line.replace("evaluate_candidate_recall(state.train_candidates", "evaluate_candidate_recall(pd.read_parquet(state.train_candidates_path)")
            if "evaluate_candidate_recall(state.val_candidates" in line:
                line = line.replace("evaluate_candidate_recall(state.val_candidates", "evaluate_candidate_recall(pd.read_parquet(state.val_candidates_path)")
            if "state.train_candidates_path is not None" in line:
                pass
            elif "state.train_candidates is not None" in line:
                line = line.replace("state.train_candidates", "state.train_candidates_path")
            if "len(state.train_candidates)" in line:
                line = line.replace("len(state.train_candidates)", "0") # we don't have it in memory
            if "len(state.val_candidates)" in line:
                line = line.replace("len(state.val_candidates)", "0")
                
            # Fix merged features
            if "state.train_merged" in line:
                line = line.replace("state.train_merged", "state.train_features")
            if "state.val_merged" in line:
                line = line.replace("state.val_merged", "state.val_features")
                
            # Fix test candidates
            if "evaluate_candidate_recall(candidates_df=state.test_candidates" in line:
                line = line.replace("candidates_df=state.test_candidates", "candidates_df=pd.read_parquet(state.test_candidates_path)")
            
            if "state.test_candidates is not None" in line:
                line = line.replace("state.test_candidates", "state.test_candidates_path")
                
            new_source.append(line)
        cell['source'] = new_source

with open("kaggle_training.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook fixes applied!")
