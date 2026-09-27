import os
import re

src_dir = "/home/bharath/Desktop/projects/ml_challenge/Error_404_Not_Found_submission/code/business_entity_resolution/src"
block_file = os.path.join(src_dir, "blocking.py")

with open(block_file, "r") as f:
    code = f.read()

# Replace the unconditional appends with conditional ones
old_logic = """            # Pass 1-4: Exact
            frames.append(generate_exact_match_candidates(s1_chunk, tgt_df, "business_name_norm", tgt_label, 2, 500))
            frames.append(generate_exact_match_candidates(s1_chunk, tgt_df, "business_name_core", tgt_label, 3, 500))
            frames.append(generate_exact_match_candidates(s1_chunk, tgt_df, "business_address_norm", tgt_label, 5, 500))
            frames.append(generate_numeric_address_candidates(s1_chunk, tgt_df, tgt_label))
            
            # Pass 5-6: Rare
            frames.append(generate_rare_token_candidates(s1_chunk, tgt_df, "business_name_norm", tgt_label, 1000))
            frames.append(generate_rare_token_candidates(s1_chunk, tgt_df, "business_address_norm", tgt_label, 1000))
            
            # Pass 7-8: Fuzzy
            frames.append(generate_fuzzy_candidates(s1_chunk, tgt_df, "business_name_norm", tgt_label, 10))
            frames.append(generate_fuzzy_candidates(s1_chunk, tgt_df, "business_address_norm", tgt_label, 10))"""

new_logic = """            # Pass 1-4: Exact
            if getattr(bcfg, "pass1_exact_norm", True):
                frames.append(generate_exact_match_candidates(s1_chunk, tgt_df, "business_name_norm", tgt_label, 2, 500))
            if getattr(bcfg, "pass2_exact_core", True):
                frames.append(generate_exact_match_candidates(s1_chunk, tgt_df, "business_name_core", tgt_label, 3, 500))
            if getattr(bcfg, "pass3_address_token", True):
                frames.append(generate_exact_match_candidates(s1_chunk, tgt_df, "business_address_norm", tgt_label, 5, 500))
            if getattr(bcfg, "pass4_numeric_address", True):
                frames.append(generate_numeric_address_candidates(s1_chunk, tgt_df, tgt_label))
            
            # Pass 5-6: Rare
            if getattr(bcfg, "pass5_rare_name_token", True):
                frames.append(generate_rare_token_candidates(s1_chunk, tgt_df, "business_name_norm", tgt_label, 1000))
            if getattr(bcfg, "pass6_rare_address_token", True):
                frames.append(generate_rare_token_candidates(s1_chunk, tgt_df, "business_address_norm", tgt_label, 1000))
            
            # Pass 7-8: Fuzzy (EXTREMELY SLOW - should be False by default)
            if getattr(bcfg, "pass7_fuzz_name", False):
                frames.append(generate_fuzzy_candidates(s1_chunk, tgt_df, "business_name_norm", tgt_label, 10))
            if getattr(bcfg, "pass8_fuzz_address", False):
                frames.append(generate_fuzzy_candidates(s1_chunk, tgt_df, "business_address_norm", tgt_label, 10))"""

if old_logic in code:
    code = code.replace(old_logic, new_logic)
    with open(block_file, "w") as f:
        f.write(code)
    print("blocking.py patched!")
else:
    print("Could not find old logic in blocking.py")
