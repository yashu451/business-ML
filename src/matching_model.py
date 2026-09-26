import pandas as pd
import difflib

print("====================================")
print("1. LOADING CLEANED TRAIN DATA")
print("====================================")

train_s1 = pd.read_csv("output/cleaned/train_source1_clean.tsv", sep="\t")
train_s2 = pd.read_csv("output/cleaned/train_source2_clean.tsv", sep="\t")
train_s3 = pd.read_csv("output/cleaned/train_source3_clean.tsv", sep="\t")
train_gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")

for df in [train_s1, train_s2, train_s3]:
    df["clean_business_name"] = df["clean_business_name"].fillna("")
    df["clean_business_address"] = df["clean_business_address"].fillna("")
    df["clean_country"] = df["clean_country"].fillna("")

print(f"Loaded {len(train_s1)} Source 1, {len(train_s2)} Source 2, {len(train_s3)} Source 3 entities.")

print("\n====================================")
print("2. GENERATING TRAINING CANDIDATE PAIRS")
print("====================================")

from candidate_generation import generate_candidates

train_candidates_df = generate_candidates(train_s1, train_s2, train_s3)

train_candidates = {}
total_train_candidates = 0

for _, row in train_candidates_df.iterrows():
    s1_id = row["source1_entity_id"]
    cands_str = str(row["candidate_entity_ids"])
    if cands_str.strip() != "" and cands_str.strip() != "nan":
        cand_ids = cands_str.split(",")
    else:
        cand_ids = []
    
    train_candidates[s1_id] = cand_ids
    total_train_candidates += len(cand_ids)

print(f"Total training candidate pairs generated: {total_train_candidates}")

print("\n====================================")
print("3. EXTRACTING FEATURES FOR TRAINING")
print("====================================")

import re

def string_similarity(s1, s2):
    return difflib.SequenceMatcher(None, str(s1), str(s2)).ratio()

def jaccard_sim(s1, s2):
    set1 = set(str(s1).lower().split())
    set2 = set(str(s2).lower().split())
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def get_num_agreement(a1, a2):
    nums1 = set(re.findall(r'\b\d+\b', str(a1)))
    nums2 = set(re.findall(r'\b\d+\b', str(a2)))
    if not nums1 and not nums2:
        return 1.0
    if not nums1 or not nums2:
        return 0.5
    return 1.0 if len(nums1.intersection(nums2)) > 0 else 0.0

s2_dict = train_s2.set_index("clean_entity_id").to_dict(orient="index")
s3_dict = train_s3.set_index("clean_entity_id").to_dict(orient="index")

def get_entity_data(entity_id, d2, d3):
    if entity_id in d2: return d2[entity_id]
    if entity_id in d3: return d3[entity_id]
    return None

features = []
for _, row in train_s1.iterrows():
    s1_id = row["clean_entity_id"]
    s1_name = row["clean_business_name"]
    s1_address = row["clean_business_address"]
    
    for c_id in train_candidates.get(s1_id, []):
        c_data = get_entity_data(c_id, s2_dict, s3_dict)
        if c_data:
            c_name = c_data["clean_business_name"]
            c_addr = c_data["clean_business_address"]
            
            name_sim = string_similarity(s1_name, c_name)
            addr_sim = string_similarity(s1_address, c_addr)
            name_jac = jaccard_sim(s1_name, c_name)
            addr_jac = jaccard_sim(s1_address, c_addr)
            num_agree = get_num_agreement(s1_address, c_addr)
            
            features.append({
                "s1_id": s1_id, 
                "c_id": c_id, 
                "name_sim": name_sim, 
                "addr_sim": addr_sim,
                "name_jac": name_jac,
                "addr_jac": addr_jac,
                "num_agree": num_agree
            })

features_df = pd.DataFrame(features)

print("\n====================================")
print("4. EVALUATING THRESHOLDS ON TRAIN DATA")
print("====================================")

true_matches = {}
for _, row in train_gt.iterrows():
    s1_id = row["source1_entity_id"]
    matched = str(row["matched_entity_ids"])
    if pd.notna(row["matched_entity_ids"]) and matched.strip() != "" and matched.strip() != "nan":
        true_matches[s1_id] = set(matched.split(","))
    else:
        true_matches[s1_id] = set()

# Ensure all training Source 1 entities have an entry
for s1_id in train_s1["clean_entity_id"]:
    if s1_id not in true_matches:
        true_matches[s1_id] = set()

best_f05 = -1
best_params = {}
best_macro_p, best_macro_r = 0, 0

if not features_df.empty:
    all_s1_ids = train_s1["clean_entity_id"].tolist()
    
    weight_combinations = [
        {"name_seq": 0.4, "addr_seq": 0.2, "name_jac": 0.2, "addr_jac": 0.1, "num_agree": 0.1},
        {"name_seq": 0.5, "addr_seq": 0.1, "name_jac": 0.2, "addr_jac": 0.1, "num_agree": 0.1},
        {"name_seq": 0.3, "addr_seq": 0.2, "name_jac": 0.3, "addr_jac": 0.1, "num_agree": 0.1},
        {"name_seq": 0.4, "addr_seq": 0.1, "name_jac": 0.3, "addr_jac": 0.05, "num_agree": 0.15},
    ]
    
    for w in weight_combinations:
        scores = (features_df["name_sim"] * w["name_seq"] + 
                  features_df["addr_sim"] * w["addr_seq"] +
                  features_df["name_jac"] * w["name_jac"] +
                  features_df["addr_jac"] * w["addr_jac"] +
                  features_df["num_agree"] * w["num_agree"])
        
        for th in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            # Generate predictions per source 1 entity
            predictions = {s1_id: set() for s1_id in all_s1_ids}
            predicted_df = features_df[scores >= th]
            for _, row in predicted_df.iterrows():
                predictions[row["s1_id"]].add(row["c_id"])
            
            total_f05, total_p, total_r = 0, 0, 0
            
            for s1_id in all_s1_ids:
                pred_set = predictions[s1_id]
                true_set = true_matches[s1_id]
                
                if len(pred_set) == 0 and len(true_set) == 0:
                    p, r, f05 = 1.0, 1.0, 1.0
                elif len(pred_set) > 0 and len(true_set) == 0:
                    p, r, f05 = 0.0, 0.0, 0.0
                elif len(pred_set) == 0 and len(true_set) > 0:
                    p, r, f05 = 0.0, 0.0, 0.0
                else:
                    tp = len(pred_set.intersection(true_set))
                    fp = len(pred_set - true_set)
                    fn = len(true_set - pred_set)
                    
                    p = tp / (tp + fp) if (tp + fp) > 0 else 0
                    r = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f05 = (1.25 * p * r) / (0.25 * p + r) if (p + r) > 0 else 0
                
                total_p += p
                total_r += r
                total_f05 += f05
                
            macro_p = total_p / len(all_s1_ids)
            macro_r = total_r / len(all_s1_ids)
            macro_f05 = total_f05 / len(all_s1_ids)
            
            if macro_f05 > best_f05:
                best_f05 = macro_f05
                best_params = {"weights": w, "threshold": th}
                best_macro_p = macro_p
                best_macro_r = macro_r

print(f"1. Selected feature weights: {best_params.get('weights')}")
print(f"2. Selected threshold: {best_params.get('threshold')}")
print(f"3. Macro Training Precision: {best_macro_p:.4f}")
print(f"4. Macro Training Recall:    {best_macro_r:.4f}")
print(f"5. Macro Training F0.5:      {best_f05:.4f}")

print("\n====================================")
print("5. APPLYING RULE TO TEST DATA")
print("====================================")

test_s1 = pd.read_csv("output/cleaned/test_source1_clean.tsv", sep="\t").fillna("")
test_s2 = pd.read_csv("output/cleaned/test_source2_clean.tsv", sep="\t").fillna("")
test_s3 = pd.read_csv("output/cleaned/test_source3_clean.tsv", sep="\t").fillna("")

test_s2_dict = test_s2.set_index("clean_entity_id").to_dict(orient="index")
test_s3_dict = test_s3.set_index("clean_entity_id").to_dict(orient="index")

test_candidates_df = pd.read_csv("output/candidate_pairs.tsv", sep="\t").fillna("")

test_candidates = {}
for _, row in test_candidates_df.iterrows():
    s1_id = row["source1_entity_id"]
    cands_str = str(row["candidate_entity_ids"])
    if cands_str.strip() != "":
        test_candidates[s1_id] = cands_str.split(",")
    else:
        test_candidates[s1_id] = []

test_results = []
w = best_params.get("weights", {"name_seq": 0.4, "addr_seq": 0.2, "name_jac": 0.2, "addr_jac": 0.1, "num_agree": 0.1})
th = best_params.get("threshold", 0.70)
total_final_matches = 0
empty_matches = 0

for _, row in test_s1.iterrows():
    s1_id = row["clean_entity_id"]
    s1_name = row["clean_business_name"]
    s1_address = row["clean_business_address"]
    
    cand_ids = test_candidates.get(s1_id, [])
    matched = []
    
    for c_id in cand_ids:
        c_data = get_entity_data(c_id, test_s2_dict, test_s3_dict)
        if c_data:
            c_name = c_data["clean_business_name"]
            c_addr = c_data["clean_business_address"]
            
            name_sim = string_similarity(s1_name, c_name)
            addr_sim = string_similarity(s1_address, c_addr)
            name_jac = jaccard_sim(s1_name, c_name)
            addr_jac = jaccard_sim(s1_address, c_addr)
            num_agree = get_num_agreement(s1_address, c_addr)
            
            score = (name_sim * w["name_seq"] + 
                     addr_sim * w["addr_seq"] +
                     name_jac * w["name_jac"] +
                     addr_jac * w["addr_jac"] +
                     num_agree * w["num_agree"])
                     
            if score >= th:
                matched.append(c_id)
                total_final_matches += 1
                
    if len(matched) == 0:
        empty_matches += 1
        
    test_results.append({
        "source1_entity_id": s1_id,
        "matched_entity_ids": ",".join(matched)
    })

print(f"Number of test Source 1 entities: {len(test_s1)}")
print(f"6. Number of final test matches: {total_final_matches}")
print(f"7. Number of test Source1 entities with no matches: {empty_matches}")

results_df = pd.DataFrame(test_results)
results_df.to_csv("output/matching_results.tsv", sep="\t", index=False)
print("\nSaved output/matching_results.tsv successfully!")
