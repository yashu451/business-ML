import pandas as pd
from collections import defaultdict

def has_token_overlap(text1, text2):
    tokens1 = set(str(text1).lower().split())
    tokens2 = set(str(text2).lower().split())
    return len(tokens1.intersection(tokens2)) > 0

def get_tokens(text):
    return set(str(text).lower().split())

def build_indices(source2, source3):
    name_index = defaultdict(lambda: defaultdict(set))
    address_index = defaultdict(lambda: defaultdict(set))
    
    name_df = defaultdict(int)
    address_df = defaultdict(int)
    total_records = 0
    
    for src in [source2, source3]:
        for _, row in src.iterrows():
            total_records += 1
            country = row["clean_country"]
            c_id = row["clean_entity_id"]
            
            n_tokens = get_tokens(row["clean_business_name"])
            a_tokens = get_tokens(row["clean_business_address"])
            
            for t in n_tokens:
                name_index[country][t].add(c_id)
                name_df[t] += 1
                
            for t in a_tokens:
                address_index[country][t].add(c_id)
                address_df[t] += 1

    max_freq = max(100, total_records * 0.05)
    
    for country in list(name_index.keys()):
        for t in list(name_index[country].keys()):
            if name_df[t] > max_freq:
                del name_index[country][t]
                
    for country in list(address_index.keys()):
        for t in list(address_index[country].keys()):
            if address_df[t] > max_freq:
                del address_index[country][t]
                
    return name_index, address_index

def generate_candidates(source1, source2, source3, output_file=None):
    name_index, address_index = build_indices(source2, source3)
    
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            
    candidate_results = []
    total_s1 = len(source1)
    
    for i, (_, row) in enumerate(source1.iterrows()):
        if (i + 1) % 100000 == 0:
            print(f"Processed {i + 1}/{total_s1} Source1 records...")
            
        s1_id = row["clean_entity_id"]
        country = row["clean_country"]
        
        n_tokens = get_tokens(row["clean_business_name"])
        a_tokens = get_tokens(row["clean_business_address"])
        
        candidates = set()
        
        if country in name_index:
            c_name_idx = name_index[country]
            for t in n_tokens:
                if t in c_name_idx:
                    candidates.update(c_name_idx[t])
                    
        if country in address_index:
            c_addr_idx = address_index[country]
            for t in a_tokens:
                if t in c_addr_idx:
                    candidates.update(c_addr_idx[t])
                    
        sorted_candidates = sorted(list(candidates))
        cands_str = ",".join(sorted_candidates)
        
        if output_file:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"{s1_id}\t{cands_str}\n")
        else:
            candidate_results.append({
                "source1_entity_id": s1_id,
                "candidate_entity_ids": cands_str
            })
            
    if not output_file:
        return pd.DataFrame(
            candidate_results,
            columns=["source1_entity_id", "candidate_entity_ids"]
        )
    return None

if __name__ == "__main__":
    usecols = ["clean_entity_id", "clean_business_name", "clean_business_address", "clean_country"]
    source1 = pd.read_csv("output/cleaned/test_source1_clean.tsv", sep="\t", usecols=usecols)
    source2 = pd.read_csv("output/cleaned/test_source2_clean.tsv", sep="\t", usecols=usecols)
    source3 = pd.read_csv("output/cleaned/test_source3_clean.tsv", sep="\t", usecols=usecols)
    
    print("Source 1:", len(source1))
    print("Source 2:", len(source2))
    print("Source 3:", len(source3))

    generate_candidates(source1, source2, source3, output_file="output/candidate_pairs.tsv")
    
    print("\nCandidate pairs saved successfully to output/candidate_pairs.tsv!\n")