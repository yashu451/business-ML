import pandas as pd

# ============================================================
# 1. LOAD CLEANED TEST DATA (Moved to main block below)
# ============================================================

# ============================================================
# 2. FUNCTION TO CHECK TOKEN OVERLAP
# ============================================================

def has_token_overlap(text1, text2):

    tokens1 = set(str(text1).lower().split())
    tokens2 = set(str(text2).lower().split())

    return len(tokens1.intersection(tokens2)) > 0


# ============================================================
# 3. STORE CANDIDATES
# ============================================================

candidate_results = []


# ============================================================
# 4. GENERATE CANDIDATES FUNCTION
# ============================================================

def generate_candidates(source1, source2, source3):
    candidate_results = []
    
    for _, row in source1.iterrows():
    
        s1_id = row["clean_entity_id"]
        country = row["clean_country"]
    
        s1_name = row["clean_business_name"]
        s1_address = row["clean_business_address"]
    
        # --------------------------------------------------------
        # BLOCKING RULE 1: SAME COUNTRY
        # --------------------------------------------------------
    
        country_s2 = source2[
            source2["clean_country"] == country
        ]
    
        country_s3 = source3[
            source3["clean_country"] == country
        ]
    
        # --------------------------------------------------------
        # BLOCKING RULE 2A: BUSINESS NAME TOKEN OVERLAP
        # --------------------------------------------------------
    
        name_candidates_s2 = country_s2[
            country_s2["clean_business_name"].apply(
                lambda name: has_token_overlap(s1_name, name)
            )
        ]
    
        name_candidates_s3 = country_s3[
            country_s3["clean_business_name"].apply(
                lambda name: has_token_overlap(s1_name, name)
            )
        ]
    
        # --------------------------------------------------------
        # BLOCKING RULE 2B: BUSINESS ADDRESS TOKEN OVERLAP
        # --------------------------------------------------------
    
        address_candidates_s2 = country_s2[
            country_s2["clean_business_address"].apply(
                lambda address: has_token_overlap(s1_address, address)
            )
        ]
    
        address_candidates_s3 = country_s3[
            country_s3["clean_business_address"].apply(
                lambda address: has_token_overlap(s1_address, address)
            )
        ]
    
        # --------------------------------------------------------
        # COMBINE NAME + ADDRESS CANDIDATES
        # --------------------------------------------------------
    
        candidates_s2 = pd.concat(
            [
                name_candidates_s2,
                address_candidates_s2
            ]
        ).drop_duplicates()
    
        candidates_s3 = pd.concat(
            [
                name_candidates_s3,
                address_candidates_s3
            ]
        ).drop_duplicates()
    
        # --------------------------------------------------------
        # COMBINE SOURCE 2 + SOURCE 3 IDs
        # --------------------------------------------------------
    
        candidate_ids = (
            candidates_s2["clean_entity_id"].tolist()
            +
            candidates_s3["clean_entity_id"].tolist()
        )
    
        # Remove duplicate IDs
        candidate_ids = sorted(set(candidate_ids))
    
        # --------------------------------------------------------
        # SAVE RESULT FOR THIS SOURCE 1 ENTITY
        # --------------------------------------------------------
    
        candidate_results.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(candidate_ids)
        })
        
    return pd.DataFrame(
        candidate_results,
        columns=[
            "source1_entity_id",
            "candidate_entity_ids"
        ]
    )

if __name__ == "__main__":
    # ============================================================
    # 1. LOAD CLEANED TEST DATA
    # ============================================================
    source1 = pd.read_csv("output/cleaned/test_source1_clean.tsv", sep="\t")
    source2 = pd.read_csv("output/cleaned/test_source2_clean.tsv", sep="\t")
    source3 = pd.read_csv("output/cleaned/test_source3_clean.tsv", sep="\t")
    
    print("Source 1:", len(source1))
    print("Source 2:", len(source2))
    print("Source 3:", len(source3))

    # ============================================================
    # 5. CREATE OUTPUT DATAFRAME
    # ============================================================
    
    candidate_df = generate_candidates(source1, source2, source3)
    
    # ============================================================
    # 6. SAVE FINAL CANDIDATE FILE
    # ============================================================
    
    candidate_df.to_csv(
        "output/candidate_pairs.tsv",
        sep="\t",
        index=False
    )
    
    # ============================================================
    # 7. DISPLAY RESULTS
    # ============================================================
    
    print("\nCandidate pairs saved successfully!\n")
    print(candidate_df)