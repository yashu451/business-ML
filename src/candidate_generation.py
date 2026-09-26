import pandas as pd


# ============================================================
# 1. LOAD CLEANED TEST DATA
# ============================================================

source1 = pd.read_csv(
    "output/cleaned/test_source1_clean.tsv",
    sep="\t"
)

source2 = pd.read_csv(
    "output/cleaned/test_source2_clean.tsv",
    sep="\t"
)

source3 = pd.read_csv(
    "output/cleaned/test_source3_clean.tsv",
    sep="\t"
)


print("Source 1:", len(source1))
print("Source 2:", len(source2))
print("Source 3:", len(source3))


# ============================================================
# 2. FUNCTION TO GET NAME TOKENS
# ============================================================

def get_name_tokens(name):
    return set(str(name).lower().split())


# ============================================================
# 3. STORE CANDIDATES
# ============================================================

candidate_results = []


# ============================================================
# 4. GENERATE CANDIDATES FOR EVERY SOURCE 1 ENTITY
# ============================================================

for _, row in source1.iterrows():

    # --------------------------------------------------------
    # Source 1 information
    # --------------------------------------------------------

    s1_id = row["clean_entity_id"]
    country = row["clean_country"]
    s1_name = row["clean_business_name"]

    s1_tokens = get_name_tokens(s1_name)


    # --------------------------------------------------------
    # BLOCKING RULE 1: SAME COUNTRY
    # --------------------------------------------------------

    candidates_s2 = source2[
        source2["clean_country"] == country
    ]

    candidates_s3 = source3[
        source3["clean_country"] == country
    ]


    # --------------------------------------------------------
    # BLOCKING RULE 2: BUSINESS NAME TOKEN OVERLAP
    # --------------------------------------------------------

    candidates_s2 = candidates_s2[
        candidates_s2["clean_business_name"].apply(
            lambda name: len(
                s1_tokens.intersection(
                    get_name_tokens(name)
                )
            ) > 0
        )
    ]

    candidates_s3 = candidates_s3[
        candidates_s3["clean_business_name"].apply(
            lambda name: len(
                s1_tokens.intersection(
                    get_name_tokens(name)
                )
            ) > 0
        )
    ]


    # --------------------------------------------------------
    # COMBINE SOURCE 2 + SOURCE 3
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


# ============================================================
# 5. CREATE OUTPUT DATAFRAME
# ============================================================

candidate_df = pd.DataFrame(
    candidate_results,
    columns=[
        "source1_entity_id",
        "candidate_entity_ids"
    ]
)


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