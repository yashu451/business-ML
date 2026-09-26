import pandas as pd


# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================

source1 = pd.read_csv(
    "output/cleaned/train_source1_clean.tsv",
    sep="\t"
)

source2 = pd.read_csv(
    "output/cleaned/train_source2_clean.tsv",
    sep="\t"
)

source3 = pd.read_csv(
    "output/cleaned/train_source3_clean.tsv",
    sep="\t"
)

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)


# ============================================================
# 2. NAME TOKEN FUNCTION
# ============================================================

def get_name_tokens(name):
    return set(str(name).lower().split())


# ============================================================
# 3. EVALUATION COUNTERS
# ============================================================

total_true_matches = 0
retained_true_matches = 0
total_candidates = 0


# ============================================================
# 4. RUN BLOCKING
# ============================================================

for _, row in source1.iterrows():

    s1_id = row["clean_entity_id"]
    country = row["clean_country"]
    s1_name = row["clean_business_name"]
    s1_address = row["clean_business_address"]

    s1_tokens = get_name_tokens(s1_name)
    s1_address_tokens = get_name_tokens(s1_address)


    # Same-country blocking

    country_s2 = source2[
        source2["clean_country"] == country
    ]

    country_s3 = source3[
        source3["clean_country"] == country
    ]


    # Name-token blocking

    name_candidates_s2 = country_s2[
        country_s2["clean_business_name"].apply(
            lambda name: len(
                s1_tokens.intersection(
                    get_name_tokens(name)
                )
            ) > 0
        )
    ]

    name_candidates_s3 = country_s3[
        country_s3["clean_business_name"].apply(
            lambda name: len(
                s1_tokens.intersection(
                    get_name_tokens(name)
                )
            ) > 0
        )
    ]


    # Address-token blocking

    address_candidates_s2 = country_s2[
        country_s2["clean_business_address"].apply(
            lambda address: len(
                s1_address_tokens.intersection(
                    get_name_tokens(address)
                )
            ) > 0
        )
    ]

    address_candidates_s3 = country_s3[
        country_s3["clean_business_address"].apply(
            lambda address: len(
                s1_address_tokens.intersection(
                    get_name_tokens(address)
                )
            ) > 0
        )
    ]


    # Combine candidates

    candidates_s2 = pd.concat([name_candidates_s2, address_candidates_s2]).drop_duplicates()
    candidates_s3 = pd.concat([name_candidates_s3, address_candidates_s3]).drop_duplicates()

    candidate_ids = set(
        candidates_s2["clean_entity_id"].tolist()
        +
        candidates_s3["clean_entity_id"].tolist()
    )


    total_candidates += len(candidate_ids)


    # Get true matches

    truth_row = ground_truth[
        ground_truth["source1_entity_id"] == s1_id
    ]

    true_ids = set()

    if not truth_row.empty:

        value = truth_row.iloc[0]["matched_entity_ids"]

        if pd.notna(value) and str(value).strip() != "":
            true_ids = set(str(value).split(","))


    # Count true matches

    total_true_matches += len(true_ids)

    retained = true_ids.intersection(candidate_ids)

    retained_true_matches += len(retained)


    print("\n", s1_id)
    print("Candidates:", sorted(candidate_ids))
    print("True matches:", sorted(true_ids))
    print("Retained:", sorted(retained))


# ============================================================
# 5. FINAL METRICS
# ============================================================

print("\n====================================")
print("BLOCKING EVALUATION")
print("====================================")

print("Total true matches:", total_true_matches)
print("True matches retained:", retained_true_matches)
print("Total candidates:", total_candidates)

if total_true_matches > 0:

    recall = (
        retained_true_matches / total_true_matches
    ) * 100

    print(
        f"Candidate Recall: {recall:.2f}%"
    )