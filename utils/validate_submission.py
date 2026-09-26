import argparse
import csv
import sys
import os

def print_errors(errors):
    for i, err in enumerate(errors, 1):
        print(f"Error {i}: {err}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Path to the directory containing test_source*.tsv")
    args = parser.parse_args()

    errors = []
    
    # 1. Load valid test IDs
    s1_ids = set()
    s2_s3_ids = set()
    
    def load_ids(filename, target_set):
        path = os.path.join(args.test_dir, filename)
        if not os.path.exists(path):
            errors.append(f"Missing test file: {path}")
            return
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                # Assuming 'entity_id' is the primary key column in original datasets
                if "entity_id" in row:
                    target_set.add(row["entity_id"].strip())
                else:
                    errors.append(f"Column 'entity_id' missing in {filename}")
                    
    load_ids("test_source1.tsv", s1_ids)
    load_ids("test_source2.tsv", s2_s3_ids)
    load_ids("test_source3.tsv", s2_s3_ids)
    
    if errors:
        print_errors(errors)
        sys.exit(1)

    # 2. Function to read a submission file
    def read_submission(filepath, required_fields):
        data = {}
        if not os.path.exists(filepath):
            errors.append(f"Missing file: {filepath}")
            return data
            
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            try:
                headers = next(reader)
            except StopIteration:
                errors.append(f"File {filepath} is empty.")
                return data
                
            # Check exact column names
            if headers != required_fields:
                errors.append(f"Invalid columns in {filepath}. Expected {required_fields}, got {headers}")
                return data
            
            for row_num, row in enumerate(reader, start=2):
                if not row:
                    continue
                s1_id = row[0].strip()
                val = row[1].strip() if len(row) > 1 else ""
                
                # Check duplicate Source 1 rows
                if s1_id in data:
                    errors.append(f"Duplicate Source 1 row in {filepath}: {s1_id}")
                else:
                    data[s1_id] = val
        return data

    # 3. Parse files
    matching_data = read_submission(args.matching, ["source1_entity_id", "matched_entity_ids"])
    candidate_data = read_submission(args.candidate, ["source1_entity_id", "candidate_entity_ids"])
    
    if errors:
        print_errors(errors)
        sys.exit(1)
        
    # 4. Check if every Source 1 test entity appears exactly once
    matching_s1_keys = set(matching_data.keys())
    candidate_s1_keys = set(candidate_data.keys())
    
    missing_matching = s1_ids - matching_s1_keys
    extra_matching = matching_s1_keys - s1_ids
    if missing_matching:
        errors.append(f"Missing {len(missing_matching)} test Source 1 entities in matching file.")
    if extra_matching:
        errors.append(f"Extra {len(extra_matching)} unknown Source 1 entities in matching file.")
        
    missing_candidate = s1_ids - candidate_s1_keys
    extra_candidate = candidate_s1_keys - s1_ids
    if missing_candidate:
        errors.append(f"Missing {len(missing_candidate)} test Source 1 entities in candidate file.")
    if extra_candidate:
        errors.append(f"Extra {len(extra_candidate)} unknown Source 1 entities in candidate file.")

    if errors:
        print_errors(errors[:20]) # Limit output just in case it's huge
        if len(errors) > 20: print(f"... and {len(errors) - 20} more errors.")
        sys.exit(1)

    # 5. Check candidates and matches lists
    for s1_id, cands_str in candidate_data.items():
        cands = [c.strip() for c in cands_str.split(",") if c.strip()]
        cand_set = set(cands)
        
        # No duplicate IDs within a candidate list
        if len(cands) != len(cand_set):
            errors.append(f"Duplicate IDs in candidate list for {s1_id}")
            
        # Candidate IDs are only existing test Source 2/Source 3 IDs
        invalid_cands = cand_set - s2_s3_ids
        if invalid_cands:
            errors.append(f"Invalid candidate ID(s) {invalid_cands} for {s1_id}. Must be from test Source 2/3.")

    for s1_id, matches_str in matching_data.items():
        matches = [m.strip() for m in matches_str.split(",") if m.strip()]
        match_set = set(matches)
        
        # No duplicate IDs within a matched list
        if len(matches) != len(match_set):
            errors.append(f"Duplicate IDs in matched list for {s1_id}")
            
        # Matched IDs are only existing test Source 2/Source 3 IDs
        invalid_matches = match_set - s2_s3_ids
        if invalid_matches:
            errors.append(f"Invalid matched ID(s) {invalid_matches} for {s1_id}. Must be from test Source 2/3.")
                
        # Every final matched ID appears in the candidate list for that Source 1 entity
        cands = [c.strip() for c in candidate_data.get(s1_id, "").split(",") if c.strip()]
        cand_set = set(cands)
        if not match_set.issubset(cand_set):
            invalid_subset = match_set - cand_set
            errors.append(f"Matched ID(s) {invalid_subset} for {s1_id} are not in its candidate list.")

    if errors:
        print_errors(errors[:20])
        if len(errors) > 20: print(f"... and {len(errors) - 20} more errors.")
        sys.exit(1)
        
    print("PASS")
    sys.exit(0)

if __name__ == "__main__":
    main()
