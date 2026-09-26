# Business Entity Resolution

This project matches business records from multiple data sources that refer to the same real-world business.

## Project Pipeline

The system follows these steps:

1. Preprocessing
2. Candidate generation / blocking
3. Candidate matching
4. Final submission generation

## Project Structure

- `dataset/` - Training and test datasets
- `src/preprocessing.py` - Cleans and normalizes business data
- `src/candidate_generation.py` - Generates candidate matches using blocking
- `src/evaluate_blocking.py` - Evaluates candidate recall
- `src/matching_model.py` - Scores candidates and generates final matches
- `utils/validate_submission.py` - Validates the output format
- `output/candidate_pairs.tsv` - Generated candidate pairs
- `output/matching_results.tsv` - Final matching results
- `requirements.txt` - Python dependencies

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
