"""
src/preprocessing.py
--------------------
Data Cleaning and Preprocessing Component for Business Entity Matching.

Member 1 Responsibility:
- Clean and normalize raw business records from Source 1, Source 2, and Source 3 TSVs.
- Preserve all original entity/business IDs and raw columns.
- Generate normalized feature fields suitable for downstream Candidate Generation (Member 2).
- Pure offline execution: zero external APIs, web requests, or external databases.
"""

import argparse
import csv
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DataPreprocessing")

# ---------------------------------------------------------------------------
# Project Directories
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "output" / "cleaned"

# ---------------------------------------------------------------------------
# Dynamic Column Configuration
# ---------------------------------------------------------------------------
COLUMN_CONFIG: Dict[str, Dict[str, Optional[str]]] = {
    "source1": {
        "id_col": "entity_id",
        "name_col": "business_name",
        "address_col": "business_address",
        "country_col": "country",
        "city_col": None,
        "state_col": None,
        "postal_code_col": None,
        "phone_col": None,
    },
    "source2": {
        "id_col": "entity_id",
        "name_col": "business_name",
        "address_col": "business_address",
        "country_col": "country",
        "city_col": None,
        "state_col": None,
        "postal_code_col": None,
        "phone_col": None,
    },
    "source3": {
        "id_col": "entity_id",
        "name_col": "business_name",
        "address_col": "business_address",
        "country_col": "country",
        "city_col": None,
        "state_col": None,
        "postal_code_col": None,
        "phone_col": None,
    },
}

# ---------------------------------------------------------------------------
# Regex Normalization Rules (Self-contained, offline)
# ---------------------------------------------------------------------------

# Common Business Terms & Legal Entity Suffixes
BUSINESS_TERM_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Business types & abbreviations
    (re.compile(r"\b(rest\.?|restaurant)\b", re.IGNORECASE), "restaurant"),
    (re.compile(r"\b(gen\.?|general)\b", re.IGNORECASE), "general"),
    (re.compile(r"\b(dept\.?|department)\b", re.IGNORECASE), "department"),
    (re.compile(r"\b(intl\.?|international)\b", re.IGNORECASE), "international"),
    (re.compile(r"\b(natl\.?|national)\b", re.IGNORECASE), "national"),
    (re.compile(r"\b(mfg\.?|manufacturing)\b", re.IGNORECASE), "manufacturing"),
    # Legal Entity Suffixes
    (re.compile(r"\b(pvt\.?\s*ltd\.?|private\s+limited)\b", re.IGNORECASE), "pvt ltd"),
    (re.compile(r"\b(l\.?l\.?c\.?|limited liability company)\b", re.IGNORECASE), "llc"),
    (re.compile(r"\b(inc\.?|incorporated)\b", re.IGNORECASE), "inc"),
    (re.compile(r"\b(corp\.?|corporation)\b", re.IGNORECASE), "corp"),
    (re.compile(r"\b(ltd\.?|limited)\b", re.IGNORECASE), "ltd"),
    (re.compile(r"\b(co\.?|company)\b", re.IGNORECASE), "co"),
    (re.compile(r"\b(gmbh)\b", re.IGNORECASE), "gmbh"),
    (re.compile(r"\b(plc)\b", re.IGNORECASE), "plc"),
    (re.compile(r"\b(pte\.?|private)\b", re.IGNORECASE), "pte"),
    (re.compile(r"\b(pvt\.?)\b", re.IGNORECASE), "pvt"),
    (re.compile(r"\b(s\.?a\.?)\b", re.IGNORECASE), "sa"),
    (re.compile(r"\b(s\.?r\.?l\.?)\b", re.IGNORECASE), "srl"),
    (re.compile(r"\b(b\.?v\.?)\b", re.IGNORECASE), "bv"),
    (re.compile(r"\b(n\.?v\.?)\b", re.IGNORECASE), "nv"),
]

# Common Street / Address Abbreviations
ADDRESS_ROAD_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(st\.?|str\.?|street)\b", re.IGNORECASE), "street"),
    (re.compile(r"\b(rd\.?|road)\b", re.IGNORECASE), "road"),
    (re.compile(r"\b(ave\.?|av\.?|avenue)\b", re.IGNORECASE), "avenue"),
    (re.compile(r"\b(blvd\.?|boulevard)\b", re.IGNORECASE), "boulevard"),
    (re.compile(r"\b(dr\.?|drive)\b", re.IGNORECASE), "drive"),
    (re.compile(r"\b(ln\.?|lane)\b", re.IGNORECASE), "lane"),
    (re.compile(r"\b(ct\.?|court)\b", re.IGNORECASE), "court"),
    (re.compile(r"\b(sq\.?|square)\b", re.IGNORECASE), "square"),
    (re.compile(r"\b(pl\.?|place)\b", re.IGNORECASE), "place"),
    (re.compile(r"\b(hwy\.?|highway)\b", re.IGNORECASE), "highway"),
    (re.compile(r"\b(pkwy\.?|parkway)\b", re.IGNORECASE), "parkway"),
    (re.compile(r"\b(ste\.?|suite)\b", re.IGNORECASE), "suite"),
    (re.compile(r"\b(apt\.?|apartment)\b", re.IGNORECASE), "apartment"),
    (re.compile(r"\b(bldg\.?|building)\b", re.IGNORECASE), "building"),
    (re.compile(r"\b(fl\.?|floor)\b", re.IGNORECASE), "floor"),
    (re.compile(r"\b(rm\.?|room)\b", re.IGNORECASE), "room"),
    (re.compile(r"\b(nr\.?|near)\b", re.IGNORECASE), "near"),
    (re.compile(r"\b(p\.?o\.?\s*box|post office box)\b", re.IGNORECASE), "pobox"),
]


# ---------------------------------------------------------------------------
# Normalization Functions
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """
    Apply standard Unicode NFKD normalization.
    Decomposes accents and converts compatibility characters to standard ASCII forms.
    """
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_compatible = normalized.encode("ascii", "ignore").decode("utf-8")
    return ascii_compatible


def clean_generic_text(val: Any) -> str:
    """
    Safely handle nulls, non-string types, leading/trailing whitespace, and common null representations.
    """
    if val is None:
        return ""
    text = str(val).strip()
    if text.lower() in {"nan", "null", "none", "n/a", "na", "\\n", "\\t", ""}:
        return ""
    return text


def normalize_business_name(
    name: Any,
    standardize_terms: bool = True,
    standardize_symbols: bool = True,
) -> str:
    """
    Normalize business entity names.
    Transformations:
    1. Safe null handling.
    2. Unicode normalization (NFKD).
    3. Lowercase conversion.
    4. Symbol standardization ('&' -> ' and ', '+' -> ' and ', '@' -> ' at ').
    5. Term and legal suffix standardization ('rest.' -> 'restaurant', 'pvt ltd', 'inc', etc.).
    6. Punctuation stripping (retaining alphanumeric characters and spaces).
    7. Whitespace collapse and stripping.
    """
    text = clean_generic_text(name)
    if not text:
        return ""

    text = normalize_unicode(text)
    text = text.lower()

    if standardize_symbols:
        text = re.sub(r"&", " and ", text)
        text = re.sub(r"\+", " and ", text)
        text = re.sub(r"@", " at ", text)

    # Standardize acronyms with dots (e.g. A.B.C. -> abc)
    text = re.sub(r"\b([a-z])\.(?=[a-z]\b|\s|$)", r"\1", text)

    if standardize_terms:
        for pattern, replacement in BUSINESS_TERM_PATTERNS:
            text = pattern.sub(replacement, text)

    # Punctuation normalization
    text = re.sub(r"[^\w\s]", " ", text)

    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_address(
    address: Any,
    standardize_road_types: bool = True,
) -> str:
    """
    Normalize address fields.
    Transformations:
    1. Safe null handling.
    2. Unicode normalization.
    3. Lowercase conversion.
    4. Symbol standardization ('#' -> ' unit ', '&' -> ' and ').
    5. Road/unit abbreviation standardization ('st.' -> 'street', 'rd' -> 'road', etc.).
    6. Preserve numbers, postal codes, and alphanumeric codes.
    7. Punctuation stripping.
    8. Whitespace collapse and stripping.
    """
    text = clean_generic_text(address)
    if not text:
        return ""

    text = normalize_unicode(text)
    text = text.lower()

    text = re.sub(r"#\s*", " unit ", text)
    text = re.sub(r"&", " and ", text)

    if standardize_road_types:
        for pattern, replacement in ADDRESS_ROAD_PATTERNS:
            text = pattern.sub(replacement, text)

    # Punctuation stripping
    text = re.sub(r"[^\w\s]", " ", text)

    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_country_location(val: Any) -> str:
    """
    Normalize country and location fields.
    Treats country as an open set of string labels without hardcoding,
    filtering, or rewriting any values:
    - Safe null handling
    - Unicode normalization (NFKD)
    - Lowercase conversion
    - Whitespace trimming and collapsing
    """
    text = clean_generic_text(val)
    if not text:
        return ""

    text = normalize_unicode(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_postal_code(val: Any) -> str:
    """
    Normalize postal / ZIP codes without destroying alphanumeric formats or leading digits.
    """
    text = clean_generic_text(val)
    if not text:
        return ""
    return re.sub(r"[^\w]", "", text).lower()


def normalize_phone_number(val: Any) -> str:
    """
    Normalize phone numbers to digits only.
    """
    text = clean_generic_text(val)
    if not text:
        return ""
    return re.sub(r"\D", "", text)


# ---------------------------------------------------------------------------
# Schema Inspection Helper (Non-destructive)
# ---------------------------------------------------------------------------

def inspect_dataset_schema(tsv_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Inspect an actual TSV file non-destructively.
    Prints and returns column names, sample rows, and total row count.
    """
    path = Path(tsv_path)
    if not path.exists():
        logger.warning(f"File not found for inspection: {path}")
        return None

    try:
        sample_rows = []
        total_rows = 0
        with open(path, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            fieldnames = reader.fieldnames or []
            for row in reader:
                total_rows += 1
                if len(sample_rows) < 3:
                    sample_rows.append(row)

        logger.info("=" * 60)
        logger.info(f"INSPECTING SCHEMA: {path.name}")
        logger.info(f"Total Rows: {total_rows}")
        logger.info(f"Columns ({len(fieldnames)}): {fieldnames}")
        logger.info("Sample Rows (First 3):")
        for i, s_row in enumerate(sample_rows, 1):
            logger.info(f"  [Row {i}]: {s_row}")
        logger.info("=" * 60)

        return {
            "file_name": path.name,
            "total_rows": total_rows,
            "columns": fieldnames,
            "sample_rows": sample_rows,
        }
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Record & Dataset Preprocessing Core
# ---------------------------------------------------------------------------

def resolve_column(
    headers: List[str],
    configured_col: Optional[str],
    candidate_keys: List[str],
) -> Optional[str]:
    """
    Resolve actual header from configured mappings or standard candidate keywords.
    """
    if configured_col and configured_col in headers:
        return configured_col
    headers_lower_map = {h.lower(): h for h in headers}
    for key in candidate_keys:
        if key.lower() in headers_lower_map:
            return headers_lower_map[key.lower()]
    return None


def preprocess_record(
    row: Dict[str, str],
    col_mapping: Dict[str, Optional[str]],
) -> Dict[str, str]:
    """
    Preprocess a single record dictionary.
    Preserves all original columns and adds normalized '_clean' columns.
    """
    clean_row = dict(row)

    id_col = col_mapping.get("id")
    name_col = col_mapping.get("name")
    address_col = col_mapping.get("address")
    city_col = col_mapping.get("city")
    state_col = col_mapping.get("state")
    country_col = col_mapping.get("country")
    postal_col = col_mapping.get("postal_code")
    phone_col = col_mapping.get("phone")

    # 1. Clean / Standardized ID
    clean_row["clean_entity_id"] = clean_generic_text(row.get(id_col, "")) if id_col else ""

    # 2. Normalized Name
    clean_name = normalize_business_name(row.get(name_col, "")) if name_col else ""
    clean_row["clean_business_name"] = clean_name

    # 3. Normalized Address
    clean_address = normalize_address(row.get(address_col, "")) if address_col else ""
    clean_row["clean_business_address"] = clean_address

    # 4. Normalized Country / Location
    clean_country = normalize_country_location(row.get(country_col, "")) if country_col else ""
    clean_row["clean_country"] = clean_country

    if city_col:
        clean_row["clean_city"] = normalize_country_location(row.get(city_col, ""))
    if state_col:
        clean_row["clean_state"] = normalize_country_location(row.get(state_col, ""))
    if postal_col:
        clean_row["clean_postal_code"] = normalize_postal_code(row.get(postal_col, ""))
    if phone_col:
        clean_row["clean_phone"] = normalize_phone_number(row.get(phone_col, ""))

    # 5. Composite Clean Full Text (Unified feature for Candidate Generation / Blocking)
    tokens = [clean_name, clean_address, clean_country]
    clean_row["clean_full_text"] = " ".join([t for t in tokens if t]).strip()

    return clean_row


# ---------------------------------------------------------------------------
# File Pipeline Handlers
# ---------------------------------------------------------------------------

def process_tsv_file(
    input_path: Path,
    output_path: Path,
    source_alias: str,
    explicit_mapping: Optional[Dict[str, Optional[str]]] = None,
) -> bool:
    """
    Read an input TSV file in read-only mode, apply non-destructive normalization,
    and output a cleaned TSV file.
    """
    if not input_path.exists():
        logger.warning(f"Input file not found: {input_path}")
        return False

    logger.info(f"Reading {input_path} (read-only)...")
    try:
        with open(input_path, mode="r", encoding="utf-8", errors="replace") as infile:
            reader = csv.DictReader(infile, delimiter="\t")
            fieldnames = list(reader.fieldnames or [])

            if not fieldnames:
                logger.error(f"No headers found in {input_path}")
                return False

            cfg = explicit_mapping or COLUMN_CONFIG.get(source_alias, {})
            col_map = {
                "id": resolve_column(fieldnames, cfg.get("id_col"), ["entity_id", "id", "business_id", "record_id"]),
                "name": resolve_column(fieldnames, cfg.get("name_col"), ["business_name", "name", "company_name", "title"]),
                "address": resolve_column(fieldnames, cfg.get("address_col"), ["business_address", "address", "street", "addr"]),
                "country": resolve_column(fieldnames, cfg.get("country_col"), ["country", "country_code", "nation"]),
                "city": resolve_column(fieldnames, cfg.get("city_col"), ["city", "town"]),
                "state": resolve_column(fieldnames, cfg.get("state_col"), ["state", "province"]),
                "postal_code": resolve_column(fieldnames, cfg.get("postal_code_col"), ["postal_code", "zip", "postcode"]),
                "phone": resolve_column(fieldnames, cfg.get("phone_col"), ["phone", "telephone", "contact"]),
            }

            logger.info(f"[{source_alias}] Resolved columns: {col_map}")

            # Define output fieldnames: all original fields + new normalized fields
            new_fields = [
                "clean_entity_id",
                "clean_business_name",
                "clean_business_address",
                "clean_country",
                "clean_full_text",
            ]
            output_fieldnames = fieldnames + [f for f in new_fields if f not in fieldnames]

            output_path.parent.mkdir(parents=True, exist_ok=True)
            row_count = 0

            with open(output_path, mode="w", encoding="utf-8", newline="") as outfile:
                writer = csv.DictWriter(outfile, fieldnames=output_fieldnames, delimiter="\t")
                writer.writeheader()

                for row in reader:
                    cleaned_row = preprocess_record(row, col_map)
                    writer.writerow(cleaned_row)
                    row_count += 1

            logger.info(f"Successfully processed {row_count} records -> {output_path}")
            return True

    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}")
        return False


def run_all_preprocessing(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """
    Process all available challenge files across dataset directories.
    """
    logger.info("=" * 60)
    logger.info("Starting Data Preprocessing Pipeline (Member 1)")
    logger.info(f"Input Data Directory:  {data_dir}")
    logger.info(f"Output Directory:      {output_dir}")
    logger.info("=" * 60)

    # Search paths for train and test partitions
    file_mapping = [
        # Training files
        (data_dir / "train" / "train_source1.tsv", output_dir / "train_source1_clean.tsv", "source1"),
        (data_dir / "train" / "train_source2.tsv", output_dir / "train_source2_clean.tsv", "source2"),
        (data_dir / "train" / "train_source3.tsv", output_dir / "train_source3_clean.tsv", "source3"),
        # Test files
        (data_dir / "test" / "test_source1.tsv", output_dir / "test_source1_clean.tsv", "source1"),
        (data_dir / "test" / "test_source2.tsv", output_dir / "test_source2_clean.tsv", "source2"),
        (data_dir / "test" / "test_source3.tsv", output_dir / "test_source3_clean.tsv", "source3"),
        # Direct data/ fallback paths
        (BASE_DIR / "data" / "train_source1.tsv", output_dir / "train_source1_clean.tsv", "source1"),
        (BASE_DIR / "data" / "train_source2.tsv", output_dir / "train_source2_clean.tsv", "source2"),
        (BASE_DIR / "data" / "train_source3.tsv", output_dir / "train_source3_clean.tsv", "source3"),
    ]

    processed_count = 0
    seen_outputs = set()

    for in_path, out_path, alias in file_mapping:
        if out_path in seen_outputs:
            continue
        if in_path.exists():
            success = process_tsv_file(
                input_path=in_path,
                output_path=out_path,
                source_alias=alias,
            )
            if success:
                processed_count += 1
                seen_outputs.add(out_path)

    logger.info("=" * 60)
    logger.info(f"Preprocessing completed. Total files generated: {processed_count}")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Member 1: Data Cleaning and Preprocessing for Business Entity Matching."
    )
    parser.add_argument(
        "--inspect",
        type=str,
        default=None,
        help="Path to a TSV file to inspect schema and sample rows without processing.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_DIR),
        help="Path to dataset directory containing train/ and test/ TSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Path to directory where cleaned TSV files will be saved.",
    )

    args = parser.parse_args()

    if args.inspect:
        inspect_dataset_schema(args.inspect)
    else:
        run_all_preprocessing(
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output_dir),
        )


if __name__ == "__main__":
    main()
