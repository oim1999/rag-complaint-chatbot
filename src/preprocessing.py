import re
import pandas as pd
from pathlib import Path
from src.config import (
    PRODUCT_MAPPING,
    TARGET_PRODUCTS,
    COL_PRODUCT,
    COL_NARRATIVE,
    COL_COMPLAINT_ID,
    COL_ISSUE,
    COL_SUB_ISSUE,
    COL_COMPANY,
    COL_STATE,
    COL_DATE_RECEIVED,
    FILTERED_DATA_PATH,
    PROCESSED_DATA_DIR,
)

# ── Boilerplate patterns common in CFPB narratives ────────────────────────────
_BOILERPLATE_PATTERNS = [
    r"i am writing to (file|submit|make) (a )?complaint",
    r"to whom it may concern[,\.]?",
    r"dear (sir|madam|consumer financial protection bureau|cfpb)[,\.]?",
    r"i am writing to bring (this|your) attention",
    r"i would like to (file|submit|report)",
    r"please (help|assist) me",
]

_BOILERPLATE_RE = re.compile(
    "|".join(_BOILERPLATE_PATTERNS),
    flags=re.IGNORECASE,
)

# Redaction placeholders used by CFPB (e.g. XXXX, XX/XX/XXXX)
_REDACTION_RE = re.compile(r"\bx{2,}\b", flags=re.IGNORECASE)

# Anything that is not a basic alphanumeric character or simple punctuation
_SPECIAL_CHARS_RE = re.compile(r"[^a-z0-9\s\.,!\?'\-]")


# ── Individual transformation functions ───────────────────────────────────────

def normalize_product(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw CFPB Product strings to the four canonical category labels.
    Unmatched products produce NaN in 'product_category' and are
    dropped in filter_to_target_products().
    """
    df = df.copy()
    df["product_category"] = (
        df[COL_PRODUCT]
        .str.lower()
        .str.strip()
        .map(PRODUCT_MAPPING)
    )
    return df


def filter_to_target_products(df: pd.DataFrame) -> pd.DataFrame:
    """Retain only rows whose product_category is one of the four targets."""
    if "product_category" not in df.columns:
        raise ValueError(
            "Column 'product_category' not found. "
            "Run normalize_product() before filter_to_target_products()."
        )
    return df[df["product_category"].isin(TARGET_PRODUCTS)].copy()


def filter_empty_narratives(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where the consumer complaint narrative is null or blank."""
    mask = (
        df[COL_NARRATIVE].notna()
        & df[COL_NARRATIVE].str.strip().ne("")
    )
    return df[mask].copy()


def calculate_word_count(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'narrative_word_count' column (integer) to df."""
    df = df.copy()
    df["narrative_word_count"] = (
        df[COL_NARRATIVE]
        .fillna("")
        .str.split()
        .str.len()
    )
    return df


def clean_text(text: str) -> str:
    """
    Full normalization pipeline for a single narrative string:
      1. Lowercase
      2. Strip boilerplate opening phrases
      3. Remove CFPB redaction placeholders (XXXX)
      4. Remove special characters (keep basic punctuation)
      5. Collapse whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _BOILERPLATE_RE.sub(" ", text)
    text = _REDACTION_RE.sub("[redacted]", text)
    text = _SPECIAL_CHARS_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_narratives(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clean_text() to COL_NARRATIVE; result stored in 'cleaned_narrative'."""
    df = df.copy()
    df["cleaned_narrative"] = df[COL_NARRATIVE].apply(clean_text)
    return df


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and standardize column names for the downstream pipeline.
    Only columns that are present in df are included (tolerant of missing optionals).
    """
    rename_map = {
        COL_COMPLAINT_ID: "complaint_id",
        "product_category": "product_category",
        COL_ISSUE: "issue",
        COL_SUB_ISSUE: "sub_issue",
        COL_COMPANY: "company",
        COL_STATE: "state",
        COL_DATE_RECEIVED: "date_received",
        "cleaned_narrative": "cleaned_narrative",
        "narrative_word_count": "narrative_word_count",
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    return df[list(existing.keys())].rename(columns=existing)


def save_filtered_dataset(df: pd.DataFrame, path: Path = FILTERED_DATA_PATH) -> None:
    """Persist the cleaned, filtered DataFrame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✓ Saved filtered dataset → {path}  ({len(df):,} rows)")


# ── Full pipeline convenience function ────────────────────────────────────────

def run_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the complete Task 1 preprocessing pipeline in the correct order:
      normalize_product → filter_to_target_products → filter_empty_narratives
      → calculate_word_count → clean_narratives → select_output_columns
    """
    df = normalize_product(df)
    df = filter_to_target_products(df)
    df = filter_empty_narratives(df)
    df = calculate_word_count(df)
    df = clean_narratives(df)
    df = select_output_columns(df)
    return df