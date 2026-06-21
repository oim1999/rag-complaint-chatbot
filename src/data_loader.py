import pandas as pd
from pathlib import Path
from src.config import (
    RAW_DATA_PATH,
    REQUIRED_COLUMNS,
    COL_COMPLAINT_ID,
    COL_PRODUCT,
    COL_NARRATIVE,
    PRODUCT_MAPPING,
)

# Only load these columns from disk — ignore the rest entirely.
# This alone cuts memory by ~60% on the CFPB file.
_USECOLS = [
    "Date received",
    "Product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company",
    "State",
    "Complaint ID",
]

# Products worth keeping while streaming — filter early, before concat.
_KEEP_PRODUCTS = set(PRODUCT_MAPPING.keys())  # lowercase raw strings


def load_raw_data(
    filepath: Path = RAW_DATA_PATH,
    chunksize: int = 50_000,
) -> pd.DataFrame:
    """
    Load the raw CFPB complaint CSV using chunked streaming.

    Reads `chunksize` rows at a time, drops columns and rows that will
    never be used, then concatenates the survivors. Peak RAM usage is
    roughly (chunksize × row_bytes) rather than (full_file × row_bytes).

    Parameters
    ----------
    filepath : Path
        Path to the raw CSV file.
    chunksize : int
        Rows per chunk. 50,000 is safe on 8 GB RAM. Lower to 20,000
        if you still get memory errors.

    Returns
    -------
    pd.DataFrame
        Filtered raw data containing only the four target product groups
        and only the columns needed by the pipeline.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at filepath.
    ValueError
        If any required columns are absent from the file.
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {filepath}\n"
            f"Place the CFPB complaints CSV at:\n  {filepath}"
        )

    kept_chunks = []
    total_rows_seen = 0

    reader = pd.read_csv(
        filepath,
        usecols=_USECOLS,
        dtype={COL_COMPLAINT_ID: str},
        chunksize=chunksize,
        low_memory=True,
        on_bad_lines="skip",      # skip any malformed rows silently
    )

    for i, chunk in enumerate(reader):
        total_rows_seen += len(chunk)

        # Validate columns once on the very first chunk
        if i == 0:
            _validate_columns(chunk, filepath)

        # Filter to target products immediately — before any allocation
        mask = chunk[COL_PRODUCT].str.lower().str.strip().isin(_KEEP_PRODUCTS)
        filtered = chunk[mask]

        if not filtered.empty:
            kept_chunks.append(filtered)

        # Progress indicator — useful for a 464K-row file
        if (i + 1) % 10 == 0:
            kept = sum(len(c) for c in kept_chunks)
            print(f"  Processed {total_rows_seen:,} rows — kept {kept:,} so far...",
                  end="\r")

    print(f"\n✓ Done. Scanned {total_rows_seen:,} rows total.")

    if not kept_chunks:
        raise ValueError(
            "No rows matched the target product categories after streaming the full file.\n"
            "Check that PRODUCT_MAPPING keys in config.py match the file's Product column."
        )

    df = pd.concat(kept_chunks, ignore_index=True)
    print(f"  Kept {len(df):,} rows matching target products.")
    return df


def _validate_columns(chunk: pd.DataFrame, filepath: Path) -> None:
    """Raise ValueError listing every required column absent from the chunk."""
    # Only validate against columns we actually requested
    requested = set(_USECOLS)
    missing = [col for col in REQUIRED_COLUMNS if col in requested and col not in chunk.columns]
    if missing:
        raise ValueError(
            f"Required columns missing from {filepath}:\n"
            + "\n".join(f"  ✗ '{col}'" for col in missing)
            + f"\n\nColumns found: {list(chunk.columns)}"
        )