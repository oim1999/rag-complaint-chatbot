import pandas as pd
from src.config import SAMPLE_SIZE, RANDOM_SEED


def stratified_sample(
    df: pd.DataFrame,
    sample_size: int = SAMPLE_SIZE,
    stratify_col: str = "product_category",
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Draw a reproducible stratified sample with proportional representation
    across all values of `stratify_col`.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, filtered complaint DataFrame.
    sample_size : int
        Target total rows to sample.
    stratify_col : str
        Column to stratify on (default: 'product_category').
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Stratified sample of length == min(sample_size, len(df)).

    Raises
    ------
    ValueError
        If df is empty or stratify_col is absent.
    """
    if df.empty:
        raise ValueError("Cannot sample from an empty DataFrame.")
    if stratify_col not in df.columns:
        raise ValueError(
            f"Stratify column '{stratify_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    n = min(sample_size, len(df))
    category_counts = df[stratify_col].value_counts()
    total = category_counts.sum()

    frames = []
    for category, count in category_counts.items():
        proportion = count / total
        n_category = max(1, round(proportion * n))
        n_category = min(n_category, count)  # never sample more than available
        subset = df[df[stratify_col] == category].sample(
            n=n_category, random_state=seed
        )
        frames.append(subset)

    sample = pd.concat(frames, ignore_index=True)

    # Correct for rounding drift
    if len(sample) > n:
        sample = sample.sample(n=n, random_state=seed)
    elif len(sample) < n:
        remainder = df.drop(index=sample.index, errors="ignore")
        shortfall = n - len(sample)
        if len(remainder) >= shortfall:
            extra = remainder.sample(n=shortfall, random_state=seed)
            sample = pd.concat([sample, extra], ignore_index=True)

    if sample.empty:
        raise ValueError(
            "Stratified sampling produced an empty result. "
            "Verify that the input DataFrame is filtered and non-empty."
        )

    return sample.reset_index(drop=True)