"""
Unit tests for src/preprocessing.py.
Run with: pytest tests/test_preprocessing.py -v
"""
import pandas as pd
import pytest

from src.preprocessing import (
    clean_text,
    normalize_product,
    filter_to_target_products,
    filter_empty_narratives,
    calculate_word_count,
    clean_narratives,
)
from src.config import COL_PRODUCT, COL_NARRATIVE, COL_COMPLAINT_ID


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal DataFrame mimicking CFPB structure for testing."""
    return pd.DataFrame({
        COL_COMPLAINT_ID: ["1", "2", "3", "4", "5"],
        COL_PRODUCT: [
            "Credit card",
            "Credit card or prepaid card",
            "Mortgage",
            "Checking or savings account",
            "Personal loan",
        ],
        COL_NARRATIVE: [
            "I was charged twice for the same purchase.",
            "  ",                        # blank — should be filtered
            "My mortgage payment was not applied.",
            None,                         # NaN — should be filtered
            "I Am Writing To File A Complaint about my loan.",
        ],
    })


# ── clean_text ────────────────────────────────────────────────────────────────

class TestCleanText:
    def test_lowercases_text(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_removes_boilerplate(self):
        result = clean_text("I am writing to file a complaint about my card.")
        assert "i am writing to file a complaint" not in result

    def test_removes_special_characters(self):
        result = clean_text("My card# was charged $500!")
        # Digits, letters, and basic punctuation should remain
        assert "#" not in result
        assert "$" not in result

    def test_collapses_whitespace(self):
        result = clean_text("too    many    spaces")
        assert "  " not in result

    def test_returns_empty_string_for_non_string(self):
        assert clean_text(None) == ""
        assert clean_text(123) == ""

    def test_preserves_meaningful_content(self):
        result = clean_text("I was charged twice for the same purchase.")
        assert "charged" in result
        assert "purchase" in result


# ── normalize_product ─────────────────────────────────────────────────────────

class TestNormalizeProduct:
    def test_maps_credit_card(self, sample_df):
        df = normalize_product(sample_df)
        assert df.loc[df[COL_COMPLAINT_ID] == "1", "product_category"].values[0] == "Credit Card"

    def test_maps_credit_card_or_prepaid_card(self, sample_df):
        df = normalize_product(sample_df)
        assert df.loc[df[COL_COMPLAINT_ID] == "2", "product_category"].values[0] == "Credit Card"

    def test_maps_savings_account(self, sample_df):
        df = normalize_product(sample_df)
        assert df.loc[df[COL_COMPLAINT_ID] == "4", "product_category"].values[0] == "Savings Account"

    def test_unmapped_product_gives_nan(self, sample_df):
        df = normalize_product(sample_df)
        assert pd.isna(df.loc[df[COL_COMPLAINT_ID] == "3", "product_category"].values[0])


# ── filter_to_target_products ─────────────────────────────────────────────────

class TestFilterToTargetProducts:
    def test_removes_unmapped_products(self, sample_df):
        df = normalize_product(sample_df)
        df = filter_to_target_products(df)
        assert "3" not in df[COL_COMPLAINT_ID].values  # Mortgage excluded

    def test_retains_target_products(self, sample_df):
        df = normalize_product(sample_df)
        df = filter_to_target_products(df)
        assert set(df["product_category"].unique()).issubset(
            {"Credit Card", "Personal Loan", "Savings Account", "Money Transfer"}
        )

    def test_raises_if_column_missing(self, sample_df):
        with pytest.raises(ValueError, match="product_category"):
            filter_to_target_products(sample_df)


# ── filter_empty_narratives ───────────────────────────────────────────────────

class TestFilterEmptyNarratives:
    def test_removes_nan_narratives(self, sample_df):
        df = filter_empty_narratives(sample_df)
        assert df[COL_NARRATIVE].isna().sum() == 0

    def test_removes_blank_string_narratives(self, sample_df):
        df = filter_empty_narratives(sample_df)
        assert (df[COL_NARRATIVE].str.strip() == "").sum() == 0

    def test_retains_valid_rows(self, sample_df):
        df = filter_empty_narratives(sample_df)
        assert "1" in df[COL_COMPLAINT_ID].values


# ── calculate_word_count ──────────────────────────────────────────────────────

class TestCalculateWordCount:
    def test_correct_word_count(self):
        df = pd.DataFrame({COL_NARRATIVE: ["one two three"]})
        df = calculate_word_count(df)
        assert df["narrative_word_count"].iloc[0] == 3

    def test_zero_for_empty_string(self):
        df = pd.DataFrame({COL_NARRATIVE: [""]})
        df = calculate_word_count(df)
        assert df["narrative_word_count"].iloc[0] == 0

    def test_zero_for_nan(self):
        df = pd.DataFrame({COL_NARRATIVE: [None]})
        df = calculate_word_count(df)
        assert df["narrative_word_count"].iloc[0] == 0