from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROCESSED_DATA_DIR / "figures"

RAW_DATA_PATH = RAW_DATA_DIR / "complaints.csv"
FILTERED_DATA_PATH = PROCESSED_DATA_DIR / "filtered_complaints.csv"

# ── Vector store ──────────────────────────────────────────────────────────────
VECTOR_STORE_DIR = ROOT_DIR / "vector_store"
CHROMA_COLLECTION_NAME = "complaint_chunks"

# ── Sampling ──────────────────────────────────────────────────────────────────
SAMPLE_SIZE = 10_000
RANDOM_SEED = 42

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ── Target product categories (canonical names used throughout pipeline) ───────
TARGET_PRODUCTS = [
    "Credit Card",
    "Personal Loan",
    "Savings Account",
    "Money Transfer",
]

# ── Raw CFPB column names (as confirmed from actual dataset header) ────────────
COL_DATE_RECEIVED = "Date received"
COL_PRODUCT = "Product"
COL_SUB_PRODUCT = "Sub-product"
COL_ISSUE = "Issue"
COL_SUB_ISSUE = "Sub-issue"
COL_NARRATIVE = "Consumer complaint narrative"
COL_COMPANY_PUBLIC_RESPONSE = "Company public response"
COL_COMPANY = "Company"
COL_STATE = "State"
COL_ZIP = "ZIP code"
COL_TAGS = "Tags"
COL_CONSENT = "Consumer consent provided?"
COL_SUBMITTED_VIA = "Submitted via"
COL_DATE_SENT = "Date sent to company"
COL_COMPANY_RESPONSE = "Company response to consumer"
COL_TIMELY = "Timely response?"
COL_DISPUTED = "Consumer disputed?"
COL_COMPLAINT_ID = "Complaint ID"

# Minimum columns the pipeline cannot run without
REQUIRED_COLUMNS = [
    COL_PRODUCT,
    COL_NARRATIVE,
    COL_COMPLAINT_ID,
    COL_ISSUE,
    COL_SUB_ISSUE,
    COL_COMPANY,
    COL_STATE,
    COL_DATE_RECEIVED,
]

# ── Product mapping: raw CFPB strings (lowercase) → canonical category ─────────
PRODUCT_MAPPING = {
    # Credit Card
    "credit card": "Credit Card",
    "credit card or prepaid card": "Credit Card",
    "prepaid card": "Credit Card",
    # Personal Loan
    "personal loan": "Personal Loan",
    "payday loan, title loan, or personal loan": "Personal Loan",
    "payday loan": "Personal Loan",
    "title loan": "Personal Loan",
    # Savings Account
    "checking or savings account": "Savings Account",
    "bank account or service": "Savings Account",
    "savings account": "Savings Account",
    # Money Transfer
    "money transfers": "Money Transfer",
    "money transfer, virtual currency, or money service": "Money Transfer",
    "money transfer": "Money Transfer",
}