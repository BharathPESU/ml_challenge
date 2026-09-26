"""
cleaning.py — Text normalization and cleaning for Business Entity Resolution.

All cleaning functions are designed to be applied identically to BOTH training
and test data to prevent train/test distribution shift.

Key design decisions:
  - Preserve raw columns; add new cleaned columns alongside them.
  - Country is treated as an open-set string feature — NEVER hard-code US/India.
  - Legal suffix stripping is conservative — only remove clearly non-distinctive suffixes.
  - Address cleaning preserves all numeric tokens (street numbers, PINs, ZIP codes).
  - Unicode/accent normalization handles French characters (é, è, ç, etc.)
    which appear in the test set.
"""

import logging
import re
import unicodedata
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legal suffix vocabulary (conservative — do NOT add distinctive words)
# ---------------------------------------------------------------------------

# Ordered longest-first so longer phrases match before sub-phrases
_LEGAL_SUFFIXES = [
    "private limited",
    "pvt ltd",
    "pvt. ltd.",
    "pvt. ltd",
    "pvt ltd.",
    "p. ltd.",
    "p.ltd",
    "(p) ltd",
    "(p) limited",
    "public limited",
    "incorporated",
    "corporation",
    "unlimited",
    "limited",
    "private",
    "company",
    "llp",
    "llc",
    "l.l.c.",
    "l.l.c",
    "l.l.p.",
    "l.l.p",
    "corp.",
    "corp",
    "inc.",
    "inc",
    "ltd.",
    "ltd",
    "pvt.",
    "pvt",
    "co.",
    r"\bco\b",
    "plc",
    "s.a.",
    "s.a",
    "s.a.r.l.",
    "s.a.r.l",
    "sarl",
    "sas",
    "gmbh",
    "ag",
    "bv",
    "nv",
    "oy",
    "ab",
    "as",
]

# Pre-compile suffix patterns for efficiency
_LEGAL_SUFFIX_PATTERNS = [
    re.compile(r"\b" + re.escape(s) + r"\b", re.IGNORECASE)
    if not s.startswith(r"\b")
    else re.compile(s, re.IGNORECASE)
    for s in _LEGAL_SUFFIXES
]


# ---------------------------------------------------------------------------
# Address abbreviation expansion mapping
# ---------------------------------------------------------------------------

_ADDRESS_ABBREV_MAP = {
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b": "highway",
    r"\bfwy\b": "freeway",
    r"\bste\b": "suite",
    r"\bfl\b": "floor",
    r"\bflr\b": "floor",
    r"\bbldg\b": "building",
    r"\bapt\b": "apartment",
    r"\bno\b": "number",
    r"\bnear\b": "near",
    r"\bopp\b": "opposite",
    r"\bopps\b": "opposite",
    r"\bnth\b": "north",
    r"\bsth\b": "south",
    r"\best\b": "east",
    r"\bwst\b": "west",
}

_ADDRESS_ABBREV_PATTERNS = [
    (re.compile(pat, re.IGNORECASE), expansion)
    for pat, expansion in _ADDRESS_ABBREV_MAP.items()
]


# ---------------------------------------------------------------------------
# Core normalization functions
# ---------------------------------------------------------------------------

def strip_accents(text: str) -> str:
    """
    Convert Unicode characters to their ASCII equivalents by stripping diacritics.

    Examples:
        "Société" → "Societe"
        "Café"    → "Cafe"
        "Zürich"  → "Zurich"

    This is essential for the test set which contains French business names.
    """
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space and strip ends."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_punctuation(text: str) -> str:
    """
    Standardize punctuation:
      - Replace '&' with 'and'
      - Replace '/' with ' '
      - Remove quotation marks, parentheses that are not part of IDs
      - Normalize hyphens and dashes to spaces
    """
    text = text.replace("&", " and ")
    text = re.sub(r"[/\\|]", " ", text)
    text = re.sub(r"[-–—]", " ", text)
    text = re.sub(r"[\"'`''""«»]", "", text)
    text = re.sub(r"[(){}[\]]", " ", text)
    text = re.sub(r"[,;:!?]", " ", text)
    return text


def normalize_text(text: str) -> str:
    """
    General-purpose text normalization pipeline:
      1. Strip accents / Unicode → ASCII
      2. Lowercase
      3. Normalize punctuation
      4. Normalize whitespace

    Applied to all text fields before feature computation.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = strip_accents(text)
    text = text.lower()
    text = normalize_punctuation(text)
    text = normalize_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# Business name normalization
# ---------------------------------------------------------------------------

def normalize_business_name(name: str) -> str:
    """
    Produce a normalized business name (business_name_norm).

    Normalization:
      - Full normalize_text() pipeline
      - Period removal after abbreviations (e.g., "Corp." → "Corp")

    Legal suffixes are NOT removed in this version (that is core name).
    """
    name = normalize_text(name)
    # Remove trailing periods (e.g. "Inc." → "Inc")
    name = re.sub(r"\.(?=\s|$)", " ", name)
    name = normalize_whitespace(name)
    return name


def normalize_business_name_core(name: str) -> str:
    """
    Produce a core business name with legal suffixes removed (business_name_core).

    Conservative suffix stripping:
      - Only well-known legal entity markers are removed.
      - Meaningful words are preserved.
      - Multiple passes handle compound suffixes (e.g., "Pvt. Ltd." → "").

    Example:
        "Acme Solutions Private Limited" → "acme solutions"
        "Google LLC"                     → "google"
        "SBI Co. Ltd"                    → "sbi"
    """
    name_norm = normalize_business_name(name)

    # Iteratively strip suffixes until stable
    prev = None
    while prev != name_norm:
        prev = name_norm
        for pattern in _LEGAL_SUFFIX_PATTERNS:
            name_norm = pattern.sub(" ", name_norm)
        name_norm = normalize_whitespace(name_norm)

    return name_norm


# ---------------------------------------------------------------------------
# Business address normalization
# ---------------------------------------------------------------------------

def normalize_business_address(address: str) -> str:
    """
    Produce a normalized business address (business_address_norm).

    Normalization:
      - Full normalize_text() pipeline
      - Street abbreviation expansion
      - Preserve all numeric tokens (house numbers, PINs, ZIP codes)

    Numbers are NOT removed because exact numeric overlap is a strong matching signal.
    """
    if not address or not address.strip():
        return ""

    address = normalize_text(address)

    # Expand common street abbreviations
    for pattern, expansion in _ADDRESS_ABBREV_PATTERNS:
        address = pattern.sub(expansion, address)

    address = normalize_whitespace(address)
    return address


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def tokenize_text(text: str) -> List[str]:
    """
    Tokenize normalized text into individual tokens (words).

    Returns an empty list for empty input.
    """
    if not text or not text.strip():
        return []
    return text.split()


def extract_numeric_tokens(text: str) -> List[str]:
    """
    Extract all numeric tokens (digit sequences) from text.

    Examples:
        "12 MG Road 560001" → ["12", "560001"]
        "Floor 3, Tower B"  → ["3"]

    Used for numeric overlap features (street numbers, PIN/ZIP codes).
    """
    return re.findall(r"\d+", text)


def extract_digit_sequences(text: str) -> List[str]:
    """
    Extract purely numeric sequences of length >= 3 (PINs, ZIPs, building numbers).
    Shorter digit strings (like floor numbers) are filtered out.
    """
    return [d for d in re.findall(r"\d+", text) if len(d) >= 3]


# ---------------------------------------------------------------------------
# DataFrame-level cleaning
# ---------------------------------------------------------------------------

def _add_missingness_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary missingness indicator columns for name, address, and country."""
    df["name_missing"] = (df["business_name"].str.strip() == "").astype(int)
    df["address_missing"] = (df["business_address"].str.strip() == "").astype(int)
    df["country_missing"] = (df["country"].str.strip() == "").astype(int)
    return df


def _add_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic length / token count features for raw fields."""
    df["name_char_length"] = df["business_name"].str.len()
    df["address_char_length"] = df["business_address"].str.len()
    df["name_token_count"] = df["business_name"].apply(
        lambda x: len(x.split()) if x.strip() else 0
    )
    df["address_token_count"] = df["business_address"].apply(
        lambda x: len(x.split()) if x.strip() else 0
    )
    return df


def clean_source_dataframe(df: pd.DataFrame, source_label: str = "") -> pd.DataFrame:
    """
    Apply all cleaning/normalization to a source dataframe (S1, S2, or S3).

    New columns added:
      - business_name_norm    : normalized business name
      - business_name_core    : normalized name with legal suffixes stripped
      - business_address_norm : normalized address
      - name_missing          : 1 if name is empty
      - address_missing       : 1 if address is empty
      - country_missing       : 1 if country is empty
      - name_char_length      : character length of raw name
      - address_char_length   : character length of raw address
      - name_token_count      : word count of raw name
      - address_token_count   : word count of raw address

    IMPORTANT:
      - country is normalized (lowercased, stripped) but treated as OPEN-SET string.
      - Raw original columns are preserved unchanged.
      - Identical logic is applied to training AND test data.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe with columns: entity_id, business_name,
        business_address, country.
    source_label : str
        Label for logging (e.g., "S1_train", "S2_test").

    Returns
    -------
    pd.DataFrame
        Original dataframe with additional cleaned columns.
    """
    logger.info("[%s] Cleaning %d records...", source_label, len(df))

    df = df.copy()

    # Ensure all string columns are non-null strings to prevent AttributeError/TypeError
    for col in ["business_name", "business_address", "country"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # Normalize names
    df["business_name_norm"] = df["business_name"].apply(normalize_business_name)
    df["business_name_core"] = df["business_name"].apply(normalize_business_name_core)

    # Normalize addresses
    df["business_address_norm"] = df["business_address"].apply(normalize_business_address)

    # Normalize country as open-set lowercase string (NOT one-hot encoded)
    df["country_norm"] = df["country"].apply(
        lambda c: normalize_whitespace(strip_accents(c).lower())
    )

    # Add missingness indicators
    df = _add_missingness_indicators(df)

    # Add length features
    df = _add_length_features(df)

    logger.info("[%s] Cleaning complete. Columns: %s", source_label, list(df.columns))
    return df
