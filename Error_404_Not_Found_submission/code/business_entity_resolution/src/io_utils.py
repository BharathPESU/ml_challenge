"""
io_utils.py — I/O utilities for the Business Entity Resolution pipeline.

Supports:
  - Reading and writing TSV files from local filesystem
  - Reading and writing TSV files from/to S3 (s3:// URIs)
  - Column validation
  - Safe handling of NaN / empty strings
  - Directory creation helpers

All TSV files use sep="\t" as required by the challenge specification.
"""

import io
import logging
import os
from pathlib import Path
from typing import List, Optional
import urllib.request

import pandas as pd

try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import ClientError, NoCredentialsError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

logger = logging.getLogger(__name__)

# Required columns per file type
REQUIRED_COLUMNS_SOURCE = {"entity_id", "business_name", "business_address", "country"}
REQUIRED_COLUMNS_GROUND_TRUTH = {"source1_entity_id", "matched_entity_ids"}
REQUIRED_COLUMNS_MATCHING_RESULTS = {"source1_entity_id", "matched_entity_ids"}
REQUIRED_COLUMNS_CANDIDATE_PAIRS = {"source1_entity_id", "candidate_entity_ids"}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def is_s3_path(path: str) -> bool:
    """Return True if path is an S3 URI (starts with s3://)."""
    return path.startswith("s3://")


def parse_s3_path(s3_path: str):
    """
    Parse an S3 URI into (bucket, key).

    Example:
        parse_s3_path("s3://my-bucket/prefix/file.tsv")
        -> ("my-bucket", "prefix/file.tsv")
    """
    assert s3_path.startswith("s3://"), f"Not a valid S3 path: {s3_path}"
    parts = s3_path[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def get_s3_client(unsigned: bool = False, region_name: Optional[str] = None):
    """
    Get a boto3 S3 client.
    
    If unsigned=True or if default initialization fails due to missing credentials,
    returns an unsigned client for accessing public S3 buckets.
    """
    if not HAS_BOTO3:
        raise RuntimeError("boto3 is not installed.")
    
    region = region_name or os.environ.get("AWS_REGION", "eu-north-1")
    
    if unsigned:
        return boto3.client("s3", region_name=region, config=BotoConfig(signature_version=UNSIGNED))
    try:
        return boto3.client("s3", region_name=region)
    except Exception:
        return boto3.client("s3", region_name=region, config=BotoConfig(signature_version=UNSIGNED))


def fetch_s3_object_body(bucket: str, key: str) -> bytes:
    """
    Fetch raw bytes of an object from S3.
    Tries authenticated boto3 client first, then unsigned boto3 client,
    and falls back to regional HTTP GET for public S3 buckets.
    """
    regions = [os.environ.get("AWS_REGION", "eu-north-1"), "eu-north-1", "us-east-1"]

    if HAS_BOTO3:
        for r in regions:
            for unsigned in [False, True]:
                try:
                    s3_client = get_s3_client(unsigned=unsigned, region_name=r)
                    response = s3_client.get_object(Bucket=bucket, Key=key)
                    return response["Body"].read()
                except Exception as e:
                    logger.debug("S3 fetch attempt failed (region=%s, unsigned=%s): %s", r, unsigned, e)
        logger.info("Boto3 fetch attempts exhausted. Falling back to HTTP fetch...")

    # Fallback to direct regional HTTP request for public S3 bucket
    for r in ["eu-north-1", "us-east-1"]:
        urls = [
            f"https://{bucket}.s3.{r}.amazonaws.com/{key}",
            f"https://s3.{r}.amazonaws.com/{bucket}/{key}",
        ]
        for url in urls:
            try:
                logger.info("Fetching public S3 file via HTTP URL: %s", url)
                req = urllib.request.Request(url, headers={"User-Agent": "BER-Pipeline/1.0"})
                with urllib.request.urlopen(req) as resp:
                    return resp.read()
            except Exception as e:
                logger.debug("HTTP fetch failed for %s: %s", url, e)

    raise RuntimeError(
        f"Failed to fetch s3://{bucket}/{key}. Ensure the S3 bucket is public (Bucket Policy allows s3:GetObject) "
        f"or your SageMaker IAM role has AmazonS3ReadOnlyAccess permissions."
    )


def ensure_local_dir(path: str) -> None:
    """Create local directory (and parents) if it does not exist."""
    dir_path = os.path.dirname(os.path.abspath(path))
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    logger.debug("Ensured directory exists: %s", dir_path)


def make_output_dir(output_dir: str) -> None:
    """Create the output directory if it does not exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Output directory ready: %s", output_dir)


# ---------------------------------------------------------------------------
# TSV reading
# ---------------------------------------------------------------------------

def _read_tsv_from_bytes(raw_bytes: bytes, source_label: str) -> pd.DataFrame:
    """Parse raw bytes as a UTF-8 tab-separated file."""
    try:
        df = pd.read_csv(
            io.BytesIO(raw_bytes),
            sep="\t",
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
        )
    except UnicodeDecodeError:
        logger.warning("UTF-8 decode failed for %s; retrying with latin-1.", source_label)
        df = pd.read_csv(
            io.BytesIO(raw_bytes),
            sep="\t",
            dtype=str,
            keep_default_na=False,
            encoding="latin-1",
        )
    return df


def read_tsv(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Read a tab-separated file from a local path or S3 URI.

    Parameters
    ----------
    path : str
        Local file path or s3:// URI.
    nrows : int, optional
        If provided, read only the first `nrows` rows (for debug sampling).

    Returns
    -------
    pd.DataFrame
        DataFrame with all columns as str, no NaN values (empty → "").
    """
    if is_s3_path(path):
        logger.info("Reading TSV from S3: %s", path)
        bucket, key = parse_s3_path(path)
        raw_bytes = fetch_s3_object_body(bucket, key)
        df = _read_tsv_from_bytes(raw_bytes, path)
        if nrows is not None:
            df = df.head(nrows)
    else:
        logger.info("Reading TSV from local path: %s", path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            nrows=nrows,
        )

    df = _sanitize_dataframe(df)
    logger.info("Loaded %d rows, %d columns from: %s", len(df), len(df.columns), path)
    return df


def _sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace any NaN values with empty strings and strip column names.
    Downcast 'country' to categorical to save memory.
    """
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        df[col] = df[col].fillna("").astype(str)
        
    if "country" in df.columns:
        df["country"] = df["country"].astype("category")
    
    return df


# ---------------------------------------------------------------------------
# TSV writing
# ---------------------------------------------------------------------------

def write_tsv(df: pd.DataFrame, path: str) -> None:
    """
    Write a DataFrame as a tab-separated file to a local path or S3 URI.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to write.
    path : str
        Local file path or s3:// URI.
    """
    tsv_content = df.to_csv(sep="\t", index=False)

    if is_s3_path(path):
        logger.info("Writing TSV to S3: %s", path)
        bucket, key = parse_s3_path(path)
        s3_client = boto3.client("s3")
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=tsv_content.encode("utf-8"),
        )
        logger.info("Successfully wrote %d rows to S3: %s", len(df), path)
    else:
        logger.info("Writing TSV to local path: %s", path)
        ensure_local_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(tsv_content)
        logger.info("Successfully wrote %d rows to: %s", len(df), path)

# ---------------------------------------------------------------------------
# Parquet Caching (Memory Optimization)
# ---------------------------------------------------------------------------

def save_intermediate_parquet(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to compressed Parquet format for fast, memory-efficient reloading."""
    ensure_local_dir(path)
    # Convert any categoricals back to string if necessary, or just save
    df.to_parquet(path, engine="pyarrow", compression="snappy")
    logger.debug("Saved intermediate parquet: %s", path)

def load_intermediate_parquet(path: str) -> pd.DataFrame:
    """Load intermediate DataFrame from Parquet."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet file not found: {path}")
    logger.debug("Loading intermediate parquet: %s", path)
    return pd.read_parquet(path, engine="pyarrow")


# ---------------------------------------------------------------------------
# Column validation
# ---------------------------------------------------------------------------

def validate_columns(df: pd.DataFrame, required: set, source_label: str) -> None:
    """
    Validate that a DataFrame contains all required columns.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"[{source_label}] Missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    logger.debug("[%s] Column validation passed. Columns: %s", source_label, list(df.columns))


def validate_source_file(df: pd.DataFrame, source_label: str) -> None:
    """Validate a source data file (S1, S2, or S3)."""
    validate_columns(df, REQUIRED_COLUMNS_SOURCE, source_label)


def validate_ground_truth_file(df: pd.DataFrame) -> None:
    """Validate the ground truth file."""
    validate_columns(df, REQUIRED_COLUMNS_GROUND_TRUTH, "ground_truth")


def validate_matching_results(df: pd.DataFrame) -> None:
    """Validate the matching_results.tsv output file."""
    validate_columns(df, REQUIRED_COLUMNS_MATCHING_RESULTS, "matching_results")


def validate_candidate_pairs(df: pd.DataFrame) -> None:
    """Validate the candidate_pairs.tsv output file."""
    validate_columns(df, REQUIRED_COLUMNS_CANDIDATE_PAIRS, "candidate_pairs")


# ---------------------------------------------------------------------------
# S3 upload/download helpers for artifacts
# ---------------------------------------------------------------------------

def upload_file_to_s3(local_path: str, s3_bucket: str, s3_key: str) -> str:
    """
    Upload a local file to S3.

    Returns
    -------
    str
        The full S3 URI of the uploaded file.
    """
    s3_client = boto3.client("s3")
    s3_client.upload_file(local_path, s3_bucket, s3_key)
    s3_uri = f"s3://{s3_bucket}/{s3_key}"
    logger.info("Uploaded %s → %s", local_path, s3_uri)
    return s3_uri


def download_file_from_s3(s3_bucket: str, s3_key: str, local_path: str) -> None:
    """Download a file from S3 to a local path."""
    ensure_local_dir(local_path)
    raw_bytes = fetch_s3_object_body(s3_bucket, s3_key)
    with open(local_path, "wb") as f:
        f.write(raw_bytes)
    logger.info("Downloaded s3://%s/%s → %s", s3_bucket, s3_key, local_path)


def read_text_from_s3(s3_bucket: str, s3_key: str) -> str:
    """Read a text file from S3 and return its content as a string."""
    raw_bytes = fetch_s3_object_body(s3_bucket, s3_key)
    return raw_bytes.decode("utf-8")


def write_text_to_s3(content: str, s3_bucket: str, s3_key: str) -> None:
    """Write a string as a text file to S3."""
    s3_client = boto3.client("s3")
    s3_client.put_object(Bucket=s3_bucket, Key=s3_key, Body=content.encode("utf-8"))
    logger.info("Wrote text content to s3://%s/%s", s3_bucket, s3_key)
