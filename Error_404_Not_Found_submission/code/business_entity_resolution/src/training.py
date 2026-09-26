"""
training.py — XGBoost model training via Amazon SageMaker SDK.

Model task:
  Binary classification: given a (S1, candidate) pair's feature vector,
  predict probability that both records refer to the same real-world business.

  Input  : pairwise feature vector (no entity IDs)
  Output : match probability ∈ [0, 1]
  Labels : 1 = true match, 0 = non-match

LEAKAGE PREVENTION:
  - entity_id columns are EXCLUDED from training features.
  - matched_entity_ids is EXCLUDED from training features.
  - Validation labels are NEVER used during training.
  - Test data is NEVER used during training or for negative sampling.
  - Feature column order is deterministic and saved with the model.

EVALUATION METRIC NOTE:
  AUC-PR (aucpr) is used as the XGBoost internal monitoring metric.
  The OFFICIAL competition metric is entity-level macro F0.5,
  computed separately via evaluation.py after threshold optimization.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature / label extraction helpers
# ---------------------------------------------------------------------------

# Columns that must NEVER enter the XGBoost feature matrix
_FORBIDDEN_FEATURE_COLS = {
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "is_match",
    "matched_entity_ids",
    "entity_id",
}


def get_feature_matrix(
    features_df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Extract the pure numerical feature matrix from a features DataFrame.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature DataFrame (output of features.extract_features).
    feature_columns : List[str], optional
        Explicit list of feature columns to use (for consistent ordering
        between training and inference). If None, inferred automatically.

    Returns
    -------
    np.ndarray
        2D float array of shape (n_pairs, n_features).
    """
    if feature_columns is not None:
        cols = [c for c in feature_columns if c in features_df.columns]
    else:
        cols = sorted([c for c in features_df.columns if c not in _FORBIDDEN_FEATURE_COLS])

    missing = [c for c in (feature_columns or []) if c not in features_df.columns]
    if missing:
        logger.warning("Feature columns missing from DataFrame: %s", missing)

    X = features_df[cols].values.astype(np.float32)
    return X


def get_labels(features_df: pd.DataFrame) -> np.ndarray:
    """
    Extract the binary label vector from a labelled features DataFrame.

    Returns
    -------
    np.ndarray
        1D int array of 0/1 labels.
    """
    if "is_match" not in features_df.columns:
        raise ValueError("'is_match' column not found. Use labelled pairs for training.")
    return features_df["is_match"].values.astype(np.int32)


# ---------------------------------------------------------------------------
# Local XGBoost training (for dev / quick experiments)
# ---------------------------------------------------------------------------

def train_xgboost_local(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any],
    num_round: int = 300,
    early_stopping_rounds: int = 30,
    feature_columns: Optional[List[str]] = None,
    model_output_path: Optional[str] = None,
    random_seed: int = 42,
) -> Any:
    """
    Train an XGBoost model locally (without SageMaker).

    Used for:
      - Local development and quick iteration
      - Hyperparameter search experiments before launching SageMaker jobs

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training feature matrix and labels.
    X_val, y_val : np.ndarray
        Validation feature matrix and labels.
    params : Dict[str, Any]
        XGBoost hyperparameters (excludes num_round and seed).
    num_round : int
        Maximum boosting rounds.
    early_stopping_rounds : int
        Stop if validation metric does not improve for this many rounds.
    feature_columns : List[str], optional
        List of feature column names for artifact saving.
    model_output_path : str, optional
        Local path to save the trained model artifact (.json).
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    xgboost.Booster
        Trained XGBoost booster.
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost is required. Install with: pip install xgboost")

    params = dict(params)
    params["seed"] = random_seed

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    evals = [(dtrain, "train"), (dval, "validation")]

    logger.info(
        "Starting local/Kaggle XGBoost training on GPU: %d train pairs, %d val pairs, %d rounds",
        len(X_train), len(X_val), num_round,
    )
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=50,
    )

    if model_output_path:
        os.makedirs(os.path.dirname(os.path.abspath(model_output_path)), exist_ok=True)
        booster.save_model(model_output_path)
        logger.info("Model saved to: %s", model_output_path)

        # Save feature schema alongside model
        if feature_columns:
            schema_path = model_output_path.replace(".json", "_feature_schema.json")
            with open(schema_path, "w") as f:
                json.dump({"feature_columns": feature_columns}, f, indent=2)
            logger.info("Feature schema saved to: %s", schema_path)

    return booster


# ---------------------------------------------------------------------------
# SageMaker XGBoost training
# ---------------------------------------------------------------------------

def prepare_sagemaker_csv(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    output_path: str,
    feature_columns: List[str],
) -> None:
    """
    Write a feature matrix + labels to CSV in SageMaker XGBoost format.

    SageMaker's built-in XGBoost container expects:
      - First column  = label
      - Remaining cols = features
      - No header row

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature DataFrame.
    labels : np.ndarray
        Binary labels (0 or 1).
    output_path : str
        Local path to write the CSV.
    feature_columns : List[str]
        Ordered feature columns (MUST match training and inference).
    """
    X = get_feature_matrix(features_df, feature_columns=feature_columns)
    label_col = labels.reshape(-1, 1)
    data = np.hstack([label_col, X])

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savetxt(output_path, data, delimiter=",", fmt="%.6f")
    logger.info("SageMaker CSV written: %s (%d rows)", output_path, len(data))


def upload_training_data_to_s3(
    train_csv_path: str,
    val_csv_path: str,
    s3_bucket: str,
    s3_prefix: str,
    experiment_id: str,
) -> Tuple[str, str]:
    """
    Upload training and validation CSVs to S3 for SageMaker training.

    Returns
    -------
    Tuple[str, str]
        (train_s3_uri, val_s3_uri)
    """
    try:
        import boto3
    except ImportError:
        raise ImportError("boto3 is required. Install with: pip install boto3")

    s3_client = boto3.client("s3")

    train_key = f"{s3_prefix}/{experiment_id}/train/train.csv"
    val_key = f"{s3_prefix}/{experiment_id}/validation/validation.csv"

    s3_client.upload_file(train_csv_path, s3_bucket, train_key)
    s3_client.upload_file(val_csv_path, s3_bucket, val_key)

    train_s3_uri = f"s3://{s3_bucket}/{s3_prefix}/{experiment_id}/train/"
    val_s3_uri = f"s3://{s3_bucket}/{s3_prefix}/{experiment_id}/validation/"

    logger.info("Uploaded training data → %s", train_s3_uri)
    logger.info("Uploaded validation data → %s", val_s3_uri)
    return train_s3_uri, val_s3_uri


def launch_sagemaker_training_job(
    train_s3_uri: str,
    val_s3_uri: str,
    model_output_s3_uri: str,
    sagemaker_role: str,
    hyperparams: Dict[str, Any],
    experiment_id: str,
    aws_region: str = "us-east-1",
    instance_type: str = "ml.m5.2xlarge",
    instance_count: int = 1,
    xgboost_version: str = "1.7-1",
) -> Dict[str, str]:
    """
    Launch a SageMaker managed XGBoost training job.

    Parameters
    ----------
    train_s3_uri : str
        S3 URI to the training data folder.
    val_s3_uri : str
        S3 URI to the validation data folder.
    model_output_s3_uri : str
        S3 URI where trained model artifacts will be saved.
    sagemaker_role : str
        IAM role ARN for SageMaker execution.
    hyperparams : Dict[str, Any]
        XGBoost hyperparameters (string keys and values for SageMaker).
    experiment_id : str
        Unique experiment identifier for the training job name.
    aws_region : str
        AWS region.
    instance_type : str
        SageMaker training instance type.
    instance_count : int
        Number of training instances.
    xgboost_version : str
        SageMaker XGBoost framework version.

    Returns
    -------
    Dict[str, str]
        {
            "job_name": str,
            "model_artifact_s3": str,
            "experiment_id": str,
        }
    """
    try:
        import sagemaker
        from sagemaker.xgboost import XGBoost
    except ImportError:
        raise ImportError("sagemaker is required. Install with: pip install sagemaker")

    job_name = f"entity-resolution-{experiment_id}-{int(time.time())}"

    # Convert all hyperparams to strings (SageMaker requirement)
    str_hyperparams = {k: str(v) for k, v in hyperparams.items()}

    sm_session = sagemaker.Session()
    estimator = XGBoost(
        entry_point=None,  # Using built-in XGBoost container
        framework_version=xgboost_version,
        role=sagemaker_role,
        instance_count=instance_count,
        instance_type=instance_type,
        output_path=model_output_s3_uri,
        hyperparameters=str_hyperparams,
        sagemaker_session=sm_session,
        base_job_name=job_name,
    )

    train_input = sagemaker.inputs.TrainingInput(
        s3_data=train_s3_uri,
        content_type="text/csv",
    )
    val_input = sagemaker.inputs.TrainingInput(
        s3_data=val_s3_uri,
        content_type="text/csv",
    )

    logger.info("Launching SageMaker XGBoost training job: %s", job_name)
    estimator.fit({"train": train_input, "validation": val_input}, job_name=job_name)

    model_artifact_s3 = f"{model_output_s3_uri}/{job_name}/output/model.tar.gz"
    logger.info("Training complete. Model artifact: %s", model_artifact_s3)

    return {
        "job_name": job_name,
        "model_artifact_s3": model_artifact_s3,
        "experiment_id": experiment_id,
    }


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_xgboost_model_local(model_path: str) -> Any:
    """
    Load a locally saved XGBoost booster from a .json file.

    Returns
    -------
    xgboost.Booster
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost is required.")

    booster = xgb.Booster()
    booster.load_model(model_path)
    logger.info("Loaded XGBoost model from: %s", model_path)
    return booster


def predict_probabilities(
    booster: Any,
    features_df: pd.DataFrame,
    feature_columns: List[str],
) -> np.ndarray:
    """
    Score candidate pairs using a trained XGBoost booster.

    Parameters
    ----------
    booster : xgboost.Booster
        Trained booster.
    features_df : pd.DataFrame
        Feature DataFrame for inference pairs.
    feature_columns : List[str]
        Ordered feature column list from training (ensures identical schema).

    Returns
    -------
    np.ndarray
        Match probabilities ∈ [0, 1] of shape (n_pairs,).
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost is required.")

    X = get_feature_matrix(features_df, feature_columns=feature_columns)
    dmatrix = xgb.DMatrix(X)
    probs = booster.predict(dmatrix)
    logger.info("Predicted probabilities for %d pairs.", len(probs))
    return probs


# ---------------------------------------------------------------------------
# Experiment tracking
# ---------------------------------------------------------------------------

def save_experiment_config(
    experiment_id: str,
    params: Dict[str, Any],
    feature_columns: List[str],
    job_name: Optional[str],
    model_artifact: Optional[str],
    output_dir: str,
) -> str:
    """
    Save a JSON record of the experiment configuration.

    Parameters
    ----------
    experiment_id : str
    params : Dict[str, Any]
        XGBoost hyperparameters used.
    feature_columns : List[str]
        Feature columns used.
    job_name : str, optional
        SageMaker job name (if applicable).
    model_artifact : str, optional
        S3 or local path to model artifact.
    output_dir : str
        Local directory to save the config.

    Returns
    -------
    str
        Path to the saved config file.
    """
    config_record = {
        "experiment_id": experiment_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "params": params,
        "feature_columns": feature_columns,
        "sagemaker_job_name": job_name,
        "model_artifact": model_artifact,
    }
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, f"{experiment_id}_config.json")

    with open(config_path, "w") as f:
        json.dump(config_record, f, indent=2)

    logger.info("Experiment config saved: %s", config_path)
    return config_path


# ---------------------------------------------------------------------------
# Default baseline hyperparameters
# ---------------------------------------------------------------------------

BASELINE_XGBOOST_PARAMS: Dict[str, Any] = {
    "objective":        "binary:logistic",
    "eval_metric":      "aucpr",
    "tree_method":      "hist",
    "device":           "cuda:0",
    "max_depth":        6,
    "eta":              0.05,
    "min_child_weight": 3,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "gamma":            0.0,
    "reg_alpha":        0.1,
    "reg_lambda":       2.0,
}
