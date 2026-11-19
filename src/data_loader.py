"""
AeroFare - Safe Data & Artifact Loader Module
Handles caching, relative path resolution, and schema validation.
"""

from pathlib import Path
import json
import pandas as pd
import joblib

try:
    import streamlit as st
    cache_data = st.cache_data
    cache_resource = st.cache_resource
except Exception:
    def cache_data(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    def cache_resource(*args, **kwargs):
        def decorator(f):
            return f
        return decorator


def get_project_root() -> Path:
    """Returns the absolute Path of the project root directory."""
    return Path(__file__).resolve().parent.parent


@cache_data()
def load_dataset(sample_size: int = None) -> pd.DataFrame:
    """
    Loads Clean_Dataset.csv with optional subsampling for fast interactive previews.
    """
    root = get_project_root()
    dataset_path = root / "data" / "Clean_Dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Clean_Dataset.csv not found at {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    if sample_size and len(df) > sample_size:
        return df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    return df


@cache_resource()
def load_model():
    """
    Loads the trained production pipeline from models/best_model.pkl.
    """
    root = get_project_root()
    model_path = root / "models" / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)


@cache_data()
def load_model_metadata() -> dict:
    """
    Loads model training metadata and feature schema.
    """
    root = get_project_root()
    meta_path = root / "models" / "model_metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, "r") as f:
        return json.load(f)


@cache_data()
def load_model_comparison() -> pd.DataFrame:
    """
    Loads the model benchmark evaluation table.
    """
    root = get_project_root()
    path = root / "outputs" / "model_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@cache_data()
def load_feature_importance() -> pd.DataFrame:
    """
    Loads feature importance weights if available.
    """
    root = get_project_root()
    path = root / "outputs" / "feature_importance.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@cache_data()
def load_diagnostics() -> dict:
    """
    Loads diagnostic summary tables by class, booking window, and consolidated metrics.
    """
    root = get_project_root()
    outputs_dir = root / "outputs"

    diagnostics = {}
    class_p = outputs_dir / "class_performance.csv"
    if class_p.exists():
        diagnostics["class_performance"] = pd.read_csv(class_p)

    window_p = outputs_dir / "booking_window_performance.csv"
    if window_p.exists():
        diagnostics["booking_window_performance"] = pd.read_csv(window_p)

    model_diag = outputs_dir / "model_diagnostics.csv"
    if model_diag.exists():
        diagnostics["model_diagnostics"] = pd.read_csv(model_diag)

    return diagnostics
