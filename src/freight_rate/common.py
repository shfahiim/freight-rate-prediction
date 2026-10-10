"""Input contracts, metrics, and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

CORE = ["pickup", "delivery", "distance", "equipment", "weight", "date"]
COORDINATES = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
FULL = CORE + COORDINATES + ["market_index", "quote_signal"]
TARGET = "posted_rate"
SEED = 42
POSITIVE_FLOOR = 1.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def environment() -> dict:
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "catboost", "threadpoolctl", "joblib", "matplotlib", "reportlab"]
    return {"python": platform.python_version(), "packages": {p: version(p) for p in packages}}


def code_checksums() -> dict:
    root = Path(__file__).parent
    return {name: sha256(root / name) for name in ["common.py", "features.py", "forecasting.py", "modeling.py", "evaluate.py"]}


def load_inputs(path: Path, *, labeled: bool = False, reduced: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_inputs(frame, labeled=labeled, reduced=reduced)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if labeled:
        frame = frame.sort_values(["date", "load_id"], kind="stable").reset_index(drop=True)
    return frame


def validate_inputs(frame: pd.DataFrame, *, labeled: bool = False, reduced: bool = False) -> None:
    required = list(CORE if reduced else FULL)
    if not reduced:
        required.append("load_id")
    if labeled:
        required.append(TARGET)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Input dataset is empty")
    if "load_id" in frame:
        if frame.load_id.isna().any() or frame.load_id.duplicated().any():
            raise ValueError("load_id values must be nonmissing and unique")
    dates = pd.to_datetime(frame.date, errors="coerce")
    if dates.isna().any() or (dates != dates.dt.normalize()).any():
        raise ValueError("Dates must be valid calendar dates without time components")
    numeric = ["distance", "weight"]
    if not reduced:
        numeric += COORDINATES + ["market_index", "quote_signal"]
    if labeled:
        numeric.append(TARGET)
    for column in numeric:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError(f"Nonfinite values in {column}")
        if column not in ["weight", "market_index"] and np.isnan(values).any():
            raise ValueError(f"Missing values in {column}")
        if column in ["distance", TARGET] and (values <= 0).any():
            raise ValueError(f"{column} must be strictly positive")


def metrics(actual, predicted) -> dict:
    y = np.asarray(actual, dtype=float)
    prediction = np.asarray(predicted, dtype=float)
    if y.shape != prediction.shape or y.ndim != 1 or not len(y):
        raise ValueError("Metrics need matching, nonempty one-dimensional arrays")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Metrics require finite values")
    error = prediction - y
    if (y <= 0).any() or (prediction <= 0).any():
        raise ValueError("Freight dollar/log metrics require positive targets and predictions")
    log_error = np.log(prediction / y)
    return {
        "rows": len(y),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "wape_pct": float(100 * np.abs(error).sum() / np.abs(y).sum()),
        "bias": float(error.mean()),
        "mape_pct": float(100 * np.mean(np.abs(error) / y)),
        "mae_log": float(np.abs(log_error).mean()),
        "rmse_log": float(np.sqrt(np.square(log_error).mean())),
    }


def slices(frame: pd.DataFrame, prediction, training: pd.DataFrame) -> dict:
    prediction = np.asarray(prediction)
    cities = set(training.pickup) | set(training.delivery)
    lanes = set(zip(training.pickup, training.delivery))
    result = {"overall": metrics(frame[TARGET], prediction), "groups": []}
    axes = {
        "month": pd.to_datetime(frame.date).dt.strftime("%Y-%m"),
        "equipment": frame.equipment.fillna("__missing__").astype(str),
        "distance_band": pd.cut(frame.distance, [0, 250, 750, 1500, np.inf], labels=["0-250", "250-750", "750-1500", "1500+"]).astype(str),
        "weight_status": pd.Series(np.where(frame.weight.isna(), "missing", np.where(frame.weight <= 0, "invalid", "valid")), index=frame.index),
        "city_status": pd.Series(np.where(frame.pickup.isin(cities) & frame.delivery.isin(cities), "seen", "unseen"), index=frame.index),
        "route_status": pd.Series(["seen" if lane in lanes else "unseen" for lane in zip(frame.pickup, frame.delivery)], index=frame.index),
    }
    for axis, groups in axes.items():
        for group in sorted(groups.unique()):
            mask = groups.eq(group).to_numpy()
            result["groups"].append({"axis": axis, "group": str(group), **metrics(frame.loc[mask, TARGET], prediction[mask])})
    return result


def chronological_split(frame: pd.DataFrame, train_end: str, evaluation_end: str):
    dates = pd.to_datetime(frame.date)
    boundary, stop = pd.Timestamp(train_end), pd.Timestamp(evaluation_end)
    training = frame.loc[dates < boundary].copy()
    evaluation = frame.loc[(dates >= boundary) & (dates < stop)].copy()
    if training.empty or evaluation.empty or training.date.max() >= evaluation.date.min():
        raise ValueError("Partitions must be nonempty and strictly chronological")
    return training, evaluation


def validate_rates(prediction) -> np.ndarray:
    values = np.asarray(prediction, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Predictions must be finite, strictly positive, and one-dimensional")
    return values
