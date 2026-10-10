"""Refit the frozen recipes on all labeled rows after the final audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from .common import environment, load_inputs, sha256, write_json
from .evaluate import fixed_fit, load_frozen


def train(data_dir: Path, evaluation_dir: Path, artifact_dir: Path) -> None:
    frozen = load_frozen(evaluation_dir, data_dir)
    audit = json.loads((evaluation_dir / "final_audit.json").read_text())
    if audit["recipe_fingerprint"] != frozen["fingerprint"]:
        raise ValueError("Final audit does not match the frozen recipe")
    frame = load_inputs(data_dir / "train_test.csv", labeled=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    fitted = {}
    metadata = {"recipe_fingerprint": frozen["fingerprint"], "training_sha256": sha256(data_dir / "train_test.csv"),
                "training_rows": len(frame), "training_last_date": str(frame.date.max().date()),
                "environment": environment(), "models": {}}
    for label, recipe in [("main", frozen["main_recipe"]), ("december", frozen["december_recipe"])]:
        if recipe["name"] not in fitted:
            fitted[recipe["name"]] = fixed_fit(recipe, frame)
        model = fitted[recipe["name"]]
        path = artifact_dir / f"{label}.joblib"
        joblib.dump(model, path)
        metadata["models"][label] = {"recipe": recipe, "feature_names": model.feature_names, "sha256": sha256(path)}
        if recipe["algorithm"] == "catboost":
            importance = sorted(zip(model.feature_names, model.estimator.feature_importances_), key=lambda row: -row[1])
            metadata["models"][label]["feature_importance"] = [{"feature": name, "importance": float(value)} for name, value in importance]
        elif recipe["algorithm"] == "forecaster":
            metadata["models"][label]["training_diagnostics"] = model.forecaster.diagnostics
            design = model.forecaster.design
            if "linear_time_trend" in design.numeric_names:
                index = design.numeric_names.index("linear_time_trend")
                daily_slope = float(model.forecaster.estimator.coef_[index] * 5 / design.scaler.scale_[index] / 365)
                metadata["models"][label]["daily_log_trend"] = daily_slope
        print(f"Refit {label}: {recipe['name']} on {len(frame):,} labeled rows", flush=True)
    write_json(artifact_dir / "metadata.json", metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--evaluation-dir", type=Path, default=Path("evaluation"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    train(args.data_dir, args.evaluation_dir, args.artifact_dir)


if __name__ == "__main__":
    main()
