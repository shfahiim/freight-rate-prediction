"""Generate final outputs using ID joins and preserved December inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .common import load_inputs, sha256, validate_rates, write_json


def align_predictions(template: pd.DataFrame, loads: pd.DataFrame, prediction) -> pd.DataFrame:
    if list(template.columns) != ["load_id", "predicted_rate"]:
        raise ValueError("Template must contain exactly load_id,predicted_rate")
    values = validate_rates(prediction)
    if len(values) != len(loads):
        raise ValueError("Prediction count differs from load count")
    for frame in [template, loads]:
        if frame.load_id.isna().any() or frame.load_id.duplicated().any():
            raise ValueError("Missing or duplicate load IDs")
    if set(template.load_id) != set(loads.load_id):
        raise ValueError("Template and input IDs differ")
    lookup = pd.Series(values, index=loads.load_id)
    output = template.copy()
    output["predicted_rate"] = output.load_id.map(lookup)
    validate_rates(output.predicted_rate)
    return output


def predict(data_dir: Path, artifact_dir: Path, predictions: Path, december_predictions: Path, diagnostics_dir: Path) -> None:
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    models = {}
    for label in ["main", "december"]:
        path = artifact_dir / f"{label}.joblib"
        if sha256(path) != metadata["models"][label]["sha256"]:
            raise ValueError(f"{label} model artifact checksum mismatch")
        models[label] = joblib.load(path)
    loads = load_inputs(data_dir / "validation.csv")
    template = pd.read_csv(data_dir / "validation_predictions_template.csv")
    main_rates = models["main"].predict(loads)
    output = align_predictions(template, loads, main_rates)
    december = load_inputs(data_dir / "december_chart_inputs.csv", reduced=True)
    expected = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    if list(december.columns) != expected:
        raise ValueError("Unexpected December column order")
    december["predicted_rate"] = validate_rates(models["december"].predict(december))
    december["date"] = december.date.dt.strftime("%Y-%m-%d")
    predictions.parent.mkdir(parents=True, exist_ok=True)
    december_predictions.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(predictions, index=False, float_format="%.6f")
    december.to_csv(december_predictions, index=False, float_format="%.6f")
    seen = set(models["main"].seen_cities)
    summary = {"recipe_fingerprint": metadata["recipe_fingerprint"],
               "input_checksums": {p.name: sha256(p) for p in sorted(data_dir.glob("*.csv"))},
               "output_checksums": {str(p): sha256(p) for p in [predictions, december_predictions]},
               "main": {"rows": len(output), "minimum": float(output.predicted_rate.min()), "maximum": float(output.predicted_rate.max()),
                        "unseen_city_rows": int((~loads.pickup.isin(seen) | ~loads.delivery.isin(seen)).sum()),
                        "floor_activations": int((models["main"].raw_predict(loads) < 1).sum()), "model": models["main"].recipe["name"]},
               "december": {"rows": len(december), "minimum": float(december.predicted_rate.min()), "maximum": float(december.predicted_rate.max()),
                            "model": models["december"].recipe["name"]},
               "note": "Final labels are unavailable; these summaries are not accuracy metrics."}
    write_json(diagnostics_dir / "prediction_summary.json", summary)
    print(f"Wrote {len(output):,} ID-aligned final predictions and {len(december)} December predictions", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--diagnostics-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--predictions", type=Path, default=Path("validation_predictions.csv"))
    parser.add_argument("--december-predictions", type=Path, default=Path("outputs/december_predictions.csv"))
    args = parser.parse_args()
    predict(args.data_dir, args.artifact_dir, args.predictions, args.december_predictions, args.diagnostics_dir)


if __name__ == "__main__":
    main()
