"""Descriptive input audit; deliberately separate from predictive evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .common import load_inputs, sha256, write_json
from .eda import build_eda


def profile(frame: pd.DataFrame) -> dict:
    return {"rows": len(frame), "columns": list(frame.columns),
            "first_date": str(frame.date.min().date()), "last_date": str(frame.date.max().date()),
            "missing": {k: int(v) for k, v in frame.isna().sum().items()},
            "negative_weight": int((frame.weight < 0).sum()),
            "monthly_rows": {k: int(v) for k, v in frame.groupby(frame.date.dt.strftime("%Y-%m")).size().items()},
            "duplicate_ids": int(frame.load_id.duplicated().sum()) if "load_id" in frame else None,
            "duplicate_rows": int(frame.duplicated().sum())}


def audit(data_dir: Path, output_dir: Path) -> None:
    development = load_inputs(data_dir / "train_test.csv", labeled=True)
    final = load_inputs(data_dir / "validation.csv")
    december = load_inputs(data_dir / "december_chart_inputs.csv", reduced=True)
    template = pd.read_csv(data_dir / "validation_predictions_template.csv")
    expected_ids = {f"TE-{i:06d}" for i in range(1, 12001)}
    if len(template) != 12000 or list(template.columns) != ["load_id", "predicted_rate"]:
        raise ValueError("Unexpected final prediction template")
    if template.load_id.isna().any() or template.load_id.duplicated().any() or set(template.load_id) != expected_ids:
        raise ValueError("Template ID contract failed")
    if set(final.load_id) != expected_ids:
        raise ValueError("Final input ID contract failed")
    cities = set(development.pickup) | set(development.delivery)
    lanes = set(zip(development.pickup, development.delivery))
    unseen_city = ~final.pickup.isin(cities) | ~final.delivery.isin(cities)
    unseen_lane = pd.Series([lane not in lanes for lane in zip(final.pickup, final.delivery)], index=final.index)
    quote_months = []
    for month, frame in development.groupby(development.date.dt.strftime("%Y-%m")):
        rate_per_mile = frame.posted_rate / frame.distance
        discrepancy = (frame.distance * frame.quote_signal - frame.posted_rate).abs() / frame.posted_rate
        quote_months.append({"month": month, "rows": len(frame),
                             "spearman_quote_rate_per_mile": float(frame.quote_signal.corr(rate_per_mile, method="spearman")),
                             "median_relative_discrepancy_pct": float(discrepancy.median() * 100)})
    result = {"development": profile(development), "final_inputs": profile(final), "december_inputs": profile(december),
              "checksums": {p.name: sha256(p) for p in sorted(data_dir.glob("*.csv"))},
              "novelty": {"training_city_count": len(cities),
                          "new_cities": sorted((set(final.pickup) | set(final.delivery)) - cities),
                          "unseen_city_rows": int(unseen_city.sum()), "unseen_route_rows": int(unseen_lane.sum()),
                          "unseen_route_known_city_rows": int((unseen_lane & ~unseen_city).sum())},
              "target": {"minimum": float(development.posted_rate.min()), "maximum": float(development.posted_rate.max())},
              "quote_months": quote_months,
              "note": "Descriptive inspection includes September/October. These are not model validation results; final inputs are not used for fitting or selection."}
    write_json(output_dir / "data_audit.json", result)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 3.5), dpi=160)
    ax.bar([r["month"][5:] for r in quote_months], [r["spearman_quote_rate_per_mile"] for r in quote_months], color="#126475")
    ax.axhline(0, color="#263238", linewidth=0.8)
    ax.set(xlabel="Month in 2025", ylabel="Spearman correlation", title="Quote signal vs observed price per mile: descriptive association")
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    fig.savefig(figures / "quote_relationship.png")
    plt.close(fig)
    build_eda(development, final, output_dir, result["checksums"])
    print(f"Audited {len(development):,} labeled and {len(final):,} unlabeled loads; {int(unseen_city.sum()):,} final loads involve unseen cities")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eda"))
    args = parser.parse_args()
    audit(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
