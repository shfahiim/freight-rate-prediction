"""Tune first, persist a frozen recipe, then explicitly run the final audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import TARGET, chronological_split, code_checksums, environment, fingerprint, load_inputs, metrics, sha256, slices, write_json
from .modeling import LoadModel, candidates

FOLDS = [("A", "2025-05-01", "2025-07-01"), ("B", "2025-06-01", "2025-08-01"),
         ("C", "2025-07-01", "2025-09-01"), ("D", "2025-08-01", "2025-10-01")]
RULES = {
    "evaluation_status": "Development backtests: September-October was reviewed during development and September overlaps selection. Final November-December labels are unavailable.",
    "aggregation": "Average MAE across folds within each calendar month, then average May through September equally.",
    "candidate_min_improvement": 0.02,
    "candidate_max_month_degradation": 0.05,
    "candidate_max_macro_rmse_degradation": 0.05,
    "candidate_max_city_degradation": 0.10,
    "quote_min_improvement": 0.02,
    "quote_max_month_degradation": 0.02,
    "unified_max_macro_degradation": 0.02,
    "unified_max_month_degradation": 0.05,
    "unified_max_city_degradation": 0.10,
    "city_count": 4,
    "city_rule": "First four training cities ordered by SHA256(city name); selected without target values.",
    "catboost_fit_cap": 36,
    "audit_train_end_exclusive": "2025-09-01",
    "audit_end_exclusive": "2025-11-01",
}


def summarize_months(records: list[dict]) -> dict:
    monthly = {}
    rmse_monthly = {}
    for record in records:
        for item in record["evaluation"]["groups"]:
            if item["axis"] == "month":
                monthly.setdefault(item["group"], []).append(item["mae"])
                rmse_monthly.setdefault(item["group"], []).append(item.get("rmse", item["mae"]))
    month_means = {month: float(np.mean(values)) for month, values in sorted(monthly.items())}
    rmse_means = {month: float(np.mean(values)) for month, values in sorted(rmse_monthly.items())}
    return {"monthly_mae": month_means, "macro_mae": float(np.mean(list(month_means.values()))),
            "monthly_rmse": rmse_means, "macro_rmse": float(np.mean(list(rmse_means.values())))}


def select_candidates(summaries: dict, recipes: list[dict]) -> tuple[dict, dict]:
    """Quote features must improve their matched loss without unstable months."""
    decisions = {}
    winners = {}
    for recipe in recipes:
        name = recipe["name"]
        eligible = True
        reason = "Eligible quote-free candidate"
        if recipe.get("include_quote"):
            counterpart = name.replace("_quote", "_noquote")
            score, base = summaries[name], summaries[counterpart]
            stable = all(score["monthly_mae"][m] <= base["monthly_mae"][m] * 1.02 for m in base["monthly_mae"])
            improves = score["macro_mae"] <= base["macro_mae"] * 0.98
            eligible = stable and improves
            reason = "Passed matched quote stability rule" if eligible else "Failed >=2% aggregate improvement and/or <=2% degradation in every month"
        elif recipe.get("reference"):
            score, base = summaries[name], summaries[recipe["reference"]]
            stable = all(score["monthly_mae"][m] <= base["monthly_mae"][m] * 1.05 for m in base["monthly_mae"])
            improves = score["macro_mae"] <= base["macro_mae"] * 0.98
            tail_safe = score["macro_rmse"] <= base["macro_rmse"] * 1.05
            eligible = stable and improves and tail_safe
            reason = "Passed baseline improvement, monthly stability, and RMSE rules" if eligible else "Failed baseline improvement, monthly stability, and/or RMSE rules"
        decisions[name] = {"eligible": eligible, "reason": reason, **summaries[name]}
    for contract in ["full", "reduced"]:
        eligible = [r for r in recipes if r["contract"] == contract and decisions[r["name"]]["eligible"]]
        winners[contract] = dict(min(eligible, key=lambda r: (summaries[r["name"]]["macro_mae"], r["name"])))
    return winners, decisions


def fixed_fit(recipe: dict, frame: pd.DataFrame, *, estimate_iterations: bool = False) -> LoadModel:
    if estimate_iterations and recipe["algorithm"] == "catboost":
        stopping_model = LoadModel(recipe).fit(frame, early_stopping=True)
        fixed_recipe = {**recipe, "iterations": stopping_model.iterations}
        return LoadModel(fixed_recipe).fit(frame)
    return LoadModel(recipe).fit(frame)


def tune(data_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "selected_recipe.json").exists() or (output_dir / "final_audit.json").exists():
        raise ValueError("Frozen results already exist. Use a separate output directory for a new development run.")
    development = load_inputs(data_dir / "train_test.csv", labeled=True)
    recipes = candidates()
    first_training, _ = chronological_split(development, FOLDS[0][1], FOLDS[0][2])
    cities = set(first_training.pickup) | set(first_training.delivery)
    withheld = sorted(cities, key=lambda c: hashlib.sha256(c.encode()).hexdigest())[:4]
    protocol = {"rules": RULES, "candidates": recipes, "withheld_cities": withheld,
                "folds": FOLDS, "environment": environment(), "code_checksums": code_checksums(),
                "training_sha256": sha256(data_dir / "train_test.csv")}
    write_json(output_dir / "protocol.json", protocol)
    records = []
    iteration_counts = {r["name"]: [] for r in recipes}
    for fold, train_end, evaluation_end in FOLDS:
        training, evaluation = chronological_split(development, train_end, evaluation_end)
        for recipe in recipes:
            model = fixed_fit(recipe, training, estimate_iterations=False)
            prediction = model.predict(evaluation)
            result = {"fold": fold, "name": recipe["name"], "train_rows": len(training),
                      "train_end_exclusive": train_end, "evaluation_end_exclusive": evaluation_end,
                      "iterations": model.iterations, "floor_activations": int((model.raw_predict(evaluation) < 1).sum()),
                      "evaluation": slices(evaluation, prediction, training)}
            if recipe["algorithm"] == "forecaster":
                result["training_diagnostics"] = model.forecaster.diagnostics
            # Forecast-horizon drift: late-period bias can be hidden by the mean.
            dates = pd.to_datetime(evaluation.date)
            first = dates < dates.min() + pd.Timedelta(days=14)
            last = dates >= dates.max() - pd.Timedelta(days=13)
            result["horizon"] = {"first_14_days": metrics(evaluation.loc[first, TARGET], prediction[first]),
                                 "last_14_days": metrics(evaluation.loc[last, TARGET], prediction[last])}
            records.append(result)
            if model.iterations is not None:
                iteration_counts[recipe["name"]].append(model.iterations)
            print(f"Fold {fold}: {recipe['name']}: MAE=${result['evaluation']['overall']['mae']:.2f}; trees={model.iterations}", flush=True)
            write_json(output_dir / "tuning_results.json", {"records": records})
        quote_prediction = evaluation.distance.to_numpy() * evaluation.quote_signal.to_numpy()
        records.append({"fold": fold, "name": "quote_distance_diagnostic", "train_rows": 0,
                        "iterations": None, "evaluation": slices(evaluation, quote_prediction, training)})
    summaries = {r["name"]: summarize_months([item for item in records if item["name"] == r["name"]]) for r in recipes}
    winners, decisions = select_candidates(summaries, recipes)
    for recipe in winners.values():
        counts = iteration_counts[recipe["name"]]
        if counts:
            recipe["iterations"] = max(1, int(np.median(counts)))

    # A single target-independent city test, confined to the tuning date range.
    training, evaluation = chronological_split(development, "2025-07-01", "2025-09-01")
    training = training.loc[~training.pickup.isin(withheld) & ~training.delivery.isin(withheld)].copy()
    unknown = evaluation.pickup.isin(withheld) | evaluation.delivery.isin(withheld)
    if not unknown.any():
        raise ValueError("City withholding produced no evaluation rows")
    city_test = {"cities": withheld, "training_rows": len(training), "evaluation_rows": len(evaluation),
                 "unseen_rows": int(unknown.sum()), "models": {}}
    for contract, recipe in list(winners.items()):
        model = fixed_fit(recipe, training)
        prediction = model.predict(evaluation)
        city_test["models"][contract] = {"name": recipe["name"], "unseen": metrics(evaluation.loc[unknown, TARGET], prediction[unknown]),
                                         "evaluation": slices(evaluation, prediction, training)}
        if recipe.get("reference"):
            reference = next(r for r in recipes if r["name"] == recipe["reference"])
            reference_model = fixed_fit(reference, training)
            reference_prediction = reference_model.predict(evaluation)
            reference_result = {"name": reference["name"], "unseen": metrics(evaluation.loc[unknown, TARGET], reference_prediction[unknown]),
                                "evaluation": slices(evaluation, reference_prediction, training)}
            city_test.setdefault("candidate_checks", {})[contract] = {
                "candidate": city_test["models"][contract], "reference": reference_result}
            if city_test["models"][contract]["unseen"]["mae"] > reference_result["unseen"]["mae"] * 1.10:
                decisions[recipe["name"]]["eligible"] = False
                decisions[recipe["name"]]["reason"] = "Failed unseen-city safeguard; retain reference without further search"
                winners[contract] = dict(reference)
                city_test["models"][contract] = reference_result
        print(f"City stress test ({contract}): unseen MAE=${city_test['models'][contract]['unseen']['mae']:.2f}", flush=True)
    full_score, reduced_score = summaries[winners["full"]["name"]], summaries[winners["reduced"]["name"]]
    comparable = (
        reduced_score["macro_mae"] <= full_score["macro_mae"] * 1.02
        and all(reduced_score["monthly_mae"][m] <= full_score["monthly_mae"][m] * 1.05 for m in full_score["monthly_mae"])
        and city_test["models"]["reduced"]["unseen"]["mae"] <= city_test["models"]["full"]["unseen"]["mae"] * 1.10
    )
    frozen = {"evaluation_status": RULES["evaluation_status"],
              "protocol_sha256": fingerprint(protocol), "training_sha256": protocol["training_sha256"],
              "code_checksums": protocol["code_checksums"], "environment": protocol["environment"],
              "rules": RULES, "selected": winners, "decisions": decisions, "city_stress": city_test,
              "deployment": "unified_reduced" if comparable else "separate",
              "main_recipe": winners["reduced"] if comparable else winners["full"],
              "december_recipe": winners["reduced"]}
    frozen["fingerprint"] = fingerprint(frozen)
    write_json(output_dir / "tuning_results.json", {"records": records, "summaries": summaries})
    write_json(output_dir / "selected_recipe.json", frozen)
    table = []
    for name, decision in decisions.items():
        table.append({"candidate": name, "eligible": decision["eligible"], "macro_mae": decision["macro_mae"], **decision["monthly_mae"]})
    pd.DataFrame(table).to_csv(output_dir / "experiments.csv", index=False)
    print(f"Saved {frozen['deployment']} development recipes: {frozen['fingerprint']}", flush=True)


def load_frozen(output_dir: Path, data_dir: Path) -> dict:
    frozen = json.loads((output_dir / "selected_recipe.json").read_text())
    saved = frozen.pop("fingerprint")
    if fingerprint(frozen) != saved:
        raise ValueError("Frozen recipe was modified")
    frozen["fingerprint"] = saved
    if sha256(data_dir / "train_test.csv") != frozen["training_sha256"]:
        raise ValueError("Training inputs changed after model selection")
    if code_checksums() != frozen["code_checksums"]:
        raise ValueError("Model/evaluation code changed after selection; use a new development run and document changes")
    current_environment = environment()
    same_python = current_environment["python"].split(".")[:2] == frozen["environment"]["python"].split(".")[:2]
    if current_environment["packages"] != frozen["environment"]["packages"] or not same_python:
        raise ValueError("Use the frozen dependency environment for reproducibility")
    return frozen


def final_audit(data_dir: Path, output_dir: Path) -> None:
    if (output_dir / "final_audit.json").exists():
        raise ValueError("Final audit already exists; it is not automatically rerun or used for reselection")
    frozen = load_frozen(output_dir, data_dir)
    development = load_inputs(data_dir / "train_test.csv", labeled=True)
    training, evaluation = chronological_split(development, "2025-09-01", "2025-11-01")
    results = {}
    fitted = {}
    for label, recipe in [("main", frozen["main_recipe"]), ("december", frozen["december_recipe"]),
                          ("full_catboost_baseline", next(r for r in candidates() if r["name"] == "full_catboost_baseline")),
                          ("reduced_catboost_baseline", next(r for r in candidates() if r["name"] == "reduced_catboost_baseline")),
                          ("equipment_baseline", {"name": "equipment_baseline", "algorithm": "equipment_median", "contract": "reduced"})]:
        if recipe["name"] not in fitted:
            fitted[recipe["name"]] = fixed_fit(recipe, training)
        model = fitted[recipe["name"]]
        prediction = model.predict(evaluation)
        results[label] = {"name": recipe["name"], "contract": recipe["contract"],
                          "floor_activations": int((model.raw_predict(evaluation) < 1).sum()),
                          "evaluation": slices(evaluation, prediction, training)}
        print(f"Development confirmation {label}: MAE=${results[label]['evaluation']['overall']['mae']:.2f}", flush=True)
    write_json(output_dir / "final_audit.json", {"recipe_fingerprint": frozen["fingerprint"],
               "training_rows": len(training), "evaluation_rows": len(evaluation),
               "training_end_exclusive": "2025-09-01", "evaluation_end_exclusive": "2025-11-01",
               "evaluation_status": frozen.get("evaluation_status", "Original audit"),
               "note": "Previously inspected September-October development confirmation; September overlaps a selection fold. Not an independent test. The saved recipe is not changed by these scores.",
               "results": results})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["tune", "audit"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    args = parser.parse_args()
    (tune if args.stage == "tune" else final_audit)(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
