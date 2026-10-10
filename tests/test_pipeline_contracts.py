"""Tests for leakage boundaries, input contracts, and submission alignment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from freight_rate.common import CORE, TARGET, chronological_split, fingerprint, metrics, validate_inputs
from freight_rate.evaluate import load_frozen, select_candidates, summarize_months
from freight_rate.features import FeatureBuilder
from freight_rate.modeling import LoadModel
from freight_rate.predict import align_predictions


@pytest.fixture
def loads():
    count = 90
    return pd.DataFrame({"load_id": [f"row-{i}" for i in range(count)],
                         "pickup": ["A"] * count, "delivery": ["B"] * count,
                         "distance": np.linspace(100, 500, count), "equipment": ["Dry Van"] * count,
                         "weight": [32000.0] * count,
                         "date": pd.date_range("2025-01-01", periods=count),
                         "pickup_lat": [35.0] * count, "pickup_lon": [-90.0] * count,
                         "delivery_lat": [38.0] * count, "delivery_lon": [-85.0] * count,
                         "market_index": [1.0] * count, "quote_signal": [2.0] * count,
                         TARGET: np.linspace(200, 1000, count)})


def test_calendar_split_keeps_same_day_together(loads):
    train, evaluation = chronological_split(loads, "2025-03-01", "2025-04-01")
    assert train.date.max() < evaluation.date.min()
    assert set(train.load_id).isdisjoint(evaluation.load_id)
    assert len(train) + len(evaluation) == len(loads)


def test_cleaning_flags_and_source_immutability(loads):
    loads.loc[0, "weight"] = -100
    loads.loc[1, "weight"] = np.nan
    original = loads.copy(deep=True)
    features = FeatureBuilder("reduced").transform(loads)
    assert features.loc[0, "weight_invalid"] == 1
    assert features.loc[0, "weight_missing"] == 0
    assert features.loc[1, "weight_missing"] == 1
    assert features.weight.iloc[:2].isna().all()
    pd.testing.assert_frame_equal(loads, original)


def test_reduced_contract_ignores_unavailable_fields_and_labels(loads):
    builder = FeatureBuilder("reduced")
    expected = builder.transform(loads[CORE])
    loads[TARGET] = 1e9
    loads.quote_signal = -999
    pd.testing.assert_frame_equal(builder.transform(loads), expected)
    assert "load_id" not in expected and TARGET not in expected
    assert "market_index" not in expected and "pickup_lat" not in expected


def test_quote_free_features_do_not_depend_on_quote(loads):
    builder = FeatureBuilder("full", False)
    expected = builder.transform(loads)
    loads.quote_signal = 1e6
    pd.testing.assert_frame_equal(builder.transform(loads), expected)


@pytest.mark.parametrize("column,value", [("distance", 0), ("distance", np.inf), ("weight", np.inf), (TARGET, np.nan), ("quote_signal", np.inf)])
def test_input_validation_rejects_corrupt_values(loads, column, value):
    loads.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_inputs(loads, labeled=True)


def test_linear_preprocessing_learns_only_training_median(loads):
    loads.loc[0, "weight"] = np.nan
    model = LoadModel({"algorithm": "ridge", "contract": "reduced"}).fit(loads)
    numerical = model.estimator.named_steps["features"].transformers_[0][2]
    imputer = model.estimator.named_steps["features"].named_transformers_["numeric"].named_steps["imputer"]
    statistics = imputer.statistics_.copy()
    assert statistics[numerical.index("weight")] == 32000
    evaluation = loads[CORE].iloc[:2].copy()
    evaluation.pickup = "New City"
    evaluation.delivery = "Another New City"
    evaluation.equipment = "Unknown Equipment"
    evaluation.weight = 900000
    prediction = model.predict(evaluation)
    assert np.isfinite(prediction).all() and (prediction > 0).all()
    np.testing.assert_array_equal(statistics, imputer.statistics_)


def test_all_missing_training_weight_fallback(loads):
    loads.weight = np.nan
    model = LoadModel({"algorithm": "ridge", "contract": "reduced"}).fit(loads)
    assert np.isfinite(model.predict(loads[CORE])).all()


def test_catboost_reduced_inference_missing_weight_and_unseen_city(loads):
    model = LoadModel({"algorithm": "catboost", "contract": "reduced", "loss": "MAE", "iterations": 8}).fit(loads)
    evaluation = loads[CORE].iloc[:3].copy()
    evaluation.pickup = "Unseen"
    evaluation.weight = np.nan
    prediction = model.predict(evaluation)
    assert np.isfinite(prediction).all() and (prediction > 0).all()


def test_predictions_follow_ids_after_loads_are_shuffled(loads):
    template = pd.DataFrame({"load_id": loads.load_id.iloc[:3], "predicted_rate": np.nan})
    shuffled = loads.iloc[[2, 0, 1]]
    output = align_predictions(template, shuffled, [300, 100, 200])
    assert output.predicted_rate.tolist() == [100, 200, 300]


@pytest.mark.parametrize("rates", [[1, 0, 3], [1, np.nan, 3], [1, np.inf, 3], [1, 2]])
def test_output_alignment_rejects_invalid_predictions(loads, rates):
    template = pd.DataFrame({"load_id": loads.load_id.iloc[:3], "predicted_rate": np.nan})
    with pytest.raises(ValueError):
        align_predictions(template, loads.iloc[:3], rates)


def test_quote_candidate_cannot_hide_bad_month_in_average():
    recipes = [{"name": "full_mae_noquote", "contract": "full", "include_quote": False},
               {"name": "full_mae_quote", "contract": "full", "include_quote": True},
               {"name": "reduced_mae_noquote", "contract": "reduced", "include_quote": False}]
    scores = {
        "full_mae_noquote": {"macro_mae": 100, "monthly_mae": {"06": 100, "07": 100, "08": 100}},
        "full_mae_quote": {"macro_mae": 90, "monthly_mae": {"06": 70, "07": 70, "08": 130}},
        "reduced_mae_noquote": {"macro_mae": 120, "monthly_mae": {"06": 120, "07": 120, "08": 120}},
    }
    winners, decisions = select_candidates(scores, recipes)
    assert winners["full"]["name"] == "full_mae_noquote"
    assert not decisions["full_mae_quote"]["eligible"]


def test_overlapping_july_has_equal_month_weight():
    records = [{"evaluation": {"groups": [{"axis": "month", "group": "June", "mae": 10}, {"axis": "month", "group": "July", "mae": 20}]}},
               {"evaluation": {"groups": [{"axis": "month", "group": "July", "mae": 40}, {"axis": "month", "group": "August", "mae": 50}]}}]
    assert summarize_months(records)["macro_mae"] == 30


def test_frozen_recipe_tampering_is_rejected(tmp_path: Path):
    original = {"training_sha256": "unchanged", "main_recipe": {"iterations": 100}}
    original["fingerprint"] = fingerprint(original)
    original["main_recipe"]["iterations"] = 200
    (tmp_path / "selected_recipe.json").write_text(json.dumps(original))
    with pytest.raises(ValueError, match="modified"):
        load_frozen(tmp_path, tmp_path)


def test_metrics_are_in_original_dollars():
    result = metrics([1000, 2000, 3000], [1100, 1800, 3600])
    assert result["mae"] == 300
    assert result["wape_pct"] == 15
    assert result["rmse"] == pytest.approx(np.sqrt(410000 / 3))
    assert result["mape_pct"] == pytest.approx(100 * (0.1 + 0.1 + 0.2) / 3)
    assert result["mae_log"] == pytest.approx(np.mean(np.abs(np.log([1.1, 0.9, 1.2]))))


def test_structured_forecast_extrapolates_trend_and_keeps_preprocessing_fixed(loads, tmp_path):
    import joblib

    # Keep load distance independent of date so the temporal slope is identifiable.
    loads.distance = np.random.default_rng(42).permutation(loads.distance.to_numpy())
    dates = pd.to_datetime(loads.date)
    loads[TARGET] = loads.distance * np.exp(0.7 + 0.001 * (dates - dates.min()).dt.days)
    loads.loc[0, "weight"] = np.nan
    model = LoadModel({"algorithm": "forecaster", "contract": "reduced", "include_quarter": False}).fit(loads)
    scaler_mean = model.forecaster.design.scaler.mean_.copy()
    median = model.forecaster.design.weight_median
    evaluation = loads[CORE].iloc[:3].copy()
    evaluation.date = pd.date_range("2025-06-01", periods=3)
    evaluation.pickup = "Unseen City"
    evaluation.delivery = "Another Unseen City"
    evaluation.weight = np.nan
    expected = evaluation.distance * np.exp(0.7 + 0.001 * (evaluation.date - dates.min()).dt.days)
    np.testing.assert_allclose(model.predict(evaluation), expected, rtol=0.01)
    np.testing.assert_array_equal(model.forecaster.design.scaler.mean_, scaler_mean)
    assert model.forecaster.design.weight_median == median == 32000
    assert model.forecaster.diagnostics["retained_training_rows"] == len(loads)
    assert model.forecaster.diagnostics["minimum_training_weight"] > 0
    joblib.dump(model, tmp_path / "model.joblib")
    np.testing.assert_array_equal(joblib.load(tmp_path / "model.joblib").predict(evaluation), model.predict(evaluation))


def test_robust_fit_preserves_extreme_label_and_all_rows(loads):
    loads.loc[0, TARGET] = 1e8
    original = loads.copy(deep=True)
    model = LoadModel({"algorithm": "forecaster", "contract": "reduced"}).fit(loads)
    assert model.forecaster.diagnostics["retained_training_rows"] == len(loads)
    assert 0 < model.forecaster.diagnostics["minimum_training_weight"] < 1
    pd.testing.assert_frame_equal(loads, original)
    assert np.isfinite(model.predict(loads[CORE])).all()


def test_calendar_hinges_reset_at_quarter_start(loads):
    evaluation = loads.iloc[:3].copy()
    evaluation.date = pd.to_datetime(["2025-03-31", "2025-04-01", "2025-12-31"])
    features = FeatureBuilder("reduced", include_calendar=True).transform(evaluation)
    assert features.quarter_phase.iloc[0] > 0.98
    assert features.quarter_phase.iloc[1] == 0
    assert features["quarter_hinge_0.9"].iloc[0] > 0
    assert features["quarter_hinge_0.9"].iloc[1] == 0
    assert features["quarter_hinge_0.9_flatbed"].eq(0).all()


@pytest.mark.parametrize("bad_month,bad_tail", [(True, False), (False, True)])
def test_forecast_gain_cannot_hide_month_failure_or_tail_degradation(bad_month, bad_tail):
    recipes = [{"name": "full_base", "contract": "full"},
               {"name": "full_forecast", "contract": "full", "reference": "full_base"},
               {"name": "reduced_base", "contract": "reduced"}]
    baseline = {"macro_mae": 100, "macro_rmse": 200, "monthly_mae": {"May": 100, "June": 100}}
    forecast = {"macro_mae": 90, "macro_rmse": 211 if bad_tail else 190,
                "monthly_mae": {"May": 70, "June": 106 if bad_month else 90}}
    winners, decisions = select_candidates({"full_base": baseline, "reduced_base": baseline, "full_forecast": forecast}, recipes)
    assert winners["full"]["name"] == "full_base"
    assert not decisions["full_forecast"]["eligible"]
