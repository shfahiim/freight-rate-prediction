"""Small, explicit model matrix with training-side early stopping."""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .common import POSITIVE_FLOOR, SEED, TARGET
from .features import FeatureBuilder
from .forecasting import RobustForecaster


class LoadModel:
    """One serializable model and its feature/positivity contracts."""

    def __init__(self, recipe: dict):
        self.recipe = dict(recipe)
        self.builder = FeatureBuilder(recipe["contract"], recipe.get("include_quote", False), recipe.get("calendar", False))

    def fit(self, frame: pd.DataFrame, *, early_stopping: bool = False):
        features = self.builder.transform(frame)
        y = frame[TARGET].to_numpy(dtype=float)
        if not np.isfinite(y).all() or (y <= 0).any():
            raise ValueError("Training targets must be finite and positive")
        self.feature_names = list(features.columns)
        self.seen_cities = sorted(set(features.pickup) | set(features.delivery))
        self.seen_lanes = sorted(set(features.lane))
        algorithm = self.recipe["algorithm"]
        self.iterations = None
        if algorithm == "forecaster":
            if early_stopping:
                raise ValueError("Structured forecaster uses fixed complexity, not external early stopping")
            self.forecaster = RobustForecaster(self.recipe).fit(frame)
            self.feature_names = self.forecaster.feature_names
        elif algorithm == "equipment_median":
            rate = pd.Series(y / features.distance.to_numpy(), index=features.index)
            self.equipment_rates = rate.groupby(features.equipment).median().to_dict()
            self.global_rate = float(rate.median())
        elif algorithm == "ridge":
            categorical = self.builder.categorical_columns(features)
            numerical = [c for c in features if c not in categorical]
            preprocessing = ColumnTransformer([
                ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]), numerical),
                ("category", OneHotEncoder(handle_unknown="ignore", min_frequency=3), categorical),
            ], sparse_threshold=1.0)
            self.estimator = Pipeline([
                ("features", preprocessing),
                ("model", Ridge(alpha=self.recipe.get("alpha", 10.0), solver="lsqr", tol=1e-6)),
            ])
            self.estimator.fit(features, y)
        elif algorithm == "catboost":
            self.estimator = CatBoostRegressor(
                iterations=self.recipe.get("iterations", 600),
                depth=6,
                learning_rate=0.08,
                l2_leaf_reg=5,
                loss_function=self.recipe["loss"],
                eval_metric="MAE",
                random_seed=SEED,
                thread_count=4,
                has_time=True,
                allow_writing_files=False,
                verbose=False,
            )
            cat_columns = self.builder.categorical_columns(features)
            if early_stopping:
                dates = pd.to_datetime(frame.date)
                cutoff = dates.max().normalize() - pd.Timedelta(days=27)
                inner_train, inner_validation = dates < cutoff, dates >= cutoff
                if not inner_train.any() or not inner_validation.any():
                    raise ValueError("Not enough dates for a training-side 28-day stopping window")
                self.estimator.fit(features.loc[inner_train], y[inner_train], cat_features=cat_columns,
                                   eval_set=(features.loc[inner_validation], y[inner_validation]),
                                   early_stopping_rounds=60, use_best_model=True)
            else:
                self.estimator.fit(features, y, cat_features=cat_columns)
            self.iterations = int(self.estimator.tree_count_)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        return self

    def raw_predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.recipe["algorithm"] == "forecaster":
            values = np.asarray(self.forecaster.predict(frame), dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Structured model produced nonfinite predictions")
            return values
        features = self.builder.transform(frame)
        if list(features.columns) != self.feature_names:
            raise ValueError("Prediction feature schema differs from training")
        if self.recipe["algorithm"] == "equipment_median":
            rates = features.equipment.map(self.equipment_rates).fillna(self.global_rate)
            prediction = rates.to_numpy() * features.distance.to_numpy()
        else:
            prediction = self.estimator.predict(features)
        values = np.asarray(prediction, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Model produced nonfinite predictions")
        return values

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.maximum(self.raw_predict(frame), self.recipe.get("positive_floor", POSITIVE_FLOOR))


def candidates() -> list[dict]:
    """Fixed development matrix declared after the original audit was observed."""
    result = []
    for contract in ["full", "reduced"]:
        baseline_name = f"{contract}_catboost_baseline"
        shared = {"contract": contract, "include_quote": False, "positive_floor": POSITIVE_FLOOR}
        result.append({**shared, "name": f"{contract}_equipment_median", "algorithm": "equipment_median"})
        result.append({**shared, "name": baseline_name, "algorithm": "catboost", "loss": "MAE",
                       "iterations": 581 if contract == "full" else 586})
        result.append({**shared, "name": f"{contract}_calendar_catboost", "algorithm": "catboost", "loss": "MAE",
                       "iterations": 600, "calendar": True, "reference": baseline_name})
        for name, quarter, trend, residual in [("spline_no_quarter", False, True, False), ("spline_no_trend", True, False, False),
                                              ("spline_forecast", True, True, False), ("hybrid_forecast", True, True, True)]:
            result.append({**shared, "name": f"{contract}_{name}", "algorithm": "forecaster", "calendar": True,
                           "include_quarter": quarter, "include_trend": trend, "residual_trees": residual,
                           "reference": baseline_name})
    return result
