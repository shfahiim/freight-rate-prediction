"""Robust additive forecasting with train-only spline/category transforms.

Calendar position and an extrapolating trend are hypotheses tested by rolling
development evaluation. No evaluation labels are filtered or used to fit these
transforms. All training rows have strictly positive robust weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from .common import SEED, TARGET
from .features import FeatureBuilder


class ForecastDesign:
    """Smooth load/geometry effects plus shrunk city and directed-lane effects."""

    def __init__(self, contract: str, include_quarter: bool = True, include_trend: bool = True):
        self.contract = contract
        self.include_quarter = include_quarter
        self.include_trend = include_trend
        self.builder = FeatureBuilder(contract, include_calendar=True)

    def _load_values(self, features):
        return np.column_stack([
            np.log(features.distance.to_numpy()),
            features.weight.fillna(self.weight_median).to_numpy() / 10000,
        ])

    def _numerical(self, features):
        values = self._load_values(features)
        pieces = [self.load_spline.transform(values[:, self.active_load])]
        names = list(self.load_spline.get_feature_names_out(np.array(["log_distance", "weight"])[self.active_load]))
        simple = {
            "equipment_reefer": features.equipment.eq("Reefer").to_numpy(dtype=float),
            "equipment_flatbed": features.equipment.eq("Flatbed").to_numpy(dtype=float),
            "weight_missing": features.weight_missing.to_numpy(dtype=float),
            "weight_invalid": features.weight_invalid.to_numpy(dtype=float),
            "weekday_sin": features.weekday_sin.to_numpy(),
            "weekday_cos": features.weekday_cos.to_numpy(),
        }
        if self.include_trend:
            simple["linear_time_trend"] = features.elapsed_days.to_numpy() / 365
        if self.include_quarter:
            for column in features:
                if column.startswith("quarter_hinge_"):
                    simple[column] = features[column].to_numpy()
        if self.contract == "full":
            simple["market_index"] = features.market_index.fillna(self.market_median).to_numpy()
            simple["market_missing"] = features.market_index.isna().to_numpy(dtype=float)
            pickup = features[["pickup_lat", "pickup_lon"]].to_numpy()
            delivery = features[["delivery_lat", "delivery_lon"]].to_numpy()
            geometry = self.geometry_spline.transform(pickup) + self.geometry_spline.transform(delivery)
            pieces.append(geometry)
            names.extend(self.geometry_spline.get_feature_names_out(["endpoint_lat", "endpoint_lon"]))
        pieces.append(np.column_stack(list(simple.values())))
        names.extend(simple)
        return np.column_stack(pieces), names

    def fit_transform(self, frame: pd.DataFrame):
        features = self.builder.transform(frame)
        median = features.weight.median()
        self.weight_median = float(median) if np.isfinite(median) else 32000.0
        if self.contract == "full":
            median = features.market_index.median()
            self.market_median = float(median) if np.isfinite(median) else 1.0
            endpoints = np.concatenate([features[["pickup_lat", "pickup_lon"]].to_numpy(), features[["delivery_lat", "delivery_lon"]].to_numpy()])
            self.geometry_spline = SplineTransformer(n_knots=5, degree=3, knots="quantile", extrapolation="linear", include_bias=False)
            self.geometry_spline.fit(endpoints)
        values = self._load_values(features)
        self.active_load = np.std(values, axis=0) > 1e-8
        if not self.active_load.any():
            raise ValueError("Spline model requires variation in distance or valid weight")
        self.load_spline = SplineTransformer(n_knots=5, degree=3, knots="quantile", extrapolation="linear", include_bias=False)
        self.load_spline.fit(values[:, self.active_load])
        numerical, names = self._numerical(features)
        self.numeric_names = names
        self.scaler = StandardScaler().fit(numerical)
        self.city_encoder = OneHotEncoder(handle_unknown="ignore")
        self.city_encoder.fit(pd.DataFrame({"city": pd.concat([features.pickup, features.delivery], ignore_index=True)}))
        self.lane_encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=3)
        self.lane_encoder.fit(features[["lane"]])
        self.feature_names = [*names, *self.city_encoder.get_feature_names_out(["city"]), *self.lane_encoder.get_feature_names_out(["lane"])]
        return self.transform(frame)

    def transform(self, frame: pd.DataFrame):
        features = self.builder.transform(frame)
        numerical, names = self._numerical(features)
        if names != self.numeric_names:
            raise ValueError("Spline numerical schema changed")
        # Scale smooth effects up so regularization favors broad geography over
        # memorized city deviations; lane coefficients receive more shrinkage.
        city = self.city_encoder.transform(pd.DataFrame({"city": features.pickup})) + self.city_encoder.transform(pd.DataFrame({"city": features.delivery}))
        lane = self.lane_encoder.transform(features[["lane"]])
        return sparse.hstack([sparse.csr_matrix(self.scaler.transform(numerical) * 5), city, lane * 0.5], format="csr")


class RobustForecaster:
    """Huber-style IRLS in log(rate/mile), optionally with a small residual model."""

    def __init__(self, recipe: dict):
        self.recipe = dict(recipe)
        self.design = ForecastDesign(recipe["contract"], recipe.get("include_quarter", True), recipe.get("include_trend", True))

    def fit(self, frame: pd.DataFrame):
        actual = frame[TARGET].to_numpy(dtype=float)
        if not np.isfinite(actual).all() or (actual <= 0).any():
            raise ValueError("All original targets must be finite and positive")
        target = np.log(actual / frame.distance.to_numpy())
        matrix = self.design.fit_transform(frame)
        weights = np.ones(len(frame))
        self.estimator = Ridge(alpha=2.0, solver="lsqr", tol=1e-6, max_iter=500)
        with threadpool_limits(limits=4):
            for iteration in range(8):
                self.estimator.fit(matrix, target, sample_weight=weights)
                residual = target - self.estimator.predict(matrix)
                mad = 1.4826 * np.median(np.abs(residual - np.median(residual)))
                delta = max(1.5 * float(mad), 0.015)
                new_weights = np.maximum(np.minimum(1.0, delta / np.maximum(np.abs(residual), 1e-12)), 1e-4)
                if np.max(np.abs(new_weights - weights)) < 1e-5:
                    break
                weights = new_weights
            self.estimator.fit(matrix, target, sample_weight=weights)
        self.feature_names = self.design.feature_names
        self.diagnostics = {"training_rows": len(frame), "retained_training_rows": len(frame),
                            "minimum_training_weight": float(weights.min()), "downweighted_rows": int((weights < 1).sum()),
                            "irls_steps": iteration + 1, "huber_log_threshold": delta,
                            "quarter_features": self.design.include_quarter, "linear_trend": self.design.include_trend}
        self.residual_model = None
        if self.recipe.get("residual_trees", False):
            features = self.design.builder.transform(frame).drop(columns=["elapsed_days"])
            residual = target - self.estimator.predict(matrix)
            self.residual_model = CatBoostRegressor(iterations=250, depth=5, learning_rate=0.05,
                                                   loss_function="MAE", random_seed=SEED, thread_count=4,
                                                   has_time=True, allow_writing_files=False, verbose=False)
            self.residual_model.fit(features, residual, cat_features=FeatureBuilder.categorical_columns(features))
        return self

    def predict_log_rpm(self, frame: pd.DataFrame):
        matrix = self.design.transform(frame)
        prediction = self.estimator.predict(matrix)
        if self.residual_model is not None:
            features = self.design.builder.transform(frame).drop(columns=["elapsed_days"])
            prediction += self.residual_model.predict(features)
        return prediction

    def predict(self, frame: pd.DataFrame):
        log_rate = self.predict_log_rpm(frame) + np.log(frame.distance.to_numpy())
        # Raise on numerical failure instead of concealing overflow by clipping.
        if not np.isfinite(log_rate).all() or (log_rate > np.log(np.finfo(float).max)).any():
            raise ValueError("Forecast produced an invalid log price")
        return np.exp(log_rate)
