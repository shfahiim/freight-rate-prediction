"""Deterministic features: never inspect labels or fit on inference inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import COORDINATES, CORE


class FeatureBuilder:
    def __init__(self, contract: str = "full", include_quote: bool = False, include_calendar: bool = False):
        if contract not in ["full", "reduced"] or (contract == "reduced" and include_quote):
            raise ValueError("Invalid input contract/quote combination")
        self.contract = contract
        self.include_quote = include_quote
        self.include_calendar = include_calendar

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = list(CORE)
        if self.contract == "full":
            required += COORDINATES + ["market_index"]
        if self.include_quote:
            required.append("quote_signal")
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"{self.contract} model missing features: {sorted(missing)}")
        result = pd.DataFrame(index=frame.index)
        for column in ["pickup", "delivery", "equipment"]:
            values = frame[column].astype("string").str.strip().replace("", pd.NA)
            result[column] = values.fillna("__missing__").astype(str)
        result["lane"] = result.pickup + " -> " + result.delivery
        distance = pd.to_numeric(frame.distance, errors="raise").astype(float)
        if not np.isfinite(distance).all() or (distance <= 0).any():
            raise ValueError("Distance must be finite and positive")
        weight = pd.to_numeric(frame.weight, errors="raise").astype(float)
        if np.isinf(weight).any():
            raise ValueError("Weight cannot be infinite")
        result["distance"] = distance
        result["log_distance"] = np.log1p(distance)
        result["weight_missing"] = weight.isna().astype(int)
        result["weight_invalid"] = (weight <= 0).astype(int)
        result["weight"] = weight.where(weight > 0)
        result["weight_per_mile"] = result.weight / distance
        for label in ["Dry Van", "Reefer", "Flatbed"]:
            result[f"distance_{label.lower().replace(' ', '_')}"] = distance * result.equipment.eq(label)
        dates = pd.to_datetime(frame.date, errors="raise")
        if dates.isna().any():
            raise ValueError("Date cannot be missing")
        weekday = dates.dt.dayofweek
        result["weekday"] = weekday.astype(str)
        result["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
        result["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
        result["weekend"] = (weekday >= 5).astype(int)
        result["elapsed_days"] = (dates - pd.Timestamp("2025-01-01")).dt.days.astype(float)
        if getattr(self, "include_calendar", False):
            quarters = dates.dt.to_period("Q")
            start = quarters.dt.start_time
            length = (quarters.dt.end_time.dt.normalize() + pd.Timedelta(days=1) - start).dt.days
            phase = (dates - start).dt.days / length
            result["quarter_phase"] = phase
            for knot in [0.60, 0.75, 0.90]:
                hinge = np.maximum(phase - knot, 0)
                result[f"quarter_hinge_{knot}"] = hinge
                for equipment in ["Reefer", "Flatbed"]:
                    result[f"quarter_hinge_{knot}_{equipment.lower()}"] = hinge * result.equipment.eq(equipment)
        if self.contract == "full":
            for column in COORDINATES + ["market_index"]:
                result[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
                if np.isinf(result[column]).any():
                    raise ValueError(f"{column} cannot be infinite")
            result["latitude_delta"] = result.delivery_lat - result.pickup_lat
            result["longitude_delta"] = result.delivery_lon - result.pickup_lon
            lat1, lat2 = np.radians(result.pickup_lat), np.radians(result.delivery_lat)
            dlat = lat2 - lat1
            dlon = np.radians(result.longitude_delta)
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            result["straight_line_miles"] = 2 * 3958.8 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            result["distance_market"] = distance * result.market_index
            if getattr(self, "include_calendar", False):
                result["market_missing"] = result.market_index.isna().astype(int)
        if self.include_quote:
            quote = pd.to_numeric(frame.quote_signal, errors="raise").astype(float)
            if not np.isfinite(quote).all():
                raise ValueError("Quote signal must be finite")
            result["quote_signal"] = quote
            result["distance_quote"] = distance * quote
        return result

    @staticmethod
    def categorical_columns(features: pd.DataFrame) -> list[str]:
        return list(features.select_dtypes(include=["object", "string"]).columns)
