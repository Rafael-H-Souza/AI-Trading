"""Feature engineering utilities for the AI trading system."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .data_loader import PriceSeries


@dataclass
class FeatureRow:
    """Represents a single training example."""

    features: List[float]
    forward_return: float
    target: int


def _moving_average(values: Sequence[float], window: int) -> float:
    return sum(values[-window:]) / window


def _standard_deviation(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return variance ** 0.5


def compute_features(series: PriceSeries) -> List[FeatureRow]:
    """Compute features and targets from a price series."""

    closes = series.closes()
    feature_rows: List[FeatureRow] = []
    if len(closes) < 7:
        return feature_rows

    returns: List[float] = [0.0]
    for idx in range(1, len(closes)):
        previous = closes[idx - 1]
        current = closes[idx]
        returns.append((current - previous) / previous)

    for idx in range(5, len(closes) - 1):
        window_returns = returns[idx - 4 : idx + 1]
        recent_closes = closes[idx - 4 : idx + 1]
        feature_vector = [
            returns[idx],
            _moving_average(recent_closes[-3:], 3),
            _moving_average(recent_closes, 5),
            _standard_deviation(window_returns),
            (closes[idx] - closes[idx - 3]) / closes[idx - 3],
        ]
        forward_return = (closes[idx + 1] - closes[idx]) / closes[idx]
        target = 1 if forward_return > 0 else 0
        feature_rows.append(
            FeatureRow(features=feature_vector, forward_return=forward_return, target=target)
        )
    return feature_rows


__all__ = ["FeatureRow", "compute_features"]
