"""Machine learning helpers implementing a lightweight logistic regression."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = pow(2.718281828459045, -value)
        return 1.0 / (1.0 + z)
    z = pow(2.718281828459045, value)
    return z / (1.0 + z)


@dataclass
class StandardScaler:
    """Simple feature scaler that normalises features using mean/std dev."""

    mean_: List[float]
    scale_: List[float]

    @classmethod
    def from_samples(cls, samples: Sequence[Sequence[float]]) -> "StandardScaler":
        transposed = list(zip(*samples))
        means: List[float] = []
        scales: List[float] = []
        for column in transposed:
            column_list = list(column)
            mean = sum(column_list) / len(column_list)
            variance = sum((value - mean) ** 2 for value in column_list) / len(column_list)
            scale = variance ** 0.5 or 1.0
            means.append(mean)
            scales.append(scale)
        return cls(mean_=means, scale_=scales)

    def transform(self, sample: Sequence[float]) -> List[float]:
        return [
            (value - mean) / scale
            for value, mean, scale in zip(sample, self.mean_, self.scale_)
        ]


@dataclass
class TradingModel:
    """Wrapper that combines feature scaling and logistic regression."""

    weights: List[float]
    bias: float
    scaler: StandardScaler

    def predict_proba(self, features: Sequence[float]) -> float:
        scaled = self.scaler.transform(features)
        value = sum(weight * feature for weight, feature in zip(self.weights, scaled)) + self.bias
        return _sigmoid(value)

    def predict(self, features: Sequence[float]) -> int:
        return 1 if self.predict_proba(features) >= 0.5 else 0


def train_model(
    feature_rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    learning_rate: float = 0.1,
    epochs: int = 400,
    regularisation: float = 0.001,
) -> TradingModel:
    """Train a logistic regression classifier on the provided features."""

    scaler = StandardScaler.from_samples(feature_rows)
    scaled_rows = [scaler.transform(row) for row in feature_rows]

    weights = [0.0 for _ in scaled_rows[0]]
    bias = 0.0

    for _ in range(epochs):
        for features, label in zip(scaled_rows, labels):
            linear_output = sum(w * x for w, x in zip(weights, features)) + bias
            prediction = _sigmoid(linear_output)
            error = prediction - label
            for idx, value in enumerate(features):
                gradient = error * value + regularisation * weights[idx]
                weights[idx] -= learning_rate * gradient
            bias -= learning_rate * error

    return TradingModel(weights=weights, bias=bias, scaler=scaler)


def train_from_feature_rows(
    rows: Sequence[Sequence[float]], labels: Sequence[int]
) -> TradingModel:
    return train_model(rows, labels)


__all__ = ["TradingModel", "train_model", "train_from_feature_rows", "StandardScaler"]
