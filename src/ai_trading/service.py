"""Service layer orchestrating data loading, feature generation and trading logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

from . import data_loader, features, model
from .data_loader import PriceSeries


@dataclass
class PredictionResult:
    """Representation of a trading prediction."""

    probability_up: float
    signal: str
    confidence: float


@dataclass
class PerformanceReport:
    """Summary of backtest results."""

    total_return: float
    avg_daily_return: float
    accuracy: float
    trades: int


def _extract_feature_matrix(rows: Sequence[features.FeatureRow]) -> List[List[float]]:
    return [row.features for row in rows]


def _extract_labels(rows: Sequence[features.FeatureRow]) -> List[int]:
    return [row.target for row in rows]


@dataclass
class AITradingService:
    """High level interface combining data, features and the ML model."""

    price_data: PriceSeries = field(default_factory=data_loader.load_price_data)
    trained_model: Optional[model.TradingModel] = None
    feature_rows: List[features.FeatureRow] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.price_data.points:
            raise ValueError("Price data must contain at least one row")
        self._ensure_model()

    def _ensure_model(self) -> None:
        if self.trained_model is None:
            self.feature_rows = features.compute_features(self.price_data)
            if not self.feature_rows:
                raise ValueError("Not enough data to train the model")
            feature_matrix = _extract_feature_matrix(self.feature_rows)
            labels = _extract_labels(self.feature_rows)
            self.trained_model = model.train_from_feature_rows(feature_matrix, labels)

    def train(self, csv_path: Optional[str] = None) -> None:
        """(Re)train the model optionally using an alternative dataset."""

        self.price_data = data_loader.load_price_data(csv_path)
        self.feature_rows = features.compute_features(self.price_data)
        if not self.feature_rows:
            raise ValueError("Not enough data to train the model")
        feature_matrix = _extract_feature_matrix(self.feature_rows)
        labels = _extract_labels(self.feature_rows)
        self.trained_model = model.train_from_feature_rows(feature_matrix, labels)

    def _prepare_features(self, closes: Iterable[float]) -> List[float]:
        series = PriceSeries(points=[data_loader.PricePoint(date=str(idx), close=value) for idx, value in enumerate(closes)])
        rows = features.compute_features(series)
        if not rows:
            raise ValueError("Not enough observations to compute indicators")
        return rows[-1].features

    def predict_from_prices(self, closes: Iterable[float]) -> PredictionResult:
        """Predict the next price movement given a sequence of closes."""

        self._ensure_model()
        assert self.trained_model is not None
        feature_vector = self._prepare_features(list(closes))
        prob_up = float(self.trained_model.predict_proba(feature_vector))
        confidence = abs(prob_up - 0.5) * 2
        signal = "buy" if prob_up >= 0.5 else "sell"
        return PredictionResult(probability_up=prob_up, signal=signal, confidence=confidence)

    def backtest(self) -> PerformanceReport:
        """Run a simple long-only backtest using the internal dataset."""

        self._ensure_model()
        assert self.trained_model is not None
        if not self.feature_rows:
            raise ValueError("Not enough data to run backtest")

        cumulative = 1.0
        returns: List[float] = []
        correct = 0
        trades = 0
        previous_signal = 0

        for row in self.feature_rows:
            prediction = self.trained_model.predict(row.features)
            trades += prediction
            if prediction == row.target:
                correct += 1
            strategy_return = previous_signal * row.forward_return
            returns.append(strategy_return)
            cumulative *= 1 + strategy_return
            previous_signal = prediction

        avg_daily_return = sum(returns) / len(returns) if returns else 0.0
        accuracy = correct / len(self.feature_rows) if self.feature_rows else 0.0
        total_return = cumulative - 1
        return PerformanceReport(
            total_return=total_return,
            avg_daily_return=avg_daily_return,
            accuracy=accuracy,
            trades=trades,
        )


__all__ = ["PredictionResult", "PerformanceReport", "AITradingService"]
