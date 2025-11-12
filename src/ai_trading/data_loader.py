"""Utility helpers for loading market data used by the AI trading system."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_DATASET = DATA_DIR / "sample_prices.csv"


@dataclass(frozen=True)
class PricePoint:
    """Single end-of-day price observation."""

    date: str
    close: float


@dataclass
class PriceSeries:
    """Collection of price points with convenience helpers."""

    points: List[PricePoint]

    def closes(self) -> List[float]:
        return [p.close for p in self.points]

    def dates(self) -> List[str]:
        return [p.date for p in self.points]

    def tail(self, count: int) -> "PriceSeries":
        return PriceSeries(points=self.points[-count:])

    def append(self, point: PricePoint) -> None:
        self.points.append(point)


def load_price_data(csv_path: str | Path | None = None) -> PriceSeries:
    """Load price data from ``csv_path`` or the packaged sample dataset."""

    target = Path(csv_path) if csv_path is not None else DEFAULT_DATASET
    if not target.exists():
        raise ValueError(f"CSV file not found: {target}")

    points: List[PricePoint] = []
    with target.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        if "Close" not in reader.fieldnames or "Date" not in reader.fieldnames:
            raise ValueError("CSV must provide 'Date' and 'Close' columns")
        for row in reader:
            try:
                close = float(row["Close"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid close price: {row['Close']!r}") from exc
            points.append(PricePoint(date=row["Date"], close=close))

    if not points:
        raise ValueError("CSV did not contain any price rows")
    points.sort(key=lambda p: p.date)
    return PriceSeries(points=points)


__all__ = ["PricePoint", "PriceSeries", "load_price_data", "DEFAULT_DATASET", "DATA_DIR"]
