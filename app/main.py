"""HTTP API exposing the AI trading service without third-party dependencies."""
from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Tuple

from ai_trading.service import AITradingService, PerformanceReport, PredictionResult

service = AITradingService()


class TradingRequestHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler implementing the trading API contract."""

    server_version = "AITradingHTTP/1.0"

    def _json_response(self, status: HTTPStatus, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _parse_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(data.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON body") from exc

    def _handle_error(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        self._json_response(status, {"error": message})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003 - API requirement
        """Silence default logging to keep output clean during tests."""

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path == "/health":
            self._json_response(HTTPStatus.OK, {"status": "ok"})
            return
        if self.path == "/backtest":
            try:
                report = service.backtest()
            except ValueError as exc:
                self._handle_error(str(exc))
                return
            self._json_response(HTTPStatus.OK, _report_to_dict(report))
            return
        self._handle_error("Endpoint not found", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path == "/train":
            try:
                payload = self._parse_json()
                csv_path = payload.get("csv_path")
                service.train(csv_path)
            except ValueError as exc:
                self._handle_error(str(exc))
                return
            self._json_response(HTTPStatus.OK, {"status": "retrained"})
            return
        if self.path == "/predict":
            try:
                payload = self._parse_json()
                closes = payload.get("closes")
                if not isinstance(closes, list) or not closes:
                    raise ValueError("Payload must contain a non-empty 'closes' array")
                if not all(isinstance(value, (int, float)) for value in closes):
                    raise ValueError("'closes' must be a list of numbers")
                result = service.predict_from_prices(closes)
            except ValueError as exc:
                self._handle_error(str(exc))
                return
            self._json_response(HTTPStatus.OK, _prediction_to_dict(result))
            return
        self._handle_error("Endpoint not found", HTTPStatus.NOT_FOUND)


def _prediction_to_dict(result: PredictionResult) -> Dict[str, Any]:
    return {
        "probability_up": result.probability_up,
        "signal": result.signal,
        "confidence": result.confidence,
    }


def _report_to_dict(report: PerformanceReport) -> Dict[str, Any]:
    return {
        "total_return": report.total_return,
        "avg_daily_return": report.avg_daily_return,
        "accuracy": report.accuracy,
        "trades": report.trades,
    }


def run(host: str = "0.0.0.0", port: int = 8000) -> Tuple[str, int]:
    """Start the HTTP server and return the host/port for confirmation."""

    server = HTTPServer((host, port), TradingRequestHandler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return host, port


__all__ = ["run", "TradingRequestHandler", "service"]
