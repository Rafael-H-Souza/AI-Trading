from ai_trading.service import AITradingService


def test_prediction_pipeline_runs():
    service = AITradingService()
    result = service.predict_from_prices([150, 151, 152, 153, 154, 155, 156])
    assert result.signal in {"buy", "sell"}
    assert 0 <= result.probability_up <= 1


def test_backtest_returns_report():
    service = AITradingService()
    report = service.backtest()
    assert -1 < report.total_return < 2
    assert report.trades >= 0
