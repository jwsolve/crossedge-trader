"""Basic Auxo validation smoke tests. Run: python smoke_test.py"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import bot_server
import validation_engine

candles = []
for i in range(260):
    close = 100 + 0.04 * i + math.sin(i / 5.0) * 2.0
    candles.append(bot_server.Candle(
        time=1700000000 + i * 3600,
        open=close - 0.1,
        high=close + 0.4,
        low=close - 0.4,
        close=close,
        volume=1000.0,
    ))

settings = dict(bot_server.DEFAULT_SETTINGS)
settings.update({
    "strategy": "sma_cross",
    "short_window": 10,
    "long_window": 30,
    "starting_cash": 1000,
    "trade_fee": 0.001,
    "backtest_slippage_pct": 0.1,
    "use_sr_filter": False,
    "use_dynamic_sr_exits": False,
    "partial_take_profit_enabled": False,
    "trailing_stop_enabled": False,
    "position_sizing_mode": "risk_based",
    "risk_per_trade_pct": 1.0,
    "max_position_pct": 0.25,
})

result = bot_server.run_backtest_for_symbol("SMOKE", candles, settings)
assert "total_pnl_pct" in result
assert "max_drawdown_pct" in result

captured = {}
original = bot_server.coinbase_api_request
bot_server.coinbase_api_request = lambda method, path, body=None: captured.update(
    {"method": method, "path": path, "body": body}
) or {"order_id": "smoke"}

try:
    bot_server.coinbase_stop_limit_order("BTC-GBP", "SELL", 0.01, 90000, 89500)
finally:
    bot_server.coinbase_api_request = original

assert "stop_limit_stop_limit_gtc" in captured["body"]["order_configuration"]

print("Auxo smoke tests passed.")
