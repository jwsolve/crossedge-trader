# Auxo Validation Lab — implementation notes

This build adds two changes to the Auxo project.

## 1. Live native-stop fail-closed behaviour

`bot_server.py` now treats a native stop submission failure differently depending on mode:

- Paper mode: retains the existing simulated-stop fallback.
- Armed Coinbase live mode: does **not** leave the position relying on a process-local stop.
  It attempts an immediate Coinbase market exit and reconciles the fill.
- If the emergency exit also fails, Auxo raises a critical error and journals that the
  position may be unprotected.

The Coinbase native stop configuration uses `stop_limit_stop_limit_gtc`.

## 2. Validation Lab

A new `validation_engine.py` module is exposed through:

`POST /api/validation`

The dashboard now contains a **Validation Lab** tab.

The current suite runs:

- Core vs S/R vs current configuration comparison
- Slippage stress at 0%, 0.05%, 0.10%, 0.20%, 0.50%
- Stop/target parameter sensitivity
- Monte Carlo resampling when there are enough closed trades
- Conservative evidence score and verdict

The suite deliberately reports regime detection, XGBoost, expectancy filtering,
self-learning and genetic optimisation as **not scored** because the current candle
backtester does not execute those components. This avoids producing misleading
validation results.

## 3. Historical Coinbase candles

`fetch_coinbase_candles()` now retrieves historical data in 300-candle chunks,
up to 3,000 candles, instead of silently truncating every request to 300 candles.

## Important limitation

The Validation Lab is a research tool, not a profitability guarantee. A serious
production validation run should use a sufficiently long historical sample and
should eventually add rolling multi-window walk-forward validation and a completely
untouched final test period.
