"""
Auxo Validation Lab
===================
Research-only validation harness for the existing Auxo backtester.

It deliberately does NOT claim to validate ML/regime/expectancy components that
the candle backtester does not execute. Those are reported as "not scored".
"""
from __future__ import annotations

import math
import random
import statistics
from datetime import datetime, timezone
from typing import Any

def _imports():
    from bot_server import (
        fetch_candles,
        run_backtest_for_symbol,
        backtest_runtime_settings,
        parse_watchlist,
        strategy_minimum_candles,
    )
    return fetch_candles, run_backtest_for_symbol, backtest_runtime_settings, parse_watchlist, strategy_minimum_candles


def _trade_returns(result: dict[str, Any]) -> list[float]:
    """Extract approximate realised trade returns from the backtest trade ledger."""
    trades = sorted(
        [dict(trade) for trade in (result.get("trades") or [])],
        key=lambda item: int(item.get("time") or 0),
    )
    buys: list[dict[str, Any]] = []
    returns: list[float] = []
    for trade in trades:
        if trade.get("side") == "BUY":
            buys.append(trade)
            continue
        if trade.get("side") != "SELL" or not buys:
            continue
        sell_qty = float(trade.get("quantity") or 0)
        remaining = sell_qty
        while remaining > 1e-12 and buys:
            buy = buys[0]
            qty = min(remaining, float(buy.get("quantity") or 0))
            if qty <= 0:
                buys.pop(0)
                continue
            entry = float(buy.get("price") or 0)
            exit_price = float(trade.get("price") or 0)
            fee = float(trade.get("fee_paid") or 0)
            entry_fee = float(buy.get("fee_paid") or 0)
            if entry > 0 and exit_price > 0:
                pnl = (exit_price - entry) * qty - fee - (entry_fee * (qty / max(float(buy.get("quantity") or 1), 1e-12)))
                returns.append(pnl / max(entry * qty, 1e-12))
            remaining -= qty
            original_qty = float(buy.get("quantity") or 0)
            buy["quantity"] = original_qty - qty
            if buy["quantity"] <= 1e-12:
                buys.pop(0)
    return returns


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "symbols": 0, "trades": 0, "closed_trades": 0, "total_pnl": 0.0,
            "return_pct": 0.0, "win_rate": 0.0, "profit_factor": 0.0,
            "expectancy_pct": 0.0, "max_drawdown_pct": 0.0,
            "trade_returns": [],
        }

    starting = sum(float(r.get("final_equity", 0)) - float(r.get("total_pnl", 0)) for r in results)
    pnl = sum(float(r.get("total_pnl", 0)) for r in results)
    trade_returns = []
    gross_profit = gross_loss = 0.0
    wins = losses = 0
    for r in results:
        trs = _trade_returns(r)
        trade_returns.extend(trs)
        for x in trs:
            if x > 0:
                wins += 1
                gross_profit += x
            elif x < 0:
                losses += 1
                gross_loss += abs(x)

    # Approximate portfolio drawdown conservatively from per-symbol worst DD.
    max_dd = max((abs(float(r.get("max_drawdown_pct", 0))) for r in results), default=0.0)
    return {
        "symbols": len(results),
        "trades": sum(int(r.get("trades_count", 0)) for r in results),
        "closed_trades": sum(int(r.get("closed_trades", 0)) for r in results),
        "total_pnl": round(pnl, 8),
        "return_pct": round((pnl / starting) * 100, 4) if starting else 0.0,
        "win_rate": round((wins / (wins + losses)) * 100, 2) if wins + losses else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss else (999.0 if gross_profit else 0.0),
        "expectancy_pct": round(statistics.mean(trade_returns) * 100, 5) if trade_returns else 0.0,
        "max_drawdown_pct": round(max_dd, 4),
        "trade_returns": trade_returns,
    }


def _run_on_candles(symbol_candles: dict[str, list[Any]], settings: dict[str, Any]) -> list[dict[str, Any]]:
    _, run_backtest_for_symbol, _, _, strategy_minimum_candles = _imports()
    results = []
    for symbol, candles in symbol_candles.items():
        if len(candles) < strategy_minimum_candles(settings):
            continue
        results.append(run_backtest_for_symbol(symbol, candles, settings))
    return results


def _variant_settings(base: dict[str, Any], name: str) -> dict[str, Any]:
    s = dict(base)
    if name == "core":
        s.update({
            "use_sr_filter": False,
            "use_dynamic_sr_exits": False,
            "partial_take_profit_enabled": False,
            "trailing_stop_enabled": False,
        })
    elif name == "sr":
        s.update({
            "use_sr_filter": True,
            "use_dynamic_sr_exits": True,
        })
    elif name == "current":
        pass
    return s


def _monte_carlo(trade_returns: list[float], simulations: int = 1000, seed: int = 42) -> dict[str, Any]:
    if len(trade_returns) < 10:
        return {"status": "insufficient_data", "trades": len(trade_returns)}
    rng = random.Random(seed)
    n = len(trade_returns)
    terminal = []
    max_dds = []
    for _ in range(simulations):
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        for _ in range(n):
            r = rng.choice(trade_returns)
            equity *= max(0.01, 1.0 + r)
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak)
        terminal.append(equity - 1.0)
        max_dds.append(max_dd)
    terminal.sort()
    max_dds.sort()
    q = lambda xs, p: xs[min(len(xs) - 1, max(0, int(len(xs) * p)))]
    return {
        "status": "ok",
        "simulations": simulations,
        "trades_per_path": n,
        "median_return_pct": round(q(terminal, .50) * 100, 2),
        "p05_return_pct": round(q(terminal, .05) * 100, 2),
        "p95_return_pct": round(q(terminal, .95) * 100, 2),
        "median_max_drawdown_pct": round(q(max_dds, .50) * 100, 2),
        "p95_max_drawdown_pct": round(q(max_dds, .95) * 100, 2),
        "probability_positive_pct": round(sum(x > 0 for x in terminal) / len(terminal) * 100, 2),
    }


def run_validation_suite(settings: dict[str, Any]) -> dict[str, Any]:
    fetch_candles, _, backtest_runtime_settings, parse_watchlist, strategy_minimum_candles = _imports()
    settings = backtest_runtime_settings(dict(settings))
    watchlist = parse_watchlist(settings.get("watchlist", "BTC,ETH,SOL,XRP"))
    granularity = int(settings.get("validation_granularity", settings.get("granularity", 3600)))
    candle_count = max(300, min(3000, int(settings.get("validation_candles", 300))))
    requested = candle_count

    candles_by_symbol = {}
    errors = []
    for symbol in watchlist:
        try:
            candles = fetch_candles(
                exchange=str(settings["exchange"]),
                symbol=symbol,
                quote_currency=str(settings["quote_currency"]),
                granularity=granularity,
                candle_count=candle_count,
                asset_class=str(settings.get("asset_class", "crypto")),
            )
            if len(candles) >= strategy_minimum_candles(settings):
                candles_by_symbol[symbol] = candles
            else:
                errors.append(f"{symbol}: only {len(candles)} candles available")
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")

    variants = {}
    for name in ("core", "sr", "current"):
        vs = _variant_settings(settings, name)
        rs = _run_on_candles(candles_by_symbol, vs)
        variants[name] = _aggregate(rs)

    # Slippage stress against the current configuration.
    slippage_grid = [0.0, 0.05, 0.10, 0.20, 0.50]
    slippage = []
    for pct in slippage_grid:
        vs = dict(settings)
        vs["backtest_slippage_pct"] = pct
        agg = _aggregate(_run_on_candles(candles_by_symbol, vs))
        slippage.append({
            "slippage_pct": pct,
            **{k: v for k, v in agg.items() if k != "trade_returns"},
        })

    # Parameter sensitivity around the current configuration.
    sensitivity = []
    for stop_mult in (0.9, 1.0, 1.1):
        for target_mult in (0.9, 1.0, 1.1):
            vs = dict(settings)
            vs["stop_loss_pct"] = float(settings.get("stop_loss_pct", 1.8)) * stop_mult
            vs["take_profit_pct"] = float(settings.get("take_profit_pct", 1.8)) * target_mult
            agg = _aggregate(_run_on_candles(candles_by_symbol, vs))
            sensitivity.append({
                "stop_multiplier": stop_mult,
                "target_multiplier": target_mult,
                "return_pct": agg["return_pct"],
                "profit_factor": agg["profit_factor"],
                "win_rate": agg["win_rate"],
                "max_drawdown_pct": agg["max_drawdown_pct"],
                "trades": agg["trades"],
            })

    current = variants["current"]
    monte_carlo = _monte_carlo(current["trade_returns"])

    # Evidence score is deliberately conservative. It is not a profitability claim.
    score = 0
    if current["closed_trades"] >= 50: score += 20
    elif current["closed_trades"] >= 20: score += 10
    if current["profit_factor"] >= 1.2: score += 20
    if current["expectancy_pct"] > 0: score += 15
    if current["return_pct"] > 0: score += 10
    if current["max_drawdown_pct"] <= 15: score += 10
    if monte_carlo.get("status") == "ok" and monte_carlo.get("probability_positive_pct", 0) >= 60: score += 15
    if len(candles_by_symbol) >= 3: score += 10
    verdict = "INSUFFICIENT EVIDENCE"
    if score >= 75: verdict = "PROMISING"
    elif score >= 55: verdict = "NEEDS MORE EVIDENCE"

    return {
        "ok": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "evidence_score": score,
        "data": {
            "symbols_requested": watchlist,
            "symbols_tested": list(candles_by_symbol),
            "granularity": granularity,
            "candles_per_symbol": candle_count,
            "candles_requested": requested,
            "historical_window_is_chunked": True,
        },
        "variants": variants,
        "slippage_stress": slippage,
        "parameter_sensitivity": sensitivity,
        "monte_carlo": monte_carlo,
        "components_not_scored": [
            "regime detector",
            "XGBoost confidence",
            "expectancy filter",
            "self-learning weights",
            "genetic optimiser",
        ],
        "errors": errors,
    }
