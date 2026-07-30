# backtest.py
"""
Backtesting and optimization functions.
"""

import math
import time
from datetime import datetime, timezone
from typing import Any, Optional, List, Dict

from utils import pct, parse_watchlist, strategy_minimum_candles, apply_slippage, backtest_runtime_settings, is_forex_settings, no_train_trades_message, FOREX_BASE_RATES
from indicators import sma, ema_series, exit_prices, position_spend, partial_take_profit_ready, trailing_stop_price, ewo_offset_signal, support_resistance, sr_buy_allowed, market_regime, regime_allowed
from risk_manager import calculate_atr
from order_execution import paper_buy, paper_sell
from coinbase_api import fetch_coinbase_candles, live_market_guard
from oanda_api import fetch_oanda_demo_candles, oanda_instrument, oanda_request, oanda_granularity, parse_oanda_time
from database import Candle

# ─── Candle fetching (unified) ──────────────────────────────────

def fetch_candles(
    exchange: str,
    symbol: str,
    quote_currency: str,
    granularity: int,
    candle_count: int,
    asset_class: str = "crypto",
) -> list:
    exchange = exchange.lower()
    symbol = symbol.upper()
    quote_currency = quote_currency.upper()
    asset_class = asset_class.lower()

    if asset_class == "forex":
        if exchange == "oanda_demo":
            return fetch_oanda_demo_candles(symbol, granularity, candle_count)
        return fetch_forex_demo_candles(symbol, granularity, candle_count)

    if exchange == "coinbase":
        return fetch_coinbase_candles(symbol, quote_currency, granularity, candle_count)

    if exchange == "binance":
        return fetch_binance_candles(symbol, quote_currency, granularity, candle_count)

    if exchange == "kraken":
        return fetch_kraken_candles(symbol, quote_currency, granularity, candle_count)

    raise RuntimeError("Exchange must be coinbase, binance, or kraken")

def fetch_forex_demo_candles(symbol: str, granularity: int, candle_count: int) -> list:
    from utils import normalize_forex_symbol, FOREX_BASE_RATES
    from types import SimpleNamespace
    symbol = normalize_forex_symbol(symbol)
    candle_count = max(40, min(720, int(candle_count)))
    base = FOREX_BASE_RATES.get(symbol)
    if base is None:
        raise RuntimeError("Unsupported forex demo pair")

    pip = 0.01 if symbol.endswith("JPY") else 0.0001
    now = int(time.time())
    end_time = now - (now % int(granularity))
    seed = sum(ord(char) for char in symbol)
    drift = ((seed % 11) - 5) * pip * 0.015
    amplitude = base * (0.0018 + ((seed % 7) * 0.00012))
    candles: list = []
    previous_close = base

    for index in range(candle_count):
        step = index - candle_count + 1
        timestamp = end_time + (step * int(granularity))
        wave = math.sin((index + seed) / 8.0) * amplitude
        faster_wave = math.sin((index + seed) / 2.7) * amplitude * 0.22
        close = max(pip, base + wave + faster_wave + (step * drift))
        open_price = previous_close
        spread = max(pip * 2, abs(close - open_price) * 0.7 + amplitude * 0.18)
        high = max(open_price, close) + spread
        low = max(pip, min(open_price, close) - spread)
        candles.append(
            SimpleNamespace(
                time=timestamp,
                open=round(open_price, 5 if pip < 0.01 else 3),
                high=round(high, 5 if pip < 0.01 else 3),
                low=round(low, 5 if pip < 0.01 else 3),
                close=round(close, 5 if pip < 0.01 else 3),
                volume=1_000_000 + ((index + seed) % 17) * 25_000,
            )
        )
        previous_close = close

    return candles

def fetch_binance_candles(symbol: str, quote_currency: str, granularity: int, candle_count: int) -> list:
    try:
        from binance.client import Client
        from types import SimpleNamespace
        BINANCE_AVAILABLE = True
    except ImportError:
        raise RuntimeError("python-binance package not installed")
    client = Client("", "")
    interval_map = {
        60: Client.KLINE_INTERVAL_1MINUTE,
        300: Client.KLINE_INTERVAL_5MINUTE,
        900: Client.KLINE_INTERVAL_15MINUTE,
        3600: Client.KLINE_INTERVAL_1HOUR,
        21600: Client.KLINE_INTERVAL_6HOUR,
        86400: Client.KLINE_INTERVAL_1DAY,
    }
    interval = interval_map.get(granularity, Client.KLINE_INTERVAL_1HOUR)
    pair = f"{symbol}{quote_currency}"
    try:
        candles = client.get_klines(symbol=pair, interval=interval, limit=candle_count)
        return [
            SimpleNamespace(
                time=int(c[0] / 1000),
                open=float(c[1]),
                high=float(c[2]),
                low=float(c[3]),
                close=float(c[4]),
                volume=float(c[5])
            )
            for c in candles
        ]
    except Exception as e:
        raise RuntimeError(f"Binance candle fetch error: {e}")

def fetch_kraken_candles(symbol: str, quote_currency: str, granularity: int, candle_count: int) -> list:
    from types import SimpleNamespace
    from utils import fetch_json
    interval_minutes = max(1, int(granularity / 60))
    candle_count = max(20, min(720, int(candle_count)))
    since = int(time.time() - (interval_minutes * 60 * candle_count))
    kraken_symbol_map = {"BTC": "XBT", "DOGE": "XDG"}
    pair = f"{kraken_symbol_map.get(symbol, symbol)}{quote_currency}"
    query = urllib.parse.urlencode({
        "pair": pair,
        "interval": interval_minutes,
        "since": since,
    })
    data = fetch_json(f"https://api.kraken.com/0/public/OHLC?{query}")

    if data.get("error"):
        raise RuntimeError("; ".join(data["error"]))

    result_key = next(key for key in data["result"].keys() if key != "last")
    candles = [
        SimpleNamespace(
            time=int(item[0]),
            open=float(item[1]),
            high=float(item[2]),
            low=float(item[3]),
            close=float(item[4]),
            volume=float(item[6]),
        )
        for item in data["result"][result_key]
    ]
    return sorted(candles, key=lambda item: item.time)[-candle_count:]

# ─── Backtest functions ──────────────────────────────────────────

def run_backtest(settings: dict) -> dict:
    settings = backtest_runtime_settings(settings)
    watchlist = parse_watchlist(settings.get("watchlist", "BTC"))
    if not watchlist:
        raise RuntimeError("Backtest watchlist is empty")

    granularity = int(settings.get("granularity", 3600))
    candle_count = int(settings.get("candle_count", 300))
    results: list[dict] = []
    errors: list[str] = []

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
            minimum_candles = strategy_minimum_candles(settings)
            if len(candles) < minimum_candles:
                raise RuntimeError(
                    f"Not enough candle data for {settings.get('strategy', 'sma_cross')} "
                    f"({len(candles)}/{minimum_candles} candles)"
                )
            results.append(run_backtest_for_symbol(symbol, candles, settings))
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")

    results.sort(key=lambda item: item["total_pnl_pct"], reverse=True)
    return {
        "ok": True,
        "exchange": settings["exchange"],
        "quote_currency": settings["quote_currency"],
        "granularity": granularity,
        "candle_count": candle_count,
        "results": results,
        "best": results[0] if results else None,
        "errors": errors,
    }

def run_backtest_for_symbol(symbol: str, candles: list, settings: dict) -> dict:
    if settings.get("strategy") == "opening_range":
        return run_opening_range_backtest(symbol, candles, settings)
    if settings.get("strategy") == "ewo_offset":
        return run_ewo_offset_backtest_for_symbol(symbol, candles, settings)
    if settings.get("strategy") == "ema_golden_cross":
        return run_ema_golden_cross_backtest(symbol, candles, settings)

    starting_cash = float(settings["starting_cash"])
    cash = starting_cash
    coin = 0.0
    entry_price: float | None = None
    highest_price: float | None = None
    partial_done = False
    trade_fee = float(settings["trade_fee"])
    slippage = float(settings.get("backtest_slippage_pct", 0.0)) / 100
    short_window = int(settings["short_window"])
    long_window = int(settings["long_window"])
    trade_start_time = int(settings.get("trade_start_time", 0))
    closes: list[float] = []
    trades: list[dict] = []
    equity_curve: list[float] = []
    peak_equity = starting_cash
    max_drawdown_pct = 0.0

    for index, candle in enumerate(candles):
        price = candle.close
        closes.append(price)
        active_candles = candles[:index + 1]

        equity = cash + (coin * price)
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0:
            drawdown_pct = ((equity - peak_equity) / peak_equity) * 100
            max_drawdown_pct = min(max_drawdown_pct, drawdown_pct)
        equity_curve.append(equity)

        if len(closes) < long_window + 1:
            continue

        short_now = sma(closes, short_window)
        long_now = sma(closes, long_window)
        short_prev = sma(closes[:-1], short_window)
        long_prev = sma(closes[:-1], long_window)

        if None in (short_now, long_now, short_prev, long_prev):
            continue

        can_trade = candle.time >= trade_start_time

        if can_trade and coin > 0 and entry_price:
            reason = None
            highest_price = max(highest_price or price, price)
            stop_price, target_price, exit_mode = exit_prices(
                entry_price=entry_price,
                candles=active_candles,
                settings=settings,
            )
            partial_quantity = 0.0
            if partial_take_profit_ready(price, entry_price, target_price, settings, partial_done):
                reason = f"partial {exit_mode} target"
                partial_done = True
                partial_quantity = coin * (float(settings.get("partial_take_profit_pct", 50.0)) / 100)
            elif trailing_stop_price(entry_price, highest_price, settings) and price <= trailing_stop_price(entry_price, highest_price, settings):
                reason = "trailing stop"
            elif price <= stop_price:
                reason = f"{exit_mode} stop"
            elif price >= target_price:
                reason = f"{exit_mode} target"
            elif short_prev >= long_prev and short_now < long_now:
                reason = "trend turned down"

            if reason:
                sold_quantity = min(coin, partial_quantity or coin)
                fill_price = apply_slippage(price, "SELL", slippage)
                gross = sold_quantity * fill_price
                fee_paid = gross * trade_fee
                cash += gross - fee_paid
                trades.append({
                    "time": candle.time,
                    "side": "SELL",
                    "symbol": symbol,
                    "price": fill_price,
                    "quantity": sold_quantity,
                    "cash_after": cash,
                    "reason": f"{reason} | slippage {settings.get('backtest_slippage_pct', 0.0)}%",
                    "fee_paid": fee_paid,
                })
                coin -= sold_quantity
                if coin <= 0.0000000001:
                    coin = 0.0
                    entry_price = None
                    highest_price = None
                    partial_done = False

        elif can_trade and short_prev <= long_prev and short_now > long_now:
            allowed, reason = sr_buy_allowed(
                price,
                support_resistance(active_candles, settings),
                settings,
            )
            if not allowed:
                continue

            spend, spend_reason = position_spend(cash, price, active_candles, settings)
            spend = min(spend, cash)
            if spend >= float(settings.get("min_order_value", 1.0)):
                fill_price = apply_slippage(price, "BUY", slippage)
                fee_paid = spend * trade_fee
                coin = (spend - fee_paid) / fill_price
                cash -= spend
                entry_price = fill_price
                highest_price = fill_price
                partial_done = False
                trades.append({
                    "time": candle.time,
                    "side": "BUY",
                    "symbol": symbol,
                    "price": fill_price,
                    "quantity": coin,
                    "cash_after": cash,
                    "reason": f"trend turned up | size {spend_reason} | slippage {settings.get('backtest_slippage_pct', 0.0)}%",
                    "fee_paid": fee_paid,
                })

    final_price = candles[-1].close if candles else 0.0
    final_equity = cash + (coin * final_price)
    total_pnl = final_equity - starting_cash
    total_pnl_pct = pct(total_pnl, starting_cash)
    sells = [trade for trade in trades if trade["side"] == "SELL"]
    wins = 0
    losses = 0

    for index, trade in enumerate(trades):
        if trade["side"] != "SELL":
            continue
        buy_trade = next(
            (
                prior for prior in reversed(trades[:index])
                if prior["side"] == "BUY" and prior["symbol"] == trade["symbol"]
            ),
            None,
        )
        if buy_trade and trade["price"] > buy_trade["price"]:
            wins += 1
        else:
            losses += 1

    return {
        "symbol": symbol,
        "candles": len(candles),
        "start_price": candles[0].close if candles else None,
        "end_price": final_price,
        "final_equity": round(final_equity, 8),
        "total_pnl": round(total_pnl, 8),
        "total_pnl_pct": total_pnl_pct,
        "max_drawdown_pct": round(max_drawdown_pct, 4),
        "trades_count": len(trades),
        "closed_trades": len(sells),
        "win_rate": round((wins / len(sells)) * 100, 2) if sells else 0.0,
        "slippage_pct": float(settings.get("backtest_slippage_pct", 0.0)),
        "open_position": coin > 0,
        "trades": trades[-80:][::-1],
        "equity_curve": [round(item, 8) for item in equity_curve[-300:]],
    }

def run_opening_range_backtest(symbol: str, candles: list, settings: dict) -> dict:
    starting_cash = float(settings["starting_cash"])
    cash = starting_cash
    coin = 0.0
    entry_price: float | None = None
    highest_price: float | None = None
    partial_done = False
    trade_fee = float(settings["trade_fee"])
    slippage = float(settings.get("backtest_slippage_pct", 0.0)) / 100
    trades: list[dict] = []
    equity_curve: list[float] = []
    peak_equity = starting_cash
    max_drawdown_pct = 0.0
    allow_short = settings.get("allow_short_selling", False)

    days: dict[str, list] = {}
    for candle in candles:
        date_key = datetime.fromtimestamp(candle.time, tz=timezone.utc).strftime("%Y-%m-%d")
        days.setdefault(date_key, []).append(candle)

    for date_key, day_candles in days.items():
        if len(day_candles) < int(settings.get("opening_range_atr_period", 14)) + 1:
            continue

        first_candle = day_candles[0]
        is_green = first_candle.close > first_candle.open
        candle_range = first_candle.high - first_candle.low

        atr = 0.0
        prev_days = list(days.keys())
        idx = prev_days.index(date_key)
        if idx >= int(settings.get("opening_range_atr_period", 14)):
            prev_candles = []
            for prev_date in prev_days[idx - int(settings.get("opening_range_atr_period", 14)):idx]:
                prev_candles.extend(days[prev_date])
            if prev_candles:
                atr = calculate_atr_from_candles(prev_candles, int(settings.get("opening_range_atr_period", 14)))

        if atr == 0:
            atr = candle_range

        manipulation_threshold = float(settings.get("opening_range_manipulation_threshold", 0.20))
        range_ratio = candle_range / atr if atr > 0 else 0
        manipulation = range_ratio < manipulation_threshold
        is_blowoff = range_ratio >= manipulation_threshold

        trigger = first_candle.high if is_green else first_candle.low
        stop_loss_mult = float(settings.get("opening_range_stop_loss_atr_multiplier", 1.5))
        take_profit_mult = float(settings.get("opening_range_take_profit_atr_multiplier", 2.5))

        if coin != 0:
            close_price = day_candles[-1].close
            gross = abs(coin) * close_price
            fee = gross * trade_fee
            if coin > 0:
                cash += gross - fee
                trades.append({
                    "time": day_candles[-1].time,
                    "side": "SELL",
                    "symbol": symbol,
                    "price": close_price,
                    "quantity": abs(coin),
                    "cash_after": cash,
                    "reason": "End of day close (long)",
                    "fee_paid": fee,
                })
            else:
                cash += gross - fee
                trades.append({
                    "time": day_candles[-1].time,
                    "side": "BUY",
                    "symbol": symbol,
                    "price": close_price,
                    "quantity": abs(coin),
                    "cash_after": cash,
                    "reason": "End of day close (short)",
                    "fee_paid": fee,
                })
            coin = 0.0
            entry_price = None

        if manipulation:
            if is_green:
                for candle in day_candles[1:]:
                    if candle.close > trigger:
                        entry = trigger
                        stop = entry - (atr * stop_loss_mult)
                        target = entry + (atr * take_profit_mult)
                        break
                else:
                    continue

                spend, spend_reason = position_spend(cash, entry, day_candles, settings)
                spend = min(spend, cash)
                if spend >= float(settings.get("min_order_value", 1.0)):
                    fill_price = apply_slippage(entry, "BUY", slippage)
                    fee_paid = spend * trade_fee
                    coin = (spend - fee_paid) / fill_price
                    cash -= spend
                    entry_price = fill_price
                    highest_price = fill_price
                    trades.append({
                        "time": first_candle.time,
                        "side": "BUY",
                        "symbol": symbol,
                        "price": fill_price,
                        "quantity": coin,
                        "cash_after": cash,
                        "reason": f"Opening Range BUY | {spend_reason}",
                        "fee_paid": fee_paid,
                    })

                    for candle in day_candles:
                        price = candle.close
                        if price <= stop and stop > 0:
                            gross = coin * price
                            fee = gross * trade_fee
                            cash += gross - fee
                            trades.append({
                                "time": candle.time,
                                "side": "SELL",
                                "symbol": symbol,
                                "price": price,
                                "quantity": coin,
                                "cash_after": cash,
                                "reason": "Stop loss hit (long)",
                                "fee_paid": fee,
                            })
                            coin = 0.0
                            entry_price = None
                            break
                        elif price >= target and target > 0:
                            gross = coin * price
                            fee = gross * trade_fee
                            cash += gross - fee
                            trades.append({
                                "time": candle.time,
                                "side": "SELL",
                                "symbol": symbol,
                                "price": price,
                                "quantity": coin,
                                "cash_after": cash,
                                "reason": "Take profit hit (long)",
                                "fee_paid": fee,
                            })
                            coin = 0.0
                            entry_price = None
                            break

            elif allow_short:
                for candle in day_candles[1:]:
                    if candle.close < trigger:
                        entry = trigger
                        stop = entry + (atr * stop_loss_mult)
                        target = entry - (atr * take_profit_mult)
                        break
                else:
                    continue

                spend, spend_reason = position_spend(cash, entry, day_candles, settings)
                spend = min(spend, cash)
                if spend >= float(settings.get("min_order_value", 1.0)):
                    fill_price = apply_slippage(entry, "SELL", slippage)
                    fee_paid = spend * trade_fee
                    coin = -(spend - fee_paid) / fill_price
                    cash += spend
                    entry_price = fill_price
                    highest_price = fill_price
                    trades.append({
                        "time": first_candle.time,
                        "side": "SHORT",
                        "symbol": symbol,
                        "price": fill_price,
                        "quantity": abs(coin),
                        "cash_after": cash,
                        "reason": f"Opening Range SHORT | {spend_reason}",
                        "fee_paid": fee_paid,
                    })

                    for candle in day_candles:
                        price = candle.close
                        if price >= stop and stop > 0:
                            gross = abs(coin) * price
                            fee = gross * trade_fee
                            cash += gross - fee
                            trades.append({
                                "time": candle.time,
                                "side": "BUY",
                                "symbol": symbol,
                                "price": price,
                                "quantity": abs(coin),
                                "cash_after": cash,
                                "reason": "Stop loss hit (short)",
                                "fee_paid": fee,
                            })
                            coin = 0.0
                            entry_price = None
                            break
                        elif price <= target and target > 0:
                            gross = abs(coin) * price
                            fee = gross * trade_fee
                            cash += gross - fee
                            trades.append({
                                "time": candle.time,
                                "side": "BUY",
                                "symbol": symbol,
                                "price": price,
                                "quantity": abs(coin),
                                "cash_after": cash,
                                "reason": "Take profit hit (short)",
                                "fee_paid": fee,
                            })
                            coin = 0.0
                            entry_price = None
                            break

        elif is_blowoff:
            if is_green and allow_short:
                for candle in day_candles[1:]:
                    if candle.close < first_candle.open:
                        entry = candle.close
                        stop = entry - (atr * stop_loss_mult)
                        target = entry + (atr * take_profit_mult * 1.5)
                        break
                else:
                    continue

                spend, spend_reason = position_spend(cash, entry, day_candles, settings)
                spend = min(spend, cash)
                if spend >= float(settings.get("min_order_value", 1.0)):
                    fill_price = apply_slippage(entry, "BUY", slippage)
                    fee_paid = spend * trade_fee
                    coin = (spend - fee_paid) / fill_price
                    cash -= spend
                    entry_price = fill_price
                    highest_price = fill_price
                    trades.append({
                        "time": first_candle.time,
                        "side": "BUY",
                        "symbol": symbol,
                        "price": fill_price,
                        "quantity": coin,
                        "cash_after": cash,
                        "reason": f"Blow-off pullback BUY | {spend_reason}",
                        "fee_paid": fee_paid,
                    })

                    for candle in day_candles:
                        price = candle.close
                        if price <= stop and stop > 0:
                            gross = coin * price
                            fee = gross * trade_fee
                            cash += gross - fee
                            trades.append({
                                "time": candle.time,
                                "side": "SELL",
                                "symbol": symbol,
                                "price": price,
                                "quantity": coin,
                                "cash_after": cash,
                                "reason": "Stop loss hit (blow-off)",
                                "fee_paid": fee,
                            })
                            coin = 0.0
                            entry_price = None
                            break
                        elif price >= target and target > 0:
                            gross = coin * price
                            fee = gross * trade_fee
                            cash += gross - fee
                            trades.append({
                                "time": candle.time,
                                "side": "SELL",
                                "symbol": symbol,
                                "price": price,
                                "quantity": coin,
                                "cash_after": cash,
                                "reason": "Take profit hit (blow-off)",
                                "fee_paid": fee,
                            })
                            coin = 0.0
                            entry_price = None
                            break

    final_price = candles[-1].close if candles else 0.0
    final_equity = cash + (coin * final_price)
    total_pnl = final_equity - starting_cash
    total_pnl_pct = pct(total_pnl, starting_cash)
    sells = [trade for trade in trades if trade["side"] in ["SELL", "BUY"]]
    wins = 0
    losses = 0

    for index, trade in enumerate(trades):
        if trade["side"] == "SELL":
            buy_trade = next(
                (prior for prior in reversed(trades[:index]) if prior["side"] == "BUY" and prior["symbol"] == trade["symbol"]),
                None
            )
            if buy_trade and trade["price"] > buy_trade["price"]:
                wins += 1
            else:
                losses += 1
        elif trade["side"] == "SHORT":
            buy_trade = next(
                (prior for prior in reversed(trades[:index]) if prior["side"] == "SHORT" and prior["symbol"] == trade["symbol"]),
                None
            )
            if buy_trade and trade["price"] < buy_trade["price"]:
                wins += 1
            else:
                losses += 1

    return {
        "symbol": symbol,
        "candles": len(candles),
        "start_price": candles[0].close if candles else None,
        "end_price": final_price,
        "final_equity": round(final_equity, 8),
        "total_pnl": round(total_pnl, 8),
        "total_pnl_pct": total_pnl_pct,
        "max_drawdown_pct": round(max_drawdown_pct, 4),
        "trades_count": len(trades),
        "closed_trades": len([t for t in trades if t["side"] in ["SELL", "BUY"]]),
        "win_rate": round((wins / (wins + losses)) * 100, 2) if (wins + losses) > 0 else 0.0,
        "slippage_pct": float(settings.get("backtest_slippage_pct", 0.0)),
        "open_position": coin != 0,
        "trades": trades[-80:][::-1],
        "equity_curve": [round(item, 8) for item in equity_curve[-300:]],
    }

def run_ewo_offset_backtest_for_symbol(symbol: str, candles: list, settings: dict) -> dict:
    starting_cash = float(settings["starting_cash"])
    cash = starting_cash
    coin = 0.0
    entry_price: float | None = None
    highest_price: float | None = None
    partial_done = False
    trade_fee = float(settings["trade_fee"])
    slippage = float(settings.get("backtest_slippage_pct", 0.0)) / 100
    trade_start_time = int(settings.get("trade_start_time", 0))
    trades: list[dict] = []
    equity_curve: list[float] = []
    peak_equity = starting_cash
    max_drawdown_pct = 0.0

    for index, candle in enumerate(candles):
        price = candle.close
        active_candles = candles[:index + 1]

        equity = cash + (coin * price)
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0:
            drawdown_pct = ((equity - peak_equity) / peak_equity) * 100
            max_drawdown_pct = min(max_drawdown_pct, drawdown_pct)
        equity_curve.append(equity)

        signal = ewo_offset_signal(active_candles, settings)
        if not signal["ready"]:
            continue

        can_trade = candle.time >= trade_start_time

        if can_trade and coin > 0 and entry_price:
            reason = None
            highest_price = max(highest_price or price, price)
            stop_price, target_price, exit_mode = exit_prices(
                entry_price=entry_price,
                candles=active_candles,
                settings=settings,
            )
            partial_quantity = 0.0
            if partial_take_profit_ready(price, entry_price, target_price, settings, partial_done):
                reason = f"partial {exit_mode} target"
                partial_done = True
                partial_quantity = coin * (float(settings.get("partial_take_profit_pct", 50.0)) / 100)
            elif trailing_stop_price(entry_price, highest_price, settings) and price <= trailing_stop_price(entry_price, highest_price, settings):
                reason = "trailing stop"
            elif price <= stop_price:
                reason = f"{exit_mode} stop"
            elif price >= target_price:
                reason = f"{exit_mode} target"
            elif signal["sell"]:
                reason = "EWO offset sell"

            if reason:
                sold_quantity = min(coin, partial_quantity or coin)
                fill_price = apply_slippage(price, "SELL", slippage)
                gross = sold_quantity * fill_price
                fee_paid = gross * trade_fee
                cash += gross - fee_paid
                trades.append({
                    "time": candle.time,
                    "side": "SELL",
                    "symbol": symbol,
                    "price": fill_price,
                    "quantity": sold_quantity,
                    "cash_after": cash,
                    "reason": f"{reason} | slippage {settings.get('backtest_slippage_pct', 0.0)}%",
                    "fee_paid": fee_paid,
                })
                coin -= sold_quantity
                if coin <= 0.0000000001:
                    coin = 0.0
                    entry_price = None
                    highest_price = None
                    partial_done = False

        elif can_trade and signal["buy"]:
            allowed, reason = sr_buy_allowed(
                price,
                support_resistance(active_candles, settings),
                settings,
            )
            if not allowed:
                continue

            spend, spend_reason = position_spend(cash, price, active_candles, settings)
            spend = min(spend, cash)
            if spend >= float(settings.get("min_order_value", 1.0)):
                fill_price = apply_slippage(price, "BUY", slippage)
                fee_paid = spend * trade_fee
                coin = (spend - fee_paid) / fill_price
                cash -= spend
                entry_price = fill_price
                highest_price = fill_price
                partial_done = False
                trades.append({
                    "time": candle.time,
                    "side": "BUY",
                    "symbol": symbol,
                    "price": fill_price,
                    "quantity": coin,
                    "cash_after": cash,
                    "reason": f"{signal['tag'] or 'EWO offset buy'} | size {spend_reason} | slippage {settings.get('backtest_slippage_pct', 0.0)}%",
                    "fee_paid": fee_paid,
                })

    final_price = candles[-1].close if candles else 0.0
    final_equity = cash + (coin * final_price)
    total_pnl = final_equity - starting_cash
    total_pnl_pct = pct(total_pnl, starting_cash)
    sells = [trade for trade in trades if trade["side"] == "SELL"]
    wins = 0
    losses = 0

    for index, trade in enumerate(trades):
        if trade["side"] != "SELL":
            continue
        buy_trade = next(
            (
                prior for prior in reversed(trades[:index])
                if prior["side"] == "BUY" and prior["symbol"] == trade["symbol"]
            ),
            None,
        )
        if buy_trade and trade["price"] > buy_trade["price"]:
            wins += 1
        else:
            losses += 1

    return {
        "symbol": symbol,
        "candles": len(candles),
        "start_price": candles[0].close if candles else None,
        "end_price": final_price,
        "final_equity": round(final_equity, 8),
        "total_pnl": round(total_pnl, 8),
        "total_pnl_pct": total_pnl_pct,
        "max_drawdown_pct": round(max_drawdown_pct, 4),
        "trades_count": len(trades),
        "closed_trades": len(sells),
        "win_rate": round((wins / len(sells)) * 100, 2) if sells else 0.0,
        "slippage_pct": float(settings.get("backtest_slippage_pct", 0.0)),
        "open_position": coin > 0,
        "trades": trades[-80:][::-1],
        "equity_curve": [round(item, 8) for item in equity_curve[-300:]],
    }

def run_ema_golden_cross_backtest(symbol: str, candles: list, settings: dict) -> dict:
    starting_cash = float(settings["starting_cash"])
    cash = starting_cash
    coin = 0.0
    entry_price: float | None = None
    highest_price: float | None = None
    partial_done = False
    trade_fee = float(settings["trade_fee"])
    slippage = float(settings.get("backtest_slippage_pct", 0.0)) / 100
    ema_short = int(settings.get("ema_short", 50))
    ema_long = int(settings.get("ema_long", 200))
    trade_start_time = int(settings.get("trade_start_time", 0))
    closes: list[float] = []
    trades: list[dict] = []
    equity_curve: list[float] = []
    peak_equity = starting_cash
    max_drawdown_pct = 0.0

    for index, candle in enumerate(candles):
        price = candle.close
        closes.append(price)
        active_candles = candles[:index + 1]

        equity = cash + (coin * price)
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0:
            drawdown_pct = ((equity - peak_equity) / peak_equity) * 100
            max_drawdown_pct = min(max_drawdown_pct, drawdown_pct)
        equity_curve.append(equity)

        if len(closes) < ema_long + 1:
            continue

        ema_short_value = ema_series(closes, ema_short)[-1]
        ema_long_value = ema_series(closes, ema_long)[-1]
        ema_short_prev = ema_series(closes[:-1], ema_short)[-1] if len(closes) > 1 else None
        ema_long_prev = ema_series(closes[:-1], ema_long)[-1] if len(closes) > 1 else None

        if None in (ema_short_value, ema_long_value, ema_short_prev, ema_long_prev):
            continue

        can_trade = candle.time >= trade_start_time

        if can_trade and coin > 0 and entry_price:
            if ema_short_prev >= ema_long_prev and ema_short_value < ema_long_value:
                sold_quantity = coin
                fill_price = apply_slippage(price, "SELL", slippage)
                gross = sold_quantity * fill_price
                fee_paid = gross * trade_fee
                cash += gross - fee_paid
                trades.append({
                    "time": candle.time,
                    "side": "SELL",
                    "symbol": symbol,
                    "price": fill_price,
                    "quantity": sold_quantity,
                    "cash_after": cash,
                    "reason": "Death Cross (EMA crossed down)",
                    "fee_paid": fee_paid,
                })
                coin = 0.0
                entry_price = None
                highest_price = None
                continue

            stop_price, target_price, exit_mode = exit_prices(
                entry_price=entry_price,
                candles=active_candles,
                settings=settings,
            )
            highest_price = max(highest_price or price, price)

            if price <= stop_price:
                sold_quantity = coin
                fill_price = apply_slippage(price, "SELL", slippage)
                gross = sold_quantity * fill_price
                fee_paid = gross * trade_fee
                cash += gross - fee_paid
                trades.append({
                    "time": candle.time,
                    "side": "SELL",
                    "symbol": symbol,
                    "price": fill_price,
                    "quantity": sold_quantity,
                    "cash_after": cash,
                    "reason": f"Stop loss hit ({exit_mode})",
                    "fee_paid": fee_paid,
                })
                coin = 0.0
                entry_price = None
                highest_price = None
                continue
            elif price >= target_price:
                sold_quantity = coin
                fill_price = apply_slippage(price, "SELL", slippage)
                gross = sold_quantity * fill_price
                fee_paid = gross * trade_fee
                cash += gross - fee_paid
                trades.append({
                    "time": candle.time,
                    "side": "SELL",
                    "symbol": symbol,
                    "price": fill_price,
                    "quantity": sold_quantity,
                    "cash_after": cash,
                    "reason": f"Take profit hit ({exit_mode})",
                    "fee_paid": fee_paid,
                })
                coin = 0.0
                entry_price = None
                highest_price = None
                continue

        if can_trade and coin == 0:
            if ema_short_prev <= ema_long_prev and ema_short_value > ema_long_value:
                spend, spend_reason = position_spend(cash, price, active_candles, settings)
                spend = min(spend, cash)
                if spend >= float(settings.get("min_order_value", 1.0)):
                    fill_price = apply_slippage(price, "BUY", slippage)
                    fee_paid = spend * trade_fee
                    coin = (spend - fee_paid) / fill_price
                    cash -= spend
                    entry_price = fill_price
                    highest_price = fill_price
                    partial_done = False
                    trades.append({
                        "time": candle.time,
                        "side": "BUY",
                        "symbol": symbol,
                        "price": fill_price,
                        "quantity": coin,
                        "cash_after": cash,
                        "reason": f"Golden Cross (EMA {ema_short}/{ema_long})",
                        "fee_paid": fee_paid,
                    })

    final_price = candles[-1].close if candles else 0.0
    final_equity = cash + (coin * final_price)
    total_pnl = final_equity - starting_cash
    total_pnl_pct = pct(total_pnl, starting_cash)
    sells = [trade for trade in trades if trade["side"] == "SELL"]
    wins = 0
    losses = 0

    for index, trade in enumerate(trades):
        if trade["side"] != "SELL":
            continue
        buy_trade = next(
            (
                prior for prior in reversed(trades[:index])
                if prior["side"] == "BUY" and prior["symbol"] == trade["symbol"]
            ),
            None,
        )
        if buy_trade and trade["price"] > buy_trade["price"]:
            wins += 1
        else:
            losses += 1

    return {
        "symbol": symbol,
        "candles": len(candles),
        "start_price": candles[0].close if candles else None,
        "end_price": final_price,
        "final_equity": round(final_equity, 8),
        "total_pnl": round(total_pnl, 8),
        "total_pnl_pct": total_pnl_pct,
        "max_drawdown_pct": round(max_drawdown_pct, 4),
        "trades_count": len(trades),
        "closed_trades": len(sells),
        "win_rate": round((wins / len(sells)) * 100, 2) if sells else 0.0,
        "slippage_pct": float(settings.get("backtest_slippage_pct", 0.0)),
        "open_position": coin > 0,
        "trades": trades[-80:][::-1],
        "equity_curve": [round(item, 8) for item in equity_curve[-300:]],
    }

# ─── Optimization and walk-forward ──────────────────────────────

def run_optimizer(settings: dict) -> dict:
    settings = backtest_runtime_settings(settings)
    if settings.get("strategy") == "ewo_offset":
        return run_ewo_offset_optimizer(settings)

    watchlist = parse_watchlist(settings.get("watchlist", "BTC"))
    if not watchlist:
        raise RuntimeError("Optimizer watchlist is empty")

    granularity = int(settings.get("granularity", 3600))
    candle_count = int(settings.get("candle_count", 300))
    short_values = [5, 8, 10, 12]
    long_values = [20, 30, 40, 60]
    stop_values = [1.5, 2.5, 3.5]
    take_values = [3.0, 5.0, 7.0]
    position_values = [0.15, 0.25, 0.35]
    results: list[dict] = []
    errors: list[str] = []
    combinations_tested = 0

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
            if len(candles) < max(long_values) + 1:
                raise RuntimeError("Not enough candle data for optimizer")

            for short_window in short_values:
                for long_window in long_values:
                    if long_window <= short_window:
                        continue
                    for stop_loss in stop_values:
                        for take_profit in take_values:
                            for position_fraction in position_values:
                                candidate_settings = {
                                    **settings,
                                    "short_window": short_window,
                                    "long_window": long_window,
                                    "stop_loss_pct": stop_loss,
                                    "take_profit_pct": take_profit,
                                    "max_position_pct": position_fraction,
                                }
                                result = run_backtest_for_symbol(
                                    symbol=symbol,
                                    candles=candles,
                                    settings=candidate_settings,
                                )
                                combinations_tested += 1
                                if result["trades_count"] == 0:
                                    continue

                                score = result["total_pnl_pct"] + (
                                    result["max_drawdown_pct"] * 0.75
                                )
                                results.append({
                                    "symbol": symbol,
                                    "score": round(score, 4),
                                    "short_window": short_window,
                                    "long_window": long_window,
                                    "stop_loss_pct": stop_loss,
                                    "take_profit_pct": take_profit,
                                    "max_position_pct": position_fraction,
                                    "final_equity": result["final_equity"],
                                    "total_pnl": result["total_pnl"],
                                    "total_pnl_pct": result["total_pnl_pct"],
                                    "max_drawdown_pct": result["max_drawdown_pct"],
                                    "trades_count": result["trades_count"],
                                    "win_rate": result["win_rate"],
                                    "open_position": result["open_position"],
                                })
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")

    results.sort(key=lambda item: item["score"], reverse=True)
    return {
        "ok": True,
        "exchange": settings["exchange"],
        "quote_currency": settings["quote_currency"],
        "granularity": granularity,
        "candle_count": candle_count,
        "combinations_tested": combinations_tested,
        "results": results[:20],
        "best": results[0] if results else None,
        "errors": errors,
    }

def run_ewo_offset_optimizer(settings: dict) -> dict:
    settings = backtest_runtime_settings(settings)
    watchlist = parse_watchlist(settings.get("watchlist", "BTC"))
    if not watchlist:
        raise RuntimeError("Optimizer watchlist is empty")

    granularity = int(settings.get("granularity", 3600))
    candle_count = int(settings.get("candle_count", 300))
    candidates = ewo_offset_candidate_settings(settings)
    results: list[dict] = []
    errors: list[str] = []
    combinations_tested = 0

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
            if len(candles) < strategy_minimum_candles(settings):
                raise RuntimeError("Not enough candle data for EWO offset optimizer")

            for candidate_settings in candidates:
                result = run_ewo_offset_backtest_for_symbol(
                    symbol=symbol,
                    candles=candles,
                    settings=candidate_settings,
                )
                combinations_tested += 1
                if result["trades_count"] == 0:
                    continue

                score = optimizer_score(result)
                results.append({
                    **result_settings_summary(symbol, candidate_settings),
                    "score": round(score, 4),
                    "final_equity": result["final_equity"],
                    "total_pnl": result["total_pnl"],
                    "total_pnl_pct": result["total_pnl_pct"],
                    "max_drawdown_pct": result["max_drawdown_pct"],
                    "trades_count": result["trades_count"],
                    "win_rate": result["win_rate"],
                    "open_position": result["open_position"],
                    "base_nb_candles_buy": int(candidate_settings["base_nb_candles_buy"]),
                    "base_nb_candles_sell": int(candidate_settings["base_nb_candles_sell"]),
                    "low_offset": float(candidate_settings["low_offset"]),
                    "low_offset_2": float(candidate_settings["low_offset_2"]),
                    "high_offset": float(candidate_settings["high_offset"]),
                    "high_offset_2": float(candidate_settings["high_offset_2"]),
                    "ewo_high": float(candidate_settings["ewo_high"]),
                    "ewo_high_2": float(candidate_settings["ewo_high_2"]),
                    "ewo_low": float(candidate_settings["ewo_low"]),
                    "rsi_buy": int(candidate_settings["rsi_buy"]),
                })
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")

    results.sort(key=lambda item: item["score"], reverse=True)
    return {
        "ok": True,
        "exchange": settings["exchange"],
        "quote_currency": settings["quote_currency"],
        "granularity": granularity,
        "candle_count": candle_count,
        "combinations_tested": combinations_tested,
        "results": results[:20],
        "best": results[0] if results else None,
        "errors": errors,
    }

def run_walk_forward(settings: dict) -> dict:
    settings = backtest_runtime_settings(settings)
    watchlist = parse_watchlist(settings.get("watchlist", "BTC"))
    if not watchlist:
        raise RuntimeError("Walk-forward watchlist is empty")

    granularity = int(settings.get("granularity", 3600))
    candle_count = int(settings.get("candle_count", 300))
    train_pct = float(settings.get("train_pct", 0.7))
    train_pct = min(0.85, max(0.5, train_pct))

    strategy = settings.get("strategy", "sma_cross")
    if strategy == "ewo_offset":
        candidates = ewo_offset_candidate_settings(settings)
    elif strategy == "opening_range":
        candidates = opening_range_candidate_settings(settings)
    else:
        candidates = optimizer_candidate_settings(settings)

    results: list[dict] = []
    errors: list[str] = []
    combinations_tested = 0

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
            split_index = int(len(candles) * train_pct)
            train_candles = candles[:split_index]
            test_candles = candles[split_index:]

            if len(train_candles) < 61 or len(test_candles) < 20:
                raise RuntimeError("Not enough candles for train/test split")

            best_train: dict | None = None
            best_settings: dict | None = None

            for candidate_settings in candidates:
                if len(train_candles) < strategy_minimum_candles(candidate_settings):
                    continue
                train_result = run_backtest_for_symbol(
                    symbol=symbol,
                    candles=train_candles,
                    settings=candidate_settings,
                )
                combinations_tested += 1
                if train_result["trades_count"] == 0:
                    continue

                train_score = optimizer_score(train_result)
                if not best_train or train_score > best_train["score"]:
                    best_train = {
                        **train_result,
                        "score": round(train_score, 4),
                    }
                    best_settings = candidate_settings

            if not best_train or not best_settings:
                raise RuntimeError(no_train_trades_message(settings))

            seed_count = strategy_minimum_candles(best_settings)
            test_seed_candles = train_candles[-seed_count:] + test_candles
            test_result = run_backtest_for_symbol(
                symbol=symbol,
                candles=test_seed_candles,
                settings={
                    **best_settings,
                    "trade_start_time": test_candles[0].time,
                },
            )
            test_score = optimizer_score(test_result)
            settings_summary = result_settings_summary(symbol, best_settings)

            results.append({
                **settings_summary,
                "train_score": best_train["score"],
                "train_pnl_pct": best_train["total_pnl_pct"],
                "train_drawdown_pct": best_train["max_drawdown_pct"],
                "train_trades": best_train["trades_count"],
                "test_score": round(test_score, 4),
                "test_final_equity": test_result["final_equity"],
                "test_total_pnl": test_result["total_pnl"],
                "test_total_pnl_pct": test_result["total_pnl_pct"],
                "test_drawdown_pct": test_result["max_drawdown_pct"],
                "test_trades": test_result["trades_count"],
                "test_win_rate": test_result["win_rate"],
                "test_open_position": test_result["open_position"],
                "train_candles": len(train_candles),
                "test_candles": len(test_candles),
            })
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")

    results.sort(key=lambda item: item["test_score"], reverse=True)
    return {
        "ok": True,
        "exchange": settings["exchange"],
        "quote_currency": settings["quote_currency"],
        "granularity": granularity,
        "candle_count": candle_count,
        "train_pct": train_pct,
        "combinations_tested": combinations_tested,
        "results": results[:20],
        "best": results[0] if results else None,
        "errors": errors,
    }

# ─── Helper functions for optimization ──────────────────────────

def optimizer_score(result: dict) -> float:
    return result["total_pnl_pct"] + (result["max_drawdown_pct"] * 0.75)

def result_settings_summary(symbol: str, settings: dict) -> dict:
    summary = {
        "symbol": symbol,
        "strategy": settings.get("strategy", "sma_cross"),
        "short_window": int(settings["short_window"]),
        "long_window": int(settings["long_window"]),
        "stop_loss_pct": float(settings["stop_loss_pct"]),
        "take_profit_pct": float(settings["take_profit_pct"]),
        "max_position_pct": float(settings["max_position_pct"]),
    }
    if settings.get("strategy") == "ewo_offset":
        summary.update({
            "base_nb_candles_buy": int(settings["base_nb_candles_buy"]),
            "base_nb_candles_sell": int(settings["base_nb_candles_sell"]),
            "low_offset": float(settings["low_offset"]),
            "low_offset_2": float(settings["low_offset_2"]),
            "high_offset": float(settings["high_offset"]),
            "high_offset_2": float(settings["high_offset_2"]),
            "ewo_high": float(settings["ewo_high"]),
            "ewo_high_2": float(settings["ewo_high_2"]),
            "ewo_low": float(settings["ewo_low"]),
            "rsi_buy": int(settings["rsi_buy"]),
        })
    if settings.get("strategy") == "opening_range":
        summary.update({
            "opening_range_minutes": int(settings["opening_range_minutes"]),
            "opening_range_atr_period": int(settings["opening_range_atr_period"]),
            "opening_range_manipulation_threshold": float(settings["opening_range_manipulation_threshold"]),
            "opening_range_stop_loss_atr_multiplier": float(settings["opening_range_stop_loss_atr_multiplier"]),
            "opening_range_take_profit_atr_multiplier": float(settings["opening_range_take_profit_atr_multiplier"]),
        })
    return summary

def ewo_offset_candidate_settings(settings: dict) -> list[dict]:
    forex = is_forex_settings(settings)
    if forex:
        buy_windows = [8, 14]
        sell_windows = [20, 30]
        low_offsets = [0.998, 1.0]
        high_offsets = [1.0, 1.002]
        rsi_values = [65, 72]
        ewo_high_values = [0.05, 0.15]
        ewo_high_2_values = [-0.1, 0.1]
        ewo_low_values = [-0.4, -0.15]
    else:
        buy_windows = [10, 14, 20]
        sell_windows = [20, 24, 30]
        low_offsets = [0.955, 0.975, 0.985]
        high_offsets = [0.991, 0.997, 1.01]
        rsi_values = [55, 65, 69]
        ewo_high_values = [float(settings.get("ewo_high", 2.327))]
        ewo_high_2_values = [float(settings.get("ewo_high_2", -2.327))]
        ewo_low_values = [float(settings.get("ewo_low", -20.988))]
    candidates: list[dict] = []

    for buy_window in buy_windows:
        for sell_window in sell_windows:
            for low_offset in low_offsets:
                for high_offset in high_offsets:
                    for rsi_buy in rsi_values:
                        for ewo_high in ewo_high_values:
                            for ewo_high_2 in ewo_high_2_values:
                                for ewo_low in ewo_low_values:
                                    candidates.append({
                                        **settings,
                                        "strategy": "ewo_offset",
                                        "base_nb_candles_buy": buy_window,
                                        "base_nb_candles_sell": sell_window,
                                        "low_offset": low_offset,
                                        "low_offset_2": min(low_offset, 0.998 if forex else 0.955),
                                        "high_offset": high_offset,
                                        "high_offset_2": max(high_offset, 1.0 if forex else 0.997),
                                        "ewo_high": ewo_high,
                                        "ewo_high_2": ewo_high_2,
                                        "ewo_low": ewo_low,
                                        "rsi_buy": rsi_buy,
                                    })

    return candidates

def optimizer_candidate_settings(settings: dict) -> list[dict]:
    short_values = [5, 8, 10, 12]
    long_values = [20, 30, 40, 60]
    stop_values = [1.5, 2.5, 3.5]
    take_values = [3.0, 5.0, 7.0]
    position_values = [0.15, 0.25, 0.35]
    candidates: list[dict] = []

    for short_window in short_values:
        for long_window in long_values:
            if long_window <= short_window:
                continue
            for stop_loss in stop_values:
                for take_profit in take_values:
                    for position_fraction in position_values:
                        candidates.append({
                            **settings,
                            "short_window": short_window,
                            "long_window": long_window,
                            "stop_loss_pct": stop_loss,
                            "take_profit_pct": take_profit,
                            "max_position_pct": position_fraction,
                        })

    return candidates

def opening_range_candidate_settings(settings: dict) -> list[dict]:
    forex = is_forex_settings(settings)

    if forex:
        manipulation_thresholds = [0.15, 0.25, 0.35]
        stop_multipliers = [0.5, 0.8, 1.2]
        target_multipliers = [1.0, 1.5, 2.0]
    else:
        manipulation_thresholds = [0.15, 0.20, 0.30]
        stop_multipliers = [1.0, 1.5, 2.0]
        target_multipliers = [2.0, 2.5, 3.0]

    atr_periods = [10, 14, 20]
    opening_minutes = [15, 30, 60]

    candidates: list[dict] = []

    for threshold in manipulation_thresholds:
        for stop_mult in stop_multipliers:
            for target_mult in target_multipliers:
                for atr_period in atr_periods:
                    for opening_min in opening_minutes:
                        candidates.append({
                            **settings,
                            "strategy": "opening_range",
                            "opening_range_minutes": opening_min,
                            "opening_range_atr_period": atr_period,
                            "opening_range_manipulation_threshold": threshold,
                            "opening_range_stop_loss_atr_multiplier": stop_mult,
                            "opening_range_take_profit_atr_multiplier": target_mult,
                        })

    return candidates

def calculate_atr_from_candles(candles: list, period: int) -> float:
    if len(candles) < period + 1:
        return 0.0

    true_ranges = []
    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i-1].close
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)

    if not true_ranges:
        return 0.0

    recent_tr = true_ranges[-period:]
    return sum(recent_tr) / len(recent_tr)