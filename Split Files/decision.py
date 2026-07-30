#!/usr/bin/env python3
"""
Trading decision logic: self-learning, opening range, legacy strategies.
"""

import time
from typing import Any, Optional, List, Dict
from datetime import datetime, timezone

from utils import now_iso, pct, parse_watchlist, normalize_forex_symbol
from indicators import (
    sma, ema_series, calculate_macd, calculate_rsi,
    support_resistance, sr_buy_allowed, exit_prices,
    partial_take_profit_ready, trailing_stop_price,
    ewo_offset_signal, market_regime, regime_allowed,
    closes_to_candles, signal_candles
)
from risk_manager import calculate_atr
from order_execution import should_live_trade, should_oanda_demo_trade, wants_oanda_demo_trade
from coinbase_api import live_market_guard
from oanda_api import oanda_demo_status_message

# ─── Self-Learning Decision ──────────────────────────────────────

def decide_self_learning(bot, fetched_prices: dict, watchlist: list, candles_by_symbol: dict) -> str:
    with bot.lock:
        settings = dict(bot.state.settings)
        last_action_time = bot.state.last_action_time
        active_symbol = bot.state.active_symbol
        coin = bot.state.coin
        positions = dict(bot.state.positions)
        day_start_equity = bot.state.day_start_equity
        peak_equity = bot.state.peak_equity

    if not settings.get('self_learning_enabled', True):
        return "HOLD self-learning disabled"

    if time.time() - last_action_time < float(settings["cooldown_seconds"]):
        return "Cooldown active"

    active_price = bot.price_for_active_position(fetched_prices)
    equity = bot.equity(active_price)
    daily_loss_limit = float(settings["daily_loss_limit_pct"]) / 100
    if equity <= day_start_equity * (1 - daily_loss_limit):
        return "Daily loss limit reached"

    max_drawdown_pct = float(settings.get("max_drawdown_pct", 20.0))
    if peak_equity > 0:
        current_drawdown = ((peak_equity - equity) / peak_equity) * 100
        if current_drawdown > max_drawdown_pct:
            return f"STOP Max drawdown {current_drawdown:.1f}% > {max_drawdown_pct}%"

    if settings.get("news_guard_enabled", False):
        for symbol in watchlist:
            blocked, reason = bot.is_news_blocked(symbol, settings)
            if blocked:
                return f"BLOCK {symbol} {reason}"

    trader = bot.self_learning_trader
    best_signal = None
    best_score = -999

    for symbol in watchlist:
        candles = candles_by_symbol.get(symbol)
        if not candles or len(candles) < 50:
            continue

        analysis = trader.analyze_candles_with_indicators(candles, settings)
        has_position = symbol in positions or (active_symbol == symbol and abs(coin) > 0)
        should_trade, direction, score, signal_types = trader.should_enter_trade(analysis, settings)

        if should_trade:
            if direction == 'BUY' and not has_position:
                if score > best_score:
                    best_score = score
                    best_signal = {
                        'symbol': symbol,
                        'direction': 'BUY',
                        'score': score,
                        'signal_types': signal_types,
                        'analysis': analysis,
                    }
            elif direction == 'SELL' and has_position:
                if score > best_score:
                    best_score = score
                    best_signal = {
                        'symbol': symbol,
                        'direction': 'SELL',
                        'score': score,
                        'signal_types': signal_types,
                        'analysis': analysis,
                    }

    if best_signal:
        if best_signal['direction'] == 'BUY':
            return f"BUY {best_signal['symbol']} self-learning score {best_signal['score']:.3f} | signals: {', '.join(best_signal['signal_types'][:3])}"
        else:
            return f"SELL {best_signal['symbol']} self-learning score {best_signal['score']:.3f} | signals: {', '.join(best_signal['signal_types'][:3])}"

    return "HOLD no self-learning signals"

# ─── Opening Range Decision ─────────────────────────────────────

def decide_opening_range(bot, fetched_prices: dict, watchlist: list, candles_by_symbol: dict) -> str:
    with bot.lock:
        settings = dict(bot.state.settings)
        last_action_time = bot.state.last_action_time
        active_symbol = bot.state.active_symbol
        coin = bot.state.coin
        positions = dict(bot.state.positions)
        day_start_equity = bot.state.day_start_equity
        peak_equity = bot.state.peak_equity

    if time.time() - last_action_time < float(settings["cooldown_seconds"]):
        return "Cooldown active"

    if settings.get("news_guard_enabled", False):
        for symbol in watchlist:
            blocked, reason = bot.is_news_blocked(symbol, settings)
            if blocked:
                return f"BLOCK {symbol} {reason}"

    active_price = bot.price_for_active_position(fetched_prices)
    equity = bot.equity(active_price)

    daily_loss_limit = float(settings["daily_loss_limit_pct"]) / 100
    if equity <= day_start_equity * (1 - daily_loss_limit):
        return "Daily loss limit reached"

    max_drawdown_pct = float(settings.get("max_drawdown_pct", 20.0))
    if peak_equity > 0:
        current_drawdown = ((peak_equity - equity) / peak_equity) * 100
        if current_drawdown > max_drawdown_pct:
            return f"STOP Max drawdown {current_drawdown:.1f}% > {max_drawdown_pct}%"

    if equity > peak_equity:
        with bot.lock:
            bot.state.peak_equity = equity

    symbol = watchlist[0]
    candles = candles_by_symbol.get(symbol, [])
    if len(candles) < max(20, int(settings.get("opening_range_atr_period", 14)) + 1):
        return "WAIT data loading for Opening Range"

    signal = opening_range_signal(bot, symbol, candles)
    analysis = signal.get("analysis", {})

    if signal["signal"] == "BUY":
        is_short = signal.get("is_short", False)
        if is_short:
            return f"BUY {symbol} SHORT {signal['reason']}"
        else:
            return f"BUY {symbol} {signal['reason']}"

    elif signal["signal"] == "SELL":
        has_position = (
            (symbol in positions and abs(positions.get(symbol, {}).get("quantity", 0)) > 0) or
            (active_symbol == symbol and abs(coin) > 0)
        )
        if has_position:
            return f"SELL {symbol} {signal['reason']}"
        else:
            if not settings.get("allow_short_selling", False):
                return f"HOLD Short selling disabled: {signal['reason']}"
            return f"BUY {symbol} SHORT {signal['reason']}"

    else:
        return f"HOLD {signal['reason']}"

def opening_range_signal(bot, symbol: str, candles: list) -> dict:
    analysis = fetch_daily_opening_candle(bot, symbol, candles)
    bot.state.opening_range_analysis = analysis

    if analysis.get("bias") is None:
        return {"signal": "HOLD", "reason": "No opening candle found", "analysis": analysis}

    current_price = candles[-1].close if candles else 0
    trigger = analysis["trigger_level"]
    atr = analysis["atr"]

    stop_loss_mult = float(bot.state.settings.get("opening_range_stop_loss_atr_multiplier", 1.5))
    take_profit_mult = float(bot.state.settings.get("opening_range_take_profit_atr_multiplier", 2.5))

    has_position = (
        (symbol in bot.state.positions and abs(bot.state.positions[symbol].get("quantity", 0)) > 0) or
        (bot.state.active_symbol == symbol and abs(bot.state.coin) > 0)
    )

    if has_position:
        position = bot.state.positions.get(symbol, {})
        entry_price = float(position.get("entry_price") or bot.state.entry_price or 0.0)
        is_short = position.get("is_short", False) or bot.state.is_short
        quantity = float(position.get("quantity", 0)) or bot.state.coin

        if entry_price > 0:
            if not is_short:
                stop_price = entry_price - (atr * stop_loss_mult)
                target_price = entry_price + (atr * take_profit_mult)

                bot.state.highest_price = max(bot.state.highest_price or current_price, current_price)

                trailing_stop = trailing_stop_price(
                    entry_price=bot.state.entry_price,
                    highest_price=bot.state.highest_price,
                    settings=bot.state.settings,
                )
                if trailing_stop and current_price <= trailing_stop:
                    return {
                        "signal": "SELL",
                        "reason": f"Trailing stop hit at {trailing_stop:.6f}",
                        "entry": entry_price,
                        "stop": stop_price,
                        "target": target_price,
                        "analysis": analysis,
                    }

                if current_price <= stop_price:
                    return {
                        "signal": "SELL",
                        "reason": f"Stop loss hit at {stop_price:.6f}",
                        "entry": entry_price,
                        "stop": stop_price,
                        "target": target_price,
                        "analysis": analysis,
                    }
                if current_price >= target_price:
                    return {
                        "signal": "SELL",
                        "reason": f"Take profit hit at {target_price:.6f}",
                        "entry": entry_price,
                        "stop": stop_price,
                        "target": target_price,
                        "analysis": analysis,
                    }

                return {
                    "signal": "HOLD",
                    "reason": f"Holding LONG position {symbol} @ {current_price:.6f}",
                    "entry": entry_price,
                    "stop": stop_price,
                    "target": target_price,
                    "analysis": analysis,
                }
            else:
                stop_price = entry_price + (atr * stop_loss_mult)
                target_price = entry_price - (atr * take_profit_mult)

                if current_price >= stop_price:
                    return {
                        "signal": "BUY",
                        "reason": f"Short stop loss hit at {stop_price:.6f}",
                        "entry": entry_price,
                        "stop": stop_price,
                        "target": target_price,
                        "analysis": analysis,
                        "is_short_exit": True,
                    }
                if current_price <= target_price:
                    return {
                        "signal": "BUY",
                        "reason": f"Short take profit hit at {target_price:.6f}",
                        "entry": entry_price,
                        "stop": stop_price,
                        "target": target_price,
                        "analysis": analysis,
                        "is_short_exit": True,
                    }

                return {
                    "signal": "HOLD",
                    "reason": f"Holding SHORT position {symbol} @ {current_price:.6f}",
                    "entry": entry_price,
                    "stop": stop_price,
                    "target": target_price,
                    "analysis": analysis,
                }

    if analysis["manipulation"] and analysis["bias"] == "bullish":
        if current_price > trigger:
            entry_price = trigger
            return {
                "signal": "BUY",
                "reason": f"Bullish manipulation: {analysis['range_ratio']:.2%} of ATR",
                "entry": entry_price,
                "stop": entry_price - (atr * stop_loss_mult),
                "target": entry_price + (atr * take_profit_mult),
                "analysis": analysis,
                "is_short": False,
            }
        else:
            return {
                "signal": "WAIT",
                "reason": f"Waiting for break above {trigger:.6f} (bullish)",
                "analysis": analysis,
            }

    if analysis["manipulation"] and analysis["bias"] == "bearish":
        if not bot.state.settings.get("allow_short_selling", False):
            return {
                "signal": "HOLD",
                "reason": "Short selling disabled",
                "analysis": analysis,
            }
        if current_price < trigger:
            entry_price = trigger
            return {
                "signal": "SELL",
                "reason": f"Bearish manipulation: {analysis['range_ratio']:.2%} of ATR",
                "entry": entry_price,
                "stop": entry_price + (atr * stop_loss_mult),
                "target": entry_price - (atr * take_profit_mult),
                "analysis": analysis,
                "is_short": True,
            }
        else:
            return {
                "signal": "WAIT",
                "reason": f"Waiting for break below {trigger:.6f} (bearish)",
                "analysis": analysis,
            }

    if analysis["blowoff"]:
        return {
            "signal": "WAIT",
            "reason": f"Blow-off candle: {analysis['range_ratio']:.2%} of ATR, waiting for pullback",
            "analysis": analysis,
        }

    return {
        "signal": "HOLD",
        "reason": "No setup detected",
        "analysis": analysis,
    }

def fetch_daily_opening_candle(bot, symbol: str, candles: list) -> dict:
    """Fetch the opening range candle and compute analysis."""
    if len(candles) < 2:
        return {"bias": None, "range": None, "atr": None, "manipulation": False, "blowoff": False}

    today = datetime.now(timezone.utc).date()

    first_candle = None
    for candle in candles:
        candle_date = datetime.fromtimestamp(candle.time, tz=timezone.utc).date()
        if candle_date == today:
            first_candle = candle
            break

    if not first_candle:
        if candles:
            first_candle = candles[-1]
        else:
            return {"bias": None, "range": None, "atr": None, "manipulation": False, "blowoff": False}

    is_green = first_candle.close > first_candle.open
    candle_range = first_candle.high - first_candle.low

    atr = bot.calculate_atr(candles, int(bot.state.settings.get("opening_range_atr_period", 14)))
    if atr == 0:
        atr = candle_range

    manipulation_threshold = float(bot.state.settings.get("opening_range_manipulation_threshold", 0.20))
    range_ratio = candle_range / atr if atr > 0 else 0
    manipulation = range_ratio < manipulation_threshold
    blowoff = range_ratio >= manipulation_threshold

    return {
        "bias": "bullish" if is_green else "bearish",
        "open": first_candle.open,
        "high": first_candle.high,
        "low": first_candle.low,
        "close": first_candle.close,
        "range": candle_range,
        "atr": atr,
        "range_ratio": round(range_ratio, 4),
        "manipulation": manipulation,
        "blowoff": blowoff,
        "is_green": is_green,
        "trigger_level": first_candle.high if is_green else first_candle.low,
        "stop_level": first_candle.low if is_green else first_candle.high,
        "opening_time": datetime.fromtimestamp(first_candle.time, tz=timezone.utc).isoformat(),
    }

# ─── Legacy Decision ─────────────────────────────────────────────

def decide_legacy(bot, fetched_prices: dict, watchlist: list, candles_by_symbol: dict) -> str:
    with bot.lock:
        settings = dict(bot.state.settings)
        last_action_time = bot.state.last_action_time
        active_symbol = bot.state.active_symbol
        coin = bot.state.coin
        entry_price = bot.state.entry_price
        highest_price = bot.state.highest_price
        partial_take_profit_done = bot.state.partial_take_profit_done
        positions = dict(bot.state.positions)
        day_start_equity = bot.state.day_start_equity
        peak_equity = bot.state.peak_equity
        price_history = dict(bot.state.price_history)

    strategy = settings.get("strategy", "sma_cross")

    if time.time() - last_action_time < float(settings["cooldown_seconds"]):
        return "Cooldown active"

    if settings.get("news_guard_enabled", False):
        for symbol in watchlist:
            blocked, reason = bot.is_news_blocked(symbol, settings)
            if blocked:
                return f"BLOCK {symbol} {reason}"

    active_price = bot.price_for_active_position(fetched_prices)
    equity = bot.equity(active_price)

    daily_loss_limit = float(settings["daily_loss_limit_pct"]) / 100
    if equity <= day_start_equity * (1 - daily_loss_limit):
        return "Daily loss limit reached"

    max_drawdown_pct = float(settings.get("max_drawdown_pct", 20.0))
    if peak_equity > 0:
        current_drawdown = ((peak_equity - equity) / peak_equity) * 100
        if current_drawdown > max_drawdown_pct:
            return f"STOP Max drawdown {current_drawdown:.1f}% > {max_drawdown_pct}%"

    if equity > peak_equity:
        with bot.lock:
            bot.state.peak_equity = equity

    if strategy == "ema_golden_cross":
        symbol = watchlist[0]
        candles = candles_by_symbol.get(symbol, [])
        if len(candles) < int(settings.get("ema_long", 200)) + 1:
            return "WAIT data loading for EMA Golden Cross"

        history = [candle.close for candle in candles]
        ema_short = int(settings.get("ema_short", 50))
        ema_long = int(settings.get("ema_long", 200))

        ema_short_value = ema_series(history, ema_short)[-1]
        ema_long_value = ema_series(history, ema_long)[-1]
        ema_short_prev = ema_series(history[:-1], ema_short)[-1] if len(history) > 1 else None
        ema_long_prev = ema_series(history[:-1], ema_long)[-1] if len(history) > 1 else None

        if None in (ema_short_value, ema_long_value, ema_short_prev, ema_long_prev):
            return "WAIT not enough data"

        if ema_short_prev <= ema_long_prev and ema_short_value > ema_long_value:
            return f"BUY {symbol} EMA Golden Cross (50/200)"

        elif ema_short_prev >= ema_long_prev and ema_short_value < ema_long_value:
            if settings.get("allow_short_selling", False):
                return f"BUY {symbol} SHORT EMA Death Cross"
            else:
                return f"HOLD EMA Death Cross (shorting disabled)"

        return "HOLD no signal"

    if abs(coin) > 0 and entry_price and active_symbol:
        symbol = active_symbol
        price = fetched_prices.get(symbol, active_price)
        history = price_history.get(symbol, [])
        candles = candles_by_symbol.get(symbol) or closes_to_candles(history)
        signal_candle_set = signal_candles(closes_to_candles(history), settings)
        signal_history = [candle.close for candle in signal_candle_set] or history
        current_highest = max(highest_price or price, price)
        stop_price, target_price, exit_mode = exit_prices(
            entry_price=entry_price,
            candles=candles,
            settings=settings,
        )

        # ─── ENHANCED EXIT CHECK ──────────────────────────────────
        position = positions.get(symbol, {})
        position_side = "SHORT" if position.get('is_short', False) else "LONG"
        entry_time = position.get('entry_time', time.time())

        should_exit, exit_reason = bot.should_exit_enhanced(
            symbol=symbol,
            candles=candles,
            entry_time=entry_time,
            position_side=position_side
        )

        if should_exit:
            return f"SELL {symbol} {exit_reason}"

        if partial_take_profit_ready(
            price=price,
            entry_price=entry_price,
            target_price=target_price,
            settings=settings,
            already_done=partial_take_profit_done,
        ):
            with bot.lock:
                bot.state.partial_take_profit_done = True
            return f"SELL {symbol} partial {exit_mode} target"

        trailing_stop = trailing_stop_price(
            entry_price=entry_price,
            highest_price=current_highest,
            settings=settings,
        )
        if trailing_stop and price <= trailing_stop:
            return f"SELL {symbol} trailing stop"

        if price <= stop_price:
            return f"SELL {symbol} {exit_mode} stop"
        if price >= target_price:
            return f"SELL {symbol} {exit_mode} target"

        if settings.get("strategy") == "ewo_offset":
            signal = ewo_offset_signal(signal_candle_set, settings)
            if signal["sell"]:
                return f"SELL {symbol} EWO offset sell"
        else:
            short_window = int(settings["short_window"])
            long_window = int(settings["long_window"])
            short_now = sma(signal_history, short_window)
            long_now = sma(signal_history, long_window)
            short_prev = sma(signal_history[:-1], short_window)
            long_prev = sma(signal_history[:-1], long_window)
            if None not in (short_now, long_now, short_prev, long_prev):
                if short_prev >= long_prev and short_now < long_now:
                    return f"SELL {symbol} trend turned down"

        return f"HOLD {symbol} position open"

    if strategy == "ewo_offset":
        scan_rows = bot.build_ewo_scan_rows(watchlist, candles_by_symbol)
    else:
        scan_rows = bot.build_scan_rows(watchlist, candles_by_symbol)
    bot.state.scan_rows = scan_rows

    candidates = [
        row for row in scan_rows
        if row["signal"] == "BUY" and row["price"] is not None
    ]
    if candidates:
        best = max(candidates, key=lambda row: row["score"])
        return f"BUY {best['symbol']} strongest trend score {best['score']:.3f}"

    waiting = [row for row in scan_rows if row["signal"].startswith("WAIT")]
    if len(waiting) == len(scan_rows):
        return "Waiting for enough price data"

    return "HOLD no qualifying entry"
