#!/usr/bin/env python3
"""
Technical indicators for trading strategies.
"""

import math
from typing import Optional, List, Any, Tuple

# ─── Re-export utils functions (to keep compatibility with order_execution) ──
from utils import closes_to_candles, signal_candles

# ─── Import from utils ──────────────────────────────────────────
from utils import pct, normalize_forex_symbol, FOREX_BASE_RATES, fetch_json, normalize_granularity, strategy_minimum_candles

# ─── Basic indicators ───────────────────────────────────────────

def sma(values: list[float], window: int) -> float | None:
    if len(values) < window:
        return None
    return sum(values[-window:]) / window

def ema_series(values: list[float], window: int) -> list[float | None]:
    if len(values) < window:
        return [None for _ in values]
    multiplier = 2 / (window + 1)
    result: list[float | None] = [None] * (window - 1)
    ema_value = sum(values[:window]) / window
    result.append(ema_value)
    for price in values[window:]:
        ema_value = (price - ema_value) * multiplier + ema_value
        result.append(ema_value)
    return result

def hma_series(values: list[float], window: int) -> list[float | None]:
    half_window = max(1, window // 2)
    sqrt_window = max(1, int(math.sqrt(window)))
    wma_half = wma_series(values, half_window)
    wma_full = wma_series(values, window)
    diff: list[float] = []
    diff_positions: list[int] = []

    for index, (half_value, full_value) in enumerate(zip(wma_half, wma_full)):
        if half_value is None or full_value is None:
            continue
        diff.append((2 * half_value) - full_value)
        diff_positions.append(index)

    hma_partial = wma_series(diff, sqrt_window)
    result: list[float | None] = [None for _ in values]
    for source_index, hma_value in zip(diff_positions, hma_partial):
        result[source_index] = hma_value
    return result

def wma_series(values: list[float], window: int) -> list[float | None]:
    if window <= 0:
        return [None for _ in values]
    weights = list(range(1, window + 1))
    divisor = sum(weights)
    result: list[float | None] = []

    for index in range(len(values)):
        if index + 1 < window:
            result.append(None)
            continue
        sample = values[index + 1 - window:index + 1]
        result.append(sum(value * weight for value, weight in zip(sample, weights)) / divisor)

    return result

def rsi_series(values: list[float], window: int) -> list[float | None]:
    result: list[float | None] = [None for _ in values]
    if len(values) <= window:
        return result

    gains: list[float] = []
    losses: list[float] = []
    for index in range(1, window + 1):
        change = values[index] - values[index - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = sum(gains) / window
    avg_loss = sum(losses) / window
    result[window] = 100.0 if avg_loss == 0 else 100 - (100 / (1 + (avg_gain / avg_loss)))

    for index in range(window + 1, len(values)):
        change = values[index] - values[index - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        avg_gain = ((avg_gain * (window - 1)) + gain) / window
        avg_loss = ((avg_loss * (window - 1)) + loss) / window
        result[index] = 100.0 if avg_loss == 0 else 100 - (100 / (1 + (avg_gain / avg_loss)))

    return result

def calculate_macd(prices: list[float], fast: int = 12, slow: int = 26) -> list[float]:
    """Calculate MACD line values."""
    if len(prices) < slow:
        return []

    fast_ema = []
    for i in range(len(prices)):
        if i < fast - 1:
            fast_ema.append(None)
        elif i == fast - 1:
            fast_ema.append(sum(prices[:fast]) / fast)
        else:
            multiplier = 2 / (fast + 1)
            fast_ema.append((prices[i] - fast_ema[-1]) * multiplier + fast_ema[-1])

    slow_ema = []
    for i in range(len(prices)):
        if i < slow - 1:
            slow_ema.append(None)
        elif i == slow - 1:
            slow_ema.append(sum(prices[:slow]) / slow)
        else:
            multiplier = 2 / (slow + 1)
            slow_ema.append((prices[i] - slow_ema[-1]) * multiplier + slow_ema[-1])

    macd_values = []
    for i in range(len(fast_ema)):
        if fast_ema[i] is not None and slow_ema[i] is not None:
            macd_values.append(fast_ema[i] - slow_ema[i])
        else:
            macd_values.append(None)

    return [v for v in macd_values if v is not None]

def calculate_rsi(prices: list[float], period: int = 14) -> float | None:
    """Calculate RSI value."""
    if len(prices) < period + 1:
        return None

    gains = 0
    losses = 0
    for i in range(len(prices) - period, len(prices)):
        change = prices[i] - prices[i-1]
        if change >= 0:
            gains += change
        else:
            losses -= change

    avg_gain = gains / period
    avg_loss = losses / period

    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def find_support_resistance(candles: list, lookback: int = 20) -> tuple[float | None, float | None]:
    """Find support and resistance levels."""
    if len(candles) < lookback:
        return None, None

    recent = candles[-lookback:]
    support = min(c.low for c in recent)
    resistance = max(c.high for c in recent)

    support_touches = [c.low for c in recent if abs(c.low - support) / support < 0.005]
    resistance_touches = [c.high for c in recent if abs(c.high - resistance) / resistance < 0.005]

    if len(support_touches) >= 2:
        support = sum(support_touches) / len(support_touches)
    if len(resistance_touches) >= 2:
        resistance = sum(resistance_touches) / len(resistance_touches)

    return support, resistance

def detect_engulfing_patterns(candles: list) -> list[dict[str, Any]]:
    """Detect bullish and bearish engulfing patterns."""
    patterns = []
    for i in range(1, len(candles)):
        prev = candles[i-1]
        curr = candles[i]

        if prev.close < prev.open and curr.close > curr.open:
            if curr.open <= prev.close and curr.close >= prev.open:
                patterns.append({
                    'bullish': True,
                    'price': curr.low,
                    'index': i,
                })
        elif prev.close > prev.open and curr.close < curr.open:
            if curr.open >= prev.close and curr.close <= prev.open:
                patterns.append({
                    'bullish': False,
                    'price': curr.high,
                    'index': i,
                })
    return patterns

# ─── Support/Resistance and Exit logic ─────────────────────────

def support_resistance(candles: list, settings: dict[str, Any]) -> dict[str, Any]:
    if not candles:
        return {
            "support": None,
            "resistance": None,
            "support_distance_pct": None,
            "resistance_distance_pct": None,
            "sr_range_pct": None,
            "reward_risk": None,
            "support_touches": 0,
            "resistance_touches": 0,
            "confirmed": False,
        }

    lookback = max(1, int(settings.get("sr_lookback_candles", 50)))
    sample = candles[-lookback:]
    tolerance = float(settings.get("sr_zone_tolerance_pct", 0.6)) / 100
    min_touches = int(settings.get("sr_min_touches", 2))
    raw_support = min(candle.low for candle in sample)
    raw_resistance = max(candle.high for candle in sample)
    support_zone_limit = raw_support * (1 + tolerance)
    resistance_zone_limit = raw_resistance * (1 - tolerance)
    support_lows = [candle.low for candle in sample if candle.low <= support_zone_limit]
    resistance_highs = [candle.high for candle in sample if candle.high >= resistance_zone_limit]
    support_touches = len(support_lows)
    resistance_touches = len(resistance_highs)
    support = sum(support_lows) / support_touches if support_touches else raw_support
    resistance = sum(resistance_highs) / resistance_touches if resistance_touches else raw_resistance
    confirmed = support_touches >= min_touches and resistance_touches >= min_touches
    price = sample[-1].close
    stop_buffer = float(settings.get("support_stop_buffer_pct", 2.0)) / 100
    stop_price = support * (1 - stop_buffer)
    risk = max(price - stop_price, 0.0)
    reward = max(resistance - price, 0.0)
    reward_risk = reward / risk if risk > 0 else None

    return {
        "support": support,
        "resistance": resistance,
        "support_distance_pct": pct(price - support, support),
        "resistance_distance_pct": pct(resistance - price, price),
        "sr_range_pct": pct(resistance - support, support),
        "reward_risk": round(reward_risk, 4) if reward_risk is not None else None,
        "support_touches": support_touches,
        "resistance_touches": resistance_touches,
        "confirmed": confirmed,
    }

def sr_buy_allowed(price: float | None, levels: dict, settings: dict) -> tuple[bool, str]:
    if not settings.get("use_sr_filter"):
        return True, ""
    if not price or not levels.get("support") or not levels.get("resistance"):
        return False, "no S/R"
    if not levels.get("confirmed"):
        support_touches = int(levels.get("support_touches") or 0)
        resistance_touches = int(levels.get("resistance_touches") or 0)
        min_touches = int(settings.get("sr_min_touches", 2))
        return (
            False,
            f"S/R needs touches S {support_touches}/{min_touches}, R {resistance_touches}/{min_touches}",
        )

    support_distance = float(levels.get("support_distance_pct") or 0.0)
    resistance_distance = float(levels.get("resistance_distance_pct") or 0.0)
    sr_range = float(levels.get("sr_range_pct") or 0.0)
    reward_risk = float(levels.get("reward_risk") or 0.0)
    near_support = float(settings.get("near_support_pct", 2.0))
    min_resistance_distance = float(settings.get("min_resistance_distance_pct", 1.0))
    min_sr_range = float(settings.get("min_sr_range_pct", 8.0))
    min_reward_risk = float(settings.get("min_reward_risk", 2.0))

    if support_distance > near_support:
        return False, "above support"
    if resistance_distance < min_resistance_distance:
        return False, "near resistance"
    if sr_range < min_sr_range:
        return False, "S/R range too small"
    if reward_risk < min_reward_risk:
        return False, "reward/risk too low"
    return True, ""

def exit_prices(entry_price: float, candles: list, settings: dict) -> tuple[float, float, str]:
    default_stop = entry_price * (1 - float(settings["stop_loss_pct"]) / 100)
    default_target = entry_price * (1 + float(settings["take_profit_pct"]) / 100)

    if not settings.get("use_dynamic_sr_exits"):
        return default_stop, default_target, "fixed"

    levels = support_resistance(candles, settings)
    support = levels.get("support")
    resistance = levels.get("resistance")
    if not support or not resistance or not levels.get("confirmed"):
        return default_stop, default_target, "fixed"

    stop_buffer = float(settings.get("support_stop_buffer_pct", 2.0)) / 100
    target_buffer = float(settings.get("resistance_target_buffer_pct", 0.5)) / 100
    sr_stop = float(support) * (1 - stop_buffer)
    sr_target = float(resistance) * (1 - target_buffer)

    if sr_stop >= entry_price or sr_target <= entry_price:
        return default_stop, default_target, "fixed"

    return sr_stop, sr_target, "S/R"

def position_spend(cash: float, entry_price: float, candles: list, settings: dict) -> tuple[float, str]:
    max_fraction_spend = cash * float(settings["max_position_pct"])
    if settings.get("position_sizing_mode") != "risk_based":
        return max_fraction_spend, "balance fraction"

    stop_price, _, exit_mode = exit_prices(entry_price, candles, settings)
    risk_per_unit = entry_price - stop_price
    if risk_per_unit <= 0:
        return 0.0, "risk sizing blocked: invalid stop"

    risk_cash = cash * (float(settings.get("risk_per_trade_pct", 1.0)) / 100)
    quantity = risk_cash / risk_per_unit
    spend = quantity * entry_price
    capped_spend = min(spend, max_fraction_spend, cash)
    return capped_spend, f"risk {settings.get('risk_per_trade_pct', 1.0)}% via {exit_mode} stop"

def partial_take_profit_ready(
    price: float,
    entry_price: float,
    target_price: float,
    settings: dict,
    already_done: bool,
) -> bool:
    if already_done or not settings.get("partial_take_profit_enabled"):
        return False
    trigger_fraction = float(settings.get("partial_take_profit_at_target_pct", 50.0)) / 100
    trigger_price = entry_price + ((target_price - entry_price) * trigger_fraction)
    return target_price > entry_price and price >= trigger_price

def trailing_stop_price(entry_price: float, highest_price: float | None, settings: dict) -> float | None:
    if not settings.get("trailing_stop_enabled") or not highest_price:
        return None
    activation = float(settings.get("trailing_activation_pct", 3.0)) / 100
    if highest_price < entry_price * (1 + activation):
        return None
    trail = float(settings.get("trailing_stop_pct", 2.0)) / 100
    return highest_price * (1 - trail)

def chart_trade_plan(state, chart_symbol: str, chart_row: dict) -> dict:
    levels: dict[str, Any] = {
        "entry": None,
        "stop": None,
        "target": None,
        "partial": None,
        "trailing": None,
        "exit_mode": None,
    }
    if not state.active_symbol or state.active_symbol != chart_symbol or not state.entry_price:
        return levels

    settings = state.settings
    entry = float(state.entry_price)
    stop = entry * (1 - float(settings["stop_loss_pct"]) / 100)
    target = entry * (1 + float(settings["take_profit_pct"]) / 100)
    exit_mode = "fixed"

    support = chart_row.get("support")
    resistance = chart_row.get("resistance")
    if settings.get("use_dynamic_sr_exits") and support and resistance and chart_row.get("sr_confirmed"):
        stop_buffer = float(settings.get("support_stop_buffer_pct", 2.0)) / 100
        target_buffer = float(settings.get("resistance_target_buffer_pct", 0.5)) / 100
        sr_stop = float(support) * (1 - stop_buffer)
        sr_target = float(resistance) * (1 - target_buffer)
        if sr_stop < entry and sr_target > entry:
            stop = sr_stop
            target = sr_target
            exit_mode = "S/R"

    partial = None
    if settings.get("partial_take_profit_enabled") and target > entry:
        trigger_fraction = float(settings.get("partial_take_profit_at_target_pct", 50.0)) / 100
        partial = entry + ((target - entry) * trigger_fraction)

    trailing = trailing_stop_price(entry, state.highest_price, settings)

    return {
        "entry": entry,
        "stop": stop,
        "target": target,
        "partial": partial,
        "trailing": trailing,
        "exit_mode": exit_mode,
    }

# ─── EWO Offset Signal ──────────────────────────────────────────

def ewo_offset_signal(candles: list, settings: dict[str, Any]) -> dict[str, Any]:
    closes = [candle.close for candle in candles]
    lows = [candle.low for candle in candles]
    volumes = [candle.volume for candle in candles]
    minimum = strategy_minimum_candles(settings)
    empty = {
        "ready": False,
        "buy": False,
        "sell": False,
        "tag": "",
        "score": 0.0,
        "ma_buy": None,
        "ma_sell": None,
    }
    if len(candles) < minimum:
        return empty

    buy_window = int(settings["base_nb_candles_buy"])
    sell_window = int(settings["base_nb_candles_sell"])
    ma_buy = ema_series(closes, buy_window)[-1]
    ma_sell = ema_series(closes, sell_window)[-1]
    ema_50 = ema_series(closes, 50)[-1]
    ema_100 = ema_series(closes, 100)[-1]
    ema_200 = ema_series(closes, 200)[-1]
    hma_50 = hma_series(closes, 50)[-1]
    sma_9 = sma(closes, 9)
    rsi = rsi_series(closes, 14)[-1]
    rsi_fast = rsi_series(closes, 4)[-1]
    rsi_slow = rsi_series(closes, 20)[-1]

    required = [ma_buy, ma_sell, ema_50, ema_100, ema_200, hma_50, sma_9, rsi, rsi_fast, rsi_slow]
    if any(value is None for value in required) or lows[-1] == 0:
        return empty

    close = closes[-1]
    volume = volumes[-1]
    ewo = ((ema_50 - ema_200) / lows[-1]) * 100
    buy_tag = ""

    buy_1 = (
        rsi_fast < 35
        and close < ma_buy * float(settings["low_offset"])
        and ewo > float(settings["ewo_high"])
        and rsi < float(settings["rsi_buy"])
        and volume > 0
        and close < ma_sell * float(settings["high_offset"])
    )
    if buy_1:
        buy_tag = "ewo1"

    buy_2 = (
        rsi_fast < 35
        and close < ma_buy * float(settings["low_offset_2"])
        and ewo > float(settings["ewo_high_2"])
        and rsi < float(settings["rsi_buy"])
        and volume > 0
        and close < ma_sell * float(settings["high_offset"])
        and rsi < 25
    )
    if buy_2:
        buy_tag = "ewo2"

    buy_3 = (
        rsi_fast < 35
        and close < ma_buy * float(settings["low_offset"])
        and ewo < float(settings["ewo_low"])
        and volume > 0
        and close < ma_sell * float(settings["high_offset"])
    )
    if buy_3:
        buy_tag = "ewolow"

    sell_primary = (
        close > sma_9
        and close > ma_sell * float(settings["high_offset_2"])
        and rsi > 50
        and volume > 0
        and rsi_fast > rsi_slow
    )
    sell_secondary = (
        close < hma_50
        and close > ma_sell * float(settings["high_offset"])
        and volume > 0
        and rsi_fast > rsi_slow
    )
    sell_guard = (hma_50 * 1.149 <= ema_100) or (close >= ema_100 * 0.951)
    sell = (sell_primary or sell_secondary) and sell_guard

    return {
        "ready": True,
        "buy": buy_1 or buy_2 or buy_3,
        "sell": sell,
        "tag": buy_tag,
        "score": ewo,
        "ma_buy": ma_buy,
        "ma_sell": ma_sell,
    }

# ─── Market Regime (simple version) ─────────────────────────────

def market_regime(candles: list, settings: dict[str, Any]) -> dict[str, Any]:
    if len(candles) < 30:
        return {
            "regime": "unknown",
            "trend_pct": 0.0,
            "volatility_pct": 0.0,
            "range_pct": 0.0,
            "reason": "not enough candles",
        }

    sample = candles[-50:]
    closes = [candle.close for candle in sample]
    highs = [candle.high for candle in sample]
    lows = [candle.low for candle in sample]
    latest = closes[-1]
    ema_fast = ema_series(closes, min(20, len(closes)))[-1]
    ema_slow = ema_series(closes, min(50, len(closes)))[-1]
    trend_pct = pct((ema_fast or latest) - (ema_slow or latest), latest)
    returns = [
        abs((closes[index] - closes[index - 1]) / closes[index - 1]) * 100
        for index in range(1, len(closes))
        if closes[index - 1]
    ]
    volatility_pct = round(sum(returns) / len(returns), 4) if returns else 0.0
    range_pct = pct(max(highs) - min(lows), latest)

    if range_pct < 1.0 and volatility_pct < 0.12:
        regime = "dead"
        reason = "low range and low movement"
    elif volatility_pct > 1.2 or range_pct > 14.0:
        regime = "volatile"
        reason = "wide range or large candle movement"
    elif abs(trend_pct) > 0.8:
        regime = "trending_up" if trend_pct > 0 else "trending_down"
        reason = "fast EMA separated from slow EMA"
    else:
        regime = "ranging"
        reason = "trend and volatility are balanced"

    return {
        "regime": regime,
        "trend_pct": trend_pct,
        "volatility_pct": volatility_pct,
        "range_pct": range_pct,
        "reason": reason,
    }

def regime_allowed(regime: str, settings: dict) -> tuple[bool, str]:
    if not settings.get("regime_filter_enabled"):
        return True, ""
    if regime in {"trending_up", "trending_down"}:
        return bool(settings.get("allow_trending_regime")), "regime trend blocked"
    if regime == "ranging":
        return bool(settings.get("allow_ranging_regime")), "regime range blocked"
    if regime == "volatile":
        return bool(settings.get("allow_volatile_regime")), "regime volatility blocked"
    if regime == "dead":
        return bool(settings.get("allow_dead_regime")), "regime dead blocked"
    return True, ""
