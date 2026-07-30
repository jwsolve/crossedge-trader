#!/usr/bin/env python3
"""
Market Regime Detection – advanced market state classification.

Detects:
- Trend strength (ADX, directional movement)
- Volatility regime (ATR, Bollinger Bands)
- Momentum regime (MACD, RSI)
- Range regime (price action)
- Volume regime (relative volume)
- Sentiment regime (risk-on/risk-off)

Outputs:
- Primary regime (trending, ranging, volatile, dead, breakout)
- Sub-regime (bullish_trend, bearish_trend, squeeze, etc.)
- Confidence score (0-1)
- Regime parameters for strategy adaptation
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

# ─── Helper functions (imported from bot_server) ──────────────────
# We'll keep them here to avoid circular imports.


def sma(values, window):
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def ema(values, window):
    if len(values) < window:
        return None
    multiplier = 2 / (window + 1)
    ema_val = sum(values[:window]) / window
    for price in values[window:]:
        ema_val = (price - ema_val) * multiplier + ema_val
    return ema_val


def atr(candles, period=14):
    """Calculate Average True Range from candle list."""
    if len(candles) < period + 1:
        return 0.0
    true_ranges = []
    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i - 1].close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    if not true_ranges:
        return 0.0
    return sum(true_ranges[-period:]) / period


def adx(candles, period=14):
    """
    Compute Average Directional Index (ADX) – measures trend strength.
    Returns ADX value (0-100).
    """
    if len(candles) < period * 2:
        return 0.0

    # Compute True Range and Directional Movement
    tr_values = []
    plus_dm = []
    minus_dm = []
    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_high = candles[i - 1].high
        prev_low = candles[i - 1].low
        prev_close = candles[i - 1].close

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_values.append(tr)

        up_move = high - prev_high
        down_move = prev_low - low

        if up_move > down_move and up_move > 0:
            plus_dm.append(up_move)
        else:
            plus_dm.append(0.0)

        if down_move > up_move and down_move > 0:
            minus_dm.append(down_move)
        else:
            minus_dm.append(0.0)

    # Smooth using Wilder's smoothing (similar to RMA)
    def wilder_smooth(values, period):
        if len(values) < period:
            return []
        smoothed = [sum(values[:period]) / period]
        for val in values[period:]:
            smoothed.append((smoothed[-1] * (period - 1) + val) / period)
        return smoothed

    tr_smooth = wilder_smooth(tr_values, period)
    plus_dm_smooth = wilder_smooth(plus_dm, period)
    minus_dm_smooth = wilder_smooth(minus_dm, period)

    if len(tr_smooth) < period:
        return 0.0

    # Compute ADX
    adx_values = []
    for i in range(len(tr_smooth)):
        tr = tr_smooth[i]
        if tr == 0:
            continue
        plus_di = (plus_dm_smooth[i] / tr) * 100 if i < len(plus_dm_smooth) else 0
        minus_di = (minus_dm_smooth[i] / tr) * 100 if i < len(minus_dm_smooth) else 0
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0
        adx_values.append(dx)

    if len(adx_values) < period:
        return 0.0

    # Smooth ADX with Wilder's method
    adx_smooth = [sum(adx_values[:period]) / period]
    for val in adx_values[period:]:
        adx_smooth.append((adx_smooth[-1] * (period - 1) + val) / period)

    return adx_smooth[-1] if adx_smooth else 0.0


def bollinger_bands(closes, period=20, num_std=2):
    """Return (upper, middle, lower, bandwidth)."""
    if len(closes) < period:
        return None, None, None, None
    middle = sum(closes[-period:]) / period
    variance = sum((c - middle) ** 2 for c in closes[-period:]) / period
    std = math.sqrt(variance)
    upper = middle + num_std * std
    lower = middle - num_std * std
    bandwidth = (upper - lower) / middle if middle != 0 else 0
    return upper, middle, lower, bandwidth


def rsi(closes, period=14):
    """Compute RSI value."""
    if len(closes) < period + 1:
        return None
    gains = 0
    losses = 0
    for i in range(len(closes) - period, len(closes)):
        change = closes[i] - closes[i - 1]
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


def macd(closes, fast=12, slow=26, signal=9):
    """Return (macd_line, signal_line, histogram)."""
    if len(closes) < slow + signal:
        return None, None, None
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    if ema_fast is None or ema_slow is None:
        return None, None, None
    macd_line = ema_fast - ema_slow
    # Need a full series for signal line
    macd_series = []
    for i in range(slow, len(closes)):
        ef = ema(closes[:i + 1], fast)
        es = ema(closes[:i + 1], slow)
        if ef is not None and es is not None:
            macd_series.append(ef - es)
    if len(macd_series) < signal:
        return None, None, None
    signal_line = ema(macd_series, signal)
    histogram = macd_line - signal_line if signal_line is not None else 0
    return macd_line, signal_line, histogram


@dataclass
class RegimeResult:
    """Output of the regime detector."""
    regime: str  # primary regime: trending, ranging, volatile, dead, breakout
    sub_regime: str  # bullish_trend, bearish_trend, bull_squeeze, bear_squeeze, etc.
    confidence: float  # 0-1
    trend_strength: float  # ADX value (0-100)
    volatility: float  # ATR / price (percentage)
    momentum: float  # MACD slope
    volume_ratio: float  # current volume / avg volume
    volatility_regime: str  # high, normal, low
    details: Dict[str, Any] = field(default_factory=dict)


class RegimeDetector:
    """
    Advanced market regime detector using multiple indicators.
    """

    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.cache = {}

    def detect(self, candles: List) -> RegimeResult:
        """
        Detect market regime from a list of Candle objects.
        Returns a RegimeResult.
        """
        if len(candles) < self.lookback:
            return self._unknown_result("Not enough candles")

        # Extract data
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        volumes = [c.volume for c in candles]
        latest = candles[-1]
        price = latest.close

        # ─── 1. Trend strength (ADX) ──────────────────────────────────
        adx_val = adx(candles, period=14)
        trend_strength = adx_val  # 0-100

        # ─── 2. Directional movement (using SMA slope) ──────────────
        sma_20 = sma(closes, 20)
        sma_50 = sma(closes, 50)
        sma_20_prev = sma(closes[:-1], 20) if len(closes) > 1 else None
        sma_50_prev = sma(closes[:-1], 50) if len(closes) > 1 else None

        trend_direction = 0
        if sma_20 is not None and sma_50 is not None and sma_20_prev is not None and sma_50_prev is not None:
            # Golden cross / death cross
            if sma_20_prev <= sma_50_prev and sma_20 > sma_50:
                trend_direction = 1  # bullish crossover
            elif sma_20_prev >= sma_50_prev and sma_20 < sma_50:
                trend_direction = -1  # bearish crossover
            elif sma_20 > sma_50:
                trend_direction = 1  # above, bullish
            else:
                trend_direction = -1  # below, bearish

        # ─── 3. Momentum (MACD slope) ─────────────────────────────────
        macd_line, signal_line, hist = macd(closes)
        momentum = 0
        if macd_line is not None and signal_line is not None:
            # 0 = bearish, 1 = bullish
            momentum = 1 if macd_line > signal_line else -1

        # ─── 4. Volatility (ATR) ─────────────────────────────────────
        atr_val = atr(candles, period=14)
        # Normalize by price
        volatility_pct = (atr_val / price) * 100 if price > 0 else 0

        # Compare to recent average volatility
        atr_series = []
        for i in range(14, len(candles)):
            atr_i = atr(candles[:i+1], period=14)
            atr_series.append(atr_i)
        avg_atr = sum(atr_series[-30:]) / len(atr_series[-30:]) if atr_series else atr_val
        volatility_ratio = atr_val / avg_atr if avg_atr > 0 else 1.0

        if volatility_ratio > 1.5:
            volatility_regime = "high"
        elif volatility_ratio < 0.6:
            volatility_regime = "low"
        else:
            volatility_regime = "normal"

        # ─── 5. Bollinger Bands width ────────────────────────────────
        upper, middle, lower, bb_width = bollinger_bands(closes, period=20)
        bb_width_pct = bb_width * 100 if bb_width else 0

        # Squeeze detection: low BB width
        avg_bb_width = 0
        if len(closes) > 40:
            widths = []
            for i in range(30, len(closes)):
                _, _, _, w = bollinger_bands(closes[:i+1], period=20)
                if w is not None:
                    widths.append(w)
            avg_bb_width = sum(widths) / len(widths) if widths else bb_width
        else:
            avg_bb_width = bb_width
        squeeze = bb_width < avg_bb_width * 0.5 if avg_bb_width > 0 else False

        # ─── 6. Volume ─────────────────────────────────────────────────
        avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else sum(volumes) / len(volumes)
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0

        # ─── 7. RSI ────────────────────────────────────────────────────
        rsi_val = rsi(closes, period=14)
        rsi_zone = "neutral"
        if rsi_val is not None:
            if rsi_val > 70:
                rsi_zone = "overbought"
            elif rsi_val < 30:
                rsi_zone = "oversold"

        # ─── 8. Price relative to recent range ──────────────────────
        high_20 = max(highs[-20:])
        low_20 = min(lows[-20:])
        range_pct = ((high_20 - low_20) / price) * 100 if price > 0 else 0
        price_position = (price - low_20) / (high_20 - low_20) if (high_20 - low_20) > 0 else 0.5

        # ─── 9. Candle pattern (engulfing, doji, etc.) ──────────────
        pattern = "none"
        if len(candles) >= 2:
            prev = candles[-2]
            if prev.close < prev.open and latest.close > latest.open:
                if latest.open <= prev.close and latest.close >= prev.open:
                    pattern = "bullish_engulfing"
            elif prev.close > prev.open and latest.close < latest.open:
                if latest.open >= prev.close and latest.close <= prev.open:
                    pattern = "bearish_engulfing"
        # Doji: small body
        body = abs(latest.close - latest.open)
        if body < (latest.high - latest.low) * 0.2:
            pattern = "doji"

        # ─── CLASSIFICATION ──────────────────────────────────────────

        # 1. Determine if market is trending (ADX > 25)
        is_trending = adx_val > 25
        # 2. Determine if range-bound (BB width low, ADX < 25)
        is_ranging = adx_val < 20 and bb_width_pct < 5 and volatility_regime != "high"
        # 3. Volatile regime (high volatility, wide range)
        is_volatile = volatility_regime == "high" or range_pct > 10
        # 4. Dead / low volatility regime
        is_dead = volatility_regime == "low" and range_pct < 2 and adx_val < 15
        # 5. Breakout (price near range extreme with volume spike)
        is_breakout = volume_ratio > 1.5 and (price_position > 0.9 or price_position < 0.1)

        # Determine primary regime
        if is_dead:
            primary = "dead"
        elif is_volatile:
            primary = "volatile"
        elif is_trending:
            primary = "trending"
        elif is_ranging:
            primary = "ranging"
        elif is_breakout:
            primary = "breakout"
        else:
            # default
            primary = "neutral"

        # Determine sub-regime
        sub_regime = "neutral"
        if primary == "trending":
            if trend_direction > 0 or momentum > 0:
                sub_regime = "bullish_trend"
            else:
                sub_regime = "bearish_trend"
        elif primary == "ranging":
            if squeeze:
                sub_regime = "squeeze"
            else:
                sub_regime = "neutral_range"
        elif primary == "volatile":
            if trend_direction > 0:
                sub_regime = "volatile_bull"
            elif trend_direction < 0:
                sub_regime = "volatile_bear"
            else:
                sub_regime = "volatile_choppy"
        elif primary == "dead":
            sub_regime = "low_activity"
        elif primary == "breakout":
            if price_position > 0.9:
                sub_regime = "bull_breakout"
            else:
                sub_regime = "bear_breakout"
        elif primary == "neutral":
            if rsi_zone == "overbought":
                sub_regime = "overbought"
            elif rsi_zone == "oversold":
                sub_regime = "oversold"
            else:
                sub_regime = "neutral"

        # Confidence score: based on how clear the signal is
        # Higher ADX → more confident in trend
        # Wider BB → more confident in volatility classification
        confidence = 0.5  # base
        if primary == "trending":
            confidence = min(1.0, adx_val / 60)  # ADX 60+ = 1.0
        elif primary == "ranging":
            confidence = min(1.0, (1 - (adx_val / 20)))  # low ADX = high confidence
        elif primary == "volatile":
            confidence = min(1.0, volatility_ratio / 2.0)  # 2x = 1.0
        elif primary == "dead":
            confidence = min(1.0, (1 - range_pct / 2)) if range_pct < 2 else 0.8
        elif primary == "breakout":
            confidence = min(1.0, volume_ratio / 2.0)  # 2x volume = 1.0

        return RegimeResult(
            regime=primary,
            sub_regime=sub_regime,
            confidence=round(confidence, 2),
            trend_strength=round(adx_val, 2),
            volatility=round(volatility_pct, 2),
            momentum=1 if momentum > 0 else -1 if momentum < 0 else 0,
            volume_ratio=round(volume_ratio, 2),
            volatility_regime=volatility_regime,
            details={
                'adx': round(adx_val, 2),
                'atr_pct': round(volatility_pct, 2),
                'bb_width_pct': round(bb_width_pct, 2),
                'volume_ratio': round(volume_ratio, 2),
                'range_pct': round(range_pct, 2),
                'price_position': round(price_position, 2),
                'rsi': round(rsi_val, 2) if rsi_val is not None else None,
                'pattern': pattern,
                'squeeze': squeeze,
                'sma_20': round(sma_20, 4) if sma_20 else None,
                'sma_50': round(sma_50, 4) if sma_50 else None,
                'momentum': momentum,
            }
        )

    def _unknown_result(self, reason: str) -> RegimeResult:
        return RegimeResult(
            regime="unknown",
            sub_regime="unknown",
            confidence=0.0,
            trend_strength=0.0,
            volatility=0.0,
            momentum=0,
            volume_ratio=1.0,
            volatility_regime="unknown",
            details={'error': reason}
        )

    # ─── Adaptation helpers ────────────────────────────────────────

    def get_stop_multiplier(self, regime: RegimeResult) -> float:
        """Return a stop multiplier based on regime (1.0 = normal)."""
        if regime.regime == "volatile":
            return 1.5
        elif regime.regime == "trending":
            return 0.8
        elif regime.regime == "ranging":
            return 0.7
        elif regime.regime == "dead":
            return 0.5
        elif regime.regime == "breakout":
            return 1.2
        else:
            return 1.0

    def get_take_profit_multiplier(self, regime: RegimeResult) -> float:
        """Return a take-profit multiplier based on regime."""
        if regime.regime == "trending":
            return 1.5
        elif regime.regime == "volatile":
            return 1.2
        elif regime.regime == "breakout":
            return 1.3
        elif regime.regime == "ranging":
            return 0.8
        elif regime.regime == "dead":
            return 0.6
        else:
            return 1.0

    def get_risk_adjustment(self, regime: RegimeResult) -> float:
        """Return a risk adjustment factor (0.5 = half risk, 2.0 = double)."""
        if regime.regime == "trending":
            return 1.2
        elif regime.regime == "breakout":
            return 1.3
        elif regime.regime == "volatile":
            return 0.7
        elif regime.regime == "ranging":
            return 0.8
        elif regime.regime == "dead":
            return 0.4
        else:
            return 1.0

    def get_preferred_strategy(self, regime: RegimeResult) -> Optional[str]:
        """Suggest a strategy based on the current regime."""
        mapping = {
            "trending": ["ema_golden_cross", "sma_cross", "ewo_offset"],
            "ranging": ["opening_range", "mean_reversion"],
            "volatile": ["opening_range", "breakout"],
            "breakout": ["opening_range", "breakout"],
            "dead": ["none"],
            "neutral": ["self_learning"],
        }
        strategies = mapping.get(regime.regime, ["self_learning"])
        return strategies[0] if strategies and strategies[0] != "none" else None

    def should_trade(self, regime: RegimeResult, min_confidence: float = 0.5) -> bool:
        """Check if the regime is suitable for trading."""
        if regime.confidence < min_confidence:
            return False
        if regime.regime == "dead":
            return False
        return True

    # ─── For UI ────────────────────────────────────────────────────

    def to_dict(self, regime: RegimeResult) -> Dict[str, Any]:
        """Convert regime result to dict for JSON serialization."""
        return {
            'regime': regime.regime,
            'sub_regime': regime.sub_regime,
            'confidence': regime.confidence,
            'trend_strength': regime.trend_strength,
            'volatility': regime.volatility,
            'momentum': regime.momentum,
            'volume_ratio': regime.volume_ratio,
            'volatility_regime': regime.volatility_regime,
            'details': regime.details,
            'stop_multiplier': self.get_stop_multiplier(regime),
            'take_profit_multiplier': self.get_take_profit_multiplier(regime),
            'risk_adjustment': self.get_risk_adjustment(regime),
            'preferred_strategy': self.get_preferred_strategy(regime),
            'should_trade': self.should_trade(regime),
        }
