# risk_manager.py
"""
Risk management and position sizing.
"""

from typing import Optional, List, Any, Tuple

# ─── Import from indicators and utils ──────────────────────────
from indicators import exit_prices
from utils import pct

# ─── Risk/Position sizing functions ─────────────────────────────

def calculate_kelly_risk(db, symbol: Optional[str] = None, settings: dict = None) -> float:
    """
    Calculate Kelly Criterion based on historical performance.
    Returns the optimal risk percentage for the next trade.
    """
    # Get Kelly metrics from database
    metrics = db.get_kelly_metrics(symbol)

    if metrics and metrics.get('total_trades', 0) >= 20:
        kelly_value = metrics.get('kelly_value', 0)
        # Use fractional Kelly (default 1/4)
        kelly_fraction = float(settings.get('kelly_fraction', 0.25)) if settings else 0.25

        # Apply fractional Kelly and cap
        risk_pct = max(0.001, min(0.10, kelly_value * kelly_fraction))
        return risk_pct

    # Fallback to fixed risk
    return float(settings.get('risk_per_trade_pct', 1.0)) / 100 if settings else 0.01

def calculate_atr(candles: list, period: int = 14) -> float:
    """Calculate Average True Range for volatility measurement."""
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

    return sum(true_ranges[-period:]) / period

def get_atr_volatility_scale(current_atr: float, average_atr: float) -> float:
    """
    Get a volatility-based scaling factor.
    Higher volatility = smaller position size.
    """
    if average_atr <= 0:
        return 1.0

    ratio = current_atr / average_atr

    if ratio > 2.5:
        return 0.3  # Very high volatility - reduce size by 70%
    elif ratio > 2.0:
        return 0.5  # High volatility - reduce size by 50%
    elif ratio > 1.5:
        return 0.7  # Elevated volatility - reduce size by 30%
    elif ratio > 0.8:
        return 1.0  # Normal volatility - full size
    elif ratio > 0.4:
        return 1.2  # Low volatility - increase size by 20%
    else:
        return 1.4  # Very low volatility - increase size by 40%

def get_regime_adaptations(state) -> dict:
    """
    Get adaptive stop/target multipliers and risk adjustment based on current regime.
    """
    settings = state.settings
    if not settings.get("regime_adaptation_enabled", True):
        return {
            'stop_multiplier': 1.0,
            'take_profit_multiplier': 1.0,
            'risk_adjustment': 1.0,
            'regime': None,
        }

    regime = state.current_regime
    if not regime or regime.confidence < settings.get("min_regime_confidence", 0.5):
        return {
            'stop_multiplier': 1.0,
            'take_profit_multiplier': 1.0,
            'risk_adjustment': 1.0,
            'regime': None,
        }

    # If we have a regime detector with methods, use them
    if hasattr(state, 'regime_detector'):
        return {
            'stop_multiplier': state.regime_detector.get_stop_multiplier(regime),
            'take_profit_multiplier': state.regime_detector.get_take_profit_multiplier(regime),
            'risk_adjustment': state.regime_detector.get_risk_adjustment(regime),
            'regime': regime,
        }
    else:
        # Default fallback
        return {
            'stop_multiplier': 1.0,
            'take_profit_multiplier': 1.0,
            'risk_adjustment': 1.0,
            'regime': None,
        }

def calculate_position_size(
    cash: float,
    entry_price: float,
    candles: list,
    symbol: str,
    settings: dict,
    db,
    state,
    position_side: str = "LONG"
) -> tuple[float, str]:
    """
    Enhanced position sizing with:
    - Fixed % risk (default)
    - Kelly Criterion
    - ATR volatility-based sizing
    - Hybrid mode (Kelly + ATR)
    - Regime-based risk adjustment
    """
    sizing_mode = settings.get('risk_sizing_mode', 'fixed')

    # Get stop distance
    stop_price, target_price, exit_mode = exit_prices(
        entry_price=entry_price,
        candles=candles,
        settings=settings
    )

    risk_per_unit = abs(entry_price - stop_price) if stop_price else 0
    if risk_per_unit <= 0:
        return 0.0, "Invalid stop - no position"

    # 1. Base risk amount
    base_risk_pct = float(settings.get('risk_per_trade_pct', 1.0)) / 100
    risk_cash = cash * base_risk_pct

    # 2. Apply Kelly if enabled
    if sizing_mode in ['kelly', 'hybrid']:
        kelly_risk = calculate_kelly_risk(db, symbol, settings)
        risk_cash = cash * kelly_risk
        risk_cash = min(risk_cash, cash * 0.10)  # Cap at 10%

    # 3. Apply ATR scaling if enabled
    if sizing_mode in ['atr', 'hybrid']:
        atr_period = int(settings.get('atr_period', 14))
        current_atr = calculate_atr(candles, atr_period)

        if current_atr > 0:
            avg_atr = 0
            if len(candles) > atr_period * 2:
                avg_atr = calculate_atr(candles[:atr_period * 2], atr_period)
            else:
                avg_atr = current_atr

            scale = get_atr_volatility_scale(current_atr, avg_atr)
            risk_cash = risk_cash * scale

    # 4. Apply regime-based risk adjustment
    regime_adapt = get_regime_adaptations(state)
    risk_cash = risk_cash * regime_adapt.get('risk_adjustment', 1.0)

    # 5. Calculate quantity from risk budget
    quantity = risk_cash / risk_per_unit
    spend = quantity * entry_price

    # 6. Apply caps
    max_fraction = float(settings.get('max_position_pct', 0.25))
    max_spend = cash * max_fraction
    min_spend = float(settings.get('min_order_value', 1.0))

    spend = min(spend, max_spend)

    # ── FIX: don't silently inflate risk to hit the exchange minimum.
    # If risk-based sizing wants less than the minimum order value,
    # either (a) accept the extra risk explicitly if it's within a
    # tolerance band, or (b) skip the trade rather than override it.
    reason_parts = []
    if spend < min_spend:
        max_acceptable_risk_multiple = float(
            settings.get('min_order_risk_override_multiple', 2.0)
        )
        implied_extra_risk = min_spend / spend if spend > 0 else float('inf')

        if implied_extra_risk <= max_acceptable_risk_multiple:
            # Minimum order size only modestly exceeds intended risk —
            # bump up, but log/report that we did it and by how much.
            spend = min_spend
            reason_parts.append(
                f"Bumped to min order ${min_spend:.2f} "
                f"({implied_extra_risk:.1f}x intended risk)"
            )
        else:
            # Minimum order size would force taking on way more risk
            # than the sizing model calls for — refuse the trade instead
            # of silently overriding the risk model.
            return 0.0, (
                f"Skipped: risk-sized spend ${spend:.2f} is below "
                f"min_order_value ${min_spend:.2f} "
                f"({implied_extra_risk:.1f}x intended risk, "
                f"limit {max_acceptable_risk_multiple:.1f}x)"
            )

    # Build reason string
    if sizing_mode == 'kelly':
        reason_parts.append(f"Kelly {risk_cash/cash*100:.2f}%")
    elif sizing_mode == 'atr':
        reason_parts.append(f"ATR {current_atr:.4f}")
    elif sizing_mode == 'hybrid':
        reason_parts.append("Hybrid (Kelly+ATR)")
    else:
        reason_parts.append(f"Fixed {base_risk_pct*100:.2f}%")

    if regime_adapt.get('regime'):
        reason_parts.append(
            f"Regime {regime_adapt['regime'].regime} x{regime_adapt['risk_adjustment']:.2f}"
        )

    return spend, " | ".join(reason_parts)