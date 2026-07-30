# utils.py
"""
Utilities and helper functions for the Auxo trading bot.
"""

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, List, Dict, Callable

# ─── Constants ──────────────────────────────────────────────────

FOREX_BASE_RATES = {
    "EURUSD": 1.0750,
    "GBPUSD": 1.2650,
    "USDJPY": 157.20,
    "AUDUSD": 0.6650,
    "USDCAD": 1.3650,
    "USDCHF": 0.8950,
    "NZDUSD": 0.6100,
    "EURGBP": 0.8500,
    "EURJPY": 169.00,
    "GBPJPY": 198.80,
}

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

# ─── Time/Date helpers ──────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def today_key() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")

def pct(value: float, base: float) -> float:
    if base == 0:
        return 0.0
    return round((value / base) * 100, 4)

# ─── Symbol/Currency helpers ────────────────────────────────────

def parse_watchlist(value: str) -> list[str]:
    symbols: list[str] = []
    for item in value.replace("\n", ",").split(","):
        symbol = normalize_forex_symbol(item)
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols

def normalize_forex_symbol(symbol: str) -> str:
    normalized = symbol.upper().replace("/", "").replace("-", "").replace("_", "").strip()
    aliases = {
        "GPB": "GBP",
    }
    for wrong, correct in aliases.items():
        if normalized.startswith(wrong):
            normalized = correct + normalized[len(wrong):]
    return normalized

def symbol_to_currency(symbol: str, asset_class: str) -> str:
    """Extract currency from symbol for news guard."""
    symbol = symbol.upper()
    if asset_class == "forex":
        if len(symbol) == 6 and symbol.isalpha():
            return symbol[:3]
        return symbol
    return symbol

def country_to_currency(country_code: str) -> str:
    """Convert country code to currency code for news guard."""
    mapping = {
        'USA': 'USD', 'GBR': 'GBP', 'JPN': 'JPY', 'EUR': 'EUR',
        'AUS': 'AUD', 'CAN': 'CAD', 'CHE': 'CHF', 'NZL': 'NZD',
        'CHN': 'CNY', 'IND': 'INR', 'BRA': 'BRL', 'ZAF': 'ZAR',
        'RUS': 'RUB', 'KOR': 'KRW', 'MEX': 'MXN', 'SGP': 'SGD',
        'HKG': 'HKD', 'TWN': 'TWD', 'IDN': 'IDR',
    }
    return mapping.get(country_code.upper(), country_code)

# ─── Environment helpers ────────────────────────────────────────

def decode_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value

def load_dotenv(path: Path) -> dict[str, str]:
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = decode_env_value(value)
        if key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded

DOTENV_LOADED_KEYS = set(load_dotenv(ENV_FILE).keys())

# ─── Chart/Timeframe helpers ────────────────────────────────────

def granularity_label(seconds: int | float) -> str:
    seconds = int(seconds)
    labels = {
        60: "1m",
        300: "5m",
        900: "15m",
        3600: "1h",
        21600: "6h",
        86400: "1d",
    }
    if seconds in labels:
        return labels[seconds]
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"

def latest_candle_incomplete(candles: list[dict] | list, granularity: int) -> bool:
    if not candles:
        return False
    latest = candles[-1]
    latest_time = latest.time if hasattr(latest, 'time') else latest.get("time")
    try:
        return int(time.time()) < int(latest_time) + int(granularity)
    except (TypeError, ValueError):
        return False

def signal_candles(candles, settings: dict):
    if settings.get("closed_candle_only") and len(candles) > 2:
        return candles[:-1]
    return candles

# ─── Blocked reason helpers ─────────────────────────────────────

def blocked_reason_key(message: str) -> str:
    text = message.lower()
    if "spread" in text:
        return "spread"
    if "volume" in text or "liquidity" in text:
        return "liquidity"
    if "s/r" in text or "support" in text or "resistance" in text or "reward/risk" in text:
        return "S/R"
    if "weak" in text:
        return "weak pair"
    if "minimum" in text or "below" in text:
        return "min order"
    if "regime" in text:
        return "regime"
    if "daily" in text:
        return "daily cap"
    return "other"

def blocked_summary(journal: list) -> dict[str, Any]:
    today = today_key()
    counts: dict[str, int] = {}
    total = 0
    for item in journal:
        if item.event != "BLOCK" or not item.time.startswith(today):
            continue
        key = blocked_reason_key(item.message)
        counts[key] = counts.get(key, 0) + 1
        total += 1
    return {
        "total": total,
        "counts": counts,
    }

def best_current_setup(scan_rows: list[dict]) -> dict | None:
    candidates = [
        row for row in scan_rows
        if row.get("signal") == "BUY" and row.get("price") is not None
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda row: float(row.get("score") or 0.0))
    return {
        "symbol": best.get("symbol"),
        "price": best.get("price"),
        "score": best.get("score"),
        "reward_risk": best.get("reward_risk"),
        "support_distance_pct": best.get("support_distance_pct"),
        "regime": best.get("regime"),
        "reason": "BUY",
    }

def open_trade_risk(state, chart_levels: dict, price: float | None) -> dict | None:
    if not state.active_symbol or not state.entry_price or abs(state.coin) <= 0 or not price:
        return None

    entry = float(state.entry_price)
    stop = chart_levels.get("stop")
    target = chart_levels.get("target")

    if state.is_short:
        risk_per_unit = float(stop) - entry if stop else 0.0
        target_per_unit = entry - float(target) if target else 0.0
        current_per_unit = entry - float(price)
    else:
        risk_per_unit = entry - float(stop) if stop else 0.0
        target_per_unit = float(target) - entry if target else 0.0
        current_per_unit = float(price) - entry

    return {
        "symbol": state.active_symbol,
        "entry": entry,
        "price": price,
        "stop": stop,
        "target": target,
        "risk_cash": round(max(risk_per_unit, 0.0) * abs(state.coin), 8),
        "target_cash": round(max(target_per_unit, 0.0) * abs(state.coin), 8),
        "current_cash": round(current_per_unit * abs(state.coin), 8),
        "current_r": round(current_per_unit / risk_per_unit, 4) if risk_per_unit > 0 else None,
        "distance_to_stop_pct": pct(float(price) - float(stop), float(price)) if stop else None,
        "distance_to_target_pct": pct(float(target) - float(price), float(price)) if target else None,
        "is_short": state.is_short,
    }

# ─── Position/performance helpers ──────────────────────────────

def position_rows(state) -> list[dict[str, Any]]:
    """Get position rows - uses OANDA data if available."""
    # This function uses the global _position_rows_bot reference (set elsewhere).
    # We'll keep it as is but import the global.
    from utils import _position_rows_bot
    rows: list[dict[str, Any]] = []
    settings = state.settings
    exchange = settings.get("exchange", "coinbase")

    if exchange == "oanda_demo":
        bot = _position_rows_bot
        if bot and bot.should_oanda_demo_trade():
            try:
                summary = bot.get_oanda_account_summary()
                if summary.get("ok"):
                    for pos in summary.get("positions", []):
                        symbol = pos.get("symbol", "")
                        units = pos.get("units", 0)
                        avg_price = pos.get("average_price", 0)
                        current_price = pos.get("current_price", 0)
                        is_short = pos.get("side") == "SHORT"
                        unrealized_pnl = pos.get("unrealized_pnl", 0)

                        stop_loss = pos.get("stop_loss")
                        take_profit = pos.get("take_profit")

                        if avg_price > 0:
                            if is_short:
                                pnl_pct = ((avg_price - current_price) / avg_price) * 100 if current_price else 0
                            else:
                                pnl_pct = ((current_price - avg_price) / avg_price) * 100 if current_price else 0
                        else:
                            pnl_pct = 0

                        rows.append({
                            "symbol": symbol,
                            "quantity": units,
                            "entry_price": avg_price,
                            "current_price": current_price if current_price else avg_price,
                            "highest_price": avg_price,
                            "stop_price": stop_loss,
                            "target_price": take_profit,
                            "stop": stop_loss,
                            "target": take_profit,
                            "unrealized_pnl": round(unrealized_pnl, 2),
                            "unrealized_pnl_pct": round(pnl_pct, 2),
                            "opened_at": now_iso(),
                            "trade_id": None,
                            "partial_take_profit_done": False,
                            "is_short": is_short,
                            "exchange": "OANDA",
                            "has_tp_sl": bool(stop_loss or take_profit),
                            "stop_distance_pct": None,
                            "target_distance_pct": None,
                            "entry_time": time.time(),
                        })
                    return rows
            except Exception:
                pass

    # ─── FALLBACK TO LOCAL POSITIONS ──────────────────────────────
    for symbol, position in state.positions.items():
        quantity = float(position.get("quantity", 0.0))
        entry = float(position.get("entry_price", 0.0))
        history = state.price_history.get(symbol, [])
        current = history[-1] if history else entry
        is_short = position.get("is_short", False)

        if is_short:
            unrealized = (entry - current) * abs(quantity)
        else:
            unrealized = (current - entry) * abs(quantity)

        if entry > 0:
            if is_short:
                pnl_pct = ((entry - current) / entry) * 100
            else:
                pnl_pct = ((current - entry) / entry) * 100
        else:
            pnl_pct = 0

        stop = (
            position.get("stop_price") or
            position.get("stop") or
            position.get("stop_loss") or
            position.get("stop_loss_price")
        )
        target = (
            position.get("target_price") or
            position.get("target") or
            position.get("take_profit") or
            position.get("take_profit_price")
        )

        stop_distance_pct = None
        target_distance_pct = None
        if stop and current:
            if is_short:
                stop_distance_pct = ((stop - current) / current) * 100
            else:
                stop_distance_pct = ((current - stop) / current) * 100
        if target and current:
            if is_short:
                target_distance_pct = ((current - target) / current) * 100
            else:
                target_distance_pct = ((target - current) / current) * 100

        rows.append({
            "symbol": symbol,
            "quantity": quantity,
            "entry_price": entry,
            "current_price": current,
            "highest_price": position.get("highest_price", entry),
            "stop_price": stop,
            "target_price": target,
            "stop": stop,
            "target": target,
            "unrealized_pnl": round(unrealized, 2),
            "unrealized_pnl_pct": round(pnl_pct, 2),
            "stop_distance_pct": round(stop_distance_pct, 2) if stop_distance_pct is not None else None,
            "target_distance_pct": round(target_distance_pct, 2) if target_distance_pct is not None else None,
            "opened_at": position.get("opened_at", now_iso()),
            "trade_id": position.get("trade_id"),
            "partial_take_profit_done": bool(position.get("partial_take_profit_done", False)),
            "is_short": is_short,
            "exchange": "Paper",
            "has_tp_sl": bool(stop or target),
            "entry_time": position.get("entry_time", time.time()),
        })

    rows.sort(key=lambda item: item["symbol"])
    return rows

# Global for position_rows (set by bot)
_position_rows_bot = None

def set_position_rows_bot(bot) -> None:
    global _position_rows_bot
    _position_rows_bot = bot

def setup_settings_key(settings: dict[str, Any]) -> str:
    strategy = settings.get("strategy", "sma_cross")
    if strategy == "ewo_offset":
        return (
            f"ewo {int(settings.get('base_nb_candles_buy', 14))}/"
            f"{int(settings.get('base_nb_candles_sell', 24))} "
            f"rsi<{int(settings.get('rsi_buy', 69))}"
        )
    if strategy == "opening_range":
        return (
            f"opening_range {int(settings.get('opening_range_minutes', 15))}m "
            f"ATR{int(settings.get('opening_range_atr_period', 14))} "
            f"thr{float(settings.get('opening_range_manipulation_threshold', 0.20)):.2f} "
            f"SL{float(settings.get('opening_range_stop_loss_atr_multiplier', 1.5)):.1f}x "
            f"TP{float(settings.get('opening_range_take_profit_atr_multiplier', 2.5)):.1f}x"
        )
    if strategy == "ema_golden_cross":
        return f"EMA GC {int(settings.get('ema_short', 50))}/{int(settings.get('ema_long', 200))}"
    if strategy == "self_learning":
        return "self_learning"
    return (
        f"sma {int(settings.get('short_window', 5))}/"
        f"{int(settings.get('long_window', 20))}"
    )

def setup_performance(records: list) -> list[dict[str, Any]]:
    stats: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if record.status != "CLOSED":
            continue
        key = (record.symbol, record.settings_key)
        row = stats.setdefault(key, {
            "symbol": record.symbol,
            "settings_key": record.settings_key,
            "closed_setups": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "regimes": {},
        })
        pnl = float(record.realized_pnl or 0.0)
        pnl_pct = float(record.pnl_pct or 0.0)
        row["closed_setups"] += 1
        row["total_pnl"] += pnl
        row["total_pnl_pct"] += pnl_pct
        if pnl >= 0:
            row["wins"] += 1
        else:
            row["losses"] += 1
        row["regimes"][record.regime] = row["regimes"].get(record.regime, 0) + 1

    rows = []
    for row in stats.values():
        closed = int(row["closed_setups"])
        top_regime = "-"
        if row["regimes"]:
            top_regime = max(row["regimes"], key=row["regimes"].get)
        rows.append({
            **row,
            "total_pnl": round(row["total_pnl"], 8),
            "total_pnl_pct": round(row["total_pnl_pct"], 4),
            "expectancy_pct": round(row["total_pnl_pct"] / closed, 4) if closed else 0.0,
            "win_rate": round((row["wins"] / closed) * 100, 2) if closed else 0.0,
            "top_regime": top_regime,
        })
    rows.sort(key=lambda item: item["expectancy_pct"], reverse=True)
    return rows

def weak_pair_map(records: list, settings: dict[str, Any]) -> dict[str, str]:
    if not settings.get("auto_disable_weak_pairs"):
        return {}

    min_trades = int(settings.get("weak_pair_min_trades", 6))
    expectancy_limit = float(settings.get("weak_pair_expectancy_limit_pct", -0.3))
    win_rate_limit = float(settings.get("weak_pair_win_rate_limit_pct", 35.0))
    by_symbol: dict[str, dict[str, Any]] = {}

    for record in records:
        if record.status != "CLOSED":
            continue
        row = by_symbol.setdefault(record.symbol, {
            "closed": 0,
            "wins": 0,
            "total_pnl_pct": 0.0,
        })
        row["closed"] += 1
        row["total_pnl_pct"] += float(record.pnl_pct or 0.0)
        if float(record.realized_pnl or 0.0) >= 0:
            row["wins"] += 1

    weak: dict[str, str] = {}
    for symbol, row in by_symbol.items():
        closed = int(row["closed"])
        if closed < min_trades:
            continue
        expectancy = row["total_pnl_pct"] / closed
        win_rate = (row["wins"] / closed) * 100
        if expectancy <= expectancy_limit:
            weak[symbol] = f"weak expectancy {expectancy:.2f}% over {closed} setups"
        elif row["total_pnl_pct"] < 0 and win_rate <= win_rate_limit:
            weak[symbol] = f"weak win rate {win_rate:.1f}% over {closed} setups"
    return weak

def setup_edge_score(records: list, symbol: str, settings_key: str) -> float:
    closed = [
        record for record in records
        if record.status == "CLOSED"
        and record.symbol == symbol
        and record.settings_key == settings_key
    ][-20:]
    if len(closed) < 3:
        return 0.0
    expectancy = sum(float(record.pnl_pct or 0.0) for record in closed) / len(closed)
    return round(max(-2.0, min(2.0, expectancy)), 4)

def recent_setup_records(records: list, limit: int = 40) -> list[dict[str, Any]]:
    return [asdict(record) for record in records[-limit:]][::-1]

def symbol_performance(trades: list) -> list[dict[str, Any]]:
    open_buys: dict[str, list] = {}
    stats: dict[str, dict[str, Any]] = {}

    for trade in trades:
        symbol = trade.symbol
        stats.setdefault(symbol, {
            "symbol": symbol,
            "buys": 0,
            "sells": 0,
            "closed_pnl": 0.0,
            "wins": 0,
            "losses": 0,
            "fees": 0.0,
        })
        stats[symbol]["fees"] += trade.fee_paid

        if trade.side == "BUY":
            stats[symbol]["buys"] += 1
            open_buys.setdefault(symbol, []).append(trade)
        elif trade.side == "SHORT":
            stats[symbol]["sells"] += 1
            open_buys.setdefault(symbol, []).append(trade)
            continue

        if trade.side == "SELL":
            stats[symbol]["sells"] += 1
            buy = open_buys.get(symbol, []).pop(0) if open_buys.get(symbol) else None
            if not buy:
                continue
            if buy.side == "SHORT":
                pnl = (buy.price - trade.price) * abs(trade.quantity)
            else:
                pnl = (trade.price - buy.price) * abs(trade.quantity)
            stats[symbol]["closed_pnl"] += pnl
            if pnl >= 0:
                stats[symbol]["wins"] += 1
            else:
                stats[symbol]["losses"] += 1
        elif trade.side == "BUY" and trade.quantity < 0:
            stats[symbol]["sells"] += 1

    rows = []
    for row in stats.values():
        closed = row["wins"] + row["losses"]
        rows.append({
            **row,
            "closed_pnl": round(row["closed_pnl"], 8),
            "fees": round(row["fees"], 8),
            "win_rate": round((row["wins"] / closed) * 100, 2) if closed else 0.0,
        })
    rows.sort(key=lambda item: item["closed_pnl"], reverse=True)
    return rows

# ─── Candle helpers ─────────────────────────────────────────────

def closes_to_candles(closes: list[float]) -> list:
    """Convert a list of close prices to a list of simple Candle objects (for compatibility)."""
    # We'll create a simple class or use dict; but to avoid imports, we'll use a namedtuple or dataclass? Better to use a simple class.
    # Since we don't want to import database.Candle, we'll create a lightweight object.
    # We'll return a list of dicts with keys: time, open, high, low, close, volume
    from types import SimpleNamespace
    candles = []
    for idx, price in enumerate(closes):
        candles.append(SimpleNamespace(
            time=idx,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=1.0
        ))
    return candles

def strategy_minimum_candles(settings: dict[str, Any]) -> int:
    if settings.get("strategy") == "ewo_offset":
        return max(
            205,
            int(settings.get("base_nb_candles_buy", 14)) + 1,
            int(settings.get("base_nb_candles_sell", 24)) + 1,
        )
    if settings.get("strategy") == "opening_range":
        return max(48, int(settings.get("opening_range_atr_period", 14)) + 2)
    if settings.get("strategy") == "ema_golden_cross":
        return int(settings.get("ema_long", 200)) + 1
    return int(settings["long_window"]) + 1

def fetch_json(url: str, timeout: int = 10) -> dict[str, Any]:
    import urllib.request
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "local-paper-trading-bot/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            raise RuntimeError(
                "HTTP 403 Forbidden. The exchange may be blocking this network/IP."
            ) from exc
        raise

def normalize_granularity(value: Any) -> int:
    granularity = int(value)
    allowed = [60, 300, 900, 3600, 21600, 86400]
    if granularity in allowed:
        return granularity
    return min(allowed, key=lambda item: abs(item - granularity))

# ─── Slippage / Backtest helpers ───────────────────────────────

def apply_slippage(price: float, side: str, slippage_fraction: float) -> float:
    if side.upper() == "BUY":
        return price * (1 + slippage_fraction)
    return price * (1 - slippage_fraction)

def backtest_runtime_settings(settings: dict[str, Any]) -> dict[str, Any]:
    runtime = {**settings}
    if not is_forex_settings(runtime):
        return runtime

    forex_caps = {
        "stop_loss_pct": (2.0, 0.4),
        "take_profit_pct": (3.0, 0.8),
        "min_sr_range_pct": (2.0, 0.5),
        "near_support_pct": (1.0, 0.3),
        "min_resistance_distance_pct": (1.0, 0.25),
        "min_reward_risk": (1.5, 1.2),
        "support_stop_buffer_pct": (1.0, 0.1),
        "resistance_target_buffer_pct": (0.2, 0.05),
        "sr_zone_tolerance_pct": (0.3, 0.15),
        "opening_range_manipulation_threshold": (0.5, 0.30),
        "opening_range_stop_loss_atr_multiplier": (2.0, 0.8),
        "opening_range_take_profit_atr_multiplier": (3.0, 1.2),
    }
    for key, (crypto_threshold, forex_value) in forex_caps.items():
        try:
            if float(runtime.get(key, forex_value)) >= crypto_threshold:
                runtime[key] = forex_value
        except (TypeError, ValueError):
            runtime[key] = forex_value

    return runtime

def is_forex_settings(settings: dict[str, Any]) -> bool:
    return (
        settings.get("asset_class") == "forex"
        or settings.get("exchange") in {"forex_demo", "oanda_demo"}
    )

def no_train_trades_message(settings: dict[str, Any]) -> str:
    if not is_forex_settings(settings):
        return "No train-window trades found"
    if settings.get("strategy") == "ewo_offset":
        return (
            "No train-window trades found; EWO/Freqtrade mode can be very strict on forex."
            "Try SMA Cross, more candles, or looser EWO/offset settings."
        )
    if settings.get("use_sr_filter"):
        return (
            "No train-window trades found; forex S/R filters may still be too tight. "
            "Try more candles or lower the S/R confirmation/range requirements."
        )
    if settings.get("strategy") == "opening_range":
        return (
            "No train-window trades found; Opening Range strategy needs at least 2 days of data "
            "and clear breakouts. Try more candles or a smaller timeframe."
        )
    return "No train-window trades found; try more candles or a faster signal window"
