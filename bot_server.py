#!/usr/bin/env python3
"""
Auxo trading bot – multi‑asset paper/live trading with Self-Learning capabilities.
Supports both long (BUY) and short (SELL) positions with adaptive signal weighting.
Uses SQLite database for permanent trade storage with TP/SL tracking.

Enhanced with:
- Kelly Criterion risk sizing
- ATR volatility-based position sizing
- Multiple exit strategies (RSI, MACD, MA, Breakout, Time-based)
- Coinbase precision fixes for price and size
- Expectancy Engine (advanced performance analytics)
- Market Regime Detector (ADX, volatility, momentum, volume)
- Dynamic strategy switching based on regime
- Adaptive stop/target and risk-per-trade
- XGBoost machine learning for signal confidence
"""

from __future__ import annotations

import json
import random
import math
import os
import base64
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import ssl
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

# ─── New ML imports ──────────────────────────────────────────────────
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# ─── Existing modules ────────────────────────────────────────────────
from database import (
    BotDatabase,
    Trade,
    JournalEntry,
    SetupRecord,
    Candle,
    ManagedOrder,
    SignalHistory,
    now_iso,
    today_key,
    pct,
    STATE_FILE,
    DB_FILE,
)
from exchange_connectors import (
    create_connectors,
    PriceAggregator,
    api_request_with_retry,
    BINANCE_AVAILABLE,
    KRAKEN_AVAILABLE
)
from expectancy_engine import ExpectancyEngine
from regime_detector import RegimeDetector, RegimeResult

# ─── Logging Setup ───
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auxo.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('auxo')

# ─── Strategy Creator Import ────────────────────────────────────────
try:
    from strategy_creator import StrategyManager, GeneticStrategyOptimizer, TradingStrategy
    STRATEGY_CREATOR_AVAILABLE = True
    logger.info("Strategy Creator module loaded")
except ImportError as e:
    STRATEGY_CREATOR_AVAILABLE = False
    logger.warning(f"Strategy Creator not available: {e}")

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, utils
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    hashes = serialization = ec = utils = None
    CRYPTOGRAPHY_AVAILABLE = False
    logger.warning("cryptography package not installed – live trading disabled")

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    websocket = None
    WEBSOCKET_AVAILABLE = False
    logger.warning("websocket-client package not installed – WebSocket feed disabled")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests package not installed – news guard disabled")

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
ENV_FILE = BASE_DIR / ".env"
AUDIT_LOG_FILE = BASE_DIR / "bot_audit.jsonl"

# Module-level bot reference (used by position_rows for OANDA queries)
_position_rows_bot = None

def set_position_rows_bot(bot) -> None:
    global _position_rows_bot
    _position_rows_bot = bot

# ─── Environment helpers ────────────────────────────────────────────
def decode_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value

def load_dotenv(path: Path = ENV_FILE) -> dict[str, str]:
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

DOTENV_LOADED_KEYS = set(load_dotenv().keys())
if DOTENV_LOADED_KEYS:
    logger.info(f"Loaded {len(DOTENV_LOADED_KEYS)} environment variables from .env")

# ─── Default Settings ──────────────────────────────────────────────
DEFAULT_SETTINGS = {
    "asset_class": "crypto",
    "exchange": "coinbase",
    "symbol": "BTC",
    "watchlist": "BTC,ETH,SOL,XRP,DOGE,LINK,AVAX",
    "quote_currency": "GBP",
    "chart_mode": "line",
    "strategy": "self_learning",
    "starting_cash": 38.0,
    "trade_fee": 0.004,
    "poll_seconds": 15,
    "live_granularity": 3600,
    "live_candle_count": 300,
    "short_window": 5,
    "long_window": 20,
    "base_nb_candles_buy": 14,
    "base_nb_candles_sell": 24,
    "low_offset": 0.975,
    "low_offset_2": 0.955,
    "high_offset": 0.991,
    "high_offset_2": 0.997,
    "ewo_high": 2.327,
    "ewo_high_2": -2.327,
    "ewo_low": -20.988,
    "rsi_buy": 69,
    "max_position_pct": 0.25,
    "position_sizing_mode": "balance_fraction",
    "risk_per_trade_pct": 1.0,
    "min_order_value": 1.0,
    "stop_loss_pct": 2.0,
    "take_profit_pct": 3.0,
    "daily_loss_limit_pct": 5.0,
    "cooldown_seconds": 120,
    "sr_lookback_candles": 50,
    "use_sr_filter": False,
    "near_support_pct": 2.0,
    "min_resistance_distance_pct": 1.0,
    "min_sr_range_pct": 8.0,
    "min_reward_risk": 2.0,
    "support_stop_buffer_pct": 2.0,
    "use_dynamic_sr_exits": False,
    "resistance_target_buffer_pct": 0.5,
    "partial_take_profit_enabled": False,
    "partial_take_profit_pct": 50.0,
    "partial_take_profit_at_target_pct": 50.0,
    "trailing_stop_enabled": False,
    "trailing_stop_pct": 2.0,
    "trailing_activation_pct": 3.0,
    "live_trading_enabled": False,
    "live_order_type": "market",
    "live_limit_offset_pct": 0.05,
    "native_stop_enabled": False,
    "max_live_order_gbp": 5.0,
    "max_daily_live_loss_gbp": 25.0,  # legacy key: now means max DAILY P/L loss in quote currency
    "max_daily_live_spend_quote": 250.0,
    "max_live_spread_pct": 0.35,
    "min_live_quote_volume": 1000.0,
    "backtest_slippage_pct": 0.10,
    "sr_zone_tolerance_pct": 0.6,
    "sr_min_touches": 2,
    "auto_disable_weak_pairs": True,
    "weak_pair_min_trades": 6,
    "weak_pair_expectancy_limit_pct": -0.3,
    "weak_pair_win_rate_limit_pct": 35.0,
    "regime_filter_enabled": False,
    "allow_trending_regime": True,
    "allow_ranging_regime": True,
    "allow_volatile_regime": False,
    "allow_dead_regime": False,
    "order_expiry_seconds": 180,
    "order_retry_limit": 1,
    "order_replace_enabled": True,
    "websocket_enabled": False,
    "closed_candle_only": True,
    "oanda_demo_trading_enabled": False,
    "max_oanda_open_trades": 3,
    "max_coinbase_open_trades": 3,
    "news_guard_enabled": False,
    "news_guard_before_minutes": 30,
    "news_guard_after_minutes": 30,
    "news_guard_block_high": True,
    "news_guard_block_medium": False,
    "news_guard_block_low": False,
    "opening_range_minutes": 15,
    "opening_range_atr_period": 14,
    "opening_range_manipulation_threshold": 0.20,
    "opening_range_stop_loss_atr_multiplier": 1.5,
    "opening_range_take_profit_atr_multiplier": 2.5,
    "oanda_account_type": "standard",
    "max_drawdown_pct": 20.0,
    "allow_short_selling": False,
    "telegram_enabled": False,
    "telegram_bot_token": "",
    "telegram_chat_id": "",
    "telegram_alert_on_buy": True,
    "telegram_alert_on_sell": True,
    "telegram_alert_on_error": True,
    "telegram_alert_on_daily_summary": True,
    "telegram_alert_on_drawdown": True,
    "telegram_drawdown_alert_pct": 10.0,
    "ema_short": 50,
    "ema_long": 200,
    "self_learning_enabled": True,
    "signal_confidence_threshold": 0.3,
    "min_signals_required": 1,
    "learning_history_size": 100,
    # ─── Strategy Creator Settings ──────────────────────────────────
    "strategy_creator_enabled": False,
    "strategy_evolution_enabled": False,
    "strategy_generations": 50,
    "strategy_population_size": 50,
    "strategy_confidence_threshold": 0.5,
    "strategy_auto_select": False,
    "strategy_evolution_frequency": 24,
    "strategy_max_active": 5,
    # ─── ATR-based exits ──────────────────────────────────────────────
    "use_atr_exits": True,
    "atr_stop_multiplier": 1.5,
    "atr_target_multiplier": 2.5,
    # ─── New Risk Settings ─────────────────────────────────────────
    "risk_sizing_mode": "fixed",  # "fixed", "kelly", "atr", "hybrid"
    "kelly_fraction": 0.25,
    "atr_period": 14,
    "atr_multiplier": 2.0,
    "max_hold_hours": 24,
    "exit_strategies_enabled": {
        "rsi": True,
        "macd": True,
        "ma": True,
        "breakout": True,
        "time": True,
    },
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "ma_exit_period": 20,
    "active_exchange": "coinbase",
    "regime_adaptation_enabled": True,
    "strategy_switching_enabled": True,
    "min_regime_confidence": 0.5,
    "regime_force_strategy": True,       # If True, override strategy based on regime
    "regime_block_dead": True,           # If True, block all trades when regime is "dead"
    "regime_trend_strategy": "ema_golden_cross",
    "regime_ranging_strategy": "opening_range",
    "regime_breakout_strategy": "opening_range",
    "regime_volatile_strategy": "sma_cross",   # fallback, but risk adjustment will handle sizing
    # ─── Strategy switching validation / anti-whipsaw ────────────────
    "strategy_switch_min_confidence": 0.60,
    "strategy_switch_persistence_candles": 3,
    "strategy_switch_min_hold_candles": 20,
    "strategy_switch_validation_folds": 4,
}

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

# ─── BOT STATE ──────────────────────────────────────────────────────
@dataclass
class BotState:
    running: bool = False
    settings: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_SETTINGS))
    cash: float = DEFAULT_SETTINGS["starting_cash"]
    coin: float = 0.0
    active_symbol: str | None = None
    entry_price: float | None = None
    highest_price: float | None = None
    active_stop_order_id: str | None = None
    partial_take_profit_done: bool = False
    last_price: float | None = None
    last_error: str | None = None
    last_signal: str = "Waiting for enough price data"
    last_action_time: float = 0.0
    day_start_equity: float = DEFAULT_SETTINGS["starting_cash"]
    day_start_date: str = ""
    live_day_start_date: str = ""
    live_daily_spend: float = 0.0
    prices: list[float] = field(default_factory=list)
    price_history: dict[str, list[float]] = field(default_factory=dict)
    candle_history: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    positions: dict[str, dict[str, Any]] = field(default_factory=dict)
    scan_rows: list[dict[str, Any]] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    journal: list[JournalEntry] = field(default_factory=list)
    setup_records: list[SetupRecord] = field(default_factory=list)
    active_setup_id: str | None = None
    active_setup_ids: dict[str, str] = field(default_factory=dict)
    open_orders: list[ManagedOrder] = field(default_factory=list)
    websocket_status: str = "disabled"
    websocket_last_message: str = ""
    websocket_last_seen: str = ""
    news_events: list[dict[str, Any]] = field(default_factory=list)
    news_last_update: str = ""
    news_guard_status: str = "idle"
    opening_range_analysis: dict[str, Any] = field(default_factory=dict)
    peak_equity: float = DEFAULT_SETTINGS["starting_cash"]
    stop_price: float | None = None
    target_price: float | None = None
    exit_mode: str = "fixed"
    is_short: bool = False
    signal_history: dict[str, dict[str, Any]] = field(default_factory=dict)
    learning_iterations: int = 0
    last_learning_update: str = ""
    db_initialized: bool = False
    # ─── Strategy Creator State ─────────────────────────────────────
    strategy_creator_enabled: bool = False
    strategy_evolution_enabled: bool = False
    last_strategy_evolution: float = 0.0
    strategy_history: list[dict] = field(default_factory=list)
    active_strategy_id: str | None = None
    # ─── Entry/Exit Tracking ──────────────────────────────────────
    entry_time: float | None = None
    # ─── Product Cache ────────────────────────────────────────────
    product_cache: dict[str, dict] = field(default_factory=dict)
    # ─── Regime / Expectancy ──────────────────────────────────────
    current_regime: Optional[RegimeResult] = None
    historical_loaded: bool = False

# ─── SELF-LEARNING TRADER WITH XGBOOST ──────────────────────────────
class SelfLearningTrader:
    def __init__(self, bot: 'PaperBot', db: BotDatabase):
        self.bot = bot
        self.db = db
        self.signal_history: dict[str, SignalHistory] = {}
        self.learning_iterations = bot.state.learning_iterations if hasattr(bot.state, 'learning_iterations') else 0

        # Default weights for new signals
        self.default_weights = {
            'trend_up': 1.0,
            'trend_down': 0.8,
            'engulfing_bullish': 0.9,
            'engulfing_bearish': 0.8,
            'breakout_resistance': 1.2,
            'breakdown_support': 1.0,
            'macd_buy': 1.1,
            'macd_sell': 0.9,
            'rsi_oversold': 0.7,
            'rsi_overbought': 0.6,
            'volume_spike': 0.5,
            'support_bounce': 0.8,
            'resistance_rejection': 0.7,
        }

        # Load signal history from database
        self.load_signal_history()

        # ─── XGBoost Integration ────────────────────────────────────
        self.xgb_model = None
        self.xgb_features = []
        self.last_train_time = 0
        self.model_path = BASE_DIR / "xgboost_model.pkl"
        self.features_path = BASE_DIR / "xgboost_features.pkl"
        self.load_xgboost_model()

    def load_signal_history(self):
        data = self.db.get_signal_history()
        for signal_type, history_data in data.items():
            self.signal_history[signal_type] = SignalHistory(
                signal_type=signal_type,
                total_signals=history_data.get('total_signals', 0),
                successful_trades=history_data.get('successful_trades', 0),
                total_pnl=history_data.get('total_pnl', 0.0),
                win_rate=history_data.get('win_rate', 0.0),
                avg_pnl=history_data.get('avg_pnl', 0.0),
                last_updated=history_data.get('last_updated', now_iso()),
            )
        logger.info(f"Loaded {len(self.signal_history)} signal types from database")

    def save_signal_history(self):
        for signal_type, history in self.signal_history.items():
            self.db.save_signal_history(signal_type, {
                'total_signals': history.total_signals,
                'successful_trades': history.successful_trades,
                'total_pnl': history.total_pnl,
                'win_rate': history.win_rate,
                'avg_pnl': history.avg_pnl,
                'weight': self.get_signal_weight(signal_type),
                'last_updated': history.last_updated,
            })
            if history.total_signals > 0:
                self.db.save_learning_history(
                    iteration=self.learning_iterations,
                    signal_type=signal_type,
                    weight=self.get_signal_weight(signal_type),
                    win_rate=history.win_rate,
                    total_signals=history.total_signals,
                )
        self.bot.state.learning_iterations = self.learning_iterations
        self.bot.state.last_learning_update = now_iso()
        self.bot.save_state()

    def record_signal_outcome(self, signal_types: list[str], pnl: float, success: bool):
        for signal_type in signal_types:
            if signal_type not in self.signal_history:
                self.signal_history[signal_type] = SignalHistory(signal_type=signal_type)

            history = self.signal_history[signal_type]
            history.total_signals += 1
            history.total_pnl += pnl
            history.avg_pnl = history.total_pnl / history.total_signals if history.total_signals > 0 else 0
            if success:
                history.successful_trades += 1
            history.win_rate = (history.successful_trades / history.total_signals) * 100 if history.total_signals > 0 else 0
            history.last_updated = now_iso()

        self.learning_iterations += 1
        self.save_signal_history()

    def get_signal_weight(self, signal_type: str) -> float:
        history = self.signal_history.get(signal_type)
        # Minimum samples before applying any adjustment
        MIN_SAMPLES = 10

        if history and history.total_signals >= MIN_SAMPLES:
            # Bayesian smoothing: Beta(alpha, beta) prior
            # Use a weak prior (alpha=2, beta=2) that pulls win rate toward 0.5
            alpha = 2.0
            beta = 2.0
            wins = history.successful_trades
            total = history.total_signals
            # Posterior win rate = (wins + alpha) / (total + alpha + beta)
            posterior_win_rate = (wins + alpha) / (total + alpha + beta)
            # Map win rate to weight: base 0.5, range 0.1–2.5
            # A win rate of 50% gives weight = 0.5, 70% → ~1.1, 30% → ~0.1
            # We'll use a linear mapping: weight = 0.5 + 2.0 * (posterior_win_rate - 0.5)
            weight = 0.5 + 2.0 * (posterior_win_rate - 0.5)
            # Clamp to reasonable range
            return max(0.1, min(2.5, weight))

        # Default weight (fallback)
        return self.default_weights.get(signal_type, 0.5)

    # ─── XGBoost Methods ──────────────────────────────────────────────

    def load_xgboost_model(self):
        if self.model_path.exists() and self.features_path.exists():
            try:
                self.xgb_model = joblib.load(self.model_path)
                self.xgb_features = joblib.load(self.features_path)
                logger.info(f"Loaded XGBoost model with {len(self.xgb_features)} features.")
            except Exception as e:
                logger.warning(f"Failed to load XGBoost model: {e}")
                self.xgb_model = None
                self.xgb_features = []

    def save_xgboost_model(self):
        if self.xgb_model is not None:
            joblib.dump(self.xgb_model, self.model_path)
            joblib.dump(self.xgb_features, self.features_path)
            logger.info("XGBoost model saved.")

    def _get_feature_names(self):
        return [
            'rsi', 'macd_hist', 'sma_cross', 'atr_pct', 'volume_ratio',
            'return_1', 'return_3', 'return_5', 'return_10',
            'engulfing_bull', 'engulfing_bear', 'breakout_up', 'breakout_down'
        ]

    def _extract_features(self, candles: list[dict], entry_time: int) -> list | None:
        idx = None
        for i, c in enumerate(candles):
            if c['time'] >= entry_time:
                idx = i
                break
        if idx is None or idx < 50:
            return None

        slice_candles = candles[idx-50:idx]
        closes = [c['close'] for c in slice_candles]
        if len(closes) < 50:
            return None

        # Compute features (using helper functions)
        rsi_val = calculate_rsi(closes, 14) or 50
        macd_data = calculate_macd(closes)
        macd_hist = macd_data[-1] if macd_data and len(macd_data) > 0 else 0
        short_sma = sma(closes, 5)
        long_sma = sma(closes, 20)
        sma_cross = 1 if short_sma and long_sma and short_sma > long_sma else 0

        atr_val = calculate_atr_from_candles(slice_candles, 14)
        atr_pct = (atr_val / closes[-1]) * 100 if closes[-1] != 0 else 0

        avg_vol = sum(c['volume'] for c in slice_candles[-20:]) / 20 if len(slice_candles) >= 20 else 1
        vol_ratio = slice_candles[-1]['volume'] / avg_vol if avg_vol > 0 else 1

        ret_1 = (closes[-1] - closes[-2]) / closes[-2] if len(closes) >= 2 else 0
        ret_3 = (closes[-1] - closes[-4]) / closes[-4] if len(closes) >= 4 else 0
        ret_5 = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 else 0
        ret_10 = (closes[-1] - closes[-11]) / closes[-11] if len(closes) >= 11 else 0

        engulfings = detect_engulfing_patterns(slice_candles)
        eng_bull = 1 if any(e['bullish'] for e in engulfings) else 0
        eng_bear = 1 if any(not e['bullish'] for e in engulfings) else 0

        support, resistance = find_support_resistance(slice_candles, 20)
        breakout_up = 1 if support and closes[-1] > support * 1.02 else 0
        breakout_down = 1 if resistance and closes[-1] < resistance * 0.98 else 0

        return [
            rsi_val,
            macd_hist,
            sma_cross,
            atr_pct,
            vol_ratio,
            ret_1,
            ret_3,
            ret_5,
            ret_10,
            eng_bull,
            eng_bear,
            breakout_up,
            breakout_down,
        ]

    def train_xgboost(self, force: bool = False):
        if not force and (time.time() - self.last_train_time < 3600 * 6):
            return

        trades = self.bot.state.trades
        if len(trades) < 20:
            logger.info("Not enough trades to train XGBoost (need 20).")
            return

        # Sort trades by time (oldest first)
        trades_sorted = sorted(trades, key=lambda t: t.time)

        X = []
        y = []
        symbols = set(t.symbol for t in trades_sorted if hasattr(t, 'symbol'))

        for symbol in symbols:
            candles = self.bot.state.candle_history.get(symbol, [])
            if len(candles) < 50:
                continue
            # Get trades for this symbol, sorted by time
            symbol_trades = [t for t in trades_sorted if t.symbol == symbol and hasattr(t, 'entry_price')]
            for trade in symbol_trades:
                entry_time = trade.time
                if not entry_time:
                    continue
                try:
                    dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                    entry_ts = int(dt.timestamp())
                except:
                    continue

                feature_vec = self._extract_features(candles, entry_ts)
                if feature_vec is None:
                    continue

                # ─── NEW TARGET: Did trade reach +1R before -1R? ──────────
                # Use stop_loss_price and take_profit_price if available
                stop_price = getattr(trade, 'stop_loss_price', None)
                target_price = getattr(trade, 'take_profit_price', None)
                entry_price = trade.entry_price
                if stop_price is not None and target_price is not None and entry_price:
                    # Determine if target was hit before stop
                    # We need actual exit price and reason? We can infer from exit_price and exit_reason.
                    # Better: we can check the trade's exit_reason if available.
                    exit_reason = getattr(trade, 'exit_reason', '')
                    if 'take profit' in exit_reason.lower() or 'target' in exit_reason.lower():
                        target_hit_first = True
                    elif 'stop loss' in exit_reason.lower() or 'stop' in exit_reason.lower():
                        target_hit_first = False
                    else:
                        # If we can't determine, use the actual PnL direction as fallback
                        # But better: skip this trade
                        continue
                    y.append(1 if target_hit_first else 0)
                else:
                    # Fallback to old target (price direction) if TP/SL missing
                    if hasattr(trade, 'exit_price') and trade.exit_price is not None:
                        target = 1 if trade.exit_price > trade.entry_price else 0
                    else:
                        current_price = self.bot.state.last_price
                        if current_price and trade.entry_price:
                            target = 1 if current_price > trade.entry_price else 0
                        else:
                            continue
                    y.append(target)

                X.append(feature_vec)

        if len(X) < 50:
            logger.info(f"Not enough feature vectors for training: {len(X)} < 50.")
            return

        X = np.array(X)
        y = np.array(y)

        # ─── TIME‑BASED WALK‑FORWARD SPLIT ──────────────────────────────
        # Use chronological split (no random shuffle)
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if len(X_train) < 20 or len(X_test) < 10:
            logger.warning("Not enough data for train/test after split.")
            return

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
        )
        model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        logger.info(f"XGBoost trained (walk‑forward): train acc {train_acc:.2f}, test acc {test_acc:.2f}")

        self.xgb_model = model
        self.xgb_features = self._get_feature_names()
        self.save_xgboost_model()
        self.last_train_time = time.time()

    def predict_xgboost(self, candles: list[dict]) -> float:
        if self.xgb_model is None or len(candles) < 50:
            return 0.5

        # Use last 50 candles (assuming candles is a list of Candle objects)
        if isinstance(candles[0], Candle):
            slice_candles = [{'time': c.time, 'open': c.open, 'high': c.high, 'low': c.low, 'close': c.close, 'volume': c.volume} for c in candles[-50:]]
        else:
            slice_candles = candles[-50:]

        feature_vec = self._extract_features(slice_candles, int(time.time()))
        if feature_vec is None or len(feature_vec) != len(self.xgb_features):
            return 0.5

        prob = self.xgb_model.predict_proba(np.array([feature_vec]))[0][1]
        return prob

    def analyze_candles_with_indicators(self, candles: list[Candle], settings: dict) -> dict[str, Any]:
        # Original rule-based analysis (unchanged)
        result = self._rule_based_analysis(candles, settings)

        # Integrate XGBoost prediction
        if self.xgb_model is not None and len(candles) >= 50:
            prob_up = self.predict_xgboost(candles)
            xgb_factor = prob_up * 2 - 1  # -1 to +1
            # Apply as a multiplier to composite score
            # If factor is positive, it amplifies bullish signals; negative dampens/reverses
            result['composite_score'] = result['composite_score'] * (0.5 + 0.5 * xgb_factor)
            result['confidence'] = min(1.0, result['confidence'] * (1 + abs(xgb_factor) * 0.5))
            result['xgb_prob'] = prob_up
            result['xgb_factor'] = xgb_factor

        return result

    def _rule_based_analysis(self, candles: list[Candle], settings: dict) -> dict[str, Any]:
        """Original signal detection – copy your existing method here."""
        if len(candles) < 50:
            return {'signals': [], 'composite_score': 0, 'confidence': 0, 'direction': 'NEUTRAL', 'signal_count': 0}

        signals = []
        prices = [c.close for c in candles]
        short_window = int(settings.get('short_window', 5))
        long_window = int(settings.get('long_window', 20))

        # 1. Trend detection (SMA crossover)
        if len(prices) >= long_window + 1:
            short_sma = sma(prices, short_window)
            long_sma = sma(prices, long_window)
            short_prev = sma(prices[:-1], short_window) if len(prices) > 1 else None
            long_prev = sma(prices[:-1], long_window) if len(prices) > 1 else None

            if short_prev is not None and long_prev is not None and short_sma is not None and long_sma is not None:
                if short_prev <= long_prev and short_sma > long_sma:
                    strength = abs(short_sma - long_sma) / (long_sma + 0.0001)
                    signals.append({
                        'type': 'trend_up',
                        'weight': self.get_signal_weight('trend_up'),
                        'direction': 'BUY',
                        'price': prices[-1],
                        'strength': min(1.0, strength * 5),
                    })
                elif short_prev >= long_prev and short_sma < long_sma:
                    strength = abs(short_sma - long_sma) / (long_sma + 0.0001)
                    signals.append({
                        'type': 'trend_down',
                        'weight': self.get_signal_weight('trend_down'),
                        'direction': 'SELL',
                        'price': prices[-1],
                        'strength': min(1.0, strength * 5),
                    })

        # 2. Engulfing patterns
        engulfing = detect_engulfing_patterns(candles)
        for e in engulfing[-2:]:
            signal_type = 'engulfing_bullish' if e['bullish'] else 'engulfing_bearish'
            signals.append({
                'type': signal_type,
                'weight': self.get_signal_weight(signal_type),
                'direction': 'BUY' if e['bullish'] else 'SELL',
                'price': e['price'],
                'strength': 0.8,
            })

        # 3. MACD
        macd_data = calculate_macd(prices)
        if macd_data and len(macd_data) > 3:
            if macd_data[-1] > 0 and macd_data[-2] <= 0:
                strength = abs(macd_data[-1] - macd_data[-2]) / (abs(prices[-1]) + 0.0001)
                signals.append({
                    'type': 'macd_buy',
                    'weight': self.get_signal_weight('macd_buy'),
                    'direction': 'BUY',
                    'price': prices[-1],
                    'strength': min(1.0, strength * 10),
                })
            elif macd_data[-1] < 0 and macd_data[-2] >= 0:
                strength = abs(macd_data[-1] - macd_data[-2]) / (abs(prices[-1]) + 0.0001)
                signals.append({
                    'type': 'macd_sell',
                    'weight': self.get_signal_weight('macd_sell'),
                    'direction': 'SELL',
                    'price': prices[-1],
                    'strength': min(1.0, strength * 10),
                })

        # 4. RSI
        rsi_value = calculate_rsi(prices)
        if rsi_value is not None:
            if rsi_value <= 30:
                signals.append({
                    'type': 'rsi_oversold',
                    'weight': self.get_signal_weight('rsi_oversold'),
                    'direction': 'BUY',
                    'price': prices[-1],
                    'strength': (30 - rsi_value) / 30,
                })
            elif rsi_value >= 70:
                signals.append({
                    'type': 'rsi_overbought',
                    'weight': self.get_signal_weight('rsi_overbought'),
                    'direction': 'SELL',
                    'price': prices[-1],
                    'strength': (rsi_value - 70) / 30,
                })

        # 5. Volume spikes
        if len(candles) >= 20:
            avg_volume = sum(c.volume for c in candles[-20:]) / 20
            last_volume = candles[-1].volume if candles else 0
            if avg_volume > 0 and last_volume > avg_volume * 2:
                signals.append({
                    'type': 'volume_spike',
                    'weight': self.get_signal_weight('volume_spike'),
                    'direction': 'NEUTRAL',
                    'price': prices[-1],
                    'strength': min(1.0, (last_volume / avg_volume) / 4),
                })

        # 6. Support/Resistance breakouts
        support, resistance = find_support_resistance(candles)
        if support and prices[-1] < support * 0.995:
            signals.append({
                'type': 'breakdown_support',
                'weight': self.get_signal_weight('breakdown_support'),
                'direction': 'SELL',
                'price': prices[-1],
                'strength': (support - prices[-1]) / support,
            })
        if resistance and prices[-1] > resistance * 1.005:
            signals.append({
                'type': 'breakout_resistance',
                'weight': self.get_signal_weight('breakout_resistance'),
                'direction': 'BUY',
                'price': prices[-1],
                'strength': (prices[-1] - resistance) / resistance,
            })

        # 7. Support/Resistance bounces
        if support and abs(prices[-1] - support) / support < 0.01:
            signals.append({
                'type': 'support_bounce',
                'weight': self.get_signal_weight('support_bounce'),
                'direction': 'BUY',
                'price': prices[-1],
                'strength': 0.5,
            })
        if resistance and abs(resistance - prices[-1]) / resistance < 0.01:
            signals.append({
                'type': 'resistance_rejection',
                'weight': self.get_signal_weight('resistance_rejection'),
                'direction': 'SELL',
                'price': prices[-1],
                'strength': 0.5,
            })

        total_weight = 0
        weighted_score = 0
        for signal in signals:
            total_weight += signal['weight']
            if signal['direction'] == 'BUY':
                weighted_score += signal['weight'] * signal['strength']
            elif signal['direction'] == 'SELL':
                weighted_score -= signal['weight'] * signal['strength']

        composite_score = weighted_score / total_weight if total_weight > 0 else 0
        confidence = min(1.0, total_weight / 4.0)

        return {
            'signals': signals,
            'composite_score': composite_score,
            'confidence': confidence,
            'direction': 'BUY' if composite_score > 0.15 else 'SELL' if composite_score < -0.15 else 'NEUTRAL',
            'signal_count': len(signals),
            'signal_types': [s['type'] for s in signals],
            'signal_scores': [s['weight'] * s['strength'] for s in signals],
        }

    def should_enter_trade(self, analysis: dict[str, Any], settings: dict) -> tuple[bool, str, float, list[str]]:
        if not settings.get('self_learning_enabled', True):
            return False, 'Self-learning disabled', 0, []

        if analysis['signal_count'] == 0:
            return False, 'No signals detected', 0, []

        min_confidence = settings.get('signal_confidence_threshold', 0.3)
        if analysis['confidence'] < min_confidence:
            return False, f'Low confidence ({analysis["confidence"]:.2f} < {min_confidence:.2f})', 0, []

        min_signals = settings.get('min_signals_required', 1)
        if analysis['signal_count'] < min_signals:
            return False, f'Not enough signals ({analysis["signal_count"]} < {min_signals})', 0, []

        if analysis['direction'] == 'NEUTRAL':
            return False, 'Neutral direction', 0, []

        if abs(analysis['composite_score']) < 0.1:
            return False, f'Composite score too low ({analysis["composite_score"]:.3f})', 0, []

        strong_signals = [s for s in analysis['signals'] if s['weight'] > 1.0]
        if not strong_signals and abs(analysis['composite_score']) < 0.25:
            return False, 'No strong signals and composite score moderate', 0, []

        return True, analysis['direction'], analysis['composite_score'], analysis['signal_types']

    def get_signal_dashboard(self) -> dict[str, Any]:
        dashboard = {}
        for signal_type, history in self.signal_history.items():
            dashboard[signal_type] = {
                'total_signals': history.total_signals,
                'win_rate': round(history.win_rate, 1),
                'avg_pnl': round(history.avg_pnl, 4),
                'weight': round(self.get_signal_weight(signal_type), 2),
                'last_updated': history.last_updated,
            }
        return dashboard

# ─── HELPER FUNCTIONS ──────────────────────────────────────────────

def detect_engulfing_patterns(candles: list[Candle] | list[dict]) -> list[dict[str, Any]]:
    """Detect bullish and bearish engulfing patterns."""
    patterns = []
    for i in range(1, len(candles)):
        prev = candles[i-1]
        curr = candles[i]
        prev_open = prev.open if hasattr(prev, 'open') else prev['open']
        prev_close = prev.close if hasattr(prev, 'close') else prev['close']
        curr_open = curr.open if hasattr(curr, 'open') else curr['open']
        curr_close = curr.close if hasattr(curr, 'close') else curr['close']
        curr_low = curr.low if hasattr(curr, 'low') else curr['low']
        curr_high = curr.high if hasattr(curr, 'high') else curr['high']

        if prev_close < prev_open and curr_close > curr_open:
            if curr_open <= prev_close and curr_close >= prev_open:
                patterns.append({
                    'bullish': True,
                    'price': curr_low,
                    'index': i,
                })
        elif prev_close > prev_open and curr_close < curr_open:
            if curr_open >= prev_close and curr_close <= prev_open:
                patterns.append({
                    'bullish': False,
                    'price': curr_high,
                    'index': i,
                })
    return patterns

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

def find_support_resistance(candles: list[Candle] | list[dict], lookback: int = 20) -> tuple[float | None, float | None]:
    """Find support and resistance levels."""
    if len(candles) < lookback:
        return None, None

    recent = candles[-lookback:]
    if isinstance(recent[0], Candle):
        support = min(c.low for c in recent)
        resistance = max(c.high for c in recent)
        support_touches = [c.low for c in recent if abs(c.low - support) / support < 0.005]
        resistance_touches = [c.high for c in recent if abs(c.high - resistance) / resistance < 0.005]
    else:
        support = min(c['low'] for c in recent)
        resistance = max(c['high'] for c in recent)
        support_touches = [c['low'] for c in recent if abs(c['low'] - support) / support < 0.005]
        resistance_touches = [c['high'] for c in recent if abs(c['high'] - resistance) / resistance < 0.005]

    if len(support_touches) >= 2:
        support = sum(support_touches) / len(support_touches)
    if len(resistance_touches) >= 2:
        resistance = sum(resistance_touches) / len(resistance_touches)

    return support, resistance

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

def calculate_atr_from_candles(candles: list[dict] | list[Candle], period: int) -> float:
    if len(candles) < period + 1:
        return 0.0

    true_ranges = []
    for i in range(1, len(candles)):
        if isinstance(candles[i], Candle):
            high = candles[i].high
            low = candles[i].low
            prev_close = candles[i-1].close
        else:
            high = candles[i]['high']
            low = candles[i]['low']
            prev_close = candles[i-1]['close']
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

def normalize_granularity(value: Any) -> int:
    granularity = int(value)
    allowed = [60, 300, 900, 3600, 21600, 86400]
    if granularity in allowed:
        return granularity
    return min(allowed, key=lambda item: abs(item - granularity))

def closes_to_candles(closes: list[float]) -> list[Candle]:
    return [
        Candle(time=index, open=price, high=price, low=price, close=price, volume=1.0)
        for index, price in enumerate(closes)
    ]

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

def latest_candle_incomplete(candles: list[dict[str, Any]] | list[Candle], granularity: int) -> bool:
    if not candles:
        return False
    latest = candles[-1]
    latest_time = latest.time if isinstance(latest, Candle) else latest.get("time")
    try:
        return int(time.time()) < int(latest_time) + int(granularity)
    except (TypeError, ValueError):
        return False

def signal_candles(candles: list[Candle], settings: dict[str, Any]) -> list[Candle]:
    if settings.get("closed_candle_only") and len(candles) > 2:
        return candles[:-1]
    return candles

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

def blocked_summary(journal: list[JournalEntry]) -> dict[str, Any]:
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

def best_current_setup(scan_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
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

def open_trade_risk(
    state: BotState,
    chart_levels: dict[str, Any],
    price: float | None,
) -> dict[str, Any] | None:
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

def position_rows(state: BotState) -> list[dict[str, Any]]:
    """Get position rows - uses OANDA data if available."""
    rows: list[dict[str, Any]] = []
    settings = state.settings
    exchange = settings.get("exchange", "coinbase")

    # If using OANDA, get REAL data from OANDA
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
            except Exception as e:
                logger.warning(f"Failed to get OANDA positions for display: {e}")

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

def support_resistance(candles: list[Candle], settings: dict[str, Any]) -> dict[str, Any]:
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

def market_regime(candles: list[Candle], settings: dict[str, Any]) -> dict[str, Any]:
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

def regime_allowed(regime: str, settings: dict[str, Any]) -> tuple[bool, str]:
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

def sr_buy_allowed(
    price: float | None,
    levels: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[bool, str]:
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

def ewo_offset_signal(candles: list[Candle], settings: dict[str, Any]) -> dict[str, Any]:
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

# ─── PAPER BOT CLASS ──────────────────────────────────────────────
class PaperBot:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.state = self.load_state()
        self.quote_currency = self.state.settings.get("quote_currency", "GBP")
        self.connectors = create_connectors(self.quote_currency)
        self.price_aggregator = PriceAggregator(self.connectors)
        logger.info(f"Loaded connectors: {list(self.connectors.keys())}")

        # Initialize database
        self.db = BotDatabase()

        # ─── Expectancy Engine ─────────────────────────────────────────
        self.expectancy = ExpectancyEngine(db=self.db)
        if self.state.trades:
            self.expectancy.set_trades(self.state.trades)
        else:
            self.expectancy.load_trades_from_db()

        # ─── Regime Detector ──────────────────────────────────────────
        self.regime_detector = RegimeDetector(lookback=100)

        # Migrate existing data if needed
        if STATE_FILE.exists() and not self.state.db_initialized:
            migrate_to_database()
            self.state.db_initialized = True
            self.save_state()

        # Backfill TP/SL from positions
        if self.state.positions:
            self.backfill_tpsl_from_positions()

        # Update performance metrics for Kelly
        self.db.update_performance_metrics()

        # Initialize self-learning with database
        self.self_learning_trader = SelfLearningTrader(self, self.db)

        self.thread: threading.Thread | None = None
        self.websocket_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.websocket_stop_event = threading.Event()
        self.feed_prices: dict[str, float] = {}
        self.news_guard_thread: threading.Thread | None = None
        self.news_guard_stop_event = threading.Event()
        self.restart_delay = 60
        self.shutdown_requested = False
        self.oanda_stream_thread: threading.Thread | None = None
        self.oanda_stop_event: threading.Event | None = None

        # ─── Product Cache ─────────────────────────────────────────────
        self.product_cache: dict[str, dict] = {}

        # ─── Strategy Manager ────────────────────────────────────────────
        self.strategy_manager = None
        self.last_strategy_evolution = 0

        if STRATEGY_CREATOR_AVAILABLE and self.state.settings.get("strategy_creator_enabled", False):
            try:
                self.strategy_manager = StrategyManager(self)
                logger.info("Strategy Manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Strategy Manager: {e}")
                self.strategy_manager = None

        # ─── START STRATEGY EVOLUTION TIMER ─────────────────────────────
        self._start_strategy_timer_if_needed()
        self.oanda_cache = None
        self.oanda_cache_time = 10
        self.oanda_cache_ttl = 10
        logger.info("Auxo bot initialised with XGBoost ML integration")

    # ─── COINBASE PRECISION HELPERS ─────────────────────────────────

    def get_product_details(self, product_id: str) -> dict:
        """Get product details with caching."""
        if product_id in self.product_cache:
            return self.product_cache[product_id]

        try:
            url = f"https://api.exchange.coinbase.com/products/{product_id}"
            data = fetch_json(url)
            self.product_cache[product_id] = data
            return data
        except Exception as e:
            logger.warning(f"Failed to get product details for {product_id}: {e}")
            return {}

    def coinbase_price_precision(self, product_id: str) -> int:
        """Get the price precision for a Coinbase product."""
        details = self.get_product_details(product_id)
        quote_increment = details.get('quote_increment', '0.01')
        if '.' in quote_increment:
            precision = len(quote_increment.split('.')[1].rstrip('0'))
        else:
            precision = 0
        return min(max(precision, 2), 8)

    def coinbase_size_precision(self, product_id: str) -> int:
        """Get the size precision for a Coinbase product."""
        details = self.get_product_details(product_id)
        base_increment = details.get('base_increment', '0.00000001')
        if '.' in base_increment:
            precision = len(base_increment.split('.')[1].rstrip('0'))
        else:
            precision = 0
        return min(max(precision, 2), 8)

    def coinbase_min_order_size(self, product_id: str) -> float:
        """Get the minimum order size for a Coinbase product."""
        details = self.get_product_details(product_id)
        base_min_size = details.get('base_min_size', '0.00000001')
        try:
            return float(base_min_size)
        except:
            return 0.00000001

    def coinbase_round_price(self, price: float, product_id: str) -> float:
        """Round price to the nearest multiple of quote_increment."""
        details = self.get_product_details(product_id)
        quote_increment = details.get('quote_increment', '0.01')
        try:
            increment = float(quote_increment)
        except:
            increment = 0.01
        if increment > 0:
            return math.floor(price / increment) * increment
        return price

    def coinbase_round_size(self, size: float, product_id: str) -> float:
        """Round size DOWN to Coinbase's base_increment.

        Never round a small balance UP to base_min_size: doing so can submit more
        base currency than is actually available and cause INSUFFICIENT_FUND.
        Callers that submit orders must separately check base_min_size.
        """
        details = self.get_product_details(product_id)
        base_increment = details.get('base_increment', '0.00000001')
        try:
            increment = float(base_increment)
        except Exception:
            increment = 0.00000001
        size = max(0.0, float(size or 0.0))
        if increment > 0:
            # Tiny epsilon protects against binary-float values such as
            # 0.009999999999 being floored one increment too far.
            return max(0.0, math.floor((size + increment * 1e-9) / increment) * increment)
        return size

    # ─── ENHANCED RISK MANAGEMENT ────────────────────────────────────

    def calculate_kelly_risk(self, symbol: Optional[str] = None) -> float:
        """Calculate Kelly Criterion based on historical performance."""
        metrics = self.db.get_kelly_metrics(symbol)
        if metrics and metrics.get('total_trades', 0) >= 20:
            kelly_value = metrics.get('kelly_value', 0)
            kelly_fraction = float(self.state.settings.get('kelly_fraction', 0.25))
            risk_pct = max(0.001, min(0.10, kelly_value * kelly_fraction))
            logger.debug(f"Kelly risk for {symbol or 'ALL'}: {risk_pct:.4f} (raw: {kelly_value:.4f})")
            return risk_pct
        return float(self.state.settings.get('risk_per_trade_pct', 1.0)) / 100

    def calculate_atr(self, candles: list[Candle], period: int = 14) -> float:
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

    def get_atr_volatility_scale(self, current_atr: float, average_atr: float) -> float:
        """Get a volatility-based scaling factor."""
        if average_atr <= 0:
            return 1.0

        ratio = current_atr / average_atr

        if ratio > 2.5:
            return 0.3
        elif ratio > 2.0:
            return 0.5
        elif ratio > 1.5:
            return 0.7
        elif ratio > 0.8:
            return 1.0
        elif ratio > 0.4:
            return 1.2
        else:
            return 1.4

    def get_regime_adaptations(self) -> dict:
        """Get adaptive stop/target multipliers and risk adjustment based on current regime."""
        settings = self.state.settings
        if not settings.get("regime_adaptation_enabled", True):
            return {
                'stop_multiplier': 1.0,
                'take_profit_multiplier': 1.0,
                'risk_adjustment': 1.0,
                'regime': None,
            }

        regime = self.state.current_regime
        if not regime or regime.confidence < settings.get("min_regime_confidence", 0.5):
            return {
                'stop_multiplier': 1.0,
                'take_profit_multiplier': 1.0,
                'risk_adjustment': 1.0,
                'regime': None,
            }

        return {
            'stop_multiplier': self.regime_detector.get_stop_multiplier(regime),
            'take_profit_multiplier': self.regime_detector.get_take_profit_multiplier(regime),
            'risk_adjustment': self.regime_detector.get_risk_adjustment(regime),
            'regime': regime,
        }

    def get_strategy_for_regime(self) -> Optional[str]:
        """Return the recommended strategy based on the current regime."""
        settings = self.state.settings
        if not settings.get("strategy_switching_enabled", True):
            return None

        regime = self.state.current_regime
        if not regime or regime.confidence < settings.get("min_regime_confidence", 0.5):
            return None

        preferred = self.regime_detector.get_preferred_strategy(regime)
        return preferred

    def calculate_position_size(
        self,
        cash: float,
        entry_price: float,
        candles: list[Candle],
        symbol: str,
        position_side: str = "LONG"
    ) -> tuple[float, str]:
        """Enhanced position sizing with fixed, Kelly, ATR, and hybrid modes."""
        settings = self.state.settings
        sizing_mode = settings.get('risk_sizing_mode', 'fixed')

        stop_price, target_price, exit_mode = self.exit_prices(
            entry_price=entry_price,
            candles=candles or closes_to_candles(self.state.price_history.get(symbol, [])),
            settings=self.state.settings,
        )

        risk_per_unit = abs(entry_price - stop_price) if stop_price else 0
        if risk_per_unit <= 0:
            return 0.0, "Invalid stop - no position"

        base_risk_pct = float(settings.get('risk_per_trade_pct', 1.0)) / 100
        risk_cash = cash * base_risk_pct

        if sizing_mode in ['kelly', 'hybrid']:
            kelly_risk = self.calculate_kelly_risk(symbol)
            risk_cash = cash * kelly_risk
            risk_cash = min(risk_cash, cash * 0.10)

        if sizing_mode in ['atr', 'hybrid']:
            atr_period = int(settings.get('atr_period', 14))
            current_atr = self.calculate_atr(candles, atr_period)
            if current_atr > 0:
                avg_atr = 0
                if len(candles) > atr_period * 2:
                    avg_atr = self.calculate_atr(candles[:atr_period * 2], atr_period)
                else:
                    avg_atr = current_atr
                scale = self.get_atr_volatility_scale(current_atr, avg_atr)
                risk_cash = risk_cash * scale

        regime_adapt = self.get_regime_adaptations()
        risk_cash = risk_cash * regime_adapt.get('risk_adjustment', 1.0)

        quantity = risk_cash / risk_per_unit
        spend = quantity * entry_price

        max_fraction = float(settings.get('max_position_pct', 0.25))
        max_spend = cash * max_fraction
        min_spend = float(settings.get('min_order_value', 1.0))

        spend = min(spend, max_spend)

        reason_parts = []
        if spend < min_spend:
            max_acceptable_risk_multiple = float(
                settings.get('min_order_risk_override_multiple', 2.0)
            )
            implied_extra_risk = min_spend / spend if spend > 0 else float('inf')

            if implied_extra_risk <= max_acceptable_risk_multiple:
                spend = min_spend
                reason_parts.append(
                    f"Bumped to min order ${min_spend:.2f} "
                    f"({implied_extra_risk:.1f}x intended risk)"
                )
            else:
                return 0.0, (
                    f"Skipped: risk-sized spend ${spend:.2f} is below "
                    f"min_order_value ${min_spend:.2f} "
                    f"({implied_extra_risk:.1f}x intended risk, "
                    f"limit {max_acceptable_risk_multiple:.1f}x)"
                )

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

    def exit_prices(
        self,
        entry_price: float,
        candles: list[Candle],
        settings: dict[str, Any],
    ) -> tuple[float, float, str]:
        """Calculate stop loss and take profit prices."""
        if not candles or len(candles) < 14:
            stop = entry_price * (1 - float(settings["stop_loss_pct"]) / 100)
            target = entry_price * (1 + float(settings["take_profit_pct"]) / 100)
            return stop, target, "fixed"

        atr_enabled = settings.get("use_atr_exits", True)
        if atr_enabled:
            atr_period = int(settings.get("atr_period", 14))
            atr_value = self.calculate_atr(candles, atr_period)

            if atr_value and atr_value > 0:
                atr_stop_mult = float(settings.get("atr_stop_multiplier", 1.5))
                atr_target_mult = float(settings.get("atr_target_multiplier", 2.5))

                regime = getattr(self.state, 'current_regime', None)
                if regime:
                    if regime.regime == "volatile":
                        atr_stop_mult *= 1.3
                        atr_target_mult *= 1.2
                    elif regime.regime == "dead":
                        atr_stop_mult *= 0.7
                        atr_target_mult *= 0.8

                stop = entry_price - (atr_value * atr_stop_mult)
                target = entry_price + (atr_value * atr_target_mult)

                min_stop_pct = 0.2
                if (entry_price - stop) / entry_price * 100 < min_stop_pct:
                    stop = entry_price * (1 - float(settings["stop_loss_pct"]) / 100)
                    target = entry_price * (1 + float(settings["take_profit_pct"]) / 100)
                    return stop, target, "fixed"

                return stop, target, f"ATR ({atr_value:.4f})"

        if settings.get("use_dynamic_sr_exits"):
            levels = support_resistance(candles, settings)
            support = levels.get("support")
            resistance = levels.get("resistance")
            if support and resistance and levels.get("confirmed"):
                stop_buffer = float(settings.get("support_stop_buffer_pct", 2.0)) / 100
                target_buffer = float(settings.get("resistance_target_buffer_pct", 0.5)) / 100
                sr_stop = float(support) * (1 - stop_buffer)
                sr_target = float(resistance) * (1 - target_buffer)
                if sr_stop < entry_price and sr_target > entry_price:
                    return sr_stop, sr_target, "S/R"

        stop = entry_price * (1 - float(settings["stop_loss_pct"]) / 100)
        target = entry_price * (1 + float(settings["take_profit_pct"]) / 100)
        return stop, target, "fixed"

    # ─── ENHANCED EXIT STRATEGIES ────────────────────────────────────

    def check_rsi_exit(self, candles: list[Candle], position_side: str) -> tuple[bool, str]:
        """Exit when RSI indicates overbought/oversold."""
        if len(candles) < 15:
            return False, ""

        closes = [c.close for c in candles[-15:]]
        rsi_value = calculate_rsi(closes, 14)

        if rsi_value is None:
            return False, ""

        settings = self.state.settings
        overbought = float(settings.get('rsi_overbought', 70))
        oversold = float(settings.get('rsi_oversold', 30))

        if position_side == "LONG":
            if rsi_value > overbought:
                return True, f"RSI overbought ({rsi_value:.1f} > {overbought})"
        else:  # SHORT
            if rsi_value < oversold:
                return True, f"RSI oversold ({rsi_value:.1f} < {oversold})"

        return False, ""

    def check_macd_exit(self, candles: list[Candle], position_side: str) -> tuple[bool, str]:
        """Exit when MACD crosses signal line in opposite direction."""
        if len(candles) < 35:
            return False, ""

        closes = [c.close for c in candles]
        macd_data = calculate_macd(closes)

        if not macd_data or len(macd_data) < 3:
            return False, ""

        macd_curr = macd_data[-1]
        macd_prev = macd_data[-2] if len(macd_data) > 1 else macd_curr

        if position_side == "LONG":
            if macd_curr < 0 and macd_prev >= 0:
                return True, f"MACD bearish crossover ({macd_curr:.4f} < 0)"
        else:  # SHORT
            if macd_curr > 0 and macd_prev <= 0:
                return True, f"MACD bullish crossover ({macd_curr:.4f} > 0)"

        return False, ""

    def check_ma_exit(self, candles: list[Candle], position_side: str) -> tuple[bool, str]:
        """Exit when price crosses key moving average."""
        settings = self.state.settings
        ma_period = int(settings.get('ma_exit_period', 20))

        if len(candles) < ma_period + 1:
            return False, ""

        closes = [c.close for c in candles]
        current_price = closes[-1]

        ma_value = sma(closes, ma_period)

        if ma_value is None:
            return False, ""

        prev_ma = sma(closes[:-1], ma_period) if len(closes) > 1 else ma_value

        if position_side == "LONG":
            if current_price < ma_value and prev_ma is not None and closes[-2] >= prev_ma:
                return True, f"Price broke below {ma_period}-period MA"
        else:  # SHORT
            if current_price > ma_value and prev_ma is not None and closes[-2] <= prev_ma:
                return True, f"Price broke above {ma_period}-period MA"

        return False, ""

    def check_breakout_exit(self, candles: list[Candle], position_side: str) -> tuple[bool, str]:
        """Exit when price breaks support/resistance."""
        if len(candles) < 20:
            return False, ""

        levels = support_resistance(candles, self.state.settings)
        support = levels.get('support')
        resistance = levels.get('resistance')

        if not support and not resistance:
            return False, ""

        current_price = candles[-1].close

        if position_side == "LONG":
            if support and current_price < support * 0.99:
                return True, f"Price broke support ({current_price:.4f} < {support:.4f})"
        else:  # SHORT
            if resistance and current_price > resistance * 1.01:
                return True, f"Price broke resistance ({current_price:.4f} > {resistance:.4f})"

        return False, ""

    def check_time_exit(self, entry_time: float) -> tuple[bool, str]:
        """Exit after a certain time period."""
        if entry_time is None:
            return False, ""

        settings = self.state.settings
        max_hold_hours = float(settings.get('max_hold_hours', 24))
        max_hold_seconds = max_hold_hours * 3600

        current_time = time.time()
        hold_time = current_time - entry_time

        if hold_time > max_hold_seconds:
            return True, f"Max hold time reached ({max_hold_hours}h)"

        return False, ""

    def should_exit_enhanced(
        self,
        symbol: str,
        candles: list[Candle],
        entry_time: float,
        position_side: str
    ) -> tuple[bool, str]:
        """Enhanced exit decision with multiple strategies."""
        if not candles:
            return False, ""

        settings = self.state.settings
        exit_strategies = settings.get('exit_strategies_enabled', {
            'rsi': True,
            'macd': True,
            'ma': True,
            'breakout': True,
            'time': True,
        })

        if exit_strategies.get('rsi', True):
            should_exit, reason = self.check_rsi_exit(candles, position_side)
            if should_exit:
                return True, reason

        if exit_strategies.get('macd', True):
            should_exit, reason = self.check_macd_exit(candles, position_side)
            if should_exit:
                return True, reason

        if exit_strategies.get('ma', True):
            should_exit, reason = self.check_ma_exit(candles, position_side)
            if should_exit:
                return True, reason

        if exit_strategies.get('breakout', True):
            should_exit, reason = self.check_breakout_exit(candles, position_side)
            if should_exit:
                return True, reason

        if exit_strategies.get('time', True):
            should_exit, reason = self.check_time_exit(entry_time)
            if should_exit:
                return True, reason

        return False, ""

    # ─── Load/Save State ─────────────────────────────────────────────

    def load_state(self) -> BotState:
        if not STATE_FILE.exists():
            state = BotState()
            state.day_start_date = today_key()
            state.peak_equity = float(state.settings["starting_cash"])
            state.signal_history = {}
            state.db_initialized = False
            logger.info("No state file found – starting fresh")
            return state

        try:
            raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            state = BotState(
                running=False,
                settings={**DEFAULT_SETTINGS, **raw.get("settings", {})},
                cash=float(raw.get("cash", DEFAULT_SETTINGS["starting_cash"])),
                coin=float(raw.get("coin", 0.0)),
                active_symbol=raw.get("active_symbol"),
                entry_price=raw.get("entry_price"),
                highest_price=raw.get("highest_price"),
                active_stop_order_id=raw.get("active_stop_order_id"),
                partial_take_profit_done=bool(raw.get("partial_take_profit_done", False)),
                last_price=raw.get("last_price"),
                last_error=raw.get("last_error"),
                last_signal=raw.get("last_signal", "Waiting for enough price data"),
                last_action_time=float(raw.get("last_action_time", 0.0)),
                day_start_equity=float(raw.get("day_start_equity", DEFAULT_SETTINGS["starting_cash"])),
                day_start_date=raw.get("day_start_date", today_key()),
                live_day_start_date=raw.get("live_day_start_date", today_key()),
                live_daily_spend=float(raw.get("live_daily_spend", 0.0)),
                prices=[float(item) for item in raw.get("prices", [])][-300:],
                price_history={
                    str(symbol): [float(item) for item in prices][-300:]
                    for symbol, prices in raw.get("price_history", {}).items()
                },
                candle_history={
                    str(symbol): [
                        {
                            "time": int(item.get("time", 0)),
                            "open": float(item.get("open", 0.0)),
                            "high": float(item.get("high", 0.0)),
                            "low": float(item.get("low", 0.0)),
                            "close": float(item.get("close", 0.0)),
                            "volume": float(item.get("volume", 0.0)),
                        }
                        for item in candles
                    ][-300:]
                    for symbol, candles in raw.get("candle_history", {}).items()
                    if isinstance(candles, list)
                },
                positions={
                    str(symbol): {
                        "quantity": float(item.get("quantity", 0.0)),
                        "entry_price": float(item.get("entry_price", 0.0)),
                        "highest_price": float(item.get("highest_price", item.get("entry_price", 0.0))),
                        "partial_take_profit_done": bool(item.get("partial_take_profit_done", False)),
                        "entry_cost": float(item.get("entry_cost", 0.0)),
                        "opened_at": item.get("opened_at", now_iso()),
                        "trade_id": item.get("trade_id"),
                        "stop_price": item.get("stop_price"),
                        "target_price": item.get("target_price"),
                        "exit_mode": item.get("exit_mode"),
                        "is_short": bool(item.get("is_short", False)),
                        "entry_time": float(item.get("entry_time", time.time())),
                    }
                    for symbol, item in raw.get("positions", {}).items()
                    if isinstance(item, dict) and abs(float(item.get("quantity", 0.0))) > 0
                },
                scan_rows=raw.get("scan_rows", []),
                trades=[
                    Trade(
                        time=item.get("time", now_iso()),
                        side=item.get("side", ""),
                        symbol=item.get("symbol", ""),
                        price=float(item.get("price", 0.0)),
                        quantity=float(item.get("quantity", 0.0)),
                        cash_after=float(item.get("cash_after", 0.0)),
                        coin_after=float(item.get("coin_after", 0.0)),
                        reason=item.get("reason", ""),
                        fee_paid=float(item.get("fee_paid", 0.0)),
                        exchange_order_id=item.get("exchange_order_id"),
                        exchange_order_status=item.get("exchange_order_status"),
                        exchange_average_filled_price=item.get("exchange_average_filled_price"),
                        exchange_filled_size=item.get("exchange_filled_size"),
                        stop_loss_price=item.get("stop_loss_price"),
                        take_profit_price=item.get("take_profit_price"),
                        exit_mode=item.get("exit_mode"),
                        exit_reason=item.get("exit_reason"),
                        regime=item.get("regime"),
                    )
                    for item in raw.get("trades", [])
                ],
                journal=[
                    JournalEntry(
                        time=item.get("time", now_iso()),
                        symbol=item.get("symbol", ""),
                        event=item.get("event", "INFO"),
                        message=item.get("message", ""),
                        price=item.get("price"),
                        details=item.get("details", {}),
                    )
                    for item in raw.get("journal", [])
                ],
                setup_records=[
                    SetupRecord(
                        id=item.get("id", str(uuid.uuid4())),
                        time=item.get("time", now_iso()),
                        symbol=item.get("symbol", ""),
                        strategy=item.get("strategy", "sma_cross"),
                        settings_key=item.get("settings_key", ""),
                        entry_price=float(item.get("entry_price", 0.0)),
                        entry_quantity=float(item.get("entry_quantity", 0.0)),
                        entry_cost=float(item.get("entry_cost", 0.0)),
                        entry_fee=float(item.get("entry_fee", 0.0)),
                        entry_reason=item.get("entry_reason", ""),
                        entry_score=float(item.get("entry_score", 0.0)),
                        base_score=float(item.get("base_score", 0.0)),
                        edge_score=float(item.get("edge_score", 0.0)),
                        regime=item.get("regime", "unknown"),
                        support_distance_pct=item.get("support_distance_pct"),
                        resistance_distance_pct=item.get("resistance_distance_pct"),
                        sr_range_pct=item.get("sr_range_pct"),
                        reward_risk=item.get("reward_risk"),
                        status=item.get("status", "OPEN"),
                        closed_quantity=float(item.get("closed_quantity", 0.0)),
                        realized_pnl=float(item.get("realized_pnl", 0.0)),
                        exit_fees=float(item.get("exit_fees", 0.0)),
                        exit_time=item.get("exit_time"),
                        exit_price=item.get("exit_price"),
                        exit_reason=item.get("exit_reason"),
                        pnl_pct=item.get("pnl_pct"),
                        signal_types=item.get("signal_types", []),
                        signal_scores=item.get("signal_scores", []),
                        stop_loss_price=item.get("stop_loss_price"),
                        take_profit_price=item.get("take_profit_price"),
                        exit_mode=item.get("exit_mode"),
                    )
                    for item in raw.get("setup_records", [])
                ],
                active_setup_id=raw.get("active_setup_id"),
                active_setup_ids={
                    str(symbol): str(setup_id)
                    for symbol, setup_id in raw.get("active_setup_ids", {}).items()
                },
                open_orders=[
                    ManagedOrder(
                        order_id=item.get("order_id", ""),
                        symbol=item.get("symbol", ""),
                        product_id=item.get("product_id", ""),
                        side=item.get("side", ""),
                        role=item.get("role", ""),
                        order_type=item.get("order_type", ""),
                        status=item.get("status", "OPEN"),
                        created_at=item.get("created_at", now_iso()),
                        updated_at=item.get("updated_at", now_iso()),
                        expires_at=float(item.get("expires_at", 0.0)),
                        retry_count=int(item.get("retry_count", 0)),
                        local_applied=bool(item.get("local_applied", False)),
                        price=item.get("price"),
                        base_size=item.get("base_size"),
                        quote_size=item.get("quote_size"),
                        reason=item.get("reason", ""),
                        details=item.get("details", {}),
                    )
                    for item in raw.get("open_orders", [])
                    if item.get("order_id")
                ],
                websocket_status=raw.get("websocket_status", "disabled"),
                websocket_last_message=raw.get("websocket_last_message", ""),
                websocket_last_seen=raw.get("websocket_last_seen", ""),
                news_events=raw.get("news_events", []),
                news_last_update=raw.get("news_last_update", ""),
                news_guard_status=raw.get("news_guard_status", "idle"),
                opening_range_analysis=raw.get("opening_range_analysis", {}),
                peak_equity=float(raw.get("peak_equity", float(raw.get("cash", DEFAULT_SETTINGS["starting_cash"])))),
                stop_price=raw.get("stop_price"),
                target_price=raw.get("target_price"),
                exit_mode=raw.get("exit_mode", "fixed"),
                is_short=bool(raw.get("is_short", False)),
                signal_history=raw.get("signal_history", {}),
                learning_iterations=raw.get("learning_iterations", 0),
                last_learning_update=raw.get("last_learning_update", ""),
                db_initialized=raw.get("db_initialized", False),
                strategy_creator_enabled=raw.get("strategy_creator_enabled", False),
                strategy_evolution_enabled=raw.get("strategy_evolution_enabled", False),
                last_strategy_evolution=raw.get("last_strategy_evolution", 0.0),
                strategy_history=raw.get("strategy_history", []),
                active_strategy_id=raw.get("active_strategy_id"),
                entry_time=raw.get("entry_time"),
                product_cache=raw.get("product_cache", {}),
                current_regime=None,
                historical_loaded=raw.get("historical_loaded", False),
            )
            if "oanda_account_type" not in state.settings or not state.settings["oanda_account_type"]:
                state.settings["oanda_account_type"] = "standard"

            if not state.price_history and state.prices:
                state.price_history[state.settings["symbol"]] = state.prices
            if state.positions and not state.active_symbol:
                state.active_symbol = next(iter(state.positions))
            logger.info(f"State loaded: cash={state.cash}, active_symbol={state.active_symbol}, coin={state.coin}")
            return state
        except (OSError, ValueError, TypeError) as e:
            logger.error(f"Failed to load state: {e} – starting fresh")
            state = BotState()
            state.day_start_date = today_key()
            state.peak_equity = float(state.settings["starting_cash"])
            state.last_error = "State file could not be read; started fresh."
            state.signal_history = {}
            state.db_initialized = False
            return state

    def save_state(self) -> None:
        with self.lock:
            data = asdict(self.state)
        STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug("State saved")

    def backfill_tpsl_from_positions(self) -> None:
        if self.state.positions:
            self.db.backfill_tpsl_from_positions(self.state)
            logger.info("TP/SL backfill completed")

    def load_historical_data_from_db(self) -> None:
        if self.state.historical_loaded:
            logger.debug("Historical data already loaded from DB, skipping.")
            return

        from dataclasses import fields

        def filter_fields(data, cls):
            valid = {f.name for f in fields(cls)}
            return {k: v for k, v in data.items() if k in valid}

        db_trades = self.db.get_trades(limit=999999)
        if db_trades:
            self.state.trades = [Trade(**filter_fields(t, Trade)) for t in db_trades]
            self.state.trades = self.state.trades[-500:]
            logger.info(f"Loaded {len(self.state.trades)} trades from database into state.")

        db_journal = self.db.get_journal(limit=999999)
        if db_journal:
            self.state.journal = [JournalEntry(**filter_fields(j, JournalEntry)) for j in db_journal]
            self.state.journal = self.state.journal[-500:]
            logger.info(f"Loaded {len(self.state.journal)} journal entries from database into state.")

        db_setup = self.db.get_setup_records(limit=999999)
        if db_setup:
            self.state.setup_records = [SetupRecord(**filter_fields(s, SetupRecord)) for s in db_setup]
            self.state.setup_records = self.state.setup_records[-500:]
            logger.info(f"Loaded {len(self.state.setup_records)} setup records from database into state.")

        self.state.historical_loaded = True
        self.save_state()

    # ─── Start/Stop ────────────────────────────────────────────────────

    def start(self) -> None:
        try:
            if self.should_sync_live_balance_on_start():
                self.sync_live_balance_from_coinbase()
        except Exception as exc:
            with self.lock:
                self.state.last_error = f"Start blocked: {exc}"
                self.state.last_signal = "Start blocked by live balance sync"
                self.save_state()
            logger.error(f"Start blocked: {exc}")
            raise

        with self.lock:
            if self.thread and self.thread.is_alive():
                self.state.running = True
                logger.info("Bot already running")
                return

            self.stop_event.clear()
            self.state.running = True

            if self.should_oanda_demo_trade():
                try:
                    summary = self.get_oanda_account_summary()
                    if summary.get("ok"):
                        self.state.cash = summary.get("balance", self.state.cash)
                        self.state.positions = {}
                        self.state.coin = 0.0
                        self.state.active_symbol = None
                        for pos in summary.get("positions", []):
                            symbol = pos.get("symbol")
                            units = pos.get("units", 0)
                            avg_price = pos.get("average_price", 0)
                            is_short = pos.get("side") == "SHORT"
                            if units != 0:
                                self.state.positions[symbol] = {
                                    "quantity": -units if is_short else units,
                                    "entry_price": avg_price,
                                    "highest_price": avg_price,
                                    "is_short": is_short,
                                    "opened_at": now_iso(),
                                    "entry_time": time.time(),
                                }
                                if is_short:
                                    self.state.coin = -units
                                else:
                                    self.state.coin = units
                                self.state.active_symbol = symbol
                        logger.info(f"Synced {len(summary.get('positions', []))} positions from OANDA on start")
                except Exception as e:
                    logger.warning(f"Failed to sync OANDA positions on start: {e}")

            if self.state.settings.get("asset_class") == "crypto" and self.state.settings.get("exchange") == "coinbase":
                self.sync_live_balance_always()

            set_position_rows_bot(self)

            self.thread = threading.Thread(target=self.run_loop, daemon=True)
            self.thread.start()
            self.start_websocket_feed_if_needed()
            self.start_news_guard_thread()
            self.start_oanda_stream()
            logger.info("Bot started with enhanced features")

    def stop(self) -> None:
        with self.lock:
            self.state.running = False
            self.stop_event.set()
            self.websocket_stop_event.set()
            self.news_guard_stop_event.set()
            self.shutdown_requested = True
            self.stop_oanda_stream()
            self.save_state()
            logger.info("Bot stopped")

    # ─── Reset ────────────────────────────────────────────────────────

    def reset(self) -> None:
        with self.lock:
            running = self.state.running
            settings = dict(self.state.settings)
            self.state = BotState(settings=settings)
            self.state.cash = float(settings["starting_cash"])
            self.state.day_start_equity = self.state.cash
            self.state.day_start_date = today_key()
            self.state.peak_equity = self.state.cash
            self.state.running = running
            self.state.news_events = []
            self.state.news_last_update = ""
            self.state.news_guard_status = "idle"
            self.state.opening_range_analysis = {}
            self.state.signal_history = {}
            self.state.db_initialized = True
            self.state.strategy_history = []
            self.save_state()

            self.db = BotDatabase()
            self.self_learning_trader = SelfLearningTrader(self, self.db)
            self.load_historical_data_from_db()
            logger.info("Bot state reset")

    # ─── Strategy Creator Methods ────────────────────────────────────

    def _start_strategy_timer_if_needed(self):
        if not STRATEGY_CREATOR_AVAILABLE:
            return

        if not self.state.settings.get("strategy_creator_enabled", False):
            return

        if not self.state.settings.get("strategy_evolution_enabled", False):
            return

        if not hasattr(self, '_strategy_timer') or not self._strategy_timer.is_alive():
            self._strategy_timer = threading.Thread(target=self._strategy_evolution_loop, daemon=True)
            self._strategy_timer.start()
            logger.info("Strategy evolution timer started")

    def _strategy_evolution_loop(self):
        while self.state.running:
            try:
                frequency = int(self.state.settings.get("strategy_evolution_frequency", 24)) * 3600
                if time.time() - self.last_strategy_evolution > frequency:
                    logger.info("Running scheduled strategy evolution...")
                    self.evolve_strategies(int(self.state.settings.get("strategy_generations", 50)))
                    self.last_strategy_evolution = time.time()
            except Exception as e:
                logger.error(f"Strategy evolution error: {e}")

            time.sleep(3600)

    def evolve_strategies(self, generations: int = 50) -> dict:
        if not STRATEGY_CREATOR_AVAILABLE or not self.strategy_manager:
            return {"ok": False, "error": "Strategy Creator not available"}

        try:
            settings = self.state.settings
            watchlist = parse_watchlist(settings.get("watchlist", settings["symbol"]))
            candles_history = []

            for symbol in watchlist[:3]:
                candles = fetch_candles(
                    exchange=settings["exchange"],
                    symbol=symbol,
                    quote_currency=settings["quote_currency"],
                    granularity=int(settings.get("live_granularity", 3600)),
                    candle_count=300,
                    asset_class=str(settings.get("asset_class", "crypto")),
                )
                if candles:
                    candles_history.append(candles)

            if not candles_history:
                return {"ok": False, "error": "No candle data available"}

            best_strategy = self.strategy_manager.evolve_strategies(candles_history, generations)

            if best_strategy:
                self.state.active_strategy_id = best_strategy.id
                self.state.strategy_history.append({
                    'id': best_strategy.id,
                    'name': best_strategy.name,
                    'fitness': best_strategy.fitness_score,
                    'win_rate': best_strategy.win_rate,
                    'generation': best_strategy.generation
                })
                if len(self.state.strategy_history) > int(settings.get("strategy_max_active", 5)):
                    self.state.strategy_history = self.state.strategy_history[-5:]
                self.save_state()

                return {
                    "ok": True,
                    "strategy": {
                        "id": best_strategy.id,
                        "name": best_strategy.name,
                        "entry_rules": best_strategy.entry_rules,
                        "exit_rule": best_strategy.exit_rule,
                        "parameters": best_strategy.parameters,
                        "win_rate": best_strategy.win_rate,
                        "fitness_score": best_strategy.fitness_score
                    }
                }
            else:
                return {"ok": False, "error": "No strategy evolved"}

        except Exception as e:
            logger.error(f"Error evolving strategies: {e}")
            return {"ok": False, "error": str(e)}

    def get_strategy_dashboard(self) -> dict:
        if not STRATEGY_CREATOR_AVAILABLE or not self.strategy_manager:
            return {"ok": False, "error": "Strategy Creator not available"}

        return {
            "ok": True,
            "summary": self.strategy_manager.get_performance_summary(),
            "active_strategy_id": self.state.active_strategy_id,
            "history": self.state.strategy_history[-10:]
        }

    def apply_strategy_signal(self, candles: list) -> dict:
        if not STRATEGY_CREATOR_AVAILABLE or not self.strategy_manager:
            return {"ok": False, "error": "Strategy Creator not available"}

        if len(candles) < 30:
            return {"ok": False, "error": "Not enough candles"}

        regime = market_regime(candles, self.state.settings)
        market_conditions = {
            'regime': regime['regime'],
            'volatility': regime['volatility_pct'],
            'trend': regime['trend_pct']
        }

        strategy = self.strategy_manager.select_strategy()
        if not strategy:
            return {"ok": False, "error": "No strategy available"}

        result = self.strategy_manager.apply_strategy_to_trade(strategy, candles)
        return {
            "ok": True,
            "strategy_id": strategy.id,
            "strategy": strategy.name,
            "direction": result['direction'],
            "confidence": result['confidence'],
            "signals": result['signals'],
            "parameters": result['parameters'],
            "exit_rule": result['exit_rule']
        }

    # ─── Database-backed Journal ─────────────────────────────────────

    def journal(
        self,
        symbol: str,
        event: str,
        message: str,
        price: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        entry = JournalEntry(
            time=now_iso(),
            symbol=symbol,
            event=event,
            message=message,
            price=price,
            details=details or {},
        )

        with self.lock:
            self.state.journal.append(entry)
            self.state.journal = self.state.journal[-500:]

        self.db.save_journal(entry)

        self.audit(event, symbol=symbol, message=message, price=price, details=details or {})

    def audit(self, event: str, **payload: Any) -> None:
        record = {
            "time": now_iso(),
            "event": event,
            **payload,
        }
        try:
            with AUDIT_LOG_FILE.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
        except OSError as e:
            self.state.last_error = f"Audit log write failed: {e}"
            logger.error(f"Audit log write failed: {e}")

    # ─── Database-backed Trade Methods ──────────────────────────────

    def record_trade(self, trade: Trade, pnl: float = 0.0) -> None:
        with self.lock:
            trade.pnl = pnl
            if trade.entry_price and trade.exit_price and trade.entry_price > 0:
                trade.pnl_pct = pct(trade.exit_price - trade.entry_price, trade.entry_price)

            self.state.trades.append(trade)
            self.state.trades = self.state.trades[-500:]

        self.db.save_trade(trade)
        self.db.update_symbol_performance(trade.symbol)
        self.db.update_performance_metrics(trade.symbol)

    # ─── OANDA SYNC ──────────────────────────────────────────────────

    def sync_oanda_positions(self) -> None:
        if not self.should_oanda_demo_trade():
            with self.lock:
                if self.state.positions:
                    logger.info("OANDA demo trading disabled - clearing synced positions")
                    self.state.positions = {}
                    self.state.active_symbol = None
                    self.state.coin = 0.0
                    self.state.is_short = False
                    self.save_state()
            return

        try:
            account_id = urllib.parse.quote(oanda_account_id())
            data = oanda_request(f"/v3/accounts/{account_id}/openPositions")
            positions = data.get("positions", [])

            with self.lock:
                self.state.positions = {}
                self.state.coin = 0.0
                self.state.active_symbol = None
                self.state.is_short = False

            for position in positions:
                instrument = position.get("instrument", "")
                symbol = instrument.replace("_", "")
                units = int(position.get("short", {}).get("units", 0))
                if units > 0:
                    price = float(position.get("short", {}).get("averagePrice", 0.0))
                    if price <= 0:
                        price = self.state.last_price or 0.0
                    with self.lock:
                        self.state.positions[symbol] = {
                            "quantity": -units,
                            "entry_price": price,
                            "highest_price": price,
                            "opened_at": now_iso(),
                            "trade_id": position.get("tradeID"),
                            "is_short": True,
                            "entry_time": time.time(),
                        }
                        self.state.active_symbol = symbol
                        self.state.entry_price = price
                        self.state.coin = -units
                        self.state.is_short = True
                    self.journal(symbol, "INFO", f"Synced OANDA SHORT position: {symbol} {units} @ {price}", price)
                else:
                    units = int(position.get("long", {}).get("units", 0))
                    if units > 0:
                        price = float(position.get("long", {}).get("averagePrice", 0.0))
                        if price <= 0:
                            price = self.state.last_price or 0.0
                        with self.lock:
                            self.state.positions[symbol] = {
                                "quantity": units,
                                "entry_price": price,
                                "highest_price": price,
                                "opened_at": now_iso(),
                                "trade_id": position.get("tradeID"),
                                "is_short": False,
                                "entry_time": time.time(),
                            }
                            self.state.active_symbol = symbol
                            self.state.entry_price = price
                            self.state.coin = units
                            self.state.is_short = False
                        self.journal(symbol, "INFO", f"Synced OANDA BUY position: {symbol} {units} @ {price}", price)

            if positions:
                logger.info(f"Synced {len(positions)} positions from OANDA")
            else:
                logger.info("No OANDA positions to sync")
            self.save_state()
        except Exception as exc:
            logger.warning(f"Failed to sync OANDA positions: {exc}")

    def get_oanda_account_summary(self) -> dict[str, Any]:
        if not self.should_oanda_demo_trade():
            return {"ok": False, "error": "OANDA demo trading not enabled"}

        try:
            account_id = urllib.parse.quote(oanda_account_id())

            summary = oanda_account_summary()
            account = summary.get("account", {})

            positions_data = oanda_request(f"/v3/accounts/{account_id}/openPositions")
            positions = positions_data.get("positions", [])

            orders_data = oanda_request(f"/v3/accounts/{account_id}/orders")
            orders = orders_data.get("orders", [])

            order_map = {}
            for order in orders:
                instrument = order.get("instrument", "")
                symbol = instrument.replace("_", "")

                order_type = order.get("type", "")
                if order_type == "STOP_LOSS":
                    if symbol not in order_map:
                        order_map[symbol] = {}
                    order_map[symbol]["stop_loss"] = float(order.get("price", 0))
                elif order_type == "TAKE_PROFIT":
                    if symbol not in order_map:
                        order_map[symbol] = {}
                    order_map[symbol]["take_profit"] = float(order.get("price", 0))

            if positions:
                instruments = ",".join([
                    oanda_instrument(pos.get("instrument", "").replace("_", ""))
                    for pos in positions
                ])
                pricing = oanda_request(
                    f"/v3/accounts/{account_id}/pricing",
                    {"instruments": instruments}
                )
                prices = pricing.get("prices", [])
            else:
                prices = []

            position_details = []
            for position in positions:
                instrument = position.get("instrument", "")
                symbol = instrument.replace("_", "")

                current_price = None
                for price_data in prices:
                    if price_data.get("instrument") == instrument:
                        bid = float(price_data.get("bids", [{}])[0].get("price", 0))
                        ask = float(price_data.get("asks", [{}])[0].get("price", 0))
                        current_price = (bid + ask) / 2
                        break

                long_units = int(position.get("long", {}).get("units", 0))
                short_units = int(position.get("short", {}).get("units", 0))
                long_avg = float(position.get("long", {}).get("averagePrice", 0))
                short_avg = float(position.get("short", {}).get("averagePrice", 0))

                order_info = order_map.get(symbol, {})

                if long_units > 0:
                    position_details.append({
                        "symbol": symbol,
                        "units": long_units,
                        "average_price": long_avg,
                        "current_price": current_price,
                        "side": "LONG",
                        "unrealized_pnl": float(position.get("long", {}).get("unrealizedPL", 0)),
                        "stop_loss": order_info.get("stop_loss"),
                        "take_profit": order_info.get("take_profit"),
                    })
                if short_units > 0:
                    position_details.append({
                        "symbol": symbol,
                        "units": short_units,
                        "average_price": short_avg,
                        "current_price": current_price,
                        "side": "SHORT",
                        "unrealized_pnl": float(position.get("short", {}).get("unrealizedPL", 0)),
                        "stop_loss": order_info.get("stop_loss"),
                        "take_profit": order_info.get("take_profit"),
                    })

            return {
                "ok": True,
                "balance": float(account.get("balance", 0)),
                "equity": float(account.get("NAV", 0)),
                "margin_used": float(account.get("marginUsed", 0)),
                "margin_available": float(account.get("marginAvailable", 0)),
                "unrealized_pnl": float(account.get("unrealizedPL", 0)),
                "total_pnl": float(account.get("pl", 0)),
                "currency": account.get("currency", "GBP"),
                "positions": position_details,
                "positions_count": len(position_details),
                "raw": account
            }

        except Exception as e:
            logger.error(f"Failed to get OANDA account summary: {e}")
            return {"ok": False, "error": str(e)}

    def should_oanda_demo_trade(self) -> bool:
        settings = self.state.settings
        return (
            bool(settings.get("oanda_demo_trading_enabled"))
            and settings.get("asset_class") == "forex"
            and settings.get("exchange") == "oanda_demo"
            and oanda_demo_orders_armed()
        )

    def wants_oanda_demo_trade(self) -> bool:
        settings = self.state.settings
        return (
            bool(settings.get("oanda_demo_trading_enabled"))
            and settings.get("asset_class") == "forex"
            and settings.get("exchange") == "oanda_demo"
        )

    # ─── OANDA Streaming ─────────────────────────────────────────────

    def start_oanda_stream(self) -> None:
        if not self.wants_oanda_demo_trade():
            logger.info("OANDA streaming not started: OANDA demo trading not enabled")
            return

        if self.oanda_stream_thread and self.oanda_stream_thread.is_alive():
            logger.info("OANDA stream already running")
            return

        if not self.state.settings.get("websocket_enabled", False):
            with self.lock:
                self.state.websocket_status = "disabled (settings)"
            logger.info("OANDA streaming not started: websocket disabled in settings")
            return

        if not oanda_is_configured():
            with self.lock:
                self.state.websocket_status = "error: OANDA not configured"
            logger.warning("OANDA streaming not started: OANDA not configured")
            return

        settings = self.state.settings
        watchlist = parse_watchlist(settings.get("watchlist", settings["symbol"]))
        max_symbols = min(len(watchlist), 10)
        symbols = watchlist[:max_symbols]

        if not symbols:
            logger.info("OANDA streaming not started: no symbols in watchlist")
            return

        self.oanda_stop_event = threading.Event()

        with self.lock:
            self.state.websocket_status = f"connecting to OANDA ({len(symbols)} symbols)..."
            self.state.websocket_last_seen = now_iso()

        def on_price(data):
            try:
                instrument = data.get("instrument", "")
                symbol = instrument.replace("_", "")
                price_data = data.get("bids", [{}])[0] or data.get("asks", [{}])[0]
                price = float(price_data.get("price", 0.0))

                if price > 0:
                    with self.lock:
                        self.feed_prices[symbol] = price
                        self.state.last_price = price
                        self.state.websocket_last_seen = now_iso()
                        if self.state.websocket_status != "streaming":
                            self.state.websocket_status = "streaming"
                        if symbol in self.state.price_history:
                            history = self.state.price_history[symbol]
                            if history and history[-1] != price:
                                history.append(price)
                                if len(history) > 300:
                                    history.pop(0)
                        else:
                            self.state.price_history[symbol] = [price]
            except Exception as e:
                logger.debug(f"Error processing OANDA price: {e}")

        def on_error(error):
            with self.lock:
                self.state.websocket_status = f"error: {error}"
                self.state.last_error = f"OANDA stream: {error}"
                self.state.websocket_last_seen = now_iso()
            logger.warning(f"OANDA stream error: {error}")

        self.oanda_stream_thread = threading.Thread(
            target=oanda_stream_pricing,
            args=(symbols, on_price, on_error, self.oanda_stop_event),
            daemon=True,
        )
        self.oanda_stream_thread.start()
        logger.info(f"OANDA pricing stream started for {len(symbols)} symbols")

    def stop_oanda_stream(self) -> None:
        if hasattr(self, 'oanda_stop_event') and self.oanda_stop_event:
            self.oanda_stop_event.set()
            logger.info("OANDA stop event set")

        if hasattr(self, 'oanda_stream_thread') and self.oanda_stream_thread:
            try:
                self.oanda_stream_thread.join(timeout=3.0)
            except Exception as e:
                logger.debug(f"Error joining OANDA stream thread: {e}")
            self.oanda_stream_thread = None

        with self.lock:
            self.state.websocket_status = "stopped"
            self.state.websocket_last_seen = now_iso()

        self.feed_prices = {}
        logger.info("OANDA pricing stream stopped")

    def update_streaming_status(self) -> None:
        settings = self.state.settings
        websocket_enabled = settings.get("websocket_enabled", False)
        oanda_demo_enabled = settings.get("oanda_demo_trading_enabled", False)
        asset_class = settings.get("asset_class", "crypto")
        exchange = settings.get("exchange", "coinbase")

        if asset_class == "forex" and exchange == "oanda_demo" and oanda_demo_enabled and websocket_enabled:
            if not oanda_is_configured():
                with self.lock:
                    self.state.websocket_status = "error: OANDA not configured"
                    self.state.websocket_last_seen = now_iso()
                return

            if not self.oanda_stream_thread or not self.oanda_stream_thread.is_alive():
                self.start_oanda_stream()
                logger.info("OANDA streaming started due to settings change")
        else:
            if self.oanda_stream_thread and self.oanda_stream_thread.is_alive():
                self.stop_oanda_stream()
                with self.lock:
                    if not oanda_demo_enabled:
                        self.state.websocket_status = "disabled (OANDA demo trading off)"
                    elif not websocket_enabled:
                        self.state.websocket_status = "disabled (websocket off)"
                    elif asset_class != "forex" or exchange != "oanda_demo":
                        self.state.websocket_status = "disabled (not OANDA forex)"
                    self.state.websocket_last_seen = now_iso()
                logger.info("OANDA streaming stopped due to settings change")

        if asset_class == "crypto" and exchange == "coinbase" and websocket_enabled and WEBSOCKET_AVAILABLE:
            if not self.websocket_thread or not self.websocket_thread.is_alive():
                self.start_websocket_feed_if_needed()
                logger.info("Coinbase WebSocket started due to settings change")
        else:
            if self.websocket_thread and self.websocket_thread.is_alive():
                self.websocket_stop_event.set()
                self.websocket_thread = None
                with self.lock:
                    if asset_class != "crypto":
                        self.state.websocket_status = "disabled (crypto only)"
                    elif exchange != "coinbase":
                        self.state.websocket_status = "disabled (Coinbase only)"
                    elif not websocket_enabled:
                        self.state.websocket_status = "disabled (websocket off)"
                    elif not WEBSOCKET_AVAILABLE:
                        self.state.websocket_status = "disabled (websocket-client not installed)"
                    self.state.websocket_last_seen = now_iso()
                logger.info("Coinbase WebSocket stopped due to settings change")

        self.save_state()

    # ─── WebSocket Feed ──────────────────────────────────────────────

    def start_websocket_feed_if_needed(self) -> None:
        if self.state.settings.get("asset_class") != "crypto" or self.state.settings.get("exchange") != "coinbase":
            self.state.websocket_status = "crypto websocket only"
            return
        if not self.state.settings.get("websocket_enabled"):
            self.state.websocket_status = "disabled"
            return
        if not WEBSOCKET_AVAILABLE:
            self.state.websocket_status = "websocket-client package not installed"
            return
        if self.websocket_thread and self.websocket_thread.is_alive():
            return
        self.websocket_stop_event.clear()
        self.websocket_thread = threading.Thread(target=self.websocket_loop, daemon=True)
        self.websocket_thread.start()
        logger.info("WebSocket feed started")

    def websocket_loop(self) -> None:
        while not self.websocket_stop_event.is_set():
            ws = None
            user_ws = None
            try:
                with self.lock:
                    settings = dict(self.state.settings)
                    product_ids = [
                        f"{symbol}-{settings['quote_currency']}"
                        for symbol in parse_watchlist(settings.get("watchlist", settings["symbol"]))
                    ]
                    self.state.websocket_status = "connecting"

                ws = websocket.create_connection(
                    "wss://advanced-trade-ws.coinbase.com",
                    timeout=15,
                    sslopt={"cert_reqs": ssl.CERT_REQUIRED},
                )
                subscribe = {
                    "type": "subscribe",
                    "channel": "ticker",
                    "product_ids": product_ids,
                }
                if coinbase_live_is_armed():
                    subscribe["jwt"] = coinbase_ws_jwt()
                ws.send(json.dumps(subscribe))

                if coinbase_live_is_armed():
                    try:
                        user_ws = websocket.create_connection(
                            "wss://advanced-trade-ws-user.coinbase.com",
                            timeout=15,
                            sslopt={"cert_reqs": ssl.CERT_REQUIRED},
                        )
                        user_ws.send(json.dumps({
                            "type": "subscribe",
                            "channel": "user",
                            "product_ids": product_ids,
                            "jwt": coinbase_ws_jwt(),
                        }))
                    except Exception as exc:
                        self.audit("WEBSOCKET_USER_CONNECT_FAILED", error=str(exc))
                        logger.warning(f"User WebSocket connection failed: {exc}")

                with self.lock:
                    self.state.websocket_status = "connected"
                    self.state.websocket_last_seen = now_iso()
                logger.info("WebSocket connected")

                while not self.websocket_stop_event.is_set():
                    raw = ws.recv()
                    self.handle_websocket_message(raw)
                    if user_ws:
                        try:
                            user_ws.settimeout(0.01)
                            user_raw = user_ws.recv()
                            self.handle_websocket_message(user_raw, user_stream=True)
                        except Exception:
                            pass
            except Exception as exc:
                with self.lock:
                    self.state.websocket_status = f"error: {exc}"
                    self.state.websocket_last_message = str(exc)
                self.audit("WEBSOCKET_ERROR", error=str(exc))
                logger.warning(f"WebSocket error: {exc}")
                self.websocket_stop_event.wait(10)
            finally:
                try:
                    if ws:
                        ws.close()
                except Exception:
                    pass
                try:
                    if user_ws:
                        user_ws.close()
                except Exception:
                    pass

    def handle_websocket_message(self, raw: str, user_stream: bool = False) -> None:
        data = json.loads(raw)
        with self.lock:
            self.state.websocket_last_seen = now_iso()
            self.state.websocket_last_message = str(data.get("channel") or data.get("type") or "")[:120]
            for event in data.get("events", []):
                for ticker in event.get("tickers", []):
                    product_id = ticker.get("product_id", "")
                    price = ticker.get("price")
                    if product_id and price:
                        self.feed_prices[product_id] = float(price)
                if user_stream:
                    self.audit("WEBSOCKET_USER_EVENT", event=event)

    # ─── News Guard ──────────────────────────────────────────────────

    def start_news_guard_thread(self) -> None:
        if not REQUESTS_AVAILABLE:
            self.state.news_guard_status = "requests package not installed"
            return
        if self.news_guard_thread and self.news_guard_thread.is_alive():
            return

        self.news_guard_stop_event.clear()
        self.news_guard_thread = threading.Thread(target=self.news_guard_loop, daemon=True)
        self.news_guard_thread.start()
        self.state.news_guard_status = "running"
        logger.info("News guard thread started")

    def news_guard_loop(self) -> None:
        while not self.news_guard_stop_event.is_set():
            try:
                self.update_news_events()
            except Exception as exc:
                with self.lock:
                    self.state.news_guard_status = f"error: {exc}"
                    self.state.last_error = f"News guard: {exc}"
                logger.warning(f"News guard error: {exc}")
            self.news_guard_stop_event.wait(900)

    def update_news_events(self) -> None:
        if not REQUESTS_AVAILABLE:
            return

        try:
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            logger.debug(f"News feed fetched: {len(data)} events")
        except Exception as exc:
            with self.lock:
                self.state.news_guard_status = f"error: {exc}"
                self.state.last_error = f"News guard: {exc}"
            logger.warning(f"News feed fetch error: {exc}")
            return

        events = []
        for item in data:
            impact = str(item.get("impact", "Low")).lower()
            if impact not in ("high", "medium", "low"):
                impact = "low"

            dt_str = item.get("date", "")
            if not dt_str:
                continue

            try:
                dt = datetime.fromisoformat(dt_str)
                dt = dt.astimezone(timezone.utc)
            except (ValueError, TypeError):
                continue

            country = str(item.get("country", ""))

            events.append({
                "time": dt.timestamp(),
                "datetime": dt.isoformat(),
                "country": country,
                "currency": country,
                "indicator": str(item.get("title", "Unknown")),
                "impact": impact,
                "forecast": item.get("forecast"),
                "previous": item.get("previous"),
                "actual": item.get("actual"),
            })

        with self.lock:
            self.state.news_events = events
            self.state.news_last_update = now_iso()
            self.state.news_guard_status = f"updated {len(events)} events"

    def is_news_blocked(self, symbol: str, settings: dict[str, Any]) -> tuple[bool, str]:
        if not settings.get("news_guard_enabled", False):
            return False, "news guard disabled"

        currency = symbol_to_currency(symbol, settings.get("asset_class", "crypto"))

        before_min = int(settings.get("news_guard_before_minutes", 30))
        after_min = int(settings.get("news_guard_after_minutes", 30))
        block_high = bool(settings.get("news_guard_block_high", True))
        block_medium = bool(settings.get("news_guard_block_medium", False))
        block_low = bool(settings.get("news_guard_block_low", False))

        now = time.time()
        for event in self.state.news_events:
            event_time = float(event.get("time", 0))
            if event_time == 0:
                continue

            if currency and event.get("currency") != currency:
                continue

            impact = event.get("impact", "low")
            if impact == "high" and not block_high:
                continue
            if impact == "medium" and not block_medium:
                continue
            if impact == "low" and not block_low:
                continue

            if now >= event_time - (before_min * 60) and now <= event_time + (after_min * 60):
                return True, f"news: {event['indicator']} ({impact})"

        return False, ""

    def news_guard_status(self) -> dict[str, Any]:
        with self.lock:
            enabled = bool(self.state.settings.get("news_guard_enabled", False))
            if not enabled:
                return {"enabled": False, "status": "disabled"}

            events = self.state.news_events
            now = time.time()
            upcoming = [e for e in events if e.get("time", 0) > now and e.get("time", 0) < now + 14400]

            return {
                "enabled": True,
                "status": self.state.news_guard_status,
                "events_cached": len(events),
                "upcoming_4h": len(upcoming),
                "last_update": self.state.news_last_update,
            }

    # ─── Telegram Alerts ─────────────────────────────────────────────

    def send_telegram_alert(self, message: str, parse_mode: str = "HTML") -> bool:
        settings = self.state.settings

        def get_val(keys: list[str], default=None):
            for key in keys:
                if key in settings and settings[key]:
                    return settings[key]
            return default

        enabled = get_val(["telegram_enabled", "TELEGRAM_ENABLED"], False)
        if not enabled:
            return False

        bot_token = get_val(["telegram_bot_token", "TELEGRAM_BOT_TOKEN"], "")
        chat_id = get_val(["telegram_chat_id", "TELEGRAM_CHAT_ID"], "")

        if not bot_token or not chat_id:
            logger.warning("Telegram enabled but missing bot_token or chat_id")
            return False

        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            }
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.debug(f"Telegram alert sent: {message[:50]}...")
                return True
            else:
                logger.warning(f"Telegram alert failed: {response.text}")
                return False
        except Exception as exc:
            logger.warning(f"Telegram alert error: {exc}")
            return False

    def format_alert_trade(self, trade: Trade, pnl: float | None = None) -> str:
        settings = self.state.settings
        currency = settings.get("quote_currency", "USD")
        emoji = "🟢" if trade.side == "BUY" else ("🔴" if trade.side == "SELL" else "🟣")
        pnl_text = ""
        if pnl is not None:
            pnl_emoji = "✅" if pnl >= 0 else "❌"
            pnl_text = f"\n💰 P/L: {pnl_emoji} {currency} {pnl:.2f}"
        return f"""
{emoji} <b>{trade.side}</b> {trade.symbol}
📊 Price: {trade.price:.6f}
📦 Qty: {trade.quantity:.6f}
💵 Cash: {currency} {trade.cash_after:.2f}
📝 Reason: {trade.reason}{pnl_text}
🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    def format_alert_error(self, error: str) -> str:
        return f"""
⚠️ <b>AUXO ALERT</b>

<b>Error:</b> {error}
🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    def format_alert_drawdown(self, drawdown_pct: float, equity: float, peak_equity: float) -> str:
        settings = self.state.settings
        currency = settings.get("quote_currency", "USD")
        return f"""
🔻 <b>DRAWDOWN ALERT</b>

<b>Current Drawdown:</b> {drawdown_pct:.1f}%
<b>Current Equity:</b> {currency} {equity:.2f}
<b>Peak Equity:</b> {currency} {peak_equity:.2f}
🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    def format_daily_summary(self) -> str:
        settings = self.state.settings
        currency = settings.get("quote_currency", "USD")
        equity = self.equity(self.state.last_price)
        day_pnl = equity - self.state.day_start_equity
        trades_today = [
            t for t in self.state.trades
            if t.time.startswith(today_key())
        ]
        buys = len([t for t in trades_today if t.side == "BUY"])
        sells = len([t for t in trades_today if t.side == "SELL"])
        return f"""
📊 <b>DAILY SUMMARY</b>

<b>Date:</b> {today_key()}
<b>Equity:</b> {currency} {equity:.2f}
<b>Day P/L:</b> {day_pnl:+.2f} ({pct(day_pnl, self.state.day_start_equity):+.2f}%)
<b>Trades:</b> {len(trades_today)} ({buys} buys, {sells} sells)
<b>Open Positions:</b> {len(self.state.positions)}
<b>Signal:</b> {self.state.last_signal}
"""

    # ─── Equity and Position Management ─────────────────────────────

    def equity(self, price: float | None) -> float:
        with self.lock:
            if self.should_oanda_demo_trade():
                try:
                    summary = self.get_oanda_account_summary()
                    if summary.get("ok"):
                        return summary.get("equity", self.state.cash)
                except Exception as e:
                    logger.warning(f"Failed to get OANDA equity: {e}")
                    return self._calculate_equity_local(price)

            return self._calculate_equity_local(price)

    def _calculate_equity_local(self, price: float | None) -> float:
        with self.lock:
            if not price:
                return self.state.cash

        total = self.state.cash

        for symbol, position in self.state.positions.items():
            quantity = float(position.get("quantity", 0.0))
            history = self.state.price_history.get(symbol, [])
            current_price = history[-1] if history else price
            total += quantity * current_price

        return round(total, 2)

    def price_for_active_position(self, fetched_prices: dict[str, float]) -> float:
        with self.lock:
            if self.state.active_symbol and self.state.active_symbol in fetched_prices:
                return fetched_prices[self.state.active_symbol]
        first_price = next(iter(fetched_prices.values()))
        return first_price

    def roll_daily_equity_if_needed(self, price: float) -> None:
        with self.lock:
            current_day = today_key()
            if self.state.day_start_date != current_day:
                self.state.day_start_date = current_day
                self.state.day_start_equity = self.equity(price)
                self.state.peak_equity = self.state.day_start_equity

    def roll_live_daily_spend_if_needed(self) -> None:
        with self.lock:
            current_day = today_key()
            if self.state.live_day_start_date != current_day:
                self.state.live_day_start_date = current_day
                self.state.live_daily_spend = 0.0

    # ─── Live Balance Sync ───────────────────────────────────────────

    def should_sync_live_balance_on_start(self) -> bool:
        with self.lock:
            settings = dict(self.state.settings)
            current_coin = self.state.coin
            active_symbol = self.state.active_symbol

        if current_coin != 0:
            with self.lock:
                self.state.last_signal = (
                    f"Start preserving open {active_symbol or ''} position; "
                    "skipped Coinbase cash sync"
                )
                self.journal(
                    active_symbol or "",
                    "INFO",
                    "Skipped live balance sync on start because an open position exists.",
                    self.state.last_price,
                )
                self.save_state()
            return False

        return (
            bool(settings.get("live_trading_enabled"))
            and settings.get("asset_class", "crypto") == "crypto"
            and settings.get("exchange") == "coinbase"
            and coinbase_live_is_armed()
        )

    def coinbase_live_account_snapshot(self) -> dict[str, Any]:
        """Value the real Coinbase brokerage account in the selected quote currency.

        This is authoritative for LIVE Coinbase Cash/Equity/Total P&L.  Local
        positions remain the strategy/trade ledger, but they no longer determine
        how much money actually exists at the exchange.
        """
        with self.lock:
            quote = str(self.state.settings.get("quote_currency", "GBP")).upper()
            local_prices = {
                str(symbol).upper(): float(history[-1])
                for symbol, history in self.state.price_history.items()
                if history
            }

        balances = coinbase_account_balances()
        quote_row = balances.get(quote, {})
        available_cash = float(quote_row.get("available", 0.0) or 0.0)
        quote_total = float(quote_row.get("total", available_cash) or 0.0)
        holdings: list[dict[str, Any]] = []
        holdings_value = 0.0
        unpriced: list[str] = []

        for currency, row in balances.items():
            if currency == quote:
                continue
            quantity = float(row.get("total", 0.0) or 0.0)
            if quantity <= 0:
                continue

            price = local_prices.get(currency)
            if price is None:
                try:
                    ticker = fetch_coinbase_ticker(currency, quote)
                    price = float(ticker.get("price", 0.0) or 0.0)
                except Exception:
                    # Some tiny/reward assets do not have a direct market in the
                    # selected quote currency.  Do not invent a value for them.
                    unpriced.append(currency)
                    continue

            value = quantity * price
            # Ignore only truly negligible dust while retaining real holdings.
            if value < 0.000001:
                continue
            holdings_value += value
            holdings.append({
                "currency": currency,
                "quantity": quantity,
                "price": price,
                "value": value,
            })

        return {
            "ok": True,
            "quote_currency": quote,
            "available_cash": available_cash,
            "quote_total": quote_total,
            "holdings_value": holdings_value,
            "equity": quote_total + holdings_value,
            "holdings": holdings,
            "unpriced": unpriced,
            "time": now_iso(),
        }

    def sync_live_balance_always(self) -> None:
        """Sync authoritative Coinbase cash/equity even if positions are open."""
        try:
            with self.lock:
                settings = dict(self.state.settings)

            if not settings.get("live_trading_enabled"):
                logger.debug("Live trading not enabled, skipping balance sync")
                return

            # This synchroniser is Coinbase-specific. Do not overwrite cash for
            # another live exchange/account type.
            if settings.get("asset_class", "crypto") != "crypto" or settings.get("exchange") != "coinbase":
                logger.debug("Not a live Coinbase crypto account, skipping Coinbase balance sync")
                return

            quote_currency = settings.get("quote_currency", "GBP")
            account_snapshot = self.coinbase_live_account_snapshot()
            actual_balance = float(account_snapshot.get("available_cash", 0.0))

            # Zero is a valid available balance (for example after deploying all
            # quote cash into a position), so it must be synchronised rather than
            # treated as an error. Only reject an impossible negative value.
            if actual_balance < 0:
                logger.warning(f"Coinbase balance is invalid: {actual_balance}")
                return

            with self.lock:
                # IMPORTANT: starting_cash is the immutable account baseline.
                # Automatic exchange balance sync must only update live available cash;
                # otherwise every trade silently moves the P/L baseline.
                self.state.cash = actual_balance
                self._coinbase_account_snapshot = account_snapshot

                current_day = today_key()
                if self.state.day_start_date != current_day:
                    self.state.day_start_equity = self.equity(self.state.last_price)
                    self.state.day_start_date = current_day
                    self.state.peak_equity = self.state.day_start_equity

                self.save_state()
                logger.info(f"✅ Synced Coinbase balance: £{actual_balance:.2f}")

        except Exception as e:
            logger.warning(f"Failed to sync Coinbase balance: {e}")

    def sync_live_balance_from_coinbase(self) -> dict[str, Any]:
        with self.lock:
            settings = dict(self.state.settings)
            current_coin = self.state.coin

        if settings.get("exchange") != "coinbase":
            raise RuntimeError("Live balance sync only supports Coinbase.")
        if not coinbase_live_is_armed():
            raise RuntimeError(coinbase_live_status_message())
        if current_coin != 0:
            raise RuntimeError(
                "Refusing to sync starting cash while the bot has an open paper/live position. "
                "Sell or reset first."
            )

        quote_currency = str(settings["quote_currency"]).upper()
        available_cash = coinbase_available_balance(quote_currency)

        with self.lock:
            # A balance sync updates available cash only. The starting-cash baseline
            # must remain unchanged unless the user explicitly edits that setting.
            self.state.cash = available_cash
            self.state.coin = 0.0
            self.state.active_symbol = None
            self.state.entry_price = None
            self.state.active_stop_order_id = None
            self.state.day_start_equity = available_cash
            self.state.day_start_date = today_key()
            self.state.peak_equity = available_cash
            self.state.last_signal = f"Synced {quote_currency} balance from Coinbase"
            self.save_state()

        logger.info(f"Synced Coinbase balance: {available_cash:.2f} {quote_currency}")
        return {
            "ok": True,
            "quote_currency": quote_currency,
            "available_cash": round(available_cash, 8),
        }

    def sync_paper_balance_from_oanda(self) -> dict[str, Any]:
        with self.lock:
            settings = dict(self.state.settings)

        if settings.get("asset_class") != "forex" or settings.get("exchange") != "oanda_demo":
            raise RuntimeError("OANDA balance sync requires Asset Class = Forex and Exchange = OANDA demo.")

        summary = oanda_account_summary()
        account = summary.get("account", {})
        balance = float(account.get("balance", 0.0))
        currency = str(account.get("currency") or settings.get("quote_currency", "USD")).upper()

        account_id = urllib.parse.quote(oanda_account_id())
        positions_data = oanda_request(f"/v3/accounts/{account_id}/openPositions")
        oanda_positions = positions_data.get("positions", [])

        with self.lock:
            self.state.positions = {}
            self.state.coin = 0.0
            self.state.active_symbol = None
            self.state.is_short = False
            self.state.entry_price = None
            self.state.highest_price = None
            self.state.stop_price = None
            self.state.target_price = None
            self.state.active_stop_order_id = None
            self.state.partial_take_profit_done = False

            # Keep the configured starting-cash baseline fixed; OANDA sync only
            # updates the current account cash balance.
            self.state.cash = balance
            self.state.settings["quote_currency"] = currency
            self.state.day_start_equity = balance
            self.state.peak_equity = balance

            for position in oanda_positions:
                instrument = position.get("instrument", "")
                symbol = instrument.replace("_", "")

                short_units = int(position.get("short", {}).get("units", 0))
                long_units = int(position.get("long", {}).get("units", 0))

                if short_units > 0:
                    avg_price = float(position.get("short", {}).get("averagePrice", 0.0))
                    self.state.positions[symbol] = {
                        "quantity": -short_units,
                        "entry_price": avg_price,
                        "highest_price": avg_price,
                        "is_short": True,
                        "opened_at": now_iso(),
                        "entry_time": time.time(),
                    }
                    self.state.coin = -short_units
                    self.state.active_symbol = symbol
                    self.state.is_short = True
                    self.state.entry_price = avg_price

                elif long_units > 0:
                    avg_price = float(position.get("long", {}).get("averagePrice", 0.0))
                    self.state.positions[symbol] = {
                        "quantity": long_units,
                        "entry_price": avg_price,
                        "highest_price": avg_price,
                        "is_short": False,
                        "opened_at": now_iso(),
                        "entry_time": time.time(),
                    }
                    self.state.coin = long_units
                    self.state.active_symbol = symbol
                    self.state.is_short = False
                    self.state.entry_price = avg_price

            self.state.last_signal = f"Synced {currency} balance from OANDA: {balance:.2f}"
            self.journal("", "INFO", self.state.last_signal, self.state.last_price)
            self.save_state()

        logger.info(f"Synced OANDA: balance={balance:.2f} {currency}, positions={len(oanda_positions)}")

        return {
            "ok": True,
            "quote_currency": currency,
            "available_cash": round(balance, 8),
            "positions": len(oanda_positions),
            "balance": balance
        }

    def verify_oanda_positions(self) -> dict[str, Any]:
        if not self.should_oanda_demo_trade():
            return {"ok": True, "message": "OANDA demo trading not enabled", "local_positions": len(self.state.positions), "oanda_positions": 0}

        try:
            account_id = urllib.parse.quote(oanda_account_id())
            data = oanda_request(f"/v3/accounts/{account_id}/openPositions")
            oanda_positions = data.get("positions", [])

            oanda_symbols = set()
            for pos in oanda_positions:
                instrument = pos.get("instrument", "")
                symbol = instrument.replace("_", "")
                units = int(pos.get("short", {}).get("units", 0)) or int(pos.get("long", {}).get("units", 0))
                if units != 0:
                    oanda_symbols.add(symbol)

            with self.lock:
                local_symbols = set(self.state.positions.keys())

            mismatches = []
            for sym in local_symbols:
                if sym not in oanda_symbols:
                    mismatches.append(f"{sym}: in local but not OANDA")

            for sym in oanda_symbols:
                if sym not in local_symbols:
                    mismatches.append(f"{sym}: in OANDA but not local")

            return {
                "ok": len(mismatches) == 0,
                "local_positions": len(local_symbols),
                "oanda_positions": len(oanda_symbols),
                "local_symbols": list(local_symbols),
                "oanda_symbols": list(oanda_symbols),
                "mismatches": mismatches,
                "message": "In sync" if len(mismatches) == 0 else f"Mismatches: {', '.join(mismatches)}"
            }

        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # ─── Manual Close Position ──────────────────────────────────────

    def close_position_manual(self, symbol: str, mode: str = "profit_only") -> dict[str, Any]:
        symbol = normalize_forex_symbol(symbol or "").upper()
        mode = str(mode or "profit_only").lower()

        if mode not in {"profit_only", "force"}:
            raise RuntimeError("Invalid close mode.")

        with self.lock:
            settings = dict(self.state.settings)
            position = dict((self.state.positions or {}).get(symbol, {}))
            single_active = self.state.active_symbol == symbol and abs(self.state.coin or 0.0) > 0

        if not position and not single_active:
            raise RuntimeError(f"No open position found for {symbol}.")

        candles = fetch_candles(
            exchange=settings["exchange"],
            symbol=symbol,
            quote_currency=settings["quote_currency"],
            granularity=int(settings.get("live_granularity", 3600)),
            candle_count=max(40, int(settings.get("live_candle_count", 300))),
            asset_class=str(settings.get("asset_class", "crypto")),
        )

        if not candles:
            raise RuntimeError(f"No candle data returned for {symbol}.")

        price = float(candles[-1].close)

        if position:
            entry_price = float(position.get("entry_price", 0.0) or 0.0)
            quantity = float(position.get("quantity", 0.0) or 0.0)
            is_short = bool(position.get("is_short", False))
        else:
            entry_price = float(self.state.entry_price or 0.0)
            quantity = float(self.state.coin or 0.0)
            is_short = self.state.is_short

        if entry_price <= 0 or quantity == 0:
            raise RuntimeError(f"Invalid open position state for {symbol}.")

        if is_short:
            pnl = (entry_price - price) * abs(quantity)
        else:
            pnl = (price - entry_price) * abs(quantity)

        if mode == "profit_only" and pnl <= 0:
            raise RuntimeError(
                f"{symbol} is not currently in profit. Current estimated P/L: {pnl:.4f}"
            )

        reason = (
            "Manual profit close before target"
            if mode == "profit_only"
            else "Manual force close"
        )

        if self.should_oanda_demo_trade():
            try:
                if is_short:
                    units = int(abs(quantity))
                else:
                    units = -int(abs(quantity))

                response = oanda_market_order(symbol, units)
                fill = oanda_order_fill(response)
                fill_price = fill["price"] or price
                filled_units = fill["units"] or abs(units)
                fee = fill["commission"]

                logger.info(f"OANDA position closed: {symbol} {filled_units} units @ {fill_price:.6f}")

                with self.lock:
                    self.state.positions.pop(symbol, None)
                    self.state.active_symbol = None
                    self.state.coin = 0.0
                    self.state.is_short = False
                    self.state.entry_price = None
                    self.state.highest_price = None
                    self.state.stop_price = None
                    self.state.target_price = None
                    self.state.active_stop_order_id = None
                    self.state.partial_take_profit_done = False
                    self.state.last_price = fill_price
                    self.state.last_action_time = time.time()
                    self.state.last_signal = f"{reason}: {symbol}"
                    self.save_state()

                trade = Trade(
                    time=now_iso(),
                    side="SELL" if not is_short else "BUY",
                    symbol=symbol,
                    price=fill_price,
                    quantity=filled_units,
                    cash_after=self.state.cash,
                    coin_after=0.0,
                    reason=f"{reason} | OANDA manual close",
                    fee_paid=fee,
                    exchange_order_id=fill["order_id"],
                    exchange_order_status=fill["status"],
                    exchange_average_filled_price=fill_price,
                    exchange_filled_size=filled_units,
                    pnl=pnl,
                    entry_price=entry_price,
                    exit_price=fill_price,
                    exit_reason=reason,
                    regime=self.state.current_regime.regime if self.state.current_regime else None,
                )
                self.record_trade(trade)

                self.journal(symbol, "INFO", f"OANDA manual close: {reason} at {fill_price:.6f}", fill_price, {"pnl": pnl})

                return {
                    "ok": True,
                    "symbol": symbol,
                    "mode": mode,
                    "price": fill_price,
                    "estimated_pnl": pnl,
                    "message": f"OANDA position closed at {fill_price:.6f}",
                    "exchange": "OANDA"
                }

            except Exception as exc:
                logger.error(f"OANDA close failed: {exc}")
                raise RuntimeError(f"OANDA close failed: {exc}")

        elif self.should_live_trade() and settings.get("exchange") == "coinbase":
            try:
                product_id = f"{symbol}-{settings['quote_currency']}"

                if is_short:
                    side = "BUY"
                    base_size = abs(quantity)
                else:
                    side = "SELL"
                    base_size = abs(quantity)

                base_size = self.coinbase_round_size(base_size, product_id)
                if base_size <= 0:
                    raise RuntimeError(f"Invalid base size after rounding: {base_size}")

                order = coinbase_market_order(
                    product_id=product_id,
                    side=side,
                    base_size=base_size,
                )
                order_id = coinbase_order_id(order)

                fill = coinbase_reconcile_order(order_id)
                if fill["filled_size"] <= 0:
                    raise RuntimeError(f"Order {order_id} was not filled.")

                filled_price = fill["average_price"] or price
                filled_size = fill["filled_size"]
                fee = fill["total_fee"]

                logger.info(f"Coinbase position closed: {symbol} {filled_size} @ {filled_price:.6f} (order {order_id})")

                with self.lock:
                    self.state.positions.pop(symbol, None)
                    self.state.active_symbol = None
                    self.state.coin = 0.0
                    self.state.is_short = False
                    self.state.entry_price = None
                    self.state.highest_price = None
                    self.state.stop_price = None
                    self.state.target_price = None
                    self.state.active_stop_order_id = None
                    self.state.partial_take_profit_done = False
                    self.state.last_price = filled_price
                    self.state.last_action_time = time.time()
                    self.state.last_signal = f"{reason}: {symbol}"
                    self.save_state()

                trade = Trade(
                    time=now_iso(),
                    side=side,
                    symbol=symbol,
                    price=filled_price,
                    quantity=filled_size,
                    cash_after=self.state.cash,
                    coin_after=0.0,
                    reason=f"{reason} | Coinbase manual close",
                    fee_paid=fee,
                    exchange_order_id=order_id,
                    exchange_order_status=fill["status"],
                    exchange_average_filled_price=filled_price,
                    exchange_filled_size=filled_size,
                    pnl=pnl,
                    entry_price=entry_price,
                    exit_price=filled_price,
                    exit_reason=reason,
                    regime=self.state.current_regime.regime if self.state.current_regime else None,
                )
                self.record_trade(trade)

                self.journal(symbol, "INFO", f"Coinbase manual close: {reason} at {filled_price:.6f}", filled_price, {"pnl": pnl})

                return {
                    "ok": True,
                    "symbol": symbol,
                    "mode": mode,
                    "price": filled_price,
                    "estimated_pnl": pnl,
                    "message": f"Coinbase position closed at {filled_price:.6f} (order {order_id})",
                    "exchange": "Coinbase",
                    "order_id": order_id,
                }

            except Exception as exc:
                logger.error(f"Coinbase close failed: {exc}")
                raise RuntimeError(f"Coinbase close failed: {exc}")

        else:
            if is_short:
                self.paper_buy(symbol, price, reason, None, is_short=True)
            else:
                self.paper_sell(symbol, price, reason, None)

            with self.lock:
                self.state.last_signal = f"{reason}: {symbol}"
                self.save_state()

            logger.info(f"Paper close: {symbol} {mode} at {price:.6f}")

            return {
                "ok": True,
                "symbol": symbol,
                "mode": mode,
                "price": price,
                "estimated_pnl": pnl,
                "message": f"Paper position closed at {price:.6f}",
                "exchange": "Paper"
            }

    # ─── Update Settings ─────────────────────────────────────────────

    def update_settings(self, updates: dict[str, Any]) -> None:
        numeric_fields = {
            "starting_cash", "trade_fee", "poll_seconds", "live_granularity", "live_candle_count",
            "short_window", "long_window", "base_nb_candles_buy", "base_nb_candles_sell",
            "low_offset", "low_offset_2", "high_offset", "high_offset_2",
            "ewo_high", "ewo_high_2", "ewo_low", "rsi_buy",
            "max_position_pct", "risk_per_trade_pct", "min_order_value",
            "stop_loss_pct", "take_profit_pct", "daily_loss_limit_pct", "cooldown_seconds",
            "sr_lookback_candles", "near_support_pct", "min_resistance_distance_pct",
            "min_sr_range_pct", "min_reward_risk", "support_stop_buffer_pct",
            "resistance_target_buffer_pct", "partial_take_profit_pct",
            "partial_take_profit_at_target_pct", "trailing_stop_pct", "trailing_activation_pct",
            "live_limit_offset_pct", "max_live_order_gbp", "max_daily_live_loss_gbp", "max_daily_live_spend_quote",
            "max_live_spread_pct", "min_live_quote_volume", "backtest_slippage_pct",
            "sr_zone_tolerance_pct", "sr_min_touches", "weak_pair_min_trades",
            "weak_pair_expectancy_limit_pct", "weak_pair_win_rate_limit_pct",
            "order_expiry_seconds", "order_retry_limit", "max_oanda_open_trades","max_coinbase_open_trades",
            "news_guard_before_minutes", "news_guard_after_minutes",
            "opening_range_minutes", "opening_range_atr_period",
            "opening_range_manipulation_threshold", "opening_range_stop_loss_atr_multiplier",
            "opening_range_take_profit_atr_multiplier", "max_drawdown_pct",
            "telegram_drawdown_alert_pct", "ema_short", "ema_long",
            "signal_confidence_threshold", "min_signals_required", "learning_history_size",
            "strategy_creator_enabled", "strategy_evolution_enabled", "strategy_generations",
            "strategy_population_size", "strategy_confidence_threshold", "strategy_auto_select",
            "strategy_evolution_frequency", "strategy_max_active",
            "kelly_fraction", "atr_period", "atr_multiplier", "max_hold_hours",
            "rsi_oversold", "rsi_overbought", "ma_exit_period",
            "min_regime_confidence",
        }
        bool_fields = {
            "live_trading_enabled", "use_sr_filter", "use_dynamic_sr_exits",
            "partial_take_profit_enabled", "trailing_stop_enabled", "native_stop_enabled",
            "auto_disable_weak_pairs", "regime_filter_enabled",
            "allow_trending_regime", "allow_ranging_regime", "allow_volatile_regime",
            "allow_dead_regime", "order_replace_enabled", "websocket_enabled",
            "closed_candle_only", "oanda_demo_trading_enabled", "news_guard_enabled",
            "news_guard_block_high", "news_guard_block_medium", "news_guard_block_low",
            "telegram_enabled", "telegram_alert_on_buy", "telegram_alert_on_sell",
            "telegram_alert_on_error", "telegram_alert_on_daily_summary",
            "telegram_alert_on_drawdown", "allow_short_selling", "self_learning_enabled",
            "strategy_creator_enabled", "strategy_evolution_enabled", "strategy_auto_select",
            "regime_adaptation_enabled", "strategy_switching_enabled",
        }
        text_fields = {
            "asset_class", "exchange", "symbol", "quote_currency", "strategy",
            "position_sizing_mode", "live_order_type", "risk_sizing_mode",
        }
        sensitive_fields = {
            "telegram_bot_token", "telegram_chat_id", "oanda_account_type",
        }
        lower_text_fields = {"chart_mode"}
        list_fields = {"watchlist"}
        dict_fields = {"exit_strategies_enabled"}

        with self.lock:
            for key, value in updates.items():
                if key in numeric_fields:
                    if key in ["strategy_creator_enabled", "strategy_evolution_enabled", "strategy_auto_select"]:
                        self.state.settings[key] = value in (True, "true", "on", "1", "yes")
                    else:
                        self.state.settings[key] = float(value)
                elif key in text_fields:
                    self.state.settings[key] = str(value).strip().upper()
                elif key in sensitive_fields:
                    self.state.settings[key] = str(value).strip()
                elif key in lower_text_fields:
                    self.state.settings[key] = str(value).strip().lower()
                elif key in list_fields:
                    self.state.settings[key] = str(value).strip().upper()
                elif key in bool_fields:
                    self.state.settings[key] = value in (True, "true", "on", "1", "yes")
                elif key in dict_fields:
                    if isinstance(value, dict):
                        self.state.settings[key] = value
                    else:
                        try:
                            self.state.settings[key] = json.loads(value) if isinstance(value, str) else {}
                        except:
                            self.state.settings[key] = {}

            self.state.settings["exchange"] = self.state.settings["exchange"].lower()
            self.state.settings["asset_class"] = self.state.settings.get("asset_class", "crypto").lower()
            if self.state.settings["asset_class"] not in {"crypto", "forex"}:
                self.state.settings["asset_class"] = "crypto"

            if self.state.settings["asset_class"] == "forex":
                if self.state.settings["exchange"] not in {"forex_demo", "oanda_demo"}:
                    self.state.settings["exchange"] = "forex_demo"
                self.state.settings["live_trading_enabled"] = False
                if self.state.settings["exchange"] != "oanda_demo":
                    self.state.settings["oanda_demo_trading_enabled"] = False

            self.state.settings["strategy"] = self.state.settings.get(
                "strategy", "self_learning"
            ).lower()
            if self.state.settings["strategy"] not in {"sma_cross", "ewo_offset", "opening_range", "ema_golden_cross", "self_learning"}:
                self.state.settings["strategy"] = "self_learning"

            if "exit_strategies_enabled" not in self.state.settings or not self.state.settings["exit_strategies_enabled"]:
                self.state.settings["exit_strategies_enabled"] = {
                    "rsi": True,
                    "macd": True,
                    "ma": True,
                    "breakout": True,
                    "time": True,
                }

            self.state.settings["position_sizing_mode"] = self.state.settings.get(
                "position_sizing_mode", "balance_fraction"
            ).lower()
            if self.state.settings["position_sizing_mode"] not in {"balance_fraction", "risk_based"}:
                self.state.settings["position_sizing_mode"] = "balance_fraction"

            self.state.settings["risk_sizing_mode"] = self.state.settings.get(
                "risk_sizing_mode", "fixed"
            ).lower()
            if self.state.settings["risk_sizing_mode"] not in {"fixed", "kelly", "atr", "hybrid"}:
                self.state.settings["risk_sizing_mode"] = "fixed"

            self.state.settings["live_order_type"] = self.state.settings.get(
                "live_order_type", "market"
            ).lower()
            if self.state.settings["live_order_type"] not in {"market", "limit", "bracket", "native_stop_scaffold"}:
                self.state.settings["live_order_type"] = "market"

            self.state.settings["chart_mode"] = self.state.settings.get("chart_mode", "line").lower()
            if self.state.settings["chart_mode"] not in {"line", "candles"}:
                self.state.settings["chart_mode"] = "line"

            watchlist = parse_watchlist(self.state.settings.get("watchlist", ""))
            if not watchlist:
                watchlist = [self.state.settings["symbol"]]
            self.state.settings["watchlist"] = ",".join(watchlist)
            self.state.settings["symbol"] = watchlist[0]

            self.state.settings["short_window"] = max(2, int(self.state.settings["short_window"]))
            self.state.settings["long_window"] = max(
                self.state.settings["short_window"] + 1,
                int(self.state.settings["long_window"]),
            )
            self.state.settings["base_nb_candles_buy"] = max(2, int(self.state.settings["base_nb_candles_buy"]))
            self.state.settings["base_nb_candles_sell"] = max(2, int(self.state.settings["base_nb_candles_sell"]))
            self.state.settings["poll_seconds"] = max(5, int(self.state.settings["poll_seconds"]))
            self.state.settings["live_granularity"] = normalize_granularity(
                self.state.settings["live_granularity"]
            )
            self.state.settings["live_candle_count"] = max(
                strategy_minimum_candles(self.state.settings),
                min(500, int(self.state.settings["live_candle_count"])),
            )
            self.state.settings["sr_lookback_candles"] = max(5, min(300, int(self.state.settings["sr_lookback_candles"])))
            self.state.settings["near_support_pct"] = max(0.0, float(self.state.settings["near_support_pct"]))
            self.state.settings["min_resistance_distance_pct"] = max(0.0, float(self.state.settings["min_resistance_distance_pct"]))
            self.state.settings["risk_per_trade_pct"] = max(0.01, float(self.state.settings["risk_per_trade_pct"]))
            self.state.settings["min_order_value"] = max(0.0, float(self.state.settings["min_order_value"]))
            self.state.settings["min_sr_range_pct"] = max(0.0, float(self.state.settings["min_sr_range_pct"]))
            self.state.settings["min_reward_risk"] = max(0.0, float(self.state.settings["min_reward_risk"]))
            self.state.settings["support_stop_buffer_pct"] = max(0.0, float(self.state.settings["support_stop_buffer_pct"]))
            self.state.settings["resistance_target_buffer_pct"] = max(0.0, float(self.state.settings["resistance_target_buffer_pct"]))
            self.state.settings["partial_take_profit_pct"] = min(95.0, max(1.0, float(self.state.settings["partial_take_profit_pct"])))
            self.state.settings["partial_take_profit_at_target_pct"] = min(99.0, max(1.0, float(self.state.settings["partial_take_profit_at_target_pct"])))
            self.state.settings["trailing_stop_pct"] = max(0.1, float(self.state.settings["trailing_stop_pct"]))
            self.state.settings["trailing_activation_pct"] = max(0.0, float(self.state.settings["trailing_activation_pct"]))
            self.state.settings["max_live_order_gbp"] = max(1, float(self.state.settings["max_live_order_gbp"]))
            self.state.settings["max_daily_live_loss_gbp"] = max(1, float(self.state.settings["max_daily_live_loss_gbp"]))
            self.state.settings["max_daily_live_spend_quote"] = max(1, float(self.state.settings.get("max_daily_live_spend_quote", 250.0)))
            self.state.settings["max_live_spread_pct"] = max(0.01, float(self.state.settings["max_live_spread_pct"]))
            self.state.settings["min_live_quote_volume"] = max(0.0, float(self.state.settings["min_live_quote_volume"]))
            self.state.settings["live_limit_offset_pct"] = max(0.0, float(self.state.settings["live_limit_offset_pct"]))
            self.state.settings["backtest_slippage_pct"] = max(0.0, float(self.state.settings["backtest_slippage_pct"]))
            self.state.settings["sr_zone_tolerance_pct"] = max(0.0, float(self.state.settings["sr_zone_tolerance_pct"]))
            self.state.settings["sr_min_touches"] = max(1, int(self.state.settings["sr_min_touches"]))
            self.state.settings["weak_pair_min_trades"] = max(1, int(self.state.settings["weak_pair_min_trades"]))
            self.state.settings["weak_pair_win_rate_limit_pct"] = min(100.0, max(0.0, float(self.state.settings["weak_pair_win_rate_limit_pct"])))
            self.state.settings["order_expiry_seconds"] = max(30, int(self.state.settings["order_expiry_seconds"]))
            self.state.settings["order_retry_limit"] = max(0, int(self.state.settings["order_retry_limit"]))
            self.state.settings["max_coinbase_open_trades"] = max(1, int(self.state.settings.get("max_coinbase_open_trades", 3)))
            self.state.settings["max_oanda_open_trades"] = max(1, int(self.state.settings.get("max_oanda_open_trades", 3)))
            self.state.settings["opening_range_minutes"] = max(1, int(self.state.settings["opening_range_minutes"]))
            self.state.settings["opening_range_atr_period"] = max(2, int(self.state.settings["opening_range_atr_period"]))
            self.state.settings["opening_range_manipulation_threshold"] = max(0.01, min(1.0, float(self.state.settings["opening_range_manipulation_threshold"])))
            self.state.settings["opening_range_stop_loss_atr_multiplier"] = max(0.1, float(self.state.settings["opening_range_stop_loss_atr_multiplier"]))
            self.state.settings["opening_range_take_profit_atr_multiplier"] = max(0.1, float(self.state.settings["opening_range_take_profit_atr_multiplier"]))
            self.state.settings["max_drawdown_pct"] = max(1.0, float(self.state.settings.get("max_drawdown_pct", 20.0)))
            self.state.settings["signal_confidence_threshold"] = max(0.05, min(0.95, float(self.state.settings.get("signal_confidence_threshold", 0.3))))
            self.state.settings["min_signals_required"] = max(0, int(self.state.settings.get("min_signals_required", 1)))
            self.state.settings["kelly_fraction"] = max(0.01, min(1.0, float(self.state.settings.get("kelly_fraction", 0.25))))
            self.state.settings["atr_period"] = max(2, int(self.state.settings.get("atr_period", 14)))
            self.state.settings["atr_multiplier"] = max(0.1, float(self.state.settings.get("atr_multiplier", 2.0)))
            self.state.settings["max_hold_hours"] = max(1, float(self.state.settings.get("max_hold_hours", 24)))
            self.state.settings["rsi_oversold"] = max(5, min(45, float(self.state.settings.get("rsi_oversold", 30))))
            self.state.settings["rsi_overbought"] = max(55, min(95, float(self.state.settings.get("rsi_overbought", 70))))
            self.state.settings["ma_exit_period"] = max(5, int(self.state.settings.get("ma_exit_period", 20)))
            self.state.settings["min_regime_confidence"] = max(0.1, min(0.95, float(self.state.settings.get("min_regime_confidence", 0.5))))

            self.state.strategy_creator_enabled = self.state.settings.get("strategy_creator_enabled", False)
            self.state.strategy_evolution_enabled = self.state.settings.get("strategy_evolution_enabled", False)

            if self.state.settings.get("strategy_creator_enabled", False) and not self.strategy_manager:
                if STRATEGY_CREATOR_AVAILABLE:
                    try:
                        self.strategy_manager = StrategyManager(self)
                        logger.info("Strategy Manager initialized (after settings update)")
                    except Exception as e:
                        logger.error(f"Failed to initialize Strategy Manager: {e}")
                        self.strategy_manager = None

            self.update_streaming_status()
            self._start_strategy_timer_if_needed()
            self.save_state()
            logger.info("Settings updated")

    # ─── Snapshot ────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            chart_symbol = self.state.active_symbol or self.state.settings["symbol"]
            chart_prices = self.state.price_history.get(chart_symbol, self.state.prices)
            chart_candles = self.state.candle_history.get(chart_symbol, [])
            chart_row = next(
                (
                    row for row in self.state.scan_rows
                    if row.get("symbol") == chart_symbol
                ),
                {},
            )
            chart_levels = chart_trade_plan(
                state=self.state,
                chart_symbol=chart_symbol,
                chart_row=chart_row,
            )
            setup_rows = setup_performance(self.state.setup_records)
            weak_pairs = weak_pair_map(self.state.setup_records, self.state.settings)
            chart_symbols = sorted(
                set(self.state.price_history.keys())
                | set(self.state.candle_history.keys())
                | set(self.state.positions.keys())
                | set(parse_watchlist(self.state.settings.get("watchlist", "")))
            )
            price = self.state.last_price
            granularity = int(self.state.settings.get("live_granularity", 3600))

            account_type = self.state.settings.get("oanda_account_type", "standard")
            if not account_type:
                account_type = "standard"
                self.state.settings["oanda_account_type"] = "standard"

            account_type_label = "Spread Bet" if account_type == "spreadbet" else "CFD/Forex"

            signal_dashboard = self.self_learning_trader.get_signal_dashboard() if hasattr(self, 'self_learning_trader') else {}

            # ─── GET REAL DATA FROM OANDA ──────────────────────────────
            oanda_data = {}
            cash = self.state.cash
            equity = self.state.cash
            total_pnl = 0.0
            unrealized_pnl = 0.0
            oanda_positions = []
            coinbase_data = {}

            if self.should_oanda_demo_trade():
                try:
                    oanda_data = self.get_oanda_data()
                    if oanda_data.get("ok"):
                        cash = oanda_data.get("balance", self.state.cash)
                        equity = oanda_data.get("equity", self.state.cash)
                        total_pnl = oanda_data.get("total_pnl", 0)
                        unrealized_pnl = oanda_data.get("unrealized_pnl", 0)
                        oanda_positions = oanda_data.get("positions", [])

                        self.state.cash = cash

                        self.state.positions = {}
                        self.state.coin = 0.0
                        self.state.active_symbol = None

                        for pos in oanda_positions:
                            symbol = pos.get("symbol")
                            units = pos.get("units", 0)
                            avg_price = pos.get("average_price", 0)
                            is_short = pos.get("side") == "SHORT"

                            if units != 0:
                                self.state.positions[symbol] = {
                                    "quantity": -units if is_short else units,
                                    "entry_price": avg_price,
                                    "highest_price": avg_price,
                                    "is_short": is_short,
                                    "opened_at": now_iso(),
                                    "stop_price": pos.get("stop_loss"),
                                    "target_price": pos.get("take_profit"),
                                    "entry_time": time.time(),
                                }
                                if is_short:
                                    self.state.coin = -units
                                else:
                                    self.state.coin = units
                                self.state.active_symbol = symbol

                except Exception as e:
                    logger.warning(f"Exception getting OANDA data: {e}")

            set_position_rows_bot(self)

            if not self.should_oanda_demo_trade():
                is_live_coinbase = (
                    bool(self.state.settings.get("live_trading_enabled"))
                    and self.state.settings.get("asset_class", "crypto") == "crypto"
                    and self.state.settings.get("exchange") == "coinbase"
                    and coinbase_live_is_armed()
                )
                if is_live_coinbase:
                    # Use the most recently reconciled exchange snapshot.  Refresh
                    # if one is not available yet.  This prevents stale/missing local
                    # positions from appearing as a large account loss.
                    coinbase_data = getattr(self, "_coinbase_account_snapshot", {}) or {}
                    if not coinbase_data.get("ok"):
                        try:
                            coinbase_data = self.coinbase_live_account_snapshot()
                            self._coinbase_account_snapshot = coinbase_data
                        except Exception as exc:
                            logger.warning(f"Failed to value Coinbase account for snapshot: {exc}")
                            coinbase_data = {}
                    if coinbase_data.get("ok"):
                        cash = float(coinbase_data.get("available_cash", self.state.cash))
                        self.state.cash = cash
                        equity = float(coinbase_data.get("equity", cash))
                    else:
                        equity = self._calculate_equity_local(price)
                else:
                    equity = self._calculate_equity_local(price)
                total_pnl = equity - float(self.state.settings.get("starting_cash", 0))

            # Account accounting: starting_cash is the fixed baseline; cash is the
            # current available balance; equity includes open positions.
            starting_cash = float(self.state.settings.get("starting_cash", 0.0))
            open_position_value = equity - cash
            if self.should_oanda_demo_trade() and oanda_data.get("ok"):
                # OANDA supplies its own unrealised P/L.
                unrealized_pnl = float(oanda_data.get("unrealized_pnl", unrealized_pnl) or 0.0)
            else:
                unrealized_pnl = 0.0
                live_holding_qty = {
                    str(item.get("currency", "")).upper(): float(item.get("quantity", 0.0) or 0.0)
                    for item in coinbase_data.get("holdings", [])
                } if coinbase_data.get("ok") else {}
                for symbol, position in self.state.positions.items():
                    quantity = float(position.get("quantity", 0.0))
                    if live_holding_qty:
                        # Local positions are the entry-price ledger, but never claim
                        # more live quantity than Coinbase actually holds.
                        quantity = min(max(quantity, 0.0), live_holding_qty.get(str(symbol).upper(), 0.0))
                    entry = position.get("entry_price")
                    history = self.state.price_history.get(symbol, [])
                    current = history[-1] if history else price
                    if entry is not None and current is not None:
                        unrealized_pnl += quantity * (float(current) - float(entry))
            realized_pnl = total_pnl - unrealized_pnl

            day_pnl = equity - self.state.day_start_equity
            live_risk = self.daily_live_risk_metrics()

            expectancy_summary = self.expectancy.summary() if hasattr(self, 'expectancy') else {}

            regime_dict = {}
            if hasattr(self.state, 'current_regime') and self.state.current_regime:
                regime_dict = self.regime_detector.to_dict(self.state.current_regime)

            strategy_summary = {}
            if self.strategy_manager and STRATEGY_CREATOR_AVAILABLE:
                strategy_summary = self.strategy_manager.get_performance_summary()

            kelly_metrics = self.db.get_kelly_metrics()

            return {
                "running": self.state.running,
                "settings": self.state.settings,
                "starting_cash": round(starting_cash, 2),
                "cash": round(cash, 2),
                "open_position_value": round(open_position_value, 2),
                "realized_pnl": round(realized_pnl, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "coin": round(self.state.coin, 12),
                "active_symbol": self.state.active_symbol,
                "chart_symbol": chart_symbol,
                "chart_symbols": chart_symbols,
                "chart_meta": {
                    "timeframe": granularity_label(granularity),
                    "granularity": granularity,
                    "latest_candle_incomplete": latest_candle_incomplete(chart_candles, granularity),
                    "closed_candle_only": bool(self.state.settings.get("closed_candle_only")),
                    "chart_mode": self.state.settings.get("chart_mode", "line"),
                },
                "price_history": {
                    symbol: prices[-80:]
                    for symbol, prices in self.state.price_history.items()
                },
                "candle_history": {
                    symbol: candles[-80:]
                    for symbol, candles in self.state.candle_history.items()
                },
                "chart_rows": {
                    str(row.get("symbol")): row
                    for row in self.state.scan_rows
                    if row.get("symbol")
                },
                "entry_price": self.state.entry_price,
                "last_price": price,
                "equity": round(equity, 2),
                "total_pnl": round(total_pnl, 2),
                "total_pnl_pct": pct(total_pnl, float(self.state.settings.get("starting_cash", 0))),
                "day_pnl": round(day_pnl, 2),
                "day_pnl_pct": pct(day_pnl, self.state.day_start_equity),
                "daily_realized_pnl": round(live_risk["realized"], 2),
                "daily_unrealized_pnl": round(live_risk["unrealized"], 2),
                "daily_risk_pnl": round(live_risk["risk_pnl"], 2),
                "live_daily_spend": round(self.state.live_daily_spend, 2),
                "max_daily_live_spend_quote": float(self.state.settings.get("max_daily_live_spend_quote", 250.0)),
                "max_daily_live_loss_quote": float(self.state.settings.get("max_daily_live_loss_gbp", 25.0)),
                "last_signal": self.state.last_signal,
                "last_error": self.state.last_error,
                "price_count": len(chart_prices),
                "short_sma": sma(chart_prices, int(self.state.settings["short_window"])),
                "long_sma": sma(chart_prices, int(self.state.settings["long_window"])),
                "support": chart_row.get("support"),
                "resistance": chart_row.get("resistance"),
                "chart_levels": chart_levels,
                "chart_trades": [
                    {
                        **asdict(item),
                        'stop_loss': getattr(item, 'stop_loss_price', None),
                        'take_profit': getattr(item, 'take_profit_price', None),
                        'exit_mode': getattr(item, 'exit_mode', None),
                        'exit_reason': getattr(item, 'exit_reason', None),
                        'regime': getattr(item, 'regime', None),
                    }
                    for item in self.state.trades[-40:]
                    if item.symbol == chart_symbol
                ],
                "prices": chart_prices[-80:],
                "candles": chart_candles[-80:],
                "scan_rows": self.state.scan_rows,
                "trades": [asdict(item) for item in self.state.trades[-60:]][::-1],
                "journal": [asdict(item) for item in self.state.journal[-120:]][::-1],
                "symbol_performance": symbol_performance(self.state.trades),
                "setup_performance": setup_rows,
                "setup_records": recent_setup_records(self.state.setup_records),
                "weak_pairs": weak_pairs,
                "positions": position_rows(self.state),
                "best_setup": best_current_setup(self.state.scan_rows),
                "blocked_summary": blocked_summary(self.state.journal),
                "open_trade_risk": open_trade_risk(self.state, chart_levels, price),
                "open_orders": [asdict(item) for item in self.state.open_orders[-40:]][::-1],
                "chart_regime": chart_row.get("regime"),
                "live_status": self.live_status(),
                "news_guard": self.news_guard_status(),
                "opening_range_analysis": self.state.opening_range_analysis,
                "peak_equity": self.state.peak_equity,
                "stop_price": self.state.stop_price,
                "target_price": self.state.target_price,
                "exit_mode": self.state.exit_mode,
                "is_short": self.state.is_short,
                "oanda_account_type": account_type,
                "account_type_label": account_type_label,
                "signal_dashboard": signal_dashboard,
                "learning_iterations": self.state.learning_iterations,
                "last_learning_update": self.state.last_learning_update,
                "oanda_balance": oanda_data.get("balance") if oanda_data.get("ok") else None,
                "oanda_equity": oanda_data.get("equity") if oanda_data.get("ok") else None,
                "oanda_margin_used": oanda_data.get("margin_used") if oanda_data.get("ok") else None,
                "oanda_margin_available": oanda_data.get("margin_available") if oanda_data.get("ok") else None,
                "oanda_unrealized_pnl": oanda_data.get("unrealized_pnl") if oanda_data.get("ok") else None,
                "oanda_currency": oanda_data.get("currency") if oanda_data.get("ok") else None,
                "oanda_positions_count": len(oanda_positions) if oanda_data.get("ok") else 0,
                "oanda_connected": oanda_data.get("ok", False),
                "strategy_creator": {
                    "enabled": self.state.settings.get("strategy_creator_enabled", False),
                    "evolution_enabled": self.state.settings.get("strategy_evolution_enabled", False),
                    "active_strategy_id": self.state.active_strategy_id,
                    "summary": strategy_summary,
                    "history": self.state.strategy_history[-5:] if self.state.strategy_history else []
                },
                "risk_metrics": {
                    "sizing_mode": self.state.settings.get("risk_sizing_mode", "fixed"),
                    "kelly_value": kelly_metrics.get("kelly_value", 0) if kelly_metrics else 0,
                    "kelly_fraction": self.state.settings.get("kelly_fraction", 0.25),
                    "atr_period": self.state.settings.get("atr_period", 14),
                    "atr_multiplier": self.state.settings.get("atr_multiplier", 2.0),
                    "max_hold_hours": self.state.settings.get("max_hold_hours", 24),
                },
                "exit_strategies": self.state.settings.get("exit_strategies_enabled", {
                    "rsi": True,
                    "macd": True,
                    "ma": True,
                    "breakout": True,
                    "time": True,
                }),
                "expectancy": expectancy_summary,
                "regime": regime_dict,
                "regime_adaptation": {
                    "enabled": self.state.settings.get("regime_adaptation_enabled", True),
                    "strategy_switching": self.state.settings.get("strategy_switching_enabled", True),
                    "min_confidence": self.state.settings.get("min_regime_confidence", 0.5),
                },
            }

    # ─── Run Loop ────────────────────────────────────────────────────

    def run_loop(self) -> None:
        last_summary_date = ""
        summary_sent_today = False

        while not self.stop_event.is_set() and not self.shutdown_requested:
            try:
                self.tick()

                if self.state.settings.get("telegram_alert_on_daily_summary", True):
                    now = datetime.now(timezone.utc)
                    current_day = today_key()

                    if current_day != last_summary_date:
                        last_summary_date = current_day
                        summary_sent_today = False

                    if now.hour == 23 and now.minute == 59 and not summary_sent_today:
                        self.send_telegram_alert(self.format_daily_summary())
                        summary_sent_today = True
                        logger.info("Daily summary sent")

            except Exception as exc:
                logger.error(f"Bot loop error: {exc}", exc_info=True)
                with self.lock:
                    self.state.last_error = str(exc)
                    self.state.last_signal = f"Paused by error: {exc}"
                    self.save_state()
                logger.info(f"Waiting {self.restart_delay}s before restarting loop...")
                self.stop_event.wait(self.restart_delay)

            wait_seconds = int(self.state.settings.get("poll_seconds", 15))
            self.stop_event.wait(wait_seconds)

        if self.shutdown_requested:
            logger.info("Bot loop exited (shutdown requested)")
        else:
            logger.info("Bot loop stopped")

    # ─── Tick ────────────────────────────────────────────────────────

    def tick(self) -> None:
        with self.lock:
            settings = dict(self.state.settings)

        if not hasattr(self, '_sync_counter'):
            self._sync_counter = 0
        self._sync_counter += 1

        if self._sync_counter % 10 == 0:
            if settings.get("asset_class") == "crypto" and settings.get("exchange") == "coinbase":
                self.sync_live_balance_always()

        watchlist = parse_watchlist(settings.get("watchlist", settings["symbol"]))
        fetched_prices: dict[str, float] = {}
        fetched_candles: dict[str, list[Candle]] = {}
        errors: list[str] = []
        granularity = int(settings.get("live_granularity", 3600))
        candle_count = int(settings.get("live_candle_count", 300))

        for symbol in watchlist:
            try:
                candles = fetch_candles(
                    exchange=settings["exchange"],
                    symbol=symbol,
                    quote_currency=settings["quote_currency"],
                    granularity=granularity,
                    candle_count=candle_count,
                    asset_class=str(settings.get("asset_class", "crypto")),
                )
                if not candles:
                    raise RuntimeError("No candles returned")
                fetched_candles[symbol] = candles
                fetched_prices[symbol] = candles[-1].close
            except Exception as exc:
                errors.append(f"{symbol}: {exc}")

        if not fetched_prices:
            raise RuntimeError("; ".join(errors) or "No candle data returned")

        with self.lock:
            active_price = self.price_for_active_position(fetched_prices)
            self.roll_daily_equity_if_needed(active_price)
            self.state.last_price = active_price
            self.state.last_error = None
            if errors:
                self.state.last_error = "; ".join(errors[:3])

            for symbol, candles in fetched_candles.items():
                self.state.price_history[symbol] = [
                    candle.close for candle in candles
                ][-300:]
                self.state.candle_history[symbol] = [
                    asdict(candle) for candle in candles
                ][-300:]

            chart_symbol = self.state.active_symbol or watchlist[0]
            self.state.prices = self.state.price_history.get(chart_symbol, [])[-300:]

            if candles := fetched_candles.get(chart_symbol):
                regime_result = self.regime_detector.detect(candles)
                self.state.current_regime = regime_result
                chart_row = next(
                    (row for row in self.state.scan_rows if row.get("symbol") == chart_symbol),
                    {}
                )
                chart_row["regime"] = regime_result.regime

            if settings.get("strategy_switching_enabled", True):
                recommended = self.get_strategy_for_regime()
                if recommended and recommended != settings.get("strategy"):
                    old_strategy = settings.get("strategy")
                    self.state.settings["strategy"] = recommended
                    self.journal(
                        chart_symbol,
                        "INFO",
                        f"Switched strategy from {old_strategy} to {recommended} based on regime {self.state.current_regime.regime if self.state.current_regime else 'unknown'}",
                        self.state.last_price,
                        {"old_strategy": old_strategy, "new_strategy": recommended, "regime": self.state.current_regime.regime if self.state.current_regime else None}
                    )
                    logger.info(f"Strategy switched: {old_strategy} → {recommended} (regime: {self.state.current_regime.regime if self.state.current_regime else 'unknown'})")

            # ─── Priority 4: Regime-driven strategy selection & dead market block ──────────
            if settings.get("regime_force_strategy", True):
                if self.state.current_regime and self.state.current_regime.confidence >= settings.get("min_regime_confidence", 0.5):
                    regime = self.state.current_regime.regime
                    # Block all trades if market is dead and regime_block_dead is True
                    if regime == "dead" and settings.get("regime_block_dead", True):
                        self.state.last_signal = "BLOCK: Market regime is DEAD – no trades"
                        self.journal(chart_symbol, "BLOCK", self.state.last_signal, self.state.last_price)
                        # Skip decision and wait for next tick
                        self.save_state()
                        return

                    # Map regime to strategy
                    strategy_map = {
                        "trending": settings.get("regime_trend_strategy", "ema_golden_cross"),
                        "trending_up": settings.get("regime_trend_strategy", "ema_golden_cross"),
                        "trending_down": settings.get("regime_trend_strategy", "ema_golden_cross"),
                        "ranging": settings.get("regime_ranging_strategy", "opening_range"),
                        "breakout": settings.get("regime_breakout_strategy", "opening_range"),
                        "volatile": settings.get("regime_volatile_strategy", "sma_cross"),
                    }
                    recommended = strategy_map.get(regime)
                    if recommended and recommended != settings.get("strategy"):
                        # Override the strategy in settings for this tick
                        self.state.settings["strategy"] = recommended
                        self.journal(
                            chart_symbol,
                            "INFO",
                            f"Regime {regime} → switching to {recommended}",
                            self.state.last_price,
                            {"regime": regime, "strategy": recommended}
                        )
                        logger.info(f"Regime {regime} → switched strategy to {recommended}")

            # ─── The decision logic continues as before ──────────────────────────────
            if self.should_live_trade():
                self.manage_open_orders()

            # Protective exits always have priority over entry/signal strategy logic.
            # This prevents strategy-specific early returns (for example EMA Golden
            # Cross HOLD) from bypassing TP/SL/partial-TP/trailing-stop management.
            decision = self.manage_position_exits(fetched_prices, fetched_candles)

            if not decision and settings.get("strategy_creator_enabled", False) and self.strategy_manager:
                for symbol in watchlist:
                    candles = fetched_candles.get(symbol, [])
                    if not candles:
                        continue

                    result = self.apply_strategy_signal(candles)
                    if result.get('ok') and result.get('direction') != 'NEUTRAL':
                        confidence = result.get('confidence', 0)
                        min_confidence = float(settings.get("strategy_confidence_threshold", 0.5))

                        if confidence > min_confidence:
                            has_position = (symbol in self.state.positions or
                                           (self.state.active_symbol == symbol and abs(self.state.coin) > 0))

                            if result['direction'] == 'BUY' and not has_position:
                                decision = f"BUY {symbol} strategy: {result['strategy']} (conf: {confidence:.2f})"
                                break
                            elif result['direction'] == 'SELL' and has_position:
                                decision = f"SELL {symbol} strategy: {result['strategy']} (conf: {confidence:.2f})"
                                break

            if not decision:
                strategy = settings.get("strategy", "self_learning")
                if strategy == "self_learning":
                    decision = self.decide_self_learning(fetched_prices, watchlist, fetched_candles)
                elif strategy == "opening_range":
                    decision = self.decide_opening_range(fetched_prices, watchlist, fetched_candles)
                else:
                    decision = self.decide_legacy(fetched_prices, watchlist, fetched_candles)

            self.state.last_signal = decision

            if decision.startswith("BUY"):
                symbol = decision.split()[1]
                candles = fetched_candles.get(symbol, [])
                is_short = "SHORT" in decision or "short" in decision.lower()
                if self.should_oanda_demo_trade():
                    self.oanda_demo_buy(symbol, fetched_prices[symbol], decision, candles, is_short=is_short)
                elif self.wants_oanda_demo_trade():
                    self.state.last_signal = f"OANDA BUY blocked: {oanda_demo_status_message()}"
                    self.journal(symbol, "BLOCK", self.state.last_signal, fetched_prices[symbol])
                elif self.should_live_trade():
                    self.live_buy(symbol, fetched_prices[symbol], decision, candles, is_short=is_short)
                else:
                    self.paper_buy(symbol, fetched_prices[symbol], decision, candles, is_short=is_short)
            elif decision.startswith("SELL"):
                parts = decision.split()
                symbol = parts[1] if len(parts) > 1 else self.state.active_symbol or settings["symbol"]
                sell_quantity = None
                position = self.state.positions.get(symbol)
                if position:
                    # Quantity must belong to the symbol being exited.  Using the
                    # legacy aggregate self.state.coin can sell the wrong amount
                    # when several Coinbase positions are open.
                    position_quantity = abs(float(position.get("quantity", 0.0) or 0.0))
                    if " partial " in f" {decision} ":
                        sell_quantity = position_quantity * (
                            float(settings.get("partial_take_profit_pct", 50.0)) / 100
                        )
                    else:
                        sell_quantity = position_quantity
                elif " partial " in f" {decision} ":
                    # Backward compatibility for a legacy state with no positions map.
                    sell_quantity = abs(self.state.coin) * (
                        float(settings.get("partial_take_profit_pct", 50.0)) / 100
                    )
                if self.should_oanda_demo_trade():
                    self.oanda_demo_sell(symbol, fetched_prices.get(symbol, active_price), decision, sell_quantity)
                elif self.wants_oanda_demo_trade():
                    self.state.last_signal = f"OANDA SELL blocked: {oanda_demo_status_message()}"
                    self.journal(symbol, "BLOCK", self.state.last_signal, fetched_prices.get(symbol, active_price))
                elif self.should_live_trade():
                    self.live_sell(symbol, fetched_prices.get(symbol, active_price), decision, sell_quantity)
                else:
                    self.paper_sell(symbol, fetched_prices.get(symbol, active_price), decision, sell_quantity)

            # ─── XGBoost training ──────────────────────────────────────
            if hasattr(self, 'self_learning_trader'):
                if len(self.state.trades) % 50 == 0 and len(self.state.trades) > 0:
                    self.self_learning_trader.train_xgboost()

            self.save_state()

    # ─── Self-Learning Decision ─────────────────────────────────────

    def decide_self_learning(
        self,
        fetched_prices: dict[str, float],
        watchlist: list[str],
        candles_by_symbol: dict[str, list[Candle]],
    ) -> str:
        with self.lock:
            settings = dict(self.state.settings)
            last_action_time = self.state.last_action_time
            active_symbol = self.state.active_symbol
            coin = self.state.coin
            positions = dict(self.state.positions)
            day_start_equity = self.state.day_start_equity
            peak_equity = self.state.peak_equity

        if not settings.get('self_learning_enabled', True):
            return "HOLD self-learning disabled"

        if time.time() - last_action_time < float(settings["cooldown_seconds"]):
            return "Cooldown active"

        active_price = self.price_for_active_position(fetched_prices)
        equity = self.equity(active_price)
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
                blocked, reason = self.is_news_blocked(symbol, settings)
                if blocked:
                    return f"BLOCK {symbol} {reason}"

        trader = self.self_learning_trader
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

    def decide_opening_range(self, fetched_prices: dict[str, float], watchlist: list[str], candles_by_symbol: dict[str, list[Candle]]) -> str:
        with self.lock:
            settings = dict(self.state.settings)
            last_action_time = self.state.last_action_time
            active_symbol = self.state.active_symbol
            coin = self.state.coin
            positions = dict(self.state.positions)
            day_start_equity = self.state.day_start_equity
            peak_equity = self.state.peak_equity

        if time.time() - last_action_time < float(settings["cooldown_seconds"]):
            return "Cooldown active"

        if settings.get("news_guard_enabled", False):
            for symbol in watchlist:
                blocked, reason = self.is_news_blocked(symbol, settings)
                if blocked:
                    return f"BLOCK {symbol} {reason}"

        active_price = self.price_for_active_position(fetched_prices)
        equity = self.equity(active_price)

        daily_loss_limit = float(settings["daily_loss_limit_pct"]) / 100
        if equity <= day_start_equity * (1 - daily_loss_limit):
            return "Daily loss limit reached"

        max_drawdown_pct = float(settings.get("max_drawdown_pct", 20.0))
        if peak_equity > 0:
            current_drawdown = ((peak_equity - equity) / peak_equity) * 100
            if current_drawdown > max_drawdown_pct:
                return f"STOP Max drawdown {current_drawdown:.1f}% > {max_drawdown_pct}%"

        if equity > peak_equity:
            with self.lock:
                self.state.peak_equity = equity

        symbol = watchlist[0]
        candles = candles_by_symbol.get(symbol, [])
        if len(candles) < max(20, int(settings.get("opening_range_atr_period", 14)) + 1):
            return "WAIT data loading for Opening Range"

        signal = self.opening_range_signal(symbol, candles)
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

    def opening_range_signal(self, symbol: str, candles: list[Candle]) -> dict[str, Any]:
        analysis = self.fetch_daily_opening_candle(symbol, candles)
        self.state.opening_range_analysis = analysis

        if analysis.get("bias") is None:
            return {"signal": "HOLD", "reason": "No opening candle found", "analysis": analysis}

        current_price = candles[-1].close if candles else 0
        trigger = analysis["trigger_level"]
        atr = analysis["atr"]

        stop_loss_mult = float(self.state.settings.get("opening_range_stop_loss_atr_multiplier", 1.5))
        take_profit_mult = float(self.state.settings.get("opening_range_take_profit_atr_multiplier", 2.5))

        has_position = (
            (symbol in self.state.positions and abs(self.state.positions[symbol].get("quantity", 0)) > 0) or
            (self.state.active_symbol == symbol and abs(self.state.coin) > 0)
        )

        if has_position:
            position = self.state.positions.get(symbol, {})
            entry_price = float(position.get("entry_price") or self.state.entry_price or 0.0)
            is_short = position.get("is_short", False) or self.state.is_short
            quantity = float(position.get("quantity", 0)) or self.state.coin

            if entry_price > 0:
                if not is_short:
                    stop_price = entry_price - (atr * stop_loss_mult)
                    target_price = entry_price + (atr * take_profit_mult)

                    self.state.highest_price = max(self.state.highest_price or current_price, current_price)

                    trailing_stop = trailing_stop_price(
                        entry_price=self.state.entry_price,
                        highest_price=self.state.highest_price,
                        settings=self.state.settings,
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
            if not self.state.settings.get("allow_short_selling", False):
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

    def fetch_daily_opening_candle(self, symbol: str, candles: list[Candle]) -> dict[str, Any]:
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

        atr = self.calculate_atr(candles, int(self.state.settings.get("opening_range_atr_period", 14)))
        if atr == 0:
            atr = candle_range

        manipulation_threshold = float(self.state.settings.get("opening_range_manipulation_threshold", 0.20))
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

    # ─── Universal Multi-Position Exit Management ────────────────────

    def manage_position_exits(
        self,
        fetched_prices: dict[str, float],
        candles_by_symbol: dict[str, list[Candle]],
    ) -> str | None:
        """Evaluate protective exits for every open position before entry strategy logic.

        This deliberately does not use ``active_symbol`` / root-level entry fields as
        the source of truth.  Each entry in state.positions owns its quantity, entry,
        high-water mark, TP/SL and partial-TP state.
        """
        with self.lock:
            settings = dict(self.state.settings)
            position_symbols = list(self.state.positions.keys())

        for symbol in position_symbols:
            with self.lock:
                position = self.state.positions.get(symbol)
                if not position:
                    continue
                position = dict(position)

            quantity = float(position.get("quantity", 0.0) or 0.0)
            if abs(quantity) <= 0:
                continue

            price = fetched_prices.get(symbol)
            if price is None:
                continue
            price = float(price)

            entry_price = float(position.get("entry_price", 0.0) or 0.0)
            if entry_price <= 0:
                continue

            is_short = bool(position.get("is_short", quantity < 0))
            position_side = "SHORT" if is_short else "LONG"

            history = self.state.price_history.get(symbol, [])
            candles = candles_by_symbol.get(symbol) or closes_to_candles(history)

            # Persist the favourable price extreme independently for every position.
            # For longs this is the highest price; for shorts it is the lowest price.
            stored_highest = position.get("highest_price")
            try:
                stored_highest = float(stored_highest) if stored_highest is not None else entry_price
            except (TypeError, ValueError):
                stored_highest = entry_price

            if is_short:
                favourable_extreme = min(stored_highest, price)
            else:
                favourable_extreme = max(stored_highest, price)

            with self.lock:
                live_position = self.state.positions.get(symbol)
                if live_position is not None:
                    live_position["highest_price"] = favourable_extreme
                # Keep legacy root fields coherent only when they refer to this symbol.
                if self.state.active_symbol == symbol:
                    self.state.highest_price = favourable_extreme

            stored_stop = (
                position.get("stop_price")
                or position.get("stop")
                or position.get("stop_loss")
                or position.get("stop_loss_price")
            )
            stored_target = (
                position.get("target_price")
                or position.get("target")
                or position.get("take_profit")
                or position.get("take_profit_price")
            )
            stop_price = float(stored_stop) if stored_stop is not None else None
            target_price = float(stored_target) if stored_target is not None else None
            exit_mode = str(position.get("exit_mode") or "stored")

            if stop_price is None or target_price is None:
                calculated_stop, calculated_target, calculated_mode = exit_prices(
                    entry_price=entry_price,
                    candles=candles,
                    settings=settings,
                )
                if stop_price is None:
                    stop_price = calculated_stop
                if target_price is None:
                    target_price = calculated_target
                if not position.get("exit_mode"):
                    exit_mode = calculated_mode

                # Persist recovered levels so the UI and future cycles use the same plan.
                with self.lock:
                    live_position = self.state.positions.get(symbol)
                    if live_position is not None:
                        live_position["stop_price"] = stop_price
                        live_position["target_price"] = target_price
                        live_position["exit_mode"] = exit_mode

            entry_time = float(position.get("entry_time", time.time()) or time.time())
            should_exit, exit_reason = self.should_exit_enhanced(
                symbol=symbol,
                candles=candles,
                entry_time=entry_time,
                position_side=position_side,
            )
            if should_exit:
                return f"SELL {symbol} {exit_reason}"

            partial_done = bool(position.get("partial_take_profit_done", False))

            # Partial TP is position-specific. Handle shorts symmetrically.
            partial_ready = False
            if settings.get("partial_take_profit_enabled") and not partial_done:
                trigger_fraction = float(
                    settings.get("partial_take_profit_at_target_pct", 50.0)
                ) / 100.0
                partial_trigger = entry_price + ((target_price - entry_price) * trigger_fraction)
                if is_short:
                    partial_ready = target_price < entry_price and price <= partial_trigger
                else:
                    partial_ready = target_price > entry_price and price >= partial_trigger

            if partial_ready:
                with self.lock:
                    live_position = self.state.positions.get(symbol)
                    if live_position is not None:
                        live_position["partial_take_profit_done"] = True
                    if self.state.active_symbol == symbol:
                        self.state.partial_take_profit_done = True
                return f"SELL {symbol} partial {exit_mode} target"

            # Position-specific trailing stop.  The old helper is long-only, so shorts
            # are handled explicitly using their favourable (lowest) price.
            trailing_stop = None
            if settings.get("trailing_stop_enabled"):
                activation = float(settings.get("trailing_activation_pct", 3.0)) / 100.0
                trail = float(settings.get("trailing_stop_pct", 2.0)) / 100.0
                if is_short:
                    if favourable_extreme <= entry_price * (1.0 - activation):
                        trailing_stop = favourable_extreme * (1.0 + trail)
                    if trailing_stop is not None and price >= trailing_stop:
                        return f"SELL {symbol} trailing stop"
                else:
                    if favourable_extreme >= entry_price * (1.0 + activation):
                        trailing_stop = favourable_extreme * (1.0 - trail)
                    if trailing_stop is not None and price <= trailing_stop:
                        return f"SELL {symbol} trailing stop"

            if is_short:
                if price >= stop_price:
                    return f"SELL {symbol} {exit_mode} stop"
                if price <= target_price:
                    return f"SELL {symbol} {exit_mode} target"
            else:
                if price <= stop_price:
                    return f"SELL {symbol} {exit_mode} stop"
                if price >= target_price:
                    return f"SELL {symbol} {exit_mode} target"

        return None

    # ─── Legacy Decision ─────────────────────────────────────────────

    def decide_legacy(self, fetched_prices: dict[str, float], watchlist: list[str], candles_by_symbol: dict[str, list[Candle]]) -> str:
        with self.lock:
            settings = dict(self.state.settings)
            last_action_time = self.state.last_action_time
            active_symbol = self.state.active_symbol
            coin = self.state.coin
            entry_price = self.state.entry_price
            highest_price = self.state.highest_price
            partial_take_profit_done = self.state.partial_take_profit_done
            positions = dict(self.state.positions)
            day_start_equity = self.state.day_start_equity
            peak_equity = self.state.peak_equity
            price_history = dict(self.state.price_history)

        strategy = settings.get("strategy", "sma_cross")

        if time.time() - last_action_time < float(settings["cooldown_seconds"]):
            return "Cooldown active"

        if settings.get("news_guard_enabled", False):
            for symbol in watchlist:
                blocked, reason = self.is_news_blocked(symbol, settings)
                if blocked:
                    return f"BLOCK {symbol} {reason}"

        active_price = self.price_for_active_position(fetched_prices)
        equity = self.equity(active_price)

        daily_loss_limit = float(settings["daily_loss_limit_pct"]) / 100
        if equity <= day_start_equity * (1 - daily_loss_limit):
            return "Daily loss limit reached"

        max_drawdown_pct = float(settings.get("max_drawdown_pct", 20.0))
        if peak_equity > 0:
            current_drawdown = ((peak_equity - equity) / peak_equity) * 100
            if current_drawdown > max_drawdown_pct:
                return f"STOP Max drawdown {current_drawdown:.1f}% > {max_drawdown_pct}%"

        if equity > peak_equity:
            with self.lock:
                self.state.peak_equity = equity

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

            position = positions.get(symbol, {})

            # Use the protective levels captured when this position was opened.
            # Recalculating ATR exits on every decision cycle can widen the stop as
            # volatility changes, while the UI continues to show the original stop.
            # Only fall back to a fresh calculation for legacy positions that do not
            # already have stored TP/SL levels.
            stored_stop = (
                position.get("stop_price") or
                position.get("stop") or
                position.get("stop_loss") or
                position.get("stop_loss_price")
            )
            stored_target = (
                position.get("target_price") or
                position.get("target") or
                position.get("take_profit") or
                position.get("take_profit_price")
            )
            stop_price = float(stored_stop) if stored_stop is not None else None
            target_price = float(stored_target) if stored_target is not None else None
            exit_mode = str(position.get("exit_mode") or self.state.exit_mode or "stored")

            if stop_price is None or target_price is None:
                calculated_stop, calculated_target, calculated_mode = exit_prices(
                    entry_price=entry_price,
                    candles=candles,
                    settings=settings,
                )
                if stop_price is None:
                    stop_price = calculated_stop
                if target_price is None:
                    target_price = calculated_target
                if not position.get("exit_mode"):
                    exit_mode = calculated_mode

            position_side = "SHORT" if position.get('is_short', False) else "LONG"
            entry_time = position.get('entry_time', time.time())

            should_exit, exit_reason = self.should_exit_enhanced(
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
                with self.lock:
                    self.state.partial_take_profit_done = True
                return f"SELL {symbol} partial {exit_mode} target"

            trailing_stop = trailing_stop_price(
                entry_price=entry_price,
                highest_price=current_highest,
                settings=settings,
            )
            if trailing_stop and price <= trailing_stop:
                return f"SELL {symbol} trailing stop"

            if position_side == "SHORT":
                if price >= stop_price:
                    return f"SELL {symbol} {exit_mode} stop"
                if price <= target_price:
                    return f"SELL {symbol} {exit_mode} target"
            else:
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
            scan_rows = self.build_ewo_scan_rows(watchlist, candles_by_symbol)
        else:
            scan_rows = self.build_scan_rows(watchlist, candles_by_symbol)
        self.state.scan_rows = scan_rows

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

    # ─── Paper Trading ───────────────────────────────────────────────

    def paper_buy(
        self,
        symbol: str,
        price: float,
        reason: str,
        candles: list[Candle] | None = None,
        spend_override: float | None = None,
        fee_override: float | None = None,
        quantity_override: float | None = None,
        exchange_order_id: str | None = None,
        exchange_order_status: str | None = None,
        exchange_average_filled_price: float | None = None,
        stop_override: float | None = None,
        target_override: float | None = None,
        is_short: bool = False,
    ) -> None:
        with self.lock:
            settings = dict(self.state.settings)

        trade_fee = float(settings["trade_fee"])
        spend_reason = "manual override"

        regime_adapt = self.get_regime_adaptations()
        stop_mult = regime_adapt.get('stop_multiplier', 1.0)
        tp_mult = regime_adapt.get('take_profit_multiplier', 1.0)

        if spend_override is not None:
            spend = spend_override
        else:
            if candles is None:
                with self.lock:
                    candles = closes_to_candles(self.state.price_history.get(symbol, []))

            original_stop = settings.get('stop_loss_pct', 2.0)
            original_tp = settings.get('take_profit_pct', 3.0)
            settings['stop_loss_pct'] = original_stop * stop_mult
            settings['take_profit_pct'] = original_tp * tp_mult

            try:
                with self.lock:
                    cash = self.state.cash
                spend, spend_reason = self.calculate_position_size(
                    cash=cash,
                    entry_price=price,
                    candles=candles,
                    symbol=symbol,
                    position_side="SHORT" if is_short else "LONG"
                )
            finally:
                settings['stop_loss_pct'] = original_stop
                settings['take_profit_pct'] = original_tp

        with self.lock:
            if spend > self.state.cash:
                spend = self.state.cash

            if spend < float(settings.get("min_order_value", 1.0)):
                self.state.last_signal = f"{'SHORT' if is_short else 'BUY'} blocked: order below minimum {settings['quote_currency']} {settings.get('min_order_value', 1.0)}"
                self.journal(symbol, "BLOCK", self.state.last_signal, price, {"spend": spend})
                return

            fee_paid = fee_override if fee_override is not None else spend * trade_fee
            coin_bought = quantity_override if quantity_override is not None else (spend - fee_paid) / price

            if stop_override is not None and target_override is not None:
                stop_price = stop_override
                target_price = target_override
                exit_mode = "opening_range"
            else:
                stop_price, target_price, exit_mode = self.exit_prices(
                    entry_price=price,
                    candles=candles or closes_to_candles(self.state.price_history.get(symbol, [])),
                    settings=self.state.settings,
                )

            if is_short:
                self.state.coin -= coin_bought
                self.state.cash += spend
                self.state.active_symbol = symbol
                self.state.entry_price = price
                self.state.highest_price = price
                self.state.stop_price = stop_price
                self.state.target_price = target_price
                self.state.exit_mode = exit_mode
                self.state.active_stop_order_id = None
                self.state.partial_take_profit_done = False
                self.state.last_price = price
                self.state.last_action_time = time.time()
                self.state.is_short = True

                self.state.positions[symbol] = {
                    "quantity": -coin_bought,
                    "entry_price": price,
                    "highest_price": price,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "exit_mode": exit_mode,
                    "partial_take_profit_done": False,
                    "entry_cost": spend,
                    "opened_at": now_iso(),
                    "is_short": True,
                    "entry_time": time.time(),
                }

                trade = Trade(
                    time=now_iso(),
                    side="SHORT",
                    symbol=symbol,
                    price=price,
                    quantity=-coin_bought,
                    cash_after=self.state.cash,
                    coin_after=self.state.coin,
                    reason=f"{reason} | size {spend_reason} | {exit_mode} stop/target",
                    fee_paid=fee_paid,
                    exchange_order_id=exchange_order_id,
                    exchange_order_status=exchange_order_status,
                    exchange_average_filled_price=exchange_average_filled_price,
                    exchange_filled_size=coin_bought if exchange_order_id else None,
                    stop_loss_price=stop_price,
                    take_profit_price=target_price,
                    exit_mode=exit_mode,
                    regime=self.state.current_regime.regime if self.state.current_regime else None,
                )
                self.record_trade(trade)
                self.journal(symbol, "SHORT", reason, price, {"spend": spend, "quantity": coin_bought, "stop": stop_price, "target": target_price})
                logger.info(f"SHORT {symbol}: {coin_bought:.6f} @ {price:.6f} | {reason}")
            else:
                self.state.cash -= spend
                self.state.coin += coin_bought
                self.state.active_symbol = symbol
                self.state.entry_price = price
                self.state.highest_price = price
                self.state.stop_price = stop_price
                self.state.target_price = target_price
                self.state.exit_mode = exit_mode
                self.state.active_stop_order_id = None
                self.state.partial_take_profit_done = False
                self.state.last_price = price
                self.state.last_action_time = time.time()
                self.state.is_short = False

                self.state.positions[symbol] = {
                    "quantity": coin_bought,
                    "entry_price": price,
                    "highest_price": price,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "exit_mode": exit_mode,
                    "partial_take_profit_done": False,
                    "entry_cost": spend,
                    "opened_at": now_iso(),
                    "is_short": False,
                    "entry_time": time.time(),
                }

                trade = Trade(
                    time=now_iso(),
                    side="BUY",
                    symbol=symbol,
                    price=price,
                    quantity=coin_bought,
                    cash_after=self.state.cash,
                    coin_after=self.state.coin,
                    reason=f"{reason} | size {spend_reason} | {exit_mode} stop/target",
                    fee_paid=fee_paid,
                    exchange_order_id=exchange_order_id,
                    exchange_order_status=exchange_order_status,
                    exchange_average_filled_price=exchange_average_filled_price,
                    exchange_filled_size=coin_bought if exchange_order_id else None,
                    stop_loss_price=stop_price,
                    take_profit_price=target_price,
                    exit_mode=exit_mode,
                    regime=self.state.current_regime.regime if self.state.current_regime else None,
                )
                self.record_trade(trade)
                self.record_setup_buy(symbol, price, coin_bought, spend, fee_paid, reason, stop_price, target_price, exit_mode)
                self.journal(symbol, "BUY", reason, price, {"spend": spend, "quantity": coin_bought, "stop": stop_price, "target": target_price})
                logger.info(f"BUY {symbol}: {coin_bought:.6f} @ {price:.6f} | {reason}")

            if settings.get("telegram_alert_on_buy", True):
                self.send_telegram_alert(self.format_alert_trade(self.state.trades[-1]))

    def paper_sell(
        self,
        symbol: str,
        price: float,
        reason: str,
        quantity_override: float | None = None,
        fee_override: float | None = None,
        exchange_order_id: str | None = None,
        exchange_order_status: str | None = None,
        exchange_average_filled_price: float | None = None,
    ) -> None:
        with self.lock:
            settings = dict(self.state.settings)
            trade_fee = float(settings["trade_fee"])

            if abs(self.state.coin) <= 0 and symbol not in self.state.positions:
                self.state.last_signal = "SELL blocked: no position"
                self.journal(symbol, "BLOCK", "SELL blocked: no position", price)
                return

            position = self.state.positions.get(symbol, {})
            position_quantity = float(position.get("quantity", 0.0))
            coin_available = self.state.coin if self.state.active_symbol == symbol else position_quantity

            if abs(coin_available) <= 0:
                self.state.last_signal = f"SELL blocked: no {symbol} position"
                self.journal(symbol, "BLOCK", f"SELL blocked: no {symbol} position", price)
                return

            is_short = position.get("is_short", False) or self.state.is_short
            entry_price = float(position.get("entry_price") or self.state.entry_price or 0.0)
            sold_quantity = min(abs(coin_available), abs(quantity_override or coin_available))

            if is_short:
                pnl = (entry_price - price) * sold_quantity
            else:
                pnl = (price - entry_price) * sold_quantity

            gross = sold_quantity * price
            fee_paid = fee_override if fee_override is not None else gross * trade_fee
            cash_received = gross - fee_paid

            self.state.cash += cash_received

            if is_short:
                self.state.coin += sold_quantity
                if symbol in self.state.positions:
                    remaining = position_quantity + sold_quantity
                    if remaining >= 0:
                        self.state.positions.pop(symbol, None)
                    else:
                        position["quantity"] = remaining
                        self.state.positions[symbol] = position
            else:
                self.state.coin -= sold_quantity
                if symbol in self.state.positions:
                    remaining = position_quantity - sold_quantity
                    if remaining <= 0:
                        self.state.positions.pop(symbol, None)
                    else:
                        position["quantity"] = remaining
                        self.state.positions[symbol] = position

            position_closed = abs(self.state.coin) <= 0.0000000001 and len(self.state.positions) == 0
            if position_closed:
                self.state.coin = 0.0
                self.state.active_symbol = None
                self.state.entry_price = None
                self.state.highest_price = None
                self.state.stop_price = None
                self.state.target_price = None
                self.state.active_stop_order_id = None
                self.state.partial_take_profit_done = False
                self.state.is_short = False
                self.state.positions = {}

            self.state.last_price = price
            self.state.last_action_time = time.time()
            side = "SELL" if not is_short else "BUY"

            trade = Trade(
                time=now_iso(),
                side=side,
                symbol=symbol,
                price=price,
                quantity=-sold_quantity if is_short else sold_quantity,
                cash_after=self.state.cash,
                coin_after=self.state.coin,
                reason=reason,
                fee_paid=fee_paid,
                exchange_order_id=exchange_order_id,
                exchange_order_status=exchange_order_status,
                exchange_average_filled_price=exchange_average_filled_price,
                exchange_filled_size=sold_quantity if exchange_order_id else None,
                pnl=pnl,
                entry_price=entry_price,
                exit_price=price,
                stop_loss_price=position.get('stop_price'),
                take_profit_price=position.get('target_price'),
                exit_mode=position.get('exit_mode'),
                exit_reason=reason,
                regime=self.state.current_regime.regime if self.state.current_regime else None,
            )
            self.record_trade(trade)

            self.record_setup_sell(
                symbol,
                price,
                sold_quantity,
                cash_received,
                fee_paid,
                reason,
                position_closed,
            )

            if hasattr(self, 'self_learning_trader') and position_closed:
                setup_record = next(
                    (r for r in reversed(self.state.setup_records)
                     if r.symbol == symbol and r.status == "CLOSED"),
                    None
                )
                if setup_record and setup_record.signal_types:
                    success = pnl > 0
                    self.self_learning_trader.record_signal_outcome(
                        setup_record.signal_types,
                        pnl,
                        success
                    )

            self.journal(symbol, side, reason, price, {"quantity": sold_quantity, "pnl": pnl})
            logger.info(f"{side} {symbol}: {sold_quantity:.6f} @ {price:.6f} | PnL: {pnl:.4f} | {reason}")

            if settings.get("telegram_alert_on_sell", True):
                self.send_telegram_alert(self.format_alert_trade(trade, pnl))

    # ─── Setup Recording ─────────────────────────────────────────────

    def record_setup_buy(
        self,
        symbol: str,
        price: float,
        quantity: float,
        entry_cost: float,
        entry_fee: float,
        reason: str,
        stop_price: float | None = None,
        target_price: float | None = None,
        exit_mode: str | None = None,
    ) -> None:
        row = self.active_scan_row(symbol)
        signal_types = []
        if "self-learning" in reason.lower() or "self_learning" in reason.lower():
            if "signals:" in reason:
                signals_part = reason.split("signals:")[-1].strip()
                signal_types = [s.strip() for s in signals_part.split(",") if s.strip()]

        setup_id = str(uuid.uuid4())
        record = SetupRecord(
            id=setup_id,
            time=now_iso(),
            symbol=symbol,
            strategy=str(self.state.settings.get("strategy", "self_learning")),
            settings_key=setup_settings_key(self.state.settings),
            entry_price=price,
            entry_quantity=quantity,
            entry_cost=entry_cost,
            entry_fee=entry_fee,
            entry_reason=reason,
            entry_score=float(row.get("score") or 0.0),
            base_score=float(row.get("base_score") or row.get("score") or 0.0),
            edge_score=float(row.get("edge_score") or 0.0),
            regime=str(self.state.current_regime.regime if self.state.current_regime else "unknown"),
            support_distance_pct=row.get("support_distance_pct"),
            resistance_distance_pct=row.get("resistance_distance_pct"),
            sr_range_pct=row.get("sr_range_pct"),
            reward_risk=row.get("reward_risk"),
            signal_types=signal_types,
            stop_loss_price=stop_price,
            take_profit_price=target_price,
            exit_mode=exit_mode,
        )
        self.state.setup_records.append(record)
        self.state.setup_records = self.state.setup_records[-500:]
        self.state.active_setup_id = setup_id
        self.state.active_setup_ids[symbol] = setup_id

        self.db.save_setup_record(record)

    def record_setup_sell(
        self,
        symbol: str,
        price: float,
        sold_quantity: float,
        cash_received: float,
        fee_paid: float,
        reason: str,
        position_closed: bool,
    ) -> None:
        setup_id = self.state.active_setup_ids.get(symbol) or self.state.active_setup_id
        if not setup_id:
            return

        record = next(
            (
                item for item in reversed(self.state.setup_records)
                if item.id == setup_id and item.symbol == symbol
            ),
            None,
        )
        if not record or record.entry_quantity <= 0:
            return

        sold_fraction = min(1.0, sold_quantity / record.entry_quantity)
        cost_basis = record.entry_cost * sold_fraction
        pnl = cash_received - cost_basis
        record.closed_quantity += sold_quantity
        record.realized_pnl += pnl
        record.exit_fees += fee_paid
        record.exit_price = price
        record.exit_reason = reason
        record.exit_time = now_iso()

        if position_closed:
            record.status = "CLOSED"
            if record.entry_price and record.exit_price and record.entry_price > 0:
                is_short = any("short" in str(s).lower() for s in record.signal_types) or "SHORT" in str(reason).upper()
                if is_short:
                    record.pnl_pct = pct(record.entry_price - record.exit_price, record.entry_price)
                else:
                    record.pnl_pct = pct(record.exit_price - record.entry_price, record.entry_price)
            else:
                record.pnl_pct = pct(record.realized_pnl, record.entry_cost)

            self.state.active_setup_ids.pop(symbol, None)
            if self.state.active_setup_id == setup_id:
                self.state.active_setup_id = None

            self.db.save_setup_record(record)

    def active_scan_row(self, symbol: str) -> dict[str, Any]:
        return next(
            (row for row in self.state.scan_rows if row.get("symbol") == symbol),
            {},
        )

    # ─── OANDA Demo Trading ──────────────────────────────────────────

    def oanda_demo_buy(
        self,
        symbol: str,
        price: float,
        reason: str,
        candles: list[Candle] | None = None,
        is_short: bool = False,
    ) -> None:
        settings = self.state.settings
        quote_currency = settings.get("quote_currency", "GBP")

        if symbol in self.state.positions:
            self.state.last_signal = f"OANDA {'SHORT' if is_short else 'BUY'} blocked: {symbol} already has an open position"
            self.journal(symbol, "BLOCK", self.state.last_signal, price)
            return

        max_positions = int(settings.get("max_oanda_open_trades", 3))
        if len(self.state.positions) >= max_positions:
            self.state.last_signal = f"OANDA {'SHORT' if is_short else 'BUY'} blocked: max open trades reached ({max_positions})"
            self.journal(symbol, "BLOCK", self.state.last_signal, price)
            return

        if is_short and not settings.get("allow_short_selling", False):
            self.state.last_signal = f"OANDA SHORT blocked: short selling disabled"
            self.journal(symbol, "BLOCK", self.state.last_signal, price)
            return

        cash = self.state.cash

        risk_pct = float(settings.get("risk_per_trade_pct", 1.0)) / 100
        risk_cash = cash * risk_pct

        max_pct = float(settings.get("max_position_pct", 0.25))
        max_position_cash = cash * max_pct

        stop_pct = float(settings.get("stop_loss_pct", 2.0)) / 100

        position_value_gbp = min(risk_cash / stop_pct, max_position_cash)
        position_value_gbp = max(position_value_gbp, float(settings.get("min_order_value", 1.0)))
        position_value_gbp = min(position_value_gbp, cash)

        logger.info(f"Position sizing: cash={cash:.2f}, risk={risk_cash:.4f}, max={max_position_cash:.2f}, stop_pct={stop_pct:.4f}, position_gbp={position_value_gbp:.2f}")

        min_order = float(settings.get("min_order_value", 1.0))
        if position_value_gbp < min_order:
            self.state.last_signal = f"OANDA {'SHORT' if is_short else 'BUY'} blocked: position {position_value_gbp:.2f} below minimum {min_order:.2f} {quote_currency}"
            self.journal(symbol, "BLOCK", self.state.last_signal, price, {"position_value": position_value_gbp, "min_order": min_order})
            return

        exchange_rate = self.get_exchange_rate(symbol)

        position_value_gbp = min(risk_cash / stop_pct, max_position_cash)

        price_in_gbp = price / exchange_rate
        units = int(position_value_gbp / price_in_gbp)

        units = max(1, units)

        logger.info(f"Exchange rate: 1 GBP = {exchange_rate:.4f} {symbol[-3:]}, price in GBP: {price_in_gbp:.6f}")

        stop_price, target_price, exit_mode = self.exit_prices(
            entry_price=price,
            candles=candles or closes_to_candles(self.state.price_history.get(symbol, [])),
            settings=self.state.settings,
        )

        try:
            response = oanda_market_order(symbol, units, stop_price, target_price)
            fill = oanda_order_fill(response)
            fill_price = fill["price"] or price
            filled_units = fill["units"] or units
            fee = fill.get("commission", 0.0)

            if symbol.upper().endswith("JPY"):
                gbp_to_jpy = 190.0
                position_value_actual = abs(filled_units * fill_price) / gbp_to_jpy
            else:
                gbp_to_quote = 1.2
                position_value_actual = abs(filled_units * fill_price) / gbp_to_quote

            position_value_actual = min(position_value_actual, position_value_gbp)

            logger.info(f"OANDA fill: {filled_units} units @ {fill_price:.6f}, value: {position_value_actual:.2f} {quote_currency}, fee: {fee:.4f}")

        except Exception as e:
            self.state.last_signal = f"OANDA order failed: {e}"
            self.journal(symbol, "ERROR", self.state.last_signal, price)
            logger.error(f"OANDA order failed: {e}")
            return

        if is_short:
            self.state.cash += position_value_actual
            self.state.coin -= abs(filled_units)
            self.state.is_short = True
            self.state.positions[symbol] = {
                "quantity": -abs(filled_units),
                "entry_price": fill_price,
                "highest_price": fill_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "stop": stop_price,
                "target": target_price,
                "stop_loss": stop_price,
                "take_profit": target_price,
                "exit_mode": exit_mode,
                "partial_take_profit_done": False,
                "entry_cost": position_value_actual,
                "opened_at": now_iso(),
                "trade_id": fill.get("trade_id"),
                "is_short": True,
                "entry_time": time.time(),
            }
        else:
            self.state.cash -= position_value_actual
            self.state.coin += abs(filled_units)
            self.state.is_short = False
            self.state.positions[symbol] = {
                "quantity": abs(filled_units),
                "entry_price": fill_price,
                "highest_price": fill_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "stop": stop_price,
                "target": target_price,
                "exit_mode": exit_mode,
                "partial_take_profit_done": False,
                "entry_cost": position_value_actual,
                "opened_at": now_iso(),
                "trade_id": fill.get("trade_id"),
                "is_short": False,
                "entry_time": time.time(),
            }

        self.state.active_symbol = symbol
        self.state.entry_price = fill_price
        self.state.highest_price = fill_price
        self.state.stop_price = stop_price
        self.state.target_price = target_price
        self.state.last_price = fill_price
        self.state.last_action_time = time.time()
        self.state.last_signal = f"OANDA {'SHORT' if is_short else 'BUY'} {symbol} @ {fill_price:.6f} | Cash: {self.state.cash:.2f}"

        side = "SHORT" if is_short else "BUY"
        trade_reason = f"{reason} | OANDA demo order | {exit_mode} stop/target"

        trade = Trade(
            time=now_iso(),
            side=side,
            symbol=symbol,
            price=fill_price,
            quantity=abs(filled_units),
            cash_after=self.state.cash,
            coin_after=self.state.coin,
            reason=trade_reason,
            fee_paid=fee,
            exchange_order_id=fill["order_id"],
            exchange_order_status=fill["status"],
            exchange_average_filled_price=fill_price,
            exchange_filled_size=abs(filled_units),
            stop_loss_price=stop_price,
            take_profit_price=target_price,
            exit_mode=exit_mode,
            regime=self.state.current_regime.regime if self.state.current_regime else None,
        )
        self.record_trade(trade)
        self.record_setup_buy(symbol, fill_price, abs(filled_units), position_value_actual, fee, trade_reason, stop_price, target_price, exit_mode)
        self.journal(symbol, side, trade_reason, fill_price, {"spend": position_value_actual, "quantity": abs(filled_units)})

        logger.info(f"OANDA {side} {symbol}: {abs(filled_units)} units @ {fill_price:.6f} | Cash: {self.state.cash:.2f}")

    def oanda_demo_sell(
        self,
        symbol: str,
        price: float,
        reason: str,
        quantity_override: float | None = None,
    ) -> None:
        position = self.state.positions.get(symbol)
        if not position:
            self.state.last_signal = "OANDA SELL blocked: no position"
            self.journal(symbol, "BLOCK", self.state.last_signal, price)
            return

        current_quantity = float(position.get("quantity", 0.0))
        quantity = min(abs(current_quantity), abs(quantity_override or current_quantity))
        is_short = position.get("is_short", False)
        entry_price = float(position.get("entry_price", 0.0))

        units = -int(max(1, round(quantity))) if not is_short else int(max(1, round(quantity)))

        response = oanda_market_order(symbol, units)
        fill = oanda_order_fill(response)
        fill_price = fill["price"] or price
        filled_units = fill["units"] or abs(units)
        fee = fill["commission"]

        exchange_rate = self.get_exchange_rate(symbol)

        if is_short:
            pnl_quote = (entry_price - fill_price) * filled_units
        else:
            pnl_quote = (fill_price - entry_price) * filled_units

        pnl = pnl_quote / exchange_rate

        gross = filled_units * fill_price
        cash_received = gross / exchange_rate - fee

        self.state.cash += cash_received

        remaining = abs(current_quantity) - filled_units
        position_closed = remaining <= 0.0000000001

        if position_closed:
            self.state.positions.pop(symbol, None)
        else:
            position["quantity"] = -remaining if is_short else remaining
            self.state.positions[symbol] = position

        self.state.active_symbol = next(iter(self.state.positions), None)
        active_position = self.state.positions.get(self.state.active_symbol or "", {})
        self.state.entry_price = active_position.get("entry_price")
        self.state.highest_price = active_position.get("highest_price")
        self.state.stop_price = active_position.get("stop_price")
        self.state.target_price = active_position.get("target_price")
        self.state.is_short = active_position.get("is_short", False)
        self.state.last_price = fill_price
        self.state.last_action_time = time.time()

        side = "SELL" if not is_short else "BUY"
        trade_reason = f"{reason} | OANDA demo order | PnL: {pnl:.2f} GBP"

        trade = Trade(
            time=now_iso(),
            side=side,
            symbol=symbol,
            price=fill_price,
            quantity=filled_units,
            cash_after=self.state.cash,
            coin_after=remaining,
            reason=trade_reason,
            fee_paid=fee,
            exchange_order_id=fill["order_id"],
            exchange_order_status=fill["status"],
            exchange_average_filled_price=fill_price,
            exchange_filled_size=filled_units,
            pnl=pnl,
            entry_price=entry_price,
            exit_price=fill_price,
            stop_loss_price=position.get("stop_price"),
            take_profit_price=position.get("target_price"),
            exit_mode=position.get("exit_mode"),
            exit_reason=reason,
            regime=self.state.current_regime.regime if self.state.current_regime else None,
        )
        self.record_trade(trade)
        self.record_setup_sell(symbol, fill_price, filled_units, cash_received, fee, trade_reason, position_closed)
        self.journal(symbol, side, trade_reason, fill_price, {"quantity": filled_units, "pnl": pnl})
        self.journal(symbol, "INFO", f"OANDA demo {side} filled", fill_price, fill)
        logger.info(f"OANDA {side} {symbol}: {filled_units} units @ {fill_price:.6f} | PnL: {pnl:.2f} GBP")

    # ─── Live Trading ────────────────────────────────────────────────

    def should_live_trade(self) -> bool:
        settings = self.state.settings
        return (
            bool(settings.get("live_trading_enabled"))
            and settings.get("asset_class", "crypto") == "crypto"
            and settings.get("exchange") == "coinbase"
            and coinbase_live_is_armed()
        )

    def live_status(self) -> dict[str, Any]:
        with self.lock:
            account_type = self.state.settings.get("oanda_account_type", "standard")
            account_type_label = "Spread Bet" if account_type == "spreadbet" else "CFD/Forex"

            if self.state.settings.get("asset_class", "crypto") == "forex":
                exchange = self.state.settings.get("exchange")
                demo_orders_enabled = bool(self.state.settings.get("oanda_demo_trading_enabled"))
                demo_orders_armed = exchange == "oanda_demo" and demo_orders_enabled and oanda_demo_orders_armed()

                websocket_enabled = bool(self.state.settings.get("websocket_enabled", False))
                websocket_status = self.state.websocket_status

                if websocket_enabled and websocket_status == "crypto websocket only":
                    websocket_status = "connecting..."

                message = (
                    oanda_demo_status_message()
                    if exchange == "oanda_demo" and demo_orders_enabled
                    else (
                        f"OANDA {account_type_label} provides real account candles/pricing; "
                        f"OANDA demo order placement is disabled."
                        if exchange == "oanda_demo"
                        else f"Forex demo mode uses synthetic paper data. Select OANDA demo for real OANDA practice data."
                    )
                )

                return {
                    "enabled": demo_orders_enabled,
                    "armed": demo_orders_armed,
                    "ready": demo_orders_armed,
                    "daily_spend": round(self.state.live_daily_spend, 2),
                    "max_daily_live_loss_gbp": self.state.settings.get("max_daily_live_loss_gbp"),
                    "max_live_order_gbp": self.state.settings.get("max_live_order_gbp"),
                    "live_order_type": "oanda_demo" if demo_orders_enabled else "paper",
                    "live_limit_offset_pct": self.state.settings.get("live_limit_offset_pct"),
                    "native_stop_enabled": False,
                    "max_live_spread_pct": self.state.settings.get("max_live_spread_pct"),
                    "min_live_quote_volume": self.state.settings.get("min_live_quote_volume"),
                    "open_orders": 0,
                    "websocket_available": True,
                    "websocket_status": websocket_status,
                    "websocket_last_seen": self.state.websocket_last_seen,
                    "message": message,
                    "news_guard_enabled": bool(self.state.settings.get("news_guard_enabled", False)),
                    "news_guard_status": self.state.news_guard_status,
                    "account_type": account_type,
                    "account_type_label": account_type_label,
                }

            armed = coinbase_live_is_armed()
            return {
                "enabled": bool(self.state.settings.get("live_trading_enabled")),
                "armed": armed,
                "ready": armed and self.state.settings.get("exchange") == "coinbase",
                "daily_spend": round(self.state.live_daily_spend, 2),
                "max_daily_live_loss_gbp": self.state.settings.get("max_daily_live_loss_gbp"),
                "max_live_order_gbp": self.state.settings.get("max_live_order_gbp"),
                "live_order_type": self.state.settings.get("live_order_type"),
                "live_limit_offset_pct": self.state.settings.get("live_limit_offset_pct"),
                "native_stop_enabled": self.state.settings.get("native_stop_enabled"),
                "max_live_spread_pct": self.state.settings.get("max_live_spread_pct"),
                "min_live_quote_volume": self.state.settings.get("min_live_quote_volume"),
                "open_orders": len([item for item in self.state.open_orders if item.status not in {"FILLED", "CANCELLED", "FAILED", "EXPIRED"}]),
                "websocket_enabled": bool(self.state.settings.get("websocket_enabled")),
                "websocket_available": WEBSOCKET_AVAILABLE,
                "websocket_status": self.state.websocket_status,
                "websocket_last_seen": self.state.websocket_last_seen,
                "message": coinbase_live_status_message(),
                "news_guard_enabled": bool(self.state.settings.get("news_guard_enabled", False)),
                "news_guard_status": self.state.news_guard_status,
                "account_type": "n/a",
                "account_type_label": "Crypto",
            }

    def daily_live_risk_metrics(self) -> dict[str, float]:
        """Return today's live-trading risk figures in quote currency.

        Realised P/L comes from setup records whose latest exit happened today.
        SetupRecord.realized_pnl is net of the recorded entry cost and exit fees.
        Unrealised P/L is calculated only from currently open local positions.
        The daily loss guard uses realised + unrealised P/L and NEVER blocks exits.
        """
        today = datetime.now().astimezone().date()

        def is_today(value: Any) -> bool:
            if not value:
                return False
            try:
                text = str(value).replace("Z", "+00:00")
                dt = datetime.fromisoformat(text)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
                return dt.astimezone().date() == today
            except Exception:
                return str(value)[:10] == today.isoformat()

        with self.lock:
            setup_records = list(self.state.setup_records)
            positions = {k: dict(v) for k, v in self.state.positions.items()}
            price_history = {k: list(v) for k, v in self.state.price_history.items()}
            last_price = self.state.last_price

        realised = sum(
            float(record.realized_pnl or 0.0)
            for record in setup_records
            if is_today(record.exit_time)
        )

        unrealised = 0.0
        for symbol, position in positions.items():
            quantity = float(position.get("quantity", 0.0) or 0.0)
            entry = float(position.get("entry_price", 0.0) or 0.0)
            history = price_history.get(symbol, [])
            current = float(history[-1]) if history else float(last_price or 0.0)
            if not entry or not current or not quantity:
                continue
            if bool(position.get("is_short", False)) or quantity < 0:
                unrealised += (entry - current) * abs(quantity)
            else:
                unrealised += (current - entry) * abs(quantity)

        return {
            "realized": realised,
            "unrealized": unrealised,
            "risk_pnl": realised + unrealised,
        }

    def live_buy(
        self,
        symbol: str,
        price: float,
        reason: str,
        candles: list[Candle] | None = None,
        is_short: bool = False,
    ) -> None:
        self.roll_live_daily_spend_if_needed()
        with self.lock:
            settings = dict(self.state.settings)
            cash = self.state.cash
            live_daily_spend = self.state.live_daily_spend
            positions = dict(self.state.positions)

        max_order = float(settings["max_live_order_gbp"])
        max_daily_loss = float(settings["max_daily_live_loss_gbp"])
        max_daily_spend = float(settings.get("max_daily_live_spend_quote", 250.0))
        max_coinbase_positions = int(settings.get("max_coinbase_open_trades", 3))

        if candles is None:
            with self.lock:
                candles = closes_to_candles(self.state.price_history.get(symbol, []))

        regime_adapt = self.get_regime_adaptations()
        stop_mult = regime_adapt.get('stop_multiplier', 1.0)
        tp_mult = regime_adapt.get('take_profit_multiplier', 1.0)

        original_stop = settings.get('stop_loss_pct', 2.0)
        original_tp = settings.get('take_profit_pct', 3.0)
        settings['stop_loss_pct'] = original_stop * stop_mult
        settings['take_profit_pct'] = original_tp * tp_mult

        try:
            paper_spend, spend_reason = self.calculate_position_size(
                cash=cash,
                entry_price=price,
                candles=candles,
                symbol=symbol,
                position_side="SHORT" if is_short else "LONG"
            )
        finally:
            settings['stop_loss_pct'] = original_stop
            settings['take_profit_pct'] = original_tp

        quote_size = round(min(max_order, paper_spend), 2)

        minimum_order = max(1.0, float(settings.get("min_order_value", 1.0)))
        if quote_size < minimum_order:
            with self.lock:
                self.state.last_signal = f"LIVE {'SHORT' if is_short else 'BUY'} blocked: order below {settings['quote_currency']} {minimum_order:.2f}"
                self.journal(symbol, "BLOCK", self.state.last_signal, price, {"quote_size": quote_size})
            return

        if len(positions) >= max_coinbase_positions:
            with self.lock:
                self.state.last_signal = f"LIVE {'SHORT' if is_short else 'BUY'} blocked: max coinbase trades reached"
                self.journal(symbol, "BLOCK", self.state.last_signal, price, {"quote_size": quote_size})
            return

        # Two independent live-entry guards:
        # 1) spend cap limits gross capital committed today;
        # 2) loss cap limits today's realised + current unrealised P/L.
        # Neither guard is used by SELL/STOP/emergency-exit paths.
        if live_daily_spend + quote_size > max_daily_spend:
            with self.lock:
                self.state.last_signal = (
                    f"LIVE {'SHORT' if is_short else 'BUY'} blocked: daily live spend cap reached "
                    f"({live_daily_spend:.2f} + {quote_size:.2f} > {max_daily_spend:.2f} {settings['quote_currency']})"
                )
                self.journal(symbol, "BLOCK", self.state.last_signal, price, {
                    "quote_size": quote_size,
                    "daily_spend": live_daily_spend,
                    "max_daily_spend": max_daily_spend,
                })
            return

        risk = self.daily_live_risk_metrics()
        if risk["risk_pnl"] <= -max_daily_loss:
            with self.lock:
                self.state.last_signal = (
                    f"LIVE {'SHORT' if is_short else 'BUY'} blocked: daily loss limit reached "
                    f"({risk['risk_pnl']:.2f} <= -{max_daily_loss:.2f} {settings['quote_currency']}; "
                    f"realised {risk['realized']:.2f}, unrealised {risk['unrealized']:.2f})"
                )
                self.journal(symbol, "BLOCK", self.state.last_signal, price, {
                    "daily_realized_pnl": risk["realized"],
                    "daily_unrealized_pnl": risk["unrealized"],
                    "daily_risk_pnl": risk["risk_pnl"],
                    "max_daily_loss": max_daily_loss,
                })
            return

        guard = live_market_guard(
            exchange=str(settings["exchange"]),
            symbol=symbol,
            quote_currency=str(settings["quote_currency"]),
            granularity=int(settings.get("live_granularity", 3600)),
            candle_count=int(settings.get("live_candle_count", 300)),
            max_spread_pct=float(settings["max_live_spread_pct"]),
            min_quote_volume=float(settings["min_live_quote_volume"]),
        )
        if not guard["ok"]:
            with self.lock:
                self.state.last_signal = f"LIVE {'SHORT' if is_short else 'BUY'} blocked: {guard['reason']}"
                self.journal(symbol, "BLOCK", self.state.last_signal, price, guard)
            return

        gbp_available = coinbase_available_balance(settings["quote_currency"])
        if gbp_available < quote_size:
            with self.lock:
                self.state.last_signal = f"LIVE {'SHORT' if is_short else 'BUY'} blocked: only {settings['quote_currency']} {gbp_available:.2f} available"
                self.journal(symbol, "BLOCK", self.state.last_signal, price, {"available": gbp_available, "quote_size": quote_size})
            return

        product_id = f"{symbol}-{settings['quote_currency']}"
        order_type = str(settings.get("live_order_type", "market"))
        limit_offset = float(settings.get("live_limit_offset_pct", 0.05)) / 100

        active_exchange = settings.get("active_exchange", "coinbase")
        connector = self.connectors.get(active_exchange)
        if not connector:
            raise RuntimeError(f"No connector available for {active_exchange}")

        side = "SELL" if is_short else "BUY"

        stop_price, target_price, exit_mode = self.exit_prices(
            entry_price=price,
            candles=candles or closes_to_candles(self.state.price_history.get(symbol, [])),
            settings=self.state.settings,
        )

        try:
            if order_type in {"limit", "bracket", "native_stop_scaffold"}:
                best_price, _, _ = self.price_aggregator.get_best_price(symbol, side=side)
                if side == "BUY":
                    limit_price = self.coinbase_round_price(best_price * (1 + limit_offset), product_id)
                else:
                    limit_price = self.coinbase_round_price(best_price * (1 - limit_offset), product_id)

                ok, volume, recommended = self.price_aggregator.check_liquidity(symbol, side, quote_size)
                if not ok:
                    with self.lock:
                        self.state.last_signal = f"LIVE {'SHORT' if is_short else 'BUY'} blocked: insufficient liquidity on {active_exchange} (need {quote_size}, have {volume})"
                        self.journal(symbol, "BLOCK", self.state.last_signal, price)
                    return

                base_size = quote_size / limit_price if limit_price > 0 else 0.0
                base_size = self.coinbase_round_size(base_size, product_id)
                order = coinbase_limit_order(product_id, side, base_size, limit_price)

            else:
                # For Coinbase, submit through Auxo's native order helper so the
                # client_order_id survives and uncertain submissions can be
                # recovered before any retry is considered.
                if active_exchange == "coinbase":
                    if side == "BUY":
                        order = coinbase_market_order(
                            product_id=product_id,
                            side=side,
                            quote_size=quote_size,
                        )
                    else:
                        best_price, _, _ = self.price_aggregator.get_best_price(symbol, side="SELL")
                        base_size = quote_size / best_price if best_price > 0 else 0.0
                        base_size = self.coinbase_round_size(base_size, product_id)
                        order = coinbase_market_order(
                            product_id=product_id,
                            side=side,
                            base_size=base_size,
                        )
                else:
                    # Preserve the existing connector path for non-Coinbase
                    # exchanges. These connectors have their own order IDs.
                    if side == "BUY":
                        raw_order = connector.market_buy(symbol, quote_size)
                    else:
                        best_price, _, _ = self.price_aggregator.get_best_price(symbol, side="SELL")
                        base_size = quote_size / best_price if best_price > 0 else 0.0
                        base_size = self.coinbase_round_size(base_size, product_id)
                        raw_order = connector.market_sell(symbol, base_size)

                    try:
                        order_id = raw_order.get("order_id") or raw_order.get("orderId") or str(raw_order.get("id"))
                    except Exception:
                        order_id = str(uuid.uuid4())
                    order = {"order_id": order_id, "raw": raw_order}

            order_id = coinbase_order_id(order)
            managed = self.track_order(
                order_id,
                symbol,
                product_id,
                side,
                "ENTRY",
                order_type,
                price=limit_price if order_type != "market" else price,
                base_size=base_size if order_type != "market" else None,
                quote_size=quote_size,
                reason=f"{reason} | size {spend_reason}",
                client_order_id=order.get("client_order_id"),
                details={
                    "native_stop_requested": bool(settings.get("native_stop_enabled")) or order_type in {"bracket", "native_stop_scaffold"},
                    "stop_price": stop_price,
                    "exit_mode": exit_mode,
                    "is_short": is_short,
                },
            )
            fill = coinbase_reconcile_order(order_id)
            if self.apply_reconciled_order(managed, fill):
                return
            if fill["filled_size"] <= 0:
                with self.lock:
                    self.state.last_signal = f"LIVE {'SHORT' if is_short else 'BUY'} pending/unfilled: {order_id}"
                    self.journal(symbol, "INFO", self.state.last_signal, price, {"order": order, "fill": fill})

        except Exception as e:
            with self.lock:
                self.state.last_signal = f"LIVE {'SHORT' if is_short else 'BUY'} error: {e}"
                self.journal(symbol, "ERROR", self.state.last_signal, price)
            logger.error(f"Live order error: {e}")
            raise

    def live_sell(
        self,
        symbol: str,
        price: float,
        reason: str,
        quantity_override: float | None = None,
        is_short: bool = False,
    ) -> None:
        settings = dict(self.state.settings)

        # Build the Coinbase product ID first because we need it
        # to determine the permitted quantity precision.
        product_id = f"{symbol}-{settings['quote_currency']}"

        base_available = coinbase_available_balance(symbol)

        desired_size = (
            quantity_override
            if quantity_override is not None
            else abs(self.state.coin)
        )

        base_size = min(base_available, desired_size)

        # Coinbase requires base_size to conform to the
        # product's base_increment.
        base_size = self.coinbase_round_size(
            base_size,
            product_id,
        )

        if base_size <= 0:
            with self.lock:
                self.state.last_signal = (
                    f"LIVE SELL blocked: no sellable {symbol} balance available"
                )
                self.journal(
                    symbol,
                    "BLOCK",
                    self.state.last_signal,
                    price,
                )
            return

        if self.state.active_stop_order_id:
            stop_order_id = self.state.active_stop_order_id
            try:
                cancel_response = coinbase_cancel_orders([stop_order_id])
                self.journal(
                    symbol,
                    "INFO",
                    f"Cancelled native stop before live sell: {stop_order_id}",
                    price,
                    {"cancel_response": cancel_response},
                )
                self.state.active_stop_order_id = None
            except Exception as exc:
                with self.lock:
                    self.state.last_signal = f"LIVE SELL blocked: could not cancel native stop {stop_order_id}: {exc}"
                    self.journal(symbol, "BLOCK", self.state.last_signal, price)
                return

        order_type = str(settings.get("live_order_type", "market"))

        if order_type == "limit":
            limit_offset = float(settings.get("live_limit_offset_pct", 0.05)) / 100
            limit_price = self.coinbase_round_price(price * (1 - limit_offset), product_id)
            order = coinbase_limit_order(
                product_id=product_id,
                side="SELL" if not is_short else "BUY",
                base_size=base_size,
                limit_price=limit_price,
            )
        else:
            order = coinbase_market_order(
                product_id=product_id,
                side="SELL" if not is_short else "BUY",
                base_size=base_size,
            )

        try:
            order_id = coinbase_order_id(order)
        except Exception as exc:
            with self.lock:
                self.state.last_signal = (
                    f"LIVE SELL order submission failed for {symbol}: {exc}"
                )

                self.journal(
                    symbol,
                    "ERROR",
                    self.state.last_signal,
                    price,
                    {
                        "product_id": product_id,
                        "base_size": base_size,
                        "order_type": order_type,
                        "reason": reason,
                        "coinbase_response": order,
                    },
                )

            logger.error(
                "LIVE SELL order submission failed for %s: %s",
                symbol,
                exc,
            )

            return

        managed = self.track_order(
            order_id,
            symbol,
            product_id,
            "SELL" if not is_short else "BUY",
            "EXIT",
            order_type,
            price=price,
            base_size=base_size,
            reason=reason,
            client_order_id=order.get("client_order_id"),
        )
        fill = coinbase_reconcile_order(order_id)
        if self.apply_reconciled_order(managed, fill):
            if abs(self.state.coin) > 0 and bool(settings.get("native_stop_enabled")) and self.state.entry_price:
                self.submit_native_stop_for_position(
                    ManagedOrder(
                        order_id=order_id,
                        symbol=symbol,
                        product_id=product_id,
                        side="SELL" if not is_short else "BUY",
                        role="EXIT",
                        order_type=order_type,
                        status="FILLED",
                        created_at=now_iso(),
                        updated_at=now_iso(),
                        expires_at=time.time(),
                        details={"exit_mode": "post-partial"},
                    ),
                    self.state.entry_price,
                )
            return
        if fill["filled_size"] <= 0:
            with self.lock:
                self.state.last_signal = f"LIVE SELL pending/unfilled: {order_id}"
                self.journal(symbol, "INFO", self.state.last_signal, price, {"order": order, "fill": fill})

    # ─── Order Management ────────────────────────────────────────────

    def track_order(
        self,
        order_id: str,
        symbol: str,
        product_id: str,
        side: str,
        role: str,
        order_type: str,
        price: float | None = None,
        base_size: float | None = None,
        quote_size: float | None = None,
        reason: str = "",
        details: dict[str, Any] | None = None,
        client_order_id: str | None = None,
    ) -> ManagedOrder:
        now_text = now_iso()
        order = ManagedOrder(
            order_id=order_id,
            symbol=symbol,
            product_id=product_id,
            side=side.upper(),
            role=role,
            order_type=order_type,
            status="OPEN",
            created_at=now_text,
            updated_at=now_text,
            expires_at=time.time() + int(self.state.settings.get("order_expiry_seconds", 180)),
            price=price,
            base_size=base_size,
            quote_size=quote_size,
            reason=reason,
            details=details or {},
            client_order_id=client_order_id,
        )
        self.state.open_orders.append(order)
        self.state.open_orders = self.state.open_orders[-120:]
        self.audit("ORDER_TRACKED", order=asdict(order))
        return order

    def managed_order(self, order_id: str) -> ManagedOrder | None:
        return next((item for item in self.state.open_orders if item.order_id == order_id), None)

    def manage_open_orders(self) -> None:
        for order in list(self.state.open_orders):
            if order.status in {"FILLED", "CANCELLED", "FAILED", "EXPIRED"}:
                continue

            try:
                fill = coinbase_reconcile_order(order.order_id)
            except Exception as exc:
                order.updated_at = now_iso()
                order.status = "RECONCILE_ERROR"
                self.audit("ORDER_RECONCILE_ERROR", order_id=order.order_id, error=str(exc))
                logger.warning(f"Order reconcile error: {exc}")
                continue

            self.apply_reconciled_order(order, fill)
            if order.status == "FILLED":
                continue

            if time.time() >= order.expires_at:
                self.expire_order(order)

    def apply_reconciled_order(self, order: ManagedOrder, fill: dict[str, Any]) -> bool:
        order.updated_at = now_iso()
        order.status = fill.get("status", "UNKNOWN")
        if fill["filled_size"] <= 0 or order.local_applied:
            return False

        filled_price = fill["average_price"] or order.price or self.state.last_price or 0.0
        is_short = order.details.get("is_short", False)

        if order.role == "ENTRY":
            filled_quote = (fill["filled_value"] or order.quote_size or 0.0) + fill["total_fee"]
            self.state.live_daily_spend += min(order.quote_size or filled_quote, filled_quote)
            self.paper_buy(
                order.symbol,
                filled_price,
                f"LIVE {order.order_type.upper()} BUY filled {order.order_id} | {order.reason}",
                spend_override=filled_quote,
                fee_override=fill["total_fee"],
                quantity_override=fill["filled_size"],
                exchange_order_id=order.order_id,
                exchange_order_status=order.status,
                exchange_average_filled_price=filled_price,
                stop_override=order.details.get("stop_price"),
                target_override=order.details.get("target_price"),
                is_short=is_short,
            )
            if order.details.get("native_stop_requested"):
                self.submit_native_stop_for_position(order, filled_price)
        elif order.role in {"EXIT", "STOP"}:
            self.paper_sell(
                order.symbol,
                filled_price,
                f"LIVE {order.role} filled {order.order_id} | {order.reason}",
                quantity_override=fill["filled_size"],
                fee_override=fill["total_fee"],
                exchange_order_id=order.order_id,
                exchange_order_status=order.status,
                exchange_average_filled_price=filled_price,
            )
            if order.role == "STOP" and abs(self.state.coin) <= 0 and not self.state.positions:
                self.state.active_stop_order_id = None

        order.local_applied = True
        order.status = "FILLED"
        order.updated_at = now_iso()
        self.audit("ORDER_FILLED_APPLIED", order=asdict(order), fill=fill)

        # paper_buy()/paper_sell() are also used to apply confirmed LIVE fills so
        # that Auxo keeps its trade history, realised P/L, fees and position state.
        # Their local cash arithmetic is authoritative in paper mode, but not in
        # Coinbase live mode. After every confirmed live Coinbase fill, replace
        # local quote cash with Coinbase's actual available quote balance. This
        # keeps Equity/Total P&L anchored to the immutable starting_cash baseline
        # and prevents local fill accounting from drifting away from the exchange.
        if (
            self.state.settings.get("live_trading_enabled")
            and self.state.settings.get("asset_class", "crypto") == "crypto"
            and self.state.settings.get("exchange") == "coinbase"
        ):
            self.sync_live_balance_always()

        return True

    def expire_order(self, order: ManagedOrder) -> None:
        try:
            cancel_response = coinbase_cancel_orders([order.order_id])
            order.status = "EXPIRED"
            order.updated_at = now_iso()
            if self.state.active_stop_order_id == order.order_id:
                self.state.active_stop_order_id = None
            self.audit("ORDER_EXPIRED_CANCELLED", order=asdict(order), cancel_response=cancel_response)
        except Exception as exc:
            order.status = "CANCEL_FAILED"
            order.updated_at = now_iso()
            self.audit("ORDER_EXPIRE_CANCEL_FAILED", order=asdict(order), error=str(exc))
            logger.warning(f"Order expiry/cancel failed: {exc}")
            return

        if (
            bool(self.state.settings.get("order_replace_enabled"))
            and order.retry_count < int(self.state.settings.get("order_retry_limit", 1))
            and order.role in {"ENTRY", "EXIT"}
        ):
            self.replace_order(order)

    def replace_order(self, order: ManagedOrder) -> None:
        try:
            if order.order_type == "limit" and order.price and order.base_size:
                replacement = coinbase_limit_order(
                    product_id=order.product_id,
                    side=order.side,
                    base_size=order.base_size,
                    limit_price=order.price,
                )
            elif order.side == "BUY" and order.quote_size:
                replacement = coinbase_market_order(order.product_id, order.side, quote_size=order.quote_size)
            elif order.base_size:
                replacement = coinbase_market_order(order.product_id, order.side, base_size=order.base_size)
            else:
                return
            replacement_id = coinbase_order_id(replacement)
            new_order = self.track_order(
                replacement_id,
                order.symbol,
                order.product_id,
                order.side,
                order.role,
                order.order_type,
                price=order.price,
                base_size=order.base_size,
                quote_size=order.quote_size,
                reason=order.reason,
                details=order.details,
                client_order_id=replacement.get("client_order_id"),
            )
            new_order.retry_count = order.retry_count + 1
            self.audit("ORDER_REPLACED", old_order_id=order.order_id, new_order_id=replacement_id)
            logger.info(f"Order replaced: {order.order_id} → {replacement_id}")
        except Exception as exc:
            self.audit("ORDER_REPLACE_FAILED", order_id=order.order_id, error=str(exc))
            logger.warning(f"Order replace failed: {exc}")

    def submit_native_stop_for_position(self, entry_order: ManagedOrder, entry_price: float) -> None:
        if not self.state.settings.get("native_stop_enabled", False):
            return

        symbol = entry_order.symbol
        position = self.state.positions.get(symbol, {})
        desired_size = abs(float(position.get("quantity") or entry_order.base_size or 0.0))

        # Fall back to the legacy single-position value only when this really is
        # the active symbol.  Never use the aggregate self.state.coin value for
        # a different symbol in a multi-position Coinbase account.
        if desired_size <= 0 and self.state.active_symbol == symbol:
            desired_size = abs(float(self.state.coin or 0.0))
        if desired_size <= 0:
            logger.warning("Native stop skipped for %s: no local position quantity", symbol)
            return

        stop_price = float(entry_order.details.get("stop_price") or 0.0)
        exit_mode = str(entry_order.details.get("exit_mode") or "fixed")
        if stop_price <= 0:
            candles = closes_to_candles(self.state.price_history.get(symbol, []))
            stop_price, _, exit_mode = self.exit_prices(entry_price, candles, self.state.settings)

        product_id = entry_order.product_id
        limit_price = self.coinbase_round_price(stop_price * 0.995, product_id)
        stop_price = self.coinbase_round_price(stop_price, product_id)
        min_size = self.coinbase_min_order_size(product_id)

        try:
            # A newly filled Coinbase BUY can take a short time to appear in the
            # available balance.  Retry the exchange balance before declaring the
            # native stop impossible.  The exchange remains authoritative for size.
            available_size = 0.0
            base_size = 0.0
            balance_error = None
            for attempt in range(4):
                try:
                    available_size = float(coinbase_available_balance(symbol) or 0.0)
                    base_size = self.coinbase_round_size(
                        min(desired_size, available_size),
                        product_id,
                    )
                    balance_error = None
                except Exception as exc:
                    balance_error = exc
                    available_size = 0.0
                    base_size = 0.0

                if base_size >= min_size and base_size > 0:
                    break
                if attempt < 3:
                    time.sleep(0.75)

            if balance_error is not None:
                raise RuntimeError(
                    f"Could not verify Coinbase {symbol} balance for native stop: {balance_error}"
                ) from balance_error

            if base_size <= 0 or base_size < min_size:
                raise RuntimeError(
                    f"Native stop size unavailable/too small for {product_id}: "
                    f"desired={desired_size:.12f}, available={available_size:.12f}, "
                    f"rounded={base_size:.12f}, minimum={min_size:.12f}"
                )

            stop_order = coinbase_stop_limit_order(
                product_id=product_id,
                side="SELL" if not self.state.is_short else "BUY",
                base_size=base_size,
                stop_price=stop_price,
                limit_price=limit_price,
            )
            stop_order_id = coinbase_order_id(stop_order)
            self.state.active_stop_order_id = stop_order_id
            self.track_order(
                stop_order_id,
                symbol,
                product_id,
                "SELL" if not self.state.is_short else "BUY",
                "STOP",
                "stop_limit",
                price=stop_price,
                base_size=base_size,
                reason=f"{exit_mode} native stop",
                details={
                    "entry_order_id": entry_order.order_id,
                    "desired_size": desired_size,
                    "exchange_available_at_submit": available_size,
                },
                client_order_id=stop_order.get("client_order_id"),
            )
            self.journal(
                symbol,
                "INFO",
                f"Native stop-limit submitted {stop_order_id} via {exit_mode} stop",
                stop_price,
                {
                    "entry_order_id": entry_order.order_id,
                    "desired_size": desired_size,
                    "exchange_available": available_size,
                    "submitted_size": base_size,
                    "stop_order": stop_order,
                },
            )
            logger.info(
                "Native stop submitted: %s at %.6f size=%.12f available=%.12f",
                stop_order_id, stop_price, base_size, available_size,
            )
        except Exception as e:
            error_text = str(e)
            if self.should_live_trade():
                # FAIL CLOSED: an armed live position must never be left relying only
                # on the process-local simulated stop. Attempt an immediate market exit.
                logger.critical(
                    f"Native stop submission failed for LIVE position: {error_text}. "
                    "Attempting emergency market exit."
                )
                self.journal(
                    symbol,
                    "CRITICAL",
                    "LIVE native stop failed; attempting emergency market exit",
                    stop_price,
                    {"error": error_text},
                )

                try:
                    if not self.state.is_short:
                        exchange_available = float(coinbase_available_balance(symbol) or 0.0)
                        emergency_size = self.coinbase_round_size(
                            min(desired_size, exchange_available),
                            product_id,
                        )

                        if emergency_size <= 0 or emergency_size < min_size:
                            logger.warning(
                                "Emergency exit skipped for %s: available %.12f rounds to %.12f, "
                                "below Coinbase minimum %.12f. Position may already be closed or dust.",
                                symbol, exchange_available, emergency_size, min_size,
                            )
                            self.journal(
                                symbol,
                                "WARNING",
                                "Emergency exit skipped: Coinbase balance is zero/dust; "
                                "position may already be closed",
                                stop_price,
                                {
                                    "desired_size": desired_size,
                                    "exchange_available": exchange_available,
                                    "rounded_size": emergency_size,
                                    "minimum_size": min_size,
                                    "native_stop_error": error_text,
                                },
                            )
                            return
                    else:
                        emergency_size = self.coinbase_round_size(desired_size, product_id)
                        if emergency_size <= 0 or emergency_size < min_size:
                            raise RuntimeError(
                                f"Emergency exit size {emergency_size} is below minimum {min_size}"
                            )

                    emergency = coinbase_market_order(
                        product_id=product_id,
                        side="BUY" if self.state.is_short else "SELL",
                        base_size=emergency_size,
                    )
                    emergency_id = coinbase_order_id(emergency)
                    fill = coinbase_reconcile_order(emergency_id)
                    filled_size = float(fill.get("filled_size") or 0.0)
                    filled_price = float(fill.get("average_price") or stop_price or 0.0)
                    if filled_size > 0:
                        self.paper_sell(
                            symbol,
                            filled_price,
                            "EMERGENCY MARKET EXIT after native stop failure",
                            quantity_override=filled_size,
                            fee_override=float(fill.get("total_fee") or 0.0),
                            exchange_order_id=emergency_id,
                            exchange_order_status=fill.get("status"),
                            exchange_average_filled_price=filled_price,
                        )
                        self.state.active_stop_order_id = None
                        self.state.stop_price = None
                        logger.critical(
                            f"Emergency live exit filled: {emergency_id} "
                            f"{filled_size:.10f} @ {filled_price:.8f}"
                        )
                    else:
                        raise RuntimeError(
                            f"Emergency market exit returned no filled size "
                            f"(order {emergency_id}, status={fill.get('status')})"
                        )
                except Exception as emergency_error:
                    self.state.stop_price = None
                    self.journal(
                        symbol,
                        "CRITICAL",
                        "LIVE position may be unprotected: emergency market exit failed",
                        stop_price,
                        {
                            "native_stop_error": error_text,
                            "emergency_exit_error": str(emergency_error),
                        },
                    )
                    raise RuntimeError(
                        "Native protective stop failed and emergency live exit failed: "
                        f"{emergency_error}"
                    ) from emergency_error
            else:
                # Paper mode can safely fall back to the existing process-local stop.
                logger.warning(
                    f"Native stop submission failed: {error_text}. "
                    "Falling back to simulated stop in paper mode."
                )
                self.state.stop_price = stop_price
                self.journal(
                    symbol,
                    "WARNING",
                    f"Native stop failed, using simulated stop at {stop_price:.6f}",
                    stop_price,
                    {"error": error_text},
                )

    def sync_native_stop_fill(self) -> None:
        if not self.state.active_stop_order_id or not self.state.active_symbol:
            return
        stop_order_id = self.state.active_stop_order_id
        fill = coinbase_reconcile_order(stop_order_id)
        if fill["filled_size"] <= 0:
            return

        symbol = self.state.active_symbol
        filled_price = fill["average_price"] or self.state.last_price or 0.0
        self.paper_sell(
            symbol,
            filled_price,
            f"NATIVE STOP filled {stop_order_id}",
            quantity_override=fill["filled_size"],
            fee_override=fill["total_fee"],
            exchange_order_id=stop_order_id,
            exchange_order_status=fill["status"],
            exchange_average_filled_price=filled_price,
        )
        if abs(self.state.coin) <= 0 and not self.state.positions:
            self.state.active_stop_order_id = None
        logger.info(f"Native stop filled: {stop_order_id}")

    # ─── Strategy Builders ───────────────────────────────────────────

    def build_scan_rows(
        self,
        watchlist: list[str],
        candles_by_symbol: dict[str, list[Candle]] | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        candles_by_symbol = candles_by_symbol or {}

        if self.state.settings.get("strategy") == "ewo_offset":
            return self.build_ewo_scan_rows(watchlist, candles_by_symbol)

        if self.state.settings.get("strategy") == "ema_golden_cross":
            return self.build_ema_golden_cross_scan_rows(watchlist, candles_by_symbol)

        settings = self.state.settings
        short_window = int(self.state.settings["short_window"])
        long_window = int(self.state.settings["long_window"])
        settings_key = setup_settings_key(settings)
        weak_pairs = weak_pair_map(self.state.setup_records, settings)

        for symbol in watchlist:
            candles = candles_by_symbol.get(symbol, [])
            history = [candle.close for candle in candles] if candles else self.state.price_history.get(symbol, [])
            if not candles:
                candles = closes_to_candles(history)
            price = history[-1] if history else None
            levels = support_resistance(candles, settings)
            regime = market_regime(candles, settings)
            short_now = sma(history, short_window)
            long_now = sma(history, long_window)
            short_prev = sma(history[:-1], short_window)
            long_prev = sma(history[:-1], long_window)
            base_score = 0.0
            signal = "WAIT data"

            if price and None not in (short_now, long_now, short_prev, long_prev):
                base_score = ((short_now - long_now) / price) * 100
                if short_prev <= long_prev and short_now > long_now and base_score > 0:
                    signal = "BUY"
                elif short_now > long_now:
                    signal = "WATCH uptrend"
                else:
                    signal = "HOLD"

                if signal == "BUY":
                    allowed, reason = sr_buy_allowed(price, levels, settings)
                    if not allowed:
                        signal = f"WATCH {reason}"
                if signal == "BUY":
                    allowed, reason = regime_allowed(regime["regime"], settings)
                    if not allowed:
                        signal = f"WATCH {reason}"
                if signal == "BUY" and symbol in weak_pairs:
                    signal = f"BLOCK {weak_pairs[symbol]}"

            edge_score = setup_edge_score(self.state.setup_records, symbol, settings_key)
            score = base_score + edge_score

            rows.append({
                "symbol": symbol,
                "price": price,
                "short_sma": short_now,
                "long_sma": long_now,
                "support": levels["support"],
                "resistance": levels["resistance"],
                "support_distance_pct": levels["support_distance_pct"],
                "resistance_distance_pct": levels["resistance_distance_pct"],
                "support_touches": levels["support_touches"],
                "resistance_touches": levels["resistance_touches"],
                "sr_confirmed": levels["confirmed"],
                "sr_range_pct": levels["sr_range_pct"],
                "reward_risk": levels["reward_risk"],
                "regime": regime["regime"],
                "regime_trend_pct": regime["trend_pct"],
                "regime_volatility_pct": regime["volatility_pct"],
                "regime_range_pct": regime["range_pct"],
                "regime_reason": regime["reason"],
                "base_score": round(base_score, 4),
                "edge_score": edge_score,
                "score": round(score, 4),
                "signal": signal,
            })

        return rows

    def build_ewo_scan_rows(
        self,
        watchlist: list[str],
        candles_by_symbol: dict[str, list[Candle]] | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        settings = self.state.settings
        candles_by_symbol = candles_by_symbol or {}
        settings_key = setup_settings_key(settings)
        weak_pairs = weak_pair_map(self.state.setup_records, settings)

        for symbol in watchlist:
            history = self.state.price_history.get(symbol, [])
            price = history[-1] if history else None
            candles = candles_by_symbol.get(symbol) or closes_to_candles(history)
            if candles:
                price = candles[-1].close
            levels = support_resistance(candles, settings)
            regime = market_regime(candles, settings)
            signal = ewo_offset_signal(candles, settings)
            status = "WAIT data"
            if signal["ready"]:
                if signal["buy"]:
                    status = "BUY"
                elif signal["sell"]:
                    status = "SELL signal"
                else:
                    status = "HOLD"

                if status == "BUY":
                    allowed, reason = sr_buy_allowed(price, levels, settings)
                    if not allowed:
                        status = f"WATCH {reason}"
                if status == "BUY":
                    allowed, reason = regime_allowed(regime["regime"], settings)
                    if not allowed:
                        status = f"WATCH {reason}"
                if status == "BUY" and symbol in weak_pairs:
                    status = f"BLOCK {weak_pairs[symbol]}"

            base_score = float(signal.get("score") or 0.0)
            edge_score = setup_edge_score(self.state.setup_records, symbol, settings_key)
            score = base_score + edge_score

            rows.append({
                "symbol": symbol,
                "price": price,
                "short_sma": signal.get("ma_buy"),
                "long_sma": signal.get("ma_sell"),
                "support": levels["support"],
                "resistance": levels["resistance"],
                "support_distance_pct": levels["support_distance_pct"],
                "resistance_distance_pct": levels["resistance_distance_pct"],
                "support_touches": levels["support_touches"],
                "resistance_touches": levels["resistance_touches"],
                "sr_confirmed": levels["confirmed"],
                "sr_range_pct": levels["sr_range_pct"],
                "reward_risk": levels["reward_risk"],
                "regime": regime["regime"],
                "regime_trend_pct": regime["trend_pct"],
                "regime_volatility_pct": regime["volatility_pct"],
                "regime_range_pct": regime["range_pct"],
                "regime_reason": regime["reason"],
                "base_score": round(base_score, 4),
                "edge_score": edge_score,
                "score": round(score, 4),
                "signal": status,
            })

        return rows

    def build_ema_golden_cross_scan_rows(
        self,
        watchlist: list[str],
        candles_by_symbol: dict[str, list[Candle]] | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        candles_by_symbol = candles_by_symbol or {}
        settings = self.state.settings
        settings_key = setup_settings_key(settings)
        weak_pairs = weak_pair_map(self.state.setup_records, settings)

        ema_short = int(settings.get("ema_short", 50))
        ema_long = int(settings.get("ema_long", 200))

        for symbol in watchlist:
            history = self.state.price_history.get(symbol, [])
            candles = candles_by_symbol.get(symbol) or closes_to_candles(history)
            price = candles[-1].close if candles else None

            if not price or len(history) < ema_long:
                rows.append({
                    "symbol": symbol,
                    "price": price,
                    "signal": "WAIT data",
                    "score": 0,
                    "regime": "unknown",
                    "support": None,
                    "resistance": None,
                    "ema_short": None,
                    "ema_long": None,
                    "ema_short_prev": None,
                    "ema_long_prev": None,
                })
                continue

            ema_short_value = ema_series(history, ema_short)[-1]
            ema_long_value = ema_series(history, ema_long)[-1]
            ema_short_prev = ema_series(history[:-1], ema_short)[-1] if len(history) > 1 else None
            ema_long_prev = ema_series(history[:-1], ema_long)[-1] if len(history) > 1 else None

            if None in (ema_short_value, ema_long_value, ema_short_prev, ema_long_prev):
                rows.append({
                    "symbol": symbol,
                    "price": price,
                    "signal": "WAIT data",
                    "score": 0,
                    "regime": "unknown",
                    "support": None,
                    "resistance": None,
                    "ema_short": ema_short_value,
                    "ema_long": ema_long_value,
                    "ema_short_prev": ema_short_prev,
                    "ema_long_prev": ema_long_prev,
                })
                continue

            signal = "HOLD"
            base_score = ((ema_short_value - ema_long_value) / ema_long_value) * 100

            if ema_short_prev <= ema_long_prev and ema_short_value > ema_long_value:
                signal = "BUY"
            elif ema_short_prev >= ema_long_prev and ema_short_value < ema_long_value:
                signal = "SELL"
            elif ema_short_value > ema_long_value:
                signal = "WATCH uptrend"
            elif ema_short_value < ema_long_value:
                signal = "WATCH downtrend"

            regime = market_regime(candles, settings)
            levels = support_resistance(candles, settings)

            if signal == "BUY":
                allowed, reason = sr_buy_allowed(price, levels, settings)
                if not allowed:
                    signal = f"WATCH {reason}"
            if signal == "BUY":
                allowed, reason = regime_allowed(regime["regime"], settings)
                if not allowed:
                    signal = f"WATCH {reason}"
            if signal == "BUY" and symbol in weak_pairs:
                signal = f"BLOCK {weak_pairs[symbol]}"

            edge_score = setup_edge_score(self.state.setup_records, symbol, settings_key)
            score = base_score + edge_score

            rows.append({
                "symbol": symbol,
                "price": price,
                "signal": signal,
                "score": round(score, 4),
                "regime": regime["regime"],
                "regime_trend_pct": regime["trend_pct"],
                "regime_volatility_pct": regime["volatility_pct"],
                "regime_range_pct": regime["range_pct"],
                "regime_reason": regime["reason"],
                "support": levels["support"],
                "resistance": levels["resistance"],
                "support_distance_pct": levels["support_distance_pct"],
                "resistance_distance_pct": levels["resistance_distance_pct"],
                "support_touches": levels["support_touches"],
                "resistance_touches": levels["resistance_touches"],
                "sr_confirmed": levels["confirmed"],
                "sr_range_pct": levels["sr_range_pct"],
                "reward_risk": levels["reward_risk"],
                "ema_short": ema_short_value,
                "ema_long": ema_long_value,
                "ema_short_prev": ema_short_prev,
                "ema_long_prev": ema_long_prev,
                "base_score": round(base_score, 4),
                "edge_score": edge_score,
            })

        return rows

# ─── STANDALONE FUNCTIONS ───────────────────────────────────────────

def oanda_stream_pricing(
    symbols: list[str],
    on_price: callable,
    on_error: callable = None,
    stop_event: threading.Event = None,
) -> None:
    if not oanda_is_configured():
        error = "OANDA not configured - cannot start stream"
        if on_error:
            on_error(error)
        raise RuntimeError(error)

    account_id = urllib.parse.quote(oanda_account_id())
    instruments = ",".join(oanda_instrument(sym) for sym in symbols)

    base = oanda_api_base()
    stream_base = base.replace("api-", "stream-")
    url = f"{stream_base}/v3/accounts/{account_id}/pricing/stream"

    params = {
        "instruments": instruments,
        "snapshot": "True",
    }
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}"

    logger.info(f"Starting OANDA pricing stream for {len(symbols)} instruments")

    request = urllib.request.Request(
        full_url,
        headers={
            "Authorization": f"Bearer {oanda_api_token()}",
            "Accept": "application/json",
            "User-Agent": "auxo-trading-bot/1.0",
            "Connection": "keep-alive",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=None) as response:
            buffer = ""
            while True:
                if stop_event and stop_event.is_set():
                    logger.info("OANDA stream stopped by event")
                    break

                chunk = response.read(1024).decode("utf-8")
                if not chunk:
                    break

                buffer += chunk

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        if data.get("type") == "PRICE":
                            on_price(data)
                        elif data.get("type") == "HEARTBEAT":
                            pass
                        elif data.get("type") == "SNAPSHOT":
                            for price_data in data.get("prices", []):
                                on_price(price_data)
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing stream message: {e}")

    except urllib.error.HTTPError as e:
        error_msg = f"OANDA stream HTTP error {e.code}: {e.reason}"
        body = e.read().decode("utf-8", errors="replace")
        logger.error(f"{error_msg} - {body}")
        if on_error:
            on_error(error_msg)
        raise
    except Exception as e:
        logger.error(f"OANDA stream error: {e}")
        if on_error:
            on_error(str(e))
        raise

# ─── OANDA API FUNCTIONS ────────────────────────────────────────────

def oanda_account_id() -> str:
    return os.environ.get("OANDA_ACCOUNT_ID", "").strip()

def oanda_api_token() -> str:
    return os.environ.get("OANDA_API_TOKEN", "").strip()

def oanda_is_configured() -> bool:
    return bool(oanda_account_id() and oanda_api_token())

def oanda_is_practice() -> bool:
    env = os.environ.get("OANDA_ENV", "practice").strip().lower()
    return env == "practice"

def oanda_api_base() -> str:
    if os.environ.get("OANDA_API_BASE", "").strip():
        return os.environ["OANDA_API_BASE"].strip().rstrip("/")
    if oanda_is_practice():
        return "https://api-fxpractice.oanda.com"
    return "https://api-fxtrade.oanda.com"

def oanda_instrument(symbol: str) -> str:
    symbol = normalize_forex_symbol(symbol)
    if len(symbol) != 6:
        raise RuntimeError("OANDA forex pairs must be six-letter symbols like EURUSD")
    return f"{symbol[:3]}_{symbol[3:]}"

def oanda_demo_orders_armed() -> bool:
    if not oanda_is_configured():
        return False
    if os.environ.get("OANDA_DEMO_TRADING_ENABLED", "").strip().lower() != "true":
        return False
    return True

def oanda_demo_status_message() -> str:
    missing = []
    if not oanda_account_id():
        missing.append("OANDA_ACCOUNT_ID")
    if not oanda_api_token():
        missing.append("OANDA_API_TOKEN")
    if os.environ.get("OANDA_DEMO_TRADING_ENABLED", "").strip().lower() != "true":
        missing.append("OANDA_DEMO_TRADING_ENABLED=true")
    if missing:
        return "OANDA order placement locked. Missing: " + ", ".join(missing)
    env = "practice" if oanda_is_practice() else "live"
    return f"OANDA order placement armed for {env} account."

def oanda_request(
    path: str,
    params: dict[str, Any] | None = None,
    timeout: int = 10,
    method: str = "GET",
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not oanda_is_configured():
        raise RuntimeError("Missing OANDA_ACCOUNT_ID or OANDA_API_TOKEN in .env")

    query = urllib.parse.urlencode(params or {})
    url = f"{oanda_api_base()}{path}"
    if query:
        url = f"{url}?{query}"
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        url,
        data=payload,
        method=method.upper(),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {oanda_api_token()}",
            "User-Agent": "cryptobot-oanda-paper/1.0",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OANDA API error {exc.code}: {body}") from exc

def oanda_account_summary() -> dict[str, Any]:
    account_id = urllib.parse.quote(oanda_account_id())
    return oanda_request(f"/v3/accounts/{account_id}/summary")

def oanda_order_fill(response: dict[str, Any]) -> dict[str, Any]:
    fill = response.get("orderFillTransaction")
    cancel = response.get("orderCancelTransaction") or response.get("orderCreateTransaction")
    if not fill:
        reason = cancel.get("reason") if isinstance(cancel, dict) else "not filled"
        raise RuntimeError(f"OANDA demo order was not filled: {reason}")

    units = abs(float(fill.get("units", 0.0)))
    price = float(fill.get("price", 0.0))
    commission = abs(float(fill.get("commission", 0.0)))
    trade_id = fill.get("tradeOpened", {}).get("tradeID") or fill.get("tradesClosed", [{}])[0].get("tradeID")
    return {
        "order_id": str(fill.get("id") or response.get("lastTransactionID") or ""),
        "trade_id": str(trade_id or ""),
        "status": "FILLED",
        "units": units,
        "price": price,
        "commission": commission,
        "raw": response,
    }

def oanda_market_order(
    symbol: str,
    units: int,
    stop_price: float | None = None,
    target_price: float | None = None,
) -> dict[str, Any]:
    if not oanda_demo_orders_armed():
        raise RuntimeError(oanda_demo_status_message())
    if units == 0:
        raise RuntimeError("OANDA order units cannot be zero")

    account_id = urllib.parse.quote(oanda_account_id())
    order: dict[str, Any] = {
        "type": "MARKET",
        "instrument": oanda_instrument(symbol),
        "units": str(units),
        "timeInForce": "FOK",
        "positionFill": "DEFAULT",
    }
    if stop_price and stop_price > 0:
        order["stopLossOnFill"] = {"price": oanda_decimal(stop_price, symbol)}
    if target_price and target_price > 0:
        order["takeProfitOnFill"] = {"price": oanda_decimal(target_price, symbol)}

    return oanda_request(
        f"/v3/accounts/{account_id}/orders",
        method="POST",
        body={"order": order},
        timeout=15,
    )

def oanda_decimal(value: float, symbol: str) -> str:
    places = 3 if normalize_forex_symbol(symbol).endswith("JPY") else 5
    return f"{value:.{places}f}"

# ─── COINBASE API FUNCTIONS ─────────────────────────────────────────

def coinbase_live_is_armed() -> bool:
    return (
        CRYPTOGRAPHY_AVAILABLE
        and os.environ.get("COINBASE_API_KEY_NAME", "").strip() != ""
        and coinbase_private_key_configured()
        and os.environ.get("LIVE_TRADING_CONFIRM", "") == "I_UNDERSTAND_THIS_PLACES_REAL_ORDERS"
    )

def coinbase_live_status_message() -> str:
    missing = []
    if not CRYPTOGRAPHY_AVAILABLE:
        missing.append("python package: cryptography")
    if not os.environ.get("COINBASE_API_KEY_NAME", "").strip():
        missing.append("COINBASE_API_KEY_NAME")
    if not coinbase_private_key_configured():
        missing.append("COINBASE_API_PRIVATE_KEY or COINBASE_API_PRIVATE_KEY_FILE")
    if os.environ.get("LIVE_TRADING_CONFIRM", "") != "I_UNDERSTAND_THIS_PLACES_REAL_ORDERS":
        missing.append("LIVE_TRADING_CONFIRM")
    if missing:
        return "Live trading locked. Missing: " + ", ".join(missing)
    source = ".env" if DOTENV_LOADED_KEYS else "environment variables"
    return f"Live trading armed by {source}."

def coinbase_private_key_configured() -> bool:
    if os.environ.get("COINBASE_API_PRIVATE_KEY", "").strip():
        return True
    key_file = os.environ.get("COINBASE_API_PRIVATE_KEY_FILE", "").strip()
    return bool(key_file and resolve_local_path(key_file).is_file())

def resolve_local_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path

def coinbase_private_key_value() -> str:
    raw_value = os.environ.get("COINBASE_API_PRIVATE_KEY", "").strip()
    if raw_value:
        return raw_value

    key_file = os.environ.get("COINBASE_API_PRIVATE_KEY_FILE", "").strip()
    if key_file:
        return resolve_local_path(key_file).read_text(encoding="utf-8").strip()

    raise RuntimeError("Coinbase private key is not configured.")

def coinbase_private_key():
    raw_value = coinbase_private_key_value()
    raw_key = extract_coinbase_private_key(raw_value).replace("\\n", "\n").encode("utf-8")
    return serialization.load_pem_private_key(raw_key, password=None)

def extract_coinbase_private_key(raw_value: str) -> str:
    if raw_value.startswith("{"):
        data = json.loads(raw_value)
        for key in ("privateKey", "private_key", "key_secret", "api_secret"):
            if data.get(key):
                return str(data[key])
        raise RuntimeError(
            "Coinbase key JSON found, but no privateKey/private_key/key_secret/api_secret field exists."
        )
    return raw_value

def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")

def coinbase_jwt(method: str, request_path: str) -> str:
    key_name = os.environ["COINBASE_API_KEY_NAME"]
    now = int(time.time())
    uri = f"{method.upper()} api.coinbase.com{request_path}"
    header = {
        "alg": "ES256",
        "kid": key_name,
        "nonce": uuid.uuid4().hex,
        "typ": "JWT",
    }
    payload = {
        "sub": key_name,
        "iss": "cdp",
        "nbf": now,
        "exp": now + 120,
        "uri": uri,
    }
    signing_input = (
        b64url(json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        + "."
        + b64url(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    )
    private_key = coinbase_private_key()
    der_signature = private_key.sign(signing_input.encode("ascii"), ec.ECDSA(hashes.SHA256()))
    r, s = utils.decode_dss_signature(der_signature)
    raw_signature = r.to_bytes(32, "big") + s.to_bytes(32, "big")
    return signing_input + "." + b64url(raw_signature)

def coinbase_ws_jwt() -> str:
    return coinbase_jwt("GET", "/users/self/verify")

def coinbase_api_request(
    method: str,
    request_path: str,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not coinbase_live_is_armed():
        raise RuntimeError(coinbase_live_status_message())

    url = f"https://api.coinbase.com{request_path}"
    body_bytes = json.dumps(body).encode("utf-8") if body is not None else None
    signed_path = request_path.split("?", 1)[0]
    token = coinbase_jwt(method, signed_path)
    request = urllib.request.Request(
        url,
        data=body_bytes,
        method=method.upper(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "local-paper-trading-bot/1.0",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Coinbase API error {exc.code}: {detail}") from exc

def coinbase_account_balances() -> dict[str, dict[str, float]]:
    """Return Coinbase brokerage balances keyed by currency.

    Coinbase separates funds into available and held balances.  Equity must use
    both; the Cash box intentionally uses only the available quote balance.
    """
    cursor = ""
    balances: dict[str, dict[str, float]] = {}

    while True:
        query = "?limit=250"
        if cursor:
            query += "&cursor=" + urllib.parse.quote(cursor)
        data = coinbase_api_request("GET", f"/api/v3/brokerage/accounts{query}")

        for account in data.get("accounts", []):
            currency = str(account.get("currency", "")).upper()
            if not currency:
                continue
            available = float((account.get("available_balance") or {}).get("value", 0.0) or 0.0)
            hold = float((account.get("hold") or {}).get("value", 0.0) or 0.0)
            row = balances.setdefault(currency, {"available": 0.0, "hold": 0.0, "total": 0.0})
            row["available"] += available
            row["hold"] += hold
            row["total"] += available + hold

        if not data.get("has_next"):
            break
        cursor = data.get("cursor", "")

    return balances

def coinbase_available_balance(currency: str) -> float:
    currency = currency.upper()
    return float(coinbase_account_balances().get(currency, {}).get("available", 0.0))

def coinbase_market_order(
    product_id: str,
    side: str,
    quote_size: float | None = None,
    base_size: float | None = None,
) -> dict[str, Any]:
    side = side.upper()
    if side not in {"BUY", "SELL"}:
        raise RuntimeError("Coinbase order side must be BUY or SELL")

    if side == "BUY":
        if quote_size is None or quote_size <= 0:
            raise RuntimeError("BUY order requires quote_size")
        order_configuration = {
            "market_market_ioc": {
                "quote_size": f"{quote_size:.2f}",
            }
        }
    else:
        if base_size is None or base_size <= 0:
            raise RuntimeError("SELL order requires base_size")
        order_configuration = {
            "market_market_ioc": {
                "base_size": f"{base_size:.10f}".rstrip("0").rstrip("."),
            }
        }

    return coinbase_create_order(product_id, side, order_configuration)

def coinbase_limit_order(
    product_id: str,
    side: str,
    base_size: float,
    limit_price: float,
) -> dict[str, Any]:
    if base_size <= 0 or limit_price <= 0:
        raise RuntimeError("Limit order requires positive base_size and limit_price")
    order_configuration = {
        "limit_limit_gtc": {
            "base_size": decimal_text(base_size, 10),
            "limit_price": decimal_text(limit_price, 8),
            "post_only": False,
        }
    }
    return coinbase_create_order(product_id, side, order_configuration)

def coinbase_stop_limit_order(
    product_id: str,
    side: str,
    base_size: float,
    stop_price: float,
    limit_price: float,
) -> dict[str, Any]:
    if base_size <= 0 or stop_price <= 0 or limit_price <= 0:
        raise RuntimeError("Stop-limit order requires positive size, stop, and limit")

    stop_direction = "STOP_DIRECTION_STOP_DOWN" if side.upper() == "SELL" else "STOP_DIRECTION_STOP_UP"

    order_configuration = {
        "stop_limit_stop_limit_gtc": {
            "base_size": decimal_text(base_size, 10),
            "limit_price": decimal_text(limit_price, 8),
            "stop_price": decimal_text(stop_price, 8),
            "stop_direction": stop_direction,
        }
    }

    # Route stop orders through the same submission/recovery path as market
    # and limit orders so client_order_id is retained and ambiguous network
    # failures can be recovered safely.
    return coinbase_create_order(product_id, side, order_configuration)

def coinbase_find_order_by_client_id(
    client_order_id: str,
    product_id: str | None = None,
) -> dict[str, Any] | None:
    """Find a Coinbase order using Auxo's client_order_id.

    Coinbase's List Orders endpoint returns client_order_id on each order,
    although it does not currently expose client_order_id as a list filter.
    Keep the search deliberately narrow (product + recent pages) because this
    function is only used immediately after an ambiguous submission.
    """
    cursor = ""
    pages_checked = 0

    while pages_checked < 5:
        params: list[tuple[str, str]] = [("limit", "100")]
        if product_id:
            params.append(("product_ids", product_id))
        if cursor:
            params.append(("cursor", cursor))

        query = urllib.parse.urlencode(params)
        data = coinbase_api_request(
            "GET",
            f"/api/v3/brokerage/orders/historical/batch?{query}",
        )

        for order in data.get("orders", []):
            if str(order.get("client_order_id") or "") == client_order_id:
                return order

        pages_checked += 1
        if not data.get("has_next"):
            break
        cursor = str(data.get("cursor") or "")
        if not cursor:
            break

    return None

def coinbase_recover_submitted_order(
    client_order_id: str,
    product_id: str,
    attempts: int = 4,
    delay_seconds: float = 0.75,
) -> dict[str, Any] | None:
    """Recover an order whose POST result was ambiguous.

    Never submits another order. It only asks Coinbase whether the original
    client_order_id already exists.
    """
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            recovered = coinbase_find_order_by_client_id(
                client_order_id,
                product_id=product_id,
            )
            if recovered:
                order_id = str(recovered.get("order_id") or "")
                if order_id:
                    logger.warning(
                        "Recovered uncertain Coinbase order %s using client_order_id %s",
                        order_id,
                        client_order_id,
                    )
                    return {
                        "success": True,
                        "success_response": {
                            "order_id": order_id,
                            "client_order_id": client_order_id,
                        },
                        "order": recovered,
                        "client_order_id": client_order_id,
                        "recovered_after_submission_error": True,
                    }
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Coinbase client-order recovery attempt %d/%d failed for %s: %s",
                attempt + 1,
                attempts,
                client_order_id,
                exc,
            )

        if attempt < attempts - 1:
            time.sleep(delay_seconds)

    if last_error:
        logger.warning(
            "Could not confirm Coinbase order for client_order_id %s: %s",
            client_order_id,
            last_error,
        )
    return None

def coinbase_create_order(
    product_id: str,
    side: str,
    order_configuration: dict[str, Any],
) -> dict[str, Any]:
    side = side.upper()

    if side not in {"BUY", "SELL"}:
        raise RuntimeError("Coinbase order side must be BUY or SELL")

    client_order_id = str(uuid.uuid4())

    body = {
        "client_order_id": client_order_id,
        "product_id": product_id,
        "side": side,
        "order_configuration": order_configuration,
    }

    try:
        response = coinbase_api_request(
            "POST",
            "/api/v3/brokerage/orders",
            body,
        )
    except Exception as submit_error:
        # A transport/response failure does NOT prove Coinbase failed to create
        # the order. Before the caller can retry, look up the exact client ID.
        # This prevents duplicate BUY/SELL orders after timeouts or lost replies.
        logger.error(
            "Coinbase order submission outcome uncertain for client_order_id %s: %s",
            client_order_id,
            submit_error,
        )

        recovered = coinbase_recover_submitted_order(
            client_order_id=client_order_id,
            product_id=product_id,
        )
        if recovered is not None:
            return recovered

        raise RuntimeError(
            "Coinbase order submission outcome is uncertain and recovery could "
            f"not find client_order_id={client_order_id}. Do not blindly retry "
            "this order; reconcile Coinbase orders/balances first."
        ) from submit_error

    # Retain Auxo's client order ID even if Coinbase's response
    # doesn't echo it back.
    if isinstance(response, dict):
        response.setdefault("client_order_id", client_order_id)

    return response

def coinbase_get_order(order_id: str) -> dict[str, Any]:
    return coinbase_api_request("GET", f"/api/v3/brokerage/orders/historical/{urllib.parse.quote(order_id)}")

def coinbase_list_fills(order_id: str) -> dict[str, Any]:
    query = "?order_id=" + urllib.parse.quote(order_id)
    return coinbase_api_request("GET", f"/api/v3/brokerage/orders/historical/fills{query}")

def coinbase_cancel_orders(order_ids: list[str]) -> dict[str, Any]:
    return coinbase_api_request("POST", "/api/v3/brokerage/orders/batch_cancel", {"order_ids": order_ids})

def coinbase_order_id(response: dict[str, Any]) -> str:
    """
    Extract an order ID from a Coinbase Advanced Trade order response.

    Raises a useful exception when Coinbase explicitly rejects the order,
    rather than reporting the rejection as a missing order ID.
    """
    if not isinstance(response, dict):
        raise RuntimeError(
            f"Unexpected Coinbase order response type: {type(response).__name__}: {response!r}"
        )

    # Common successful response shapes.
    order_id = response.get("order_id")
    if order_id:
        return str(order_id)

    success_response = response.get("success_response")
    if isinstance(success_response, dict):
        order_id = success_response.get("order_id")
        if order_id:
            return str(order_id)

    order_data = response.get("order")
    if isinstance(order_data, dict):
        order_id = order_data.get("order_id")
        if order_id:
            return str(order_id)

    # Coinbase Advanced Trade rejected the order.
    error_response = response.get("error_response")
    if isinstance(error_response, dict):
        error = error_response.get("error", "UNKNOWN_ERROR")
        message = error_response.get("message", "")
        details = error_response.get("error_details", "")
        preview_failure = error_response.get("preview_failure_reason", "")

        raise RuntimeError(
            "Coinbase rejected order: "
            f"error={error}; "
            f"message={message}; "
            f"details={details}; "
            f"preview_failure_reason={preview_failure}"
        )

    # Some responses expose success=False without error_response.
    if response.get("success") is False:
        raise RuntimeError(
            f"Coinbase rejected order: {response}"
        )

    raise RuntimeError(
        f"Coinbase returned an unexpected order response: {response}"
    )

def coinbase_reconcile_order(order_id: str) -> dict[str, Any]:
    order_data: dict[str, Any] = {}
    fills_data: dict[str, Any] = {}
    for attempt in range(4):
        order_data = coinbase_get_order(order_id)
        fills_data = coinbase_list_fills(order_id)
        if fills_data.get("fills") or attempt == 3:
            break
        time.sleep(0.75)
    fills = fills_data.get("fills", [])
    filled_size = 0.0
    filled_value = 0.0
    total_fee = 0.0

    for fill in fills:
        size = float(fill.get("size") or fill.get("base_size") or 0.0)
        price = float(fill.get("price") or 0.0)
        commission = float(fill.get("commission") or fill.get("fee") or 0.0)
        filled_size += size
        filled_value += size * price
        total_fee += commission

    order = order_data.get("order", order_data)
    status = str(order.get("status") or order.get("completion_percentage") or "UNKNOWN")
    average_price = filled_value / filled_size if filled_size > 0 else 0.0
    return {
        "order_id": order_id,
        "status": status,
        "filled_size": filled_size,
        "filled_value": filled_value,
        "total_fee": total_fee,
        "average_price": average_price,
        "fills_count": len(fills),
        "order": order,
    }

def decimal_text(value: float, places: int) -> str:
    return f"{value:.{places}f}".rstrip("0").rstrip(".")

# ─── TRADING HELPER FUNCTIONS ───────────────────────────────────────

def fetch_candles(
    exchange: str,
    symbol: str,
    quote_currency: str,
    granularity: int,
    candle_count: int,
    asset_class: str = "crypto",
) -> list[Candle]:
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

def fetch_oanda_demo_candles(
    symbol: str,
    granularity: int,
    candle_count: int,
) -> list[Candle]:
    instrument = oanda_instrument(symbol)
    count = max(40, min(5000, int(candle_count)))
    data = oanda_request(
        f"/v3/instruments/{urllib.parse.quote(instrument)}/candles",
        {
            "price": "M",
            "granularity": oanda_granularity(granularity),
            "count": count,
        },
        timeout=15,
    )
    candles: list[Candle] = []
    for item in data.get("candles", []):
        mid = item.get("mid") or {}
        if not mid:
            continue
        candles.append(
            Candle(
                time=parse_oanda_time(str(item["time"])),
                open=float(mid["o"]),
                high=float(mid["h"]),
                low=float(mid["l"]),
                close=float(mid["c"]),
                volume=float(item.get("volume", 0.0)),
            )
        )
    return sorted(candles, key=lambda item: item.time)[-count:]

def oanda_granularity(seconds: int | float) -> str:
    seconds = int(seconds)
    mapping = {
        60: "M1",
        300: "M5",
        900: "M15",
        3600: "H1",
        21600: "H6",
        86400: "D",
    }
    if seconds not in mapping:
        raise RuntimeError("OANDA demo supports 1m, 5m, 15m, 1h, 6h, and 1d candles")
    return mapping[seconds]

def parse_oanda_time(value: str) -> int:
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1]
    if "." in raw:
        head, fraction = raw.split(".", 1)
        raw = f"{head}.{fraction[:6].ljust(6, '0')}"
    parsed = datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())

def fetch_forex_demo_candles(
    symbol: str,
    granularity: int,
    candle_count: int,
) -> list[Candle]:
    symbol = normalize_forex_symbol(symbol)
    candle_count = max(40, min(720, int(candle_count)))
    base = FOREX_BASE_RATES.get(symbol)
    if base is None:
        raise RuntimeError("Unsupported forex demo pair")

    pip = forex_pip_size(symbol)
    now = int(time.time())
    end_time = now - (now % int(granularity))
    seed = sum(ord(char) for char in symbol)
    drift = ((seed % 11) - 5) * pip * 0.015
    amplitude = base * (0.0018 + ((seed % 7) * 0.00012))
    candles: list[Candle] = []
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
            Candle(
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

def forex_pip_size(symbol: str) -> float:
    symbol = normalize_forex_symbol(symbol)
    return 0.01 if symbol.endswith("JPY") else 0.0001

def fetch_coinbase_candles(
    symbol: str,
    quote_currency: str,
    granularity: int,
    candle_count: int,
) -> list[Candle]:
    # Coinbase's public candles endpoint returns at most 300 candles per request.
    # Fetch historical windows in chunks so the validation lab can use a genuinely
    # longer sample instead of silently truncating to the most recent 300 candles.
    candle_count = max(20, min(3000, int(candle_count)))
    product = f"{symbol}-{quote_currency}"
    end = datetime.now(timezone.utc)
    all_candles: dict[int, Candle] = {}
    remaining = candle_count
    chunk_size = 300
    step = timedelta(seconds=int(granularity))

    while remaining > 0:
        chunk = min(chunk_size, remaining)
        start = end - (step * chunk)
        query = urllib.parse.urlencode({
            "granularity": int(granularity),
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
        })
        data = fetch_json(f"https://api.exchange.coinbase.com/products/{product}/candles?{query}")
        if not data:
            break

        for item in data:
            candle = Candle(
                time=int(item[0]),
                low=float(item[1]),
                high=float(item[2]),
                open=float(item[3]),
                close=float(item[4]),
                volume=float(item[5]),
            )
            all_candles[candle.time] = candle

        earliest = min(int(item[0]) for item in data)
        end = datetime.fromtimestamp(earliest, tz=timezone.utc) - step
        received = len(data)
        remaining -= received
        if received <= 0:
            break

    return sorted(all_candles.values(), key=lambda item: item.time)[-candle_count:]

def fetch_kraken_candles(
    symbol: str,
    quote_currency: str,
    granularity: int,
    candle_count: int,
) -> list[Candle]:
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
        Candle(
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

def fetch_binance_candles(symbol: str, quote_currency: str, granularity: int, candle_count: int) -> list[Candle]:
    if not BINANCE_AVAILABLE:
        raise RuntimeError("python-binance package not installed")
    from binance.client import Client
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
            Candle(
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

def fetch_json(url: str, timeout: int = 10) -> dict[str, Any]:
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

# ─── BACKTEST FUNCTIONS ─────────────────────────────────────────────

def run_backtest(settings: dict[str, Any]) -> dict[str, Any]:
    settings = backtest_runtime_settings(settings)
    watchlist = parse_watchlist(settings.get("watchlist", "BTC"))
    if not watchlist:
        raise RuntimeError("Backtest watchlist is empty")

    granularity = int(settings.get("granularity", 3600))
    candle_count = int(settings.get("candle_count", 300))
    results: list[dict[str, Any]] = []
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
            logger.warning(f"Backtest error for {symbol}: {exc}")

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

def run_backtest_for_symbol(
    symbol: str,
    candles: list[Candle],
    settings: dict[str, Any],
) -> dict[str, Any]:

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
    locked_stop: float | None = None
    locked_target: float | None = None
    locked_exit_mode: str = "fixed"
    trade_fee = float(settings["trade_fee"])
    slippage = float(settings.get("backtest_slippage_pct", 0.0)) / 100
    short_window = int(settings["short_window"])
    long_window = int(settings["long_window"])
    trade_start_time = int(settings.get("trade_start_time", 0))
    closes: list[float] = []
    trades: list[dict[str, Any]] = []
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
            highest_price = max(highest_price or price, candle.high)
            stop_price = locked_stop if locked_stop is not None else entry_price * (1 - float(settings["stop_loss_pct"]) / 100)
            target_price = locked_target if locked_target is not None else entry_price * (1 + float(settings["take_profit_pct"]) / 100)
            exit_mode = locked_exit_mode
            partial_quantity = 0.0
            partial_trigger = entry_price + ((target_price - entry_price) * (float(settings.get("partial_take_profit_at_target_pct", 50.0)) / 100))
            trail = trailing_stop_price(entry_price, highest_price, settings)
            intrabar_reason, intrabar_fill = backtest_intrabar_long_exit(candle, stop_price, target_price, trail)
            if settings.get("partial_take_profit_enabled") and not partial_done and candle.high >= partial_trigger and not intrabar_reason:
                reason = f"partial {exit_mode} target"
                partial_done = True
                partial_quantity = coin * (float(settings.get("partial_take_profit_pct", 50.0)) / 100)
                price = max(candle.open, partial_trigger)
            elif intrabar_reason:
                reason = "trailing stop" if intrabar_reason == "trailing stop" else f"{exit_mode} {intrabar_reason}"
                price = float(intrabar_fill)
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
                    locked_stop = locked_target = None
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
                locked_stop, locked_target, locked_exit_mode = exit_prices(entry_price, active_candles, settings)
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

    metrics = backtest_metrics(trades, starting_cash, equity_curve)

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
        "closed_trades": metrics["closed_trades"],
        "win_rate": metrics["win_rate"],
        "profit_factor": metrics["profit_factor"],
        "expectancy": metrics["expectancy"],
        "expectancy_pct_starting_cash": metrics["expectancy_pct_starting_cash"],
        "average_winner": metrics["average_winner"],
        "average_loser": metrics["average_loser"],
        "payoff_ratio": metrics["payoff_ratio"],
        "total_fees": metrics["total_fees"],
        "sharpe_like": metrics["sharpe_like"],
        "sortino_like": metrics["sortino_like"],
        "_trade_pnls": metrics["trade_pnls"],
        "slippage_pct": float(settings.get("backtest_slippage_pct", 0.0)),
        "open_position": coin > 0,
        "trades": trades[-80:][::-1],
        "equity_curve": [round(item, 8) for item in equity_curve[-300:]],
    }

def run_opening_range_backtest(
    symbol: str,
    candles: list[Candle],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Backtest the Opening Range strategy with both long and short support."""
    starting_cash = float(settings["starting_cash"])
    cash = starting_cash
    coin = 0.0
    entry_price: float | None = None
    highest_price: float | None = None
    partial_done = False
    trade_fee = float(settings["trade_fee"])
    slippage = float(settings.get("backtest_slippage_pct", 0.0)) / 100
    trades: list[dict[str, Any]] = []
    equity_curve: list[float] = []
    peak_equity = starting_cash
    max_drawdown_pct = 0.0
    allow_short = settings.get("allow_short_selling", False)

    days: dict[str, list[Candle]] = {}
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
                cash -= gross + fee
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
                    if candle.high >= trigger:
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
                        reason, level_fill = backtest_intrabar_long_exit(candle, stop, target)
                        price = float(level_fill) if level_fill is not None else candle.close
                        if reason == "stop":
                            price = apply_slippage(price, "SELL", slippage)
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
                        elif reason == "target":
                            price = apply_slippage(price, "SELL", slippage)
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
                    if candle.low <= trigger:
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
                    coin = -spend / fill_price
                    cash += spend - fee_paid
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
                        reason, level_fill = backtest_intrabar_short_exit(candle, stop, target)
                        price = float(level_fill) if level_fill is not None else candle.close
                        if reason == "stop":
                            price = apply_slippage(price, "BUY", slippage)
                            gross = abs(coin) * price
                            fee = gross * trade_fee
                            cash -= gross + fee
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
                        elif reason == "target":
                            price = apply_slippage(price, "BUY", slippage)
                            gross = abs(coin) * price
                            fee = gross * trade_fee
                            cash -= gross + fee
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
                        reason, level_fill = backtest_intrabar_long_exit(candle, stop, target)
                        price = float(level_fill) if level_fill is not None else candle.close
                        if reason == "stop":
                            price = apply_slippage(price, "SELL", slippage)
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
                        elif reason == "target":
                            price = apply_slippage(price, "SELL", slippage)
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

    metrics = backtest_metrics(trades, starting_cash, equity_curve)

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
        "closed_trades": metrics["closed_trades"],
        "win_rate": metrics["win_rate"],
        "profit_factor": metrics["profit_factor"],
        "expectancy": metrics["expectancy"],
        "expectancy_pct_starting_cash": metrics["expectancy_pct_starting_cash"],
        "average_winner": metrics["average_winner"],
        "average_loser": metrics["average_loser"],
        "payoff_ratio": metrics["payoff_ratio"],
        "total_fees": metrics["total_fees"],
        "sharpe_like": metrics["sharpe_like"],
        "sortino_like": metrics["sortino_like"],
        "_trade_pnls": metrics["trade_pnls"],
        "slippage_pct": float(settings.get("backtest_slippage_pct", 0.0)),
        "open_position": coin != 0,
        "trades": trades[-80:][::-1],
        "equity_curve": [round(item, 8) for item in equity_curve[-300:]],
    }

def run_ewo_offset_backtest_for_symbol(
    symbol: str,
    candles: list[Candle],
    settings: dict[str, Any],
) -> dict[str, Any]:
    starting_cash = float(settings["starting_cash"])
    cash = starting_cash
    coin = 0.0
    entry_price: float | None = None
    highest_price: float | None = None
    partial_done = False
    locked_stop: float | None = None
    locked_target: float | None = None
    locked_exit_mode: str = "fixed"
    trade_fee = float(settings["trade_fee"])
    slippage = float(settings.get("backtest_slippage_pct", 0.0)) / 100
    trade_start_time = int(settings.get("trade_start_time", 0))
    trades: list[dict[str, Any]] = []
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
            highest_price = max(highest_price or price, candle.high)
            stop_price = locked_stop if locked_stop is not None else entry_price * (1 - float(settings["stop_loss_pct"]) / 100)
            target_price = locked_target if locked_target is not None else entry_price * (1 + float(settings["take_profit_pct"]) / 100)
            exit_mode = locked_exit_mode
            partial_quantity = 0.0
            partial_trigger = entry_price + ((target_price - entry_price) * (float(settings.get("partial_take_profit_at_target_pct", 50.0)) / 100))
            trail = trailing_stop_price(entry_price, highest_price, settings)
            intrabar_reason, intrabar_fill = backtest_intrabar_long_exit(candle, stop_price, target_price, trail)
            if settings.get("partial_take_profit_enabled") and not partial_done and candle.high >= partial_trigger and not intrabar_reason:
                reason = f"partial {exit_mode} target"
                partial_done = True
                partial_quantity = coin * (float(settings.get("partial_take_profit_pct", 50.0)) / 100)
                price = max(candle.open, partial_trigger)
            elif intrabar_reason:
                reason = "trailing stop" if intrabar_reason == "trailing stop" else f"{exit_mode} {intrabar_reason}"
                price = float(intrabar_fill)
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
                    locked_stop = locked_target = None
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
                locked_stop, locked_target, locked_exit_mode = exit_prices(entry_price, active_candles, settings)
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

    metrics = backtest_metrics(trades, starting_cash, equity_curve)

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
        "closed_trades": metrics["closed_trades"],
        "win_rate": metrics["win_rate"],
        "profit_factor": metrics["profit_factor"],
        "expectancy": metrics["expectancy"],
        "expectancy_pct_starting_cash": metrics["expectancy_pct_starting_cash"],
        "average_winner": metrics["average_winner"],
        "average_loser": metrics["average_loser"],
        "payoff_ratio": metrics["payoff_ratio"],
        "total_fees": metrics["total_fees"],
        "sharpe_like": metrics["sharpe_like"],
        "sortino_like": metrics["sortino_like"],
        "_trade_pnls": metrics["trade_pnls"],
        "slippage_pct": float(settings.get("backtest_slippage_pct", 0.0)),
        "open_position": coin > 0,
        "trades": trades[-80:][::-1],
        "equity_curve": [round(item, 8) for item in equity_curve[-300:]],
    }

def run_ema_golden_cross_backtest(
    symbol: str,
    candles: list[Candle],
    settings: dict[str, Any],
) -> dict[str, Any]:
    starting_cash = float(settings["starting_cash"])
    cash = starting_cash
    coin = 0.0
    entry_price: float | None = None
    highest_price: float | None = None
    partial_done = False
    locked_stop: float | None = None
    locked_target: float | None = None
    locked_exit_mode: str = "fixed"
    trade_fee = float(settings["trade_fee"])
    slippage = float(settings.get("backtest_slippage_pct", 0.0)) / 100
    ema_short = int(settings.get("ema_short", 50))
    ema_long = int(settings.get("ema_long", 200))
    trade_start_time = int(settings.get("trade_start_time", 0))
    closes: list[float] = []
    trades: list[dict[str, Any]] = []
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
                locked_stop = locked_target = None
                highest_price = None
                continue

            stop_price = locked_stop if locked_stop is not None else entry_price * (1 - float(settings["stop_loss_pct"]) / 100)
            target_price = locked_target if locked_target is not None else entry_price * (1 + float(settings["take_profit_pct"]) / 100)
            exit_mode = locked_exit_mode
            highest_price = max(highest_price or price, candle.high)
            trail = trailing_stop_price(entry_price, highest_price, settings)
            intrabar_reason, intrabar_fill = backtest_intrabar_long_exit(candle, stop_price, target_price, trail)

            if intrabar_reason in {"stop", "trailing stop"}:
                sold_quantity = coin
                fill_price = apply_slippage(float(intrabar_fill), "SELL", slippage)
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
                    "reason": ("Trailing stop hit" if intrabar_reason == "trailing stop" else f"Stop loss hit ({exit_mode})"),
                    "fee_paid": fee_paid,
                })
                coin = 0.0
                entry_price = None
                locked_stop = locked_target = None
                highest_price = None
                continue
            elif intrabar_reason == "target":
                sold_quantity = coin
                fill_price = apply_slippage(float(intrabar_fill), "SELL", slippage)
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
                locked_stop = locked_target = None
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
                    locked_stop, locked_target, locked_exit_mode = exit_prices(entry_price, active_candles, settings)
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

    metrics = backtest_metrics(trades, starting_cash, equity_curve)

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
        "closed_trades": metrics["closed_trades"],
        "win_rate": metrics["win_rate"],
        "profit_factor": metrics["profit_factor"],
        "expectancy": metrics["expectancy"],
        "expectancy_pct_starting_cash": metrics["expectancy_pct_starting_cash"],
        "average_winner": metrics["average_winner"],
        "average_loser": metrics["average_loser"],
        "payoff_ratio": metrics["payoff_ratio"],
        "total_fees": metrics["total_fees"],
        "sharpe_like": metrics["sharpe_like"],
        "sortino_like": metrics["sortino_like"],
        "_trade_pnls": metrics["trade_pnls"],
        "slippage_pct": float(settings.get("backtest_slippage_pct", 0.0)),
        "open_position": coin > 0,
        "trades": trades[-80:][::-1],
        "equity_curve": [round(item, 8) for item in equity_curve[-300:]],
    }

def run_optimizer(settings: dict[str, Any]) -> dict[str, Any]:
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
    results: list[dict[str, Any]] = []
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

def run_ewo_offset_optimizer(settings: dict[str, Any]) -> dict[str, Any]:
    settings = backtest_runtime_settings(settings)
    watchlist = parse_watchlist(settings.get("watchlist", "BTC"))
    if not watchlist:
        raise RuntimeError("Optimizer watchlist is empty")

    granularity = int(settings.get("granularity", 3600))
    candle_count = int(settings.get("candle_count", 300))
    candidates = ewo_offset_candidate_settings(settings)
    results: list[dict[str, Any]] = []
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

def optimizer_score(result: dict[str, Any]) -> float:
    return result["total_pnl_pct"] + (result["max_drawdown_pct"] * 0.75)

def result_settings_summary(symbol: str, settings: dict[str, Any]) -> dict[str, Any]:
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

def ewo_offset_candidate_settings(settings: dict[str, Any]) -> list[dict[str, Any]]:
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
    candidates: list[dict[str, Any]] = []

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

def optimizer_candidate_settings(settings: dict[str, Any]) -> list[dict[str, Any]]:
    short_values = [5, 8, 10, 12]
    long_values = [20, 30, 40, 60]
    stop_values = [1.5, 2.5, 3.5]
    take_values = [3.0, 5.0, 7.0]
    position_values = [0.15, 0.25, 0.35]
    candidates: list[dict[str, Any]] = []

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

def run_walk_forward(settings: dict[str, Any]) -> dict[str, Any]:
    """Rolling/expanding walk-forward validation.

    Each fold optimises only on candles available before that fold, then evaluates
    the selected settings on the immediately following unseen test window.  Test
    windows never overlap.  This is materially stronger than a single 70/30 holdout.
    """
    settings = backtest_runtime_settings(settings)
    watchlist = parse_watchlist(settings.get("watchlist", "BTC"))
    if not watchlist:
        raise RuntimeError("Walk-forward watchlist is empty")

    granularity = int(settings.get("granularity", 3600))
    candle_count = int(settings.get("candle_count", 300))
    initial_train_pct = min(0.75, max(0.45, float(settings.get("walk_forward_train_pct", 0.60))))
    requested_folds = min(8, max(2, int(settings.get("walk_forward_folds", 4))))

    strategy = settings.get("strategy", "sma_cross")
    if strategy == "ewo_offset":
        candidates = ewo_offset_candidate_settings(settings)
    elif strategy == "opening_range":
        candidates = opening_range_candidate_settings(settings)
    else:
        candidates = optimizer_candidate_settings(settings)

    results: list[dict[str, Any]] = []
    errors: list[str] = []
    combinations_tested = 0

    for symbol in watchlist:
        try:
            candles = fetch_candles(
                exchange=str(settings["exchange"]), symbol=symbol,
                quote_currency=str(settings["quote_currency"]), granularity=granularity,
                candle_count=candle_count, asset_class=str(settings.get("asset_class", "crypto")),
            )
            minimum = max(20, strategy_minimum_candles(settings))
            initial_train = max(minimum, int(len(candles) * initial_train_pct))
            remaining = len(candles) - initial_train
            if remaining < 20:
                raise RuntimeError("Not enough candles for rolling walk-forward test windows")
            folds = min(requested_folds, max(2, remaining // 20))
            test_size = remaining // folds
            if test_size < 20:
                raise RuntimeError("Walk-forward test windows are too small")

            fold_rows: list[dict[str, Any]] = []
            oos_pnls: list[float] = []
            oos_trade_pnls: list[float] = []
            oos_closed = oos_wins = 0
            worst_dd = 0.0
            total_fees = 0.0
            last_best_settings: dict[str, Any] | None = None

            for fold in range(folds):
                train_end = initial_train + fold * test_size
                test_end = len(candles) if fold == folds - 1 else min(len(candles), train_end + test_size)
                train_candles = candles[:train_end]
                test_candles = candles[train_end:test_end]
                if len(test_candles) < 5:
                    continue

                best_train = None
                best_settings = None
                for candidate_settings in candidates:
                    if len(train_candles) < strategy_minimum_candles(candidate_settings):
                        continue
                    train_result = run_backtest_for_symbol(symbol, train_candles, candidate_settings)
                    combinations_tested += 1
                    if train_result["trades_count"] == 0:
                        continue
                    score = optimizer_score(train_result)
                    if best_train is None or score > best_train["score"]:
                        best_train = {**train_result, "score": round(score, 4)}
                        best_settings = candidate_settings

                if best_train is None or best_settings is None:
                    continue

                seed_count = strategy_minimum_candles(best_settings)
                seeded = train_candles[-seed_count:] + test_candles
                test_result = run_backtest_for_symbol(
                    symbol, seeded,
                    {**best_settings, "trade_start_time": test_candles[0].time},
                )
                last_best_settings = best_settings
                pnls = [float(x) for x in test_result.get("_trade_pnls", [])]
                oos_trade_pnls.extend(pnls)
                oos_pnls.append(float(test_result["total_pnl"]))
                oos_closed += int(test_result.get("closed_trades", 0))
                oos_wins += sum(1 for x in pnls if x > 0)
                worst_dd = max(worst_dd, abs(float(test_result.get("max_drawdown_pct", 0))))
                total_fees += float(test_result.get("total_fees", 0))
                fold_rows.append({
                    "fold": fold + 1,
                    "train_candles": len(train_candles), "test_candles": len(test_candles),
                    "train_pnl_pct": best_train["total_pnl_pct"], "train_score": best_train["score"],
                    "test_pnl": test_result["total_pnl"], "test_pnl_pct": test_result["total_pnl_pct"],
                    "test_drawdown_pct": test_result["max_drawdown_pct"],
                    "test_trades": test_result["trades_count"], "test_closed_trades": test_result.get("closed_trades", 0),
                    "test_profit_factor": test_result.get("profit_factor", 0),
                    "test_expectancy": test_result.get("expectancy", 0),
                    "settings": result_settings_summary(symbol, best_settings),
                })

            if not fold_rows or last_best_settings is None:
                raise RuntimeError("No valid walk-forward folds produced trades")

            gross_profit = sum(x for x in oos_trade_pnls if x > 0)
            gross_loss = abs(sum(x for x in oos_trade_pnls if x < 0))
            pf = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
            expectancy = sum(oos_trade_pnls) / len(oos_trade_pnls) if oos_trade_pnls else 0.0
            total_pnl = sum(oos_pnls)
            starting_cash = float(settings.get("starting_cash", 1000.0))
            capital_basis = starting_cash * len(fold_rows)
            total_pct = total_pnl / capital_basis * 100 if capital_basis else 0.0
            summary = result_settings_summary(symbol, last_best_settings)
            results.append({
                **summary,
                "folds": len(fold_rows), "fold_results": fold_rows,
                "test_total_pnl": round(total_pnl, 8), "test_total_pnl_pct": round(total_pct, 4),
                "test_drawdown_pct": round(worst_dd, 4), "test_trades": sum(x["test_trades"] for x in fold_rows),
                "test_closed_trades": oos_closed,
                "test_win_rate": round(oos_wins / len(oos_trade_pnls) * 100, 2) if oos_trade_pnls else 0.0,
                "test_profit_factor": round(pf, 4), "test_expectancy": round(expectancy, 8),
                "test_total_fees": round(total_fees, 8),
                "test_score": round(total_pct - worst_dd * 0.75, 4),
                "train_pnl_pct": round(sum(x["train_pnl_pct"] for x in fold_rows) / len(fold_rows), 4),
                "train_score": round(sum(x["train_score"] for x in fold_rows) / len(fold_rows), 4),
                "train_trades": 0, "test_open_position": False,
            })
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")

    results.sort(key=lambda item: item["test_score"], reverse=True)
    return {
        "ok": True, "mode": "rolling_walk_forward", "exchange": settings["exchange"],
        "quote_currency": settings["quote_currency"], "granularity": granularity,
        "candle_count": candle_count, "initial_train_pct": initial_train_pct,
        "requested_folds": requested_folds, "combinations_tested": combinations_tested,
        "results": results[:20], "best": results[0] if results else None, "errors": errors,
    }



def run_strategy_switch_validation(settings: dict[str, Any]) -> dict[str, Any]:
    """Out-of-sample test of regime-driven strategy switching versus fixed strategies.

    The selected strategy for each test fold is decided using TRAINING candles only.
    A regime must persist for several consecutive observations and clear the configured
    confidence threshold.  The selected strategy is then held for the whole unseen test
    fold, preventing candle-by-candle strategy whipsaw and look-ahead bias.
    """
    settings = backtest_runtime_settings(settings)
    watchlist = parse_watchlist(settings.get("watchlist", "BTC"))
    if not watchlist:
        raise RuntimeError("Strategy-switch validation watchlist is empty")

    granularity = int(settings.get("granularity", 3600))
    candle_count = int(settings.get("candle_count", 300))
    folds_requested = min(8, max(2, int(settings.get("strategy_switch_validation_folds", 4))))
    train_pct = min(0.75, max(0.45, float(settings.get("walk_forward_train_pct", 0.60))))
    min_conf = min(1.0, max(0.0, float(settings.get("strategy_switch_min_confidence", 0.60))))
    persistence = min(12, max(1, int(settings.get("strategy_switch_persistence_candles", 3))))
    min_hold = max(5, int(settings.get("strategy_switch_min_hold_candles", 20)))

    configured = str(settings.get("strategy", "sma_cross"))
    strategies = ["sma_cross", "ema_golden_cross", "opening_range", "ewo_offset"]
    if configured not in strategies:
        strategies.insert(0, configured)
    strategies = list(dict.fromkeys(strategies))

    detector = RegimeDetector(lookback=100)
    all_results: list[dict[str, Any]] = []
    errors: list[str] = []

    def persistent_regime(train: list[Candle]) -> tuple[str | None, float, list[str]]:
        observations: list[RegimeResult] = []
        # Evaluate consecutive endings using only information available at each ending.
        for offset in range(persistence - 1, -1, -1):
            end = len(train) - offset
            if end < 30:
                continue
            try:
                observations.append(detector.detect(train[:end]))
            except Exception:
                pass
        names = [str(x.regime) for x in observations]
        if len(observations) < persistence:
            return None, 0.0, names
        last = observations[-1]
        if any(x.regime != last.regime for x in observations):
            return None, float(last.confidence), names
        if float(last.confidence) < min_conf:
            return None, float(last.confidence), names
        return str(last.regime), float(last.confidence), names

    def mapped_strategy(regime: str | None) -> str | None:
        if not regime:
            return None
        mapping = {
            "trending": settings.get("regime_trend_strategy", "ema_golden_cross"),
            "trending_up": settings.get("regime_trend_strategy", "ema_golden_cross"),
            "trending_down": settings.get("regime_trend_strategy", "ema_golden_cross"),
            "ranging": settings.get("regime_ranging_strategy", "opening_range"),
            "breakout": settings.get("regime_breakout_strategy", "opening_range"),
            "volatile": settings.get("regime_volatile_strategy", "sma_cross"),
        }
        return str(mapping.get(regime)) if mapping.get(regime) else None

    for symbol in watchlist:
        try:
            candles = fetch_candles(
                exchange=str(settings["exchange"]), symbol=symbol,
                quote_currency=str(settings["quote_currency"]), granularity=granularity,
                candle_count=candle_count, asset_class=str(settings.get("asset_class", "crypto")),
            )
            initial_train = max(210, int(len(candles) * train_pct))
            remaining = len(candles) - initial_train
            folds = min(folds_requested, max(2, remaining // min_hold))
            test_size = remaining // folds if folds else 0
            if test_size < min_hold:
                raise RuntimeError("Not enough candles for strategy-switch validation windows")

            fold_rows = []
            switched_pnls: list[float] = []
            fixed_totals = {name: 0.0 for name in strategies}
            fixed_trades = {name: 0 for name in strategies}
            previous_selected = configured

            for fold in range(folds):
                train_end = initial_train + fold * test_size
                test_end = len(candles) if fold == folds - 1 else min(len(candles), train_end + test_size)
                train = candles[:train_end]
                test = candles[train_end:test_end]
                if len(test) < min_hold:
                    continue

                regime, confidence, regime_history = persistent_regime(train)
                recommended = mapped_strategy(regime)
                # If the regime is not persistent/confident, retain the previous strategy.
                selected = recommended if recommended in strategies else previous_selected
                previous_selected = selected

                strategy_results: dict[str, dict[str, Any]] = {}
                for strategy_name in strategies:
                    candidate = {**settings, "strategy": strategy_name}
                    seed_count = max(strategy_minimum_candles(candidate), 30)
                    seeded = train[-seed_count:] + test
                    if len(seeded) < strategy_minimum_candles(candidate):
                        continue
                    result = run_backtest_for_symbol(
                        symbol, seeded,
                        {**candidate, "trade_start_time": test[0].time},
                    )
                    strategy_results[strategy_name] = result
                    fixed_totals[strategy_name] += float(result.get("total_pnl", 0.0))
                    fixed_trades[strategy_name] += int(result.get("closed_trades", 0))

                selected_result = strategy_results.get(selected)
                if selected_result is None:
                    selected = configured if configured in strategy_results else next(iter(strategy_results), None)
                    selected_result = strategy_results.get(selected) if selected else None
                if selected_result is None:
                    continue

                switched_pnls.append(float(selected_result.get("total_pnl", 0.0)))
                fold_rows.append({
                    "fold": fold + 1,
                    "train_candles": len(train),
                    "test_candles": len(test),
                    "regime": regime,
                    "regime_confidence": round(confidence, 4),
                    "regime_history": regime_history,
                    "recommended_strategy": recommended,
                    "active_strategy": selected,
                    "configured_strategy": configured,
                    "switch_applied": bool(recommended and selected == recommended),
                    "active_pnl": round(float(selected_result.get("total_pnl", 0.0)), 8),
                    "active_pnl_pct": float(selected_result.get("total_pnl_pct", 0.0)),
                    "active_closed_trades": int(selected_result.get("closed_trades", 0)),
                    "active_fees": float(selected_result.get("total_fees", 0.0)),
                    "fixed": {
                        name: {
                            "pnl": round(float(res.get("total_pnl", 0.0)), 8),
                            "pnl_pct": float(res.get("total_pnl_pct", 0.0)),
                            "closed_trades": int(res.get("closed_trades", 0)),
                            "fees": float(res.get("total_fees", 0.0)),
                        } for name, res in strategy_results.items()
                    },
                })

            if not fold_rows:
                raise RuntimeError("No valid strategy-switch validation folds")

            switched_total = sum(switched_pnls)
            best_fixed_name = max(fixed_totals, key=fixed_totals.get)
            best_fixed_total = fixed_totals[best_fixed_name]
            improvement = switched_total - best_fixed_total
            all_results.append({
                "symbol": symbol,
                "configured_strategy": configured,
                "folds": len(fold_rows),
                "switching_total_pnl": round(switched_total, 8),
                "best_fixed_strategy": best_fixed_name,
                "best_fixed_total_pnl": round(best_fixed_total, 8),
                "switching_vs_best_fixed": round(improvement, 8),
                "switching_beats_best_fixed": improvement > 0,
                "fixed_strategy_totals": {k: round(v, 8) for k, v in fixed_totals.items()},
                "fixed_strategy_closed_trades": fixed_trades,
                "fold_results": fold_rows,
            })
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")

    switching_total = sum(float(x["switching_total_pnl"]) for x in all_results)
    best_fixed_portfolio = None
    if all_results:
        portfolio_fixed = {name: sum(float(r["fixed_strategy_totals"].get(name, 0.0)) for r in all_results) for name in strategies}
        best_fixed_portfolio = max(portfolio_fixed, key=portfolio_fixed.get)
        best_fixed_value = portfolio_fixed[best_fixed_portfolio]
    else:
        portfolio_fixed, best_fixed_value = {}, 0.0

    return {
        "ok": True,
        "mode": "out_of_sample_strategy_switch_validation",
        "exchange": settings.get("exchange"),
        "quote_currency": settings.get("quote_currency"),
        "granularity": granularity,
        "candle_count": candle_count,
        "min_regime_confidence": min_conf,
        "persistence_candles": persistence,
        "min_strategy_hold_candles": min_hold,
        "configured_strategy": configured,
        "strategies_compared": strategies,
        "results": all_results,
        "portfolio_switching_total_pnl": round(switching_total, 8),
        "portfolio_fixed_totals": {k: round(v, 8) for k, v in portfolio_fixed.items()},
        "portfolio_best_fixed_strategy": best_fixed_portfolio,
        "portfolio_best_fixed_total_pnl": round(best_fixed_value, 8),
        "portfolio_switching_vs_best_fixed": round(switching_total - best_fixed_value, 8),
        "portfolio_switching_beats_best_fixed": switching_total > best_fixed_value,
        "errors": errors,
        "note": "All strategy choices are made from prior candles only; test windows include configured fees and slippage.",
    }


def run_monte_carlo(settings: dict[str, Any]) -> dict[str, Any]:
    """Bootstrap closed-trade P/L from the current strategy backtest."""
    settings = backtest_runtime_settings(settings)
    simulations = min(10000, max(250, int(settings.get("monte_carlo_simulations", 2000))))
    seed = int(settings.get("monte_carlo_seed", 42))
    base = run_backtest(settings)
    trade_pnls: list[float] = []
    for result in base.get("results", []):
        trade_pnls.extend(float(x) for x in result.get("_trade_pnls", []))
    if len(trade_pnls) < 5:
        return {"ok": False, "error": f"Need at least 5 closed trades for Monte Carlo; found {len(trade_pnls)}", "trades": len(trade_pnls)}

    rng = random.Random(seed)
    starting_cash = float(settings.get("starting_cash", 1000.0))
    paths = []
    dds = []
    ruin = 0
    for _ in range(simulations):
        equity = starting_cash
        peak = starting_cash
        max_dd = 0.0
        for _ in range(len(trade_pnls)):
            equity += rng.choice(trade_pnls)
            peak = max(peak, equity)
            if peak > 0:
                max_dd = max(max_dd, (peak - equity) / peak * 100)
            if equity <= 0:
                ruin += 1
                equity = 0.0
                break
        paths.append((equity - starting_cash) / starting_cash * 100 if starting_cash else 0.0)
        dds.append(max_dd)
    paths.sort(); dds.sort()
    def q(values, pct):
        idx = min(len(values)-1, max(0, int(round((len(values)-1)*pct))))
        return values[idx]
    return {
        "ok": True, "simulations": simulations, "closed_trades": len(trade_pnls),
        "median_return_pct": round(q(paths, .50), 2), "p05_return_pct": round(q(paths, .05), 2),
        "p95_return_pct": round(q(paths, .95), 2), "median_max_drawdown_pct": round(q(dds, .50), 2),
        "p95_max_drawdown_pct": round(q(dds, .95), 2),
        "probability_positive_pct": round(sum(x > 0 for x in paths) / len(paths) * 100, 2),
        "probability_loss_pct": round(sum(x < 0 for x in paths) / len(paths) * 100, 2),
        "probability_ruin_pct": round(ruin / simulations * 100, 2),
        "base_backtest": {"symbols": len(base.get("results", [])), "errors": base.get("errors", [])},
    }

def opening_range_candidate_settings(settings: dict[str, Any]) -> list[dict[str, Any]]:
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

    candidates: list[dict[str, Any]] = []

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

def is_forex_settings(settings: dict[str, Any]) -> bool:
    return (
        settings.get("asset_class") == "forex"
        or settings.get("exchange") in {"forex_demo", "oanda_demo"}
    )

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

def backtest_intrabar_long_exit(
    candle: Candle,
    stop_price: float,
    target_price: float,
    trailing_price: float | None = None,
) -> tuple[str | None, float | None]:
    """Resolve protective exits from OHLC data, conservatively if both levels hit."""
    effective_stop = max(stop_price, trailing_price) if trailing_price is not None else stop_price
    stop_hit = effective_stop > 0 and candle.low <= effective_stop
    target_hit = target_price > 0 and candle.high >= target_price
    if stop_hit:
        # Gap through a stop fills no better than the candle open; otherwise at the stop.
        return ("trailing stop" if trailing_price is not None and effective_stop == trailing_price else "stop",
                min(candle.open, effective_stop))
    if target_hit:
        # A favourable gap may fill at the open; otherwise at the target.
        return "target", max(candle.open, target_price)
    return None, None


def backtest_intrabar_short_exit(
    candle: Candle,
    stop_price: float,
    target_price: float,
) -> tuple[str | None, float | None]:
    """Resolve short protective exits from OHLC data; stop wins ambiguous same-bar hits."""
    stop_hit = stop_price > 0 and candle.high >= stop_price
    target_hit = target_price > 0 and candle.low <= target_price
    if stop_hit:
        return "stop", max(candle.open, stop_price)
    if target_hit:
        return "target", min(candle.open, target_price)
    return None, None


def backtest_metrics(trades: list[dict[str, Any]], starting_cash: float, equity_curve: list[float]) -> dict[str, Any]:
    """Trade-level analytics using FIFO long lots and explicit SHORT lots."""
    long_qty = long_cost = 0.0
    short_qty = short_proceeds = 0.0
    pnls: list[float] = []
    fees = sum(float(t.get("fee_paid", 0.0) or 0.0) for t in trades)
    for t in trades:
        side = str(t.get("side", "")).upper()
        qty = abs(float(t.get("quantity", 0.0) or 0.0))
        price = float(t.get("price", 0.0) or 0.0)
        fee = float(t.get("fee_paid", 0.0) or 0.0)
        if qty <= 0 or price <= 0:
            continue
        if side == "SHORT":
            short_qty += qty
            short_proceeds += qty * price - fee
        elif side == "BUY" and short_qty > 1e-12:
            close_qty = min(qty, short_qty)
            basis = short_proceeds * (close_qty / short_qty)
            pnls.append(basis - (close_qty * price + fee * (close_qty / qty)))
            short_proceeds -= basis
            short_qty -= close_qty
        elif side == "BUY":
            long_qty += qty
            long_cost += qty * price + fee
        elif side == "SELL" and long_qty > 1e-12:
            close_qty = min(qty, long_qty)
            basis = long_cost * (close_qty / long_qty)
            pnls.append(close_qty * price - fee * (close_qty / qty) - basis)
            long_cost -= basis
            long_qty -= close_qty

    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    avg_win = gross_profit / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    expectancy = sum(pnls) / len(pnls) if pnls else 0.0
    returns = []
    for a, b in zip(equity_curve, equity_curve[1:]):
        if a > 0:
            returns.append((b - a) / a)
    sharpe = 0.0
    sortino = 0.0
    if len(returns) > 1:
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        sd = math.sqrt(variance)
        if sd > 0:
            sharpe = mean_r / sd * math.sqrt(len(returns))
        downside = [min(r, 0.0) for r in returns]
        downside_dev = math.sqrt(sum(r * r for r in downside) / len(downside)) if downside else 0.0
        if downside_dev > 0:
            sortino = mean_r / downside_dev * math.sqrt(len(returns))
    return {
        "closed_trades": len(pnls),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(pnls) * 100, 2) if pnls else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
        "expectancy": round(expectancy, 8),
        "expectancy_pct_starting_cash": round(expectancy / starting_cash * 100, 4) if starting_cash else 0.0,
        "average_winner": round(avg_win, 8),
        "average_loser": round(avg_loss, 8),
        "payoff_ratio": round(avg_win / avg_loss, 4) if avg_loss > 0 else (999.0 if avg_win > 0 else 0.0),
        "total_fees": round(fees, 8),
        "sharpe_like": round(sharpe, 4),
        "sortino_like": round(sortino, 4),
        "trade_pnls": [round(x, 10) for x in pnls],
    }


def apply_slippage(price: float, side: str, slippage_fraction: float) -> float:
    if side.upper() == "BUY":
        return price * (1 + slippage_fraction)
    return price * (1 - slippage_fraction)

def exit_prices(
    entry_price: float,
    candles: list[Candle],
    settings: dict[str, Any],
) -> tuple[float, float, str]:
    """Backtest-safe exit calculation without PaperBot instance state."""
    if not candles or len(candles) < 14:
        return (
            entry_price * (1 - float(settings["stop_loss_pct"]) / 100),
            entry_price * (1 + float(settings["take_profit_pct"]) / 100),
            "fixed",
        )

    if settings.get("use_atr_exits", True):
        atr_period = int(settings.get("atr_period", 14))
        atr_value = calculate_atr_from_candles(candles, atr_period)
        if atr_value > 0:
            stop_mult = float(settings.get("atr_stop_multiplier", 1.5))
            target_mult = float(settings.get("atr_target_multiplier", 2.5))
            stop = entry_price - (atr_value * stop_mult)
            target = entry_price + (atr_value * target_mult)
            if (entry_price - stop) / entry_price * 100 >= 0.2:
                return stop, target, f"ATR ({atr_value:.4f})"

    if settings.get("use_dynamic_sr_exits"):
        levels = support_resistance(candles, settings)
        support = levels.get("support")
        resistance = levels.get("resistance")
        confirmed = levels.get("confirmed", levels.get("sr_confirmed", False))
        if support and resistance and confirmed:
            stop_buffer = float(settings.get("support_stop_buffer_pct", 2.0)) / 100
            target_buffer = float(settings.get("resistance_target_buffer_pct", 0.5)) / 100
            sr_stop = float(support) * (1 - stop_buffer)
            sr_target = float(resistance) * (1 - target_buffer)
            if sr_stop < entry_price < sr_target:
                return sr_stop, sr_target, "S/R"

    return (
        entry_price * (1 - float(settings["stop_loss_pct"]) / 100),
        entry_price * (1 + float(settings["take_profit_pct"]) / 100),
        "fixed",
    )

def position_spend(
    cash: float,
    entry_price: float,
    candles: list[Candle],
    settings: dict[str, Any],
) -> tuple[float, str]:
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
    settings: dict[str, Any],
    already_done: bool,
) -> bool:
    if already_done or not settings.get("partial_take_profit_enabled"):
        return False
    trigger_fraction = float(settings.get("partial_take_profit_at_target_pct", 50.0)) / 100
    trigger_price = entry_price + ((target_price - entry_price) * trigger_fraction)
    return target_price > entry_price and price >= trigger_price

def trailing_stop_price(
    entry_price: float,
    highest_price: float | None,
    settings: dict[str, Any],
) -> float | None:
    if not settings.get("trailing_stop_enabled") or not highest_price:
        return None
    activation = float(settings.get("trailing_activation_pct", 3.0)) / 100
    if highest_price < entry_price * (1 + activation):
        return None
    trail = float(settings.get("trailing_stop_pct", 2.0)) / 100
    return highest_price * (1 - trail)

def chart_trade_plan(
    state: BotState,
    chart_symbol: str,
    chart_row: dict[str, Any],
) -> dict[str, Any]:
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

def setup_performance(records: list[SetupRecord]) -> list[dict[str, Any]]:
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

def weak_pair_map(records: list[SetupRecord], settings: dict[str, Any]) -> dict[str, str]:
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

def setup_edge_score(records: list[SetupRecord], symbol: str, settings_key: str) -> float:
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

def recent_setup_records(records: list[SetupRecord], limit: int = 40) -> list[dict[str, Any]]:
    return [asdict(record) for record in records[-limit:]][::-1]

def symbol_performance(trades: list[Trade]) -> list[dict[str, Any]]:
    open_buys: dict[str, list[Trade]] = {}
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

def live_market_guard(
    exchange: str,
    symbol: str,
    quote_currency: str,
    granularity: int,
    candle_count: int,
    max_spread_pct: float,
    min_quote_volume: float,
) -> dict[str, Any]:
    if exchange.lower() != "coinbase":
        return {"ok": True, "reason": "Guard only enforced for Coinbase live trading"}

    ticker = fetch_coinbase_ticker(symbol, quote_currency)
    bid = float(ticker.get("bid") or 0.0)
    ask = float(ticker.get("ask") or 0.0)
    if bid <= 0 or ask <= 0 or ask < bid:
        return {"ok": False, "reason": "invalid Coinbase bid/ask"}

    midpoint = (bid + ask) / 2
    spread_pct = ((ask - bid) / midpoint) * 100 if midpoint else 100.0
    if spread_pct > max_spread_pct:
        return {
            "ok": False,
            "reason": f"spread {spread_pct:.3f}% > limit {max_spread_pct:.3f}%",
            "spread_pct": round(spread_pct, 4),
            "bid": bid,
            "ask": ask,
        }

    candles = fetch_candles(
        exchange=exchange,
        symbol=symbol,
        quote_currency=quote_currency,
        granularity=granularity,
        candle_count=min(50, max(20, candle_count)),
    )
    quote_volume = sum(candle.close * candle.volume for candle in candles)
    if quote_volume < min_quote_volume:
        return {
            "ok": False,
            "reason": (
                f"recent quote volume {quote_currency} {quote_volume:.2f} "
                f"< minimum {quote_currency} {min_quote_volume:.2f}"
            ),
            "spread_pct": round(spread_pct, 4),
            "quote_volume": round(quote_volume, 2),
            "bid": bid,
            "ask": ask,
        }

    return {
        "ok": True,
        "reason": "market liquid enough",
        "spread_pct": round(spread_pct, 4),
        "quote_volume": round(quote_volume, 2),
        "bid": bid,
        "ask": ask,
    }

def fetch_coinbase_ticker(symbol: str, quote_currency: str) -> dict[str, Any]:
    product = f"{symbol.upper()}-{quote_currency.upper()}"
    return fetch_json(f"https://api.exchange.coinbase.com/products/{product}/ticker")

def coinbase_products_for_quote(quote_currency: str = "GBP") -> dict[str, Any]:
    quote_currency = quote_currency.upper()
    products = fetch_json("https://api.exchange.coinbase.com/products")
    rows = []

    for product in products:
        if product.get("quote_currency") != quote_currency:
            continue
        if product.get("status") != "online":
            continue
        product_id = product.get("id", "")
        base_currency = product.get("base_currency") or product_id.split("-", 1)[0]
        rows.append({
            "product_id": product_id,
            "base_currency": base_currency,
            "quote_currency": quote_currency,
            "display_name": product.get("display_name", product_id),
            "min_market_funds": product.get("min_market_funds"),
        })

    rows.sort(key=lambda item: item["base_currency"])
    return {
        "ok": True,
        "quote_currency": quote_currency,
        "count": len(rows),
        "products": rows,
        "symbols": [row["base_currency"] for row in rows],
        "watchlist": ",".join(row["base_currency"] for row in rows),
    }

def coinbase_quote_comparison(settings: dict[str, Any]) -> dict[str, Any]:
    quote_values = str(settings.get("quote_currencies", "GBP,USD,USDC")).upper()
    quotes = [item.strip() for item in quote_values.replace("\n", ",").split(",") if item.strip()]
    if not quotes:
        quotes = [str(settings.get("quote_currency", "GBP")).upper()]

    watchlist = parse_watchlist(settings.get("watchlist", "BTC,ETH,SOL,XRP,DOGE,ADA,LINK,AVAX,LTC,BCH"))
    preferred = watchlist or ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "LINK", "AVAX", "LTC", "BCH"]
    granularity = int(settings.get("granularity", settings.get("live_granularity", 3600)))
    candle_count = min(300, max(strategy_minimum_candles(settings), int(settings.get("candle_count", settings.get("live_candle_count", 120)))))
    max_symbols = max(1, min(10, int(settings.get("max_symbols", 8))))
    rows: list[dict[str, Any]] = []

    for quote in quotes:
        errors: list[str] = []
        products = coinbase_products_for_quote(quote)
        supported = set(products["symbols"])
        candidates = [symbol for symbol in preferred if symbol in supported][:max_symbols]
        if not candidates:
            candidates = products["symbols"][:max_symbols]

        spread_values: list[float] = []
        volume_values: list[float] = []
        backtest_rows: list[dict[str, Any]] = []

        for symbol in candidates:
            try:
                guard = live_market_guard(
                    exchange="coinbase",
                    symbol=symbol,
                    quote_currency=quote,
                    granularity=granularity,
                    candle_count=candle_count,
                    max_spread_pct=999,
                    min_quote_volume=0,
                )
                if guard.get("spread_pct") is not None:
                    spread_values.append(float(guard["spread_pct"]))
                if guard.get("quote_volume") is not None:
                    volume_values.append(float(guard["quote_volume"]))

                candles = fetch_candles(
                    exchange="coinbase",
                    symbol=symbol,
                    quote_currency=quote,
                    granularity=granularity,
                    candle_count=candle_count,
                )
                if len(candles) >= strategy_minimum_candles(settings):
                    result = run_backtest_for_symbol(
                        symbol,
                        candles,
                        {**settings, "quote_currency": quote, "watchlist": symbol},
                    )
                    backtest_rows.append(result)
            except Exception as exc:
                errors.append(f"{symbol}: {exc}")

        best = max(backtest_rows, key=lambda item: item.get("total_pnl_pct", -999999), default=None)
        rows.append({
            "quote_currency": quote,
            "online_pairs": products["count"],
            "tested_symbols": candidates,
            "tested_count": len(candidates),
            "avg_spread_pct": round(sum(spread_values) / len(spread_values), 4) if spread_values else None,
            "avg_quote_volume": round(sum(volume_values) / len(volume_values), 2) if volume_values else None,
            "best_symbol": best.get("symbol") if best else None,
            "best_pnl_pct": best.get("total_pnl_pct") if best else None,
            "best_pnl": best.get("total_pnl") if best else None,
            "best_trades": best.get("trades_count") if best else 0,
            "errors": errors[:6],
        })

    rows.sort(key=lambda item: (
        item["best_pnl_pct"] if item["best_pnl_pct"] is not None else -999999,
        -(item["avg_spread_pct"] or 999999),
    ), reverse=True)
    return {
        "ok": True,
        "quotes": quotes,
        "rows": rows,
        "granularity": granularity,
        "candle_count": candle_count,
        "max_symbols": max_symbols,
    }

# ─── MIGRATION FUNCTION ────────────────────────────────────────────

def migrate_to_database():
    if not STATE_FILE.exists():
        logger.info("No state file to migrate")
        return

    db = BotDatabase()

    try:
        with open(STATE_FILE, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load state file: {e}")
        return

    migrated = 0

    for trade_data in data.get('trades', []):
        try:
            if 'stop_loss_price' not in trade_data:
                trade_data['stop_loss_price'] = None
            if 'take_profit_price' not in trade_data:
                trade_data['take_profit_price'] = None
            if 'exit_mode' not in trade_data:
                trade_data['exit_mode'] = None
            if 'exit_reason' not in trade_data:
                trade_data['exit_reason'] = None
            if 'regime' not in trade_data:
                trade_data['regime'] = None
            trade = Trade(**trade_data)
            db.save_trade(trade)
            migrated += 1
        except Exception as e:
            logger.warning(f"Failed to migrate trade: {e}")

    for entry_data in data.get('journal', []):
        try:
            entry = JournalEntry(**entry_data)
            db.save_journal(entry)
        except Exception as e:
            logger.warning(f"Failed to migrate journal entry: {e}")

    for record_data in data.get('setup_records', []):
        try:
            if 'stop_loss_price' not in record_data:
                record_data['stop_loss_price'] = None
            if 'take_profit_price' not in record_data:
                record_data['take_profit_price'] = None
            if 'exit_mode' not in record_data:
                record_data['exit_mode'] = None
            record = SetupRecord(**record_data)
            db.save_setup_record(record)
        except Exception as e:
            logger.warning(f"Failed to migrate setup record: {e}")

    for signal_type, history_data in data.get('signal_history', {}).items():
        try:
            db.save_signal_history(signal_type, history_data)
        except Exception as e:
            logger.warning(f"Failed to migrate signal history for {signal_type}: {e}")

    logger.info(f"Migration complete: {migrated} trades migrated")

# ─── HTTP SERVER ─────────────────────────────────────────────────────

def parse_json_body(handler: SimpleHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8"))

class BotRequestHandler(SimpleHTTPRequestHandler):
    bot: PaperBot

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.bot = BotRequestHandler.bot
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def _check_auth(self) -> bool:
        token = os.environ.get("BOT_API_TOKEN", "")
        if not token:
            return True
        header = self.headers.get("Authorization", "")
        expected = f"Bearer {token}"
        if header == expected:
            return True
        self.send_json({"ok": False, "error": "Unauthorized"}, HTTPStatus.UNAUTHORIZED)
        return False

    def do_GET(self) -> None:
        if self.path.startswith("/api/") and not self._check_auth():
            return
        try:
            if self.path == "/api/status":
                try:
                    self.send_json(self.bot.snapshot())
                except BrokenPipeError:
                    logger.debug("Client disconnected during /api/status")
                    return
                return

            if self.path == "/api/status-light":
                try:
                    self.send_json({
                        "running": self.bot.state.running,
                        "equity": self.bot.equity(self.bot.state.last_price),
                        "cash": self.bot.state.cash,
                        "last_price": self.bot.state.last_price,
                        "positions": len(self.bot.state.positions),
                        "last_signal": self.bot.state.last_signal,
                        "last_error": self.bot.state.last_error,
                    })
                except BrokenPipeError:
                    logger.debug("Client disconnected during /api/status-light")
                    return
                return

            if self.path == "/api/diagnostics":
                try:
                    self.send_json(diagnostics())
                except BrokenPipeError:
                    return
                return

            if self.path == "/api/sync-oanda-balance":
                try:
                    summary = self.bot.get_oanda_account_summary()
                    if summary.get("ok"):
                        with self.bot.lock:
                            self.bot.state.cash = summary.get("balance", 0)
                            self.bot.state.positions = {}
                            self.bot.state.coin = 0.0
                            self.bot.state.active_symbol = None

                            for pos in summary.get("positions", []):
                                symbol = pos.get("symbol")
                                units = pos.get("units", 0)
                                avg_price = pos.get("average_price", 0)
                                is_short = pos.get("side") == "SHORT"

                                if units != 0:
                                    self.bot.state.positions[symbol] = {
                                        "quantity": -units if is_short else units,
                                        "entry_price": avg_price,
                                        "highest_price": avg_price,
                                        "is_short": is_short,
                                        "opened_at": now_iso(),
                                        "entry_time": time.time(),
                                    }
                            self.bot.save_state()

                        self.send_json({
                            "ok": True,
                            "balance": summary.get("balance"),
                            "equity": summary.get("equity"),
                            "positions": summary.get("positions_count"),
                            "pnl": summary.get("total_pnl")
                        })
                    else:
                        self.send_json({"ok": False, "error": summary.get("error")})
                except BrokenPipeError:
                    logger.debug("Client disconnected during /api/sync-oanda-balance")
                    return
                return

            if self.path == "/api/strategy-dashboard":
                try:
                    result = self.bot.get_strategy_dashboard()
                    self.send_json(result)
                except BrokenPipeError:
                    return
                return

            if self.path.startswith("/api/candles"):
                try:
                    self.handle_candles_request()
                except BrokenPipeError:
                    logger.debug("Client disconnected during /api/candles")
                    return
                return

            if self.path == "/api/coinbase-auth-check":
                try:
                    self.send_json(coinbase_auth_check())
                except BrokenPipeError:
                    return
                return

            if self.path == "/api/oanda-auth-check":
                try:
                    self.send_json(oanda_auth_check())
                except BrokenPipeError:
                    return
                return

            if self.path == "/api/coinbase-gbp-products":
                try:
                    self.send_json(coinbase_products_for_quote("GBP"))
                except BrokenPipeError:
                    return
                return

            if self.path.startswith("/api/coinbase-products"):
                try:
                    query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    quote = query.get("quote", ["GBP"])[0]
                    self.send_json(coinbase_products_for_quote(quote))
                except BrokenPipeError:
                    return
                return

            if self.path.endswith('.css') or self.path.endswith('.js') or self.path.endswith('.json') or self.path.endswith('.png') or self.path.endswith('.jpg') or self.path.endswith('.svg'):
                try:
                    self.serve_static_file()
                except BrokenPipeError:
                    logger.debug("Client disconnected during static file transfer")
                    return
                return

            if not self.path.startswith("/api/"):
                try:
                    self.send_index()
                except BrokenPipeError:
                    logger.debug("Client disconnected during index transfer")
                    return
                return

            try:
                self.send_error(HTTPStatus.NOT_FOUND, f"Endpoint not found: {self.path}")
            except BrokenPipeError:
                return

        except BrokenPipeError:
            logger.debug(f"Client disconnected during GET request: {self.path}")
            return
        except ConnectionResetError:
            logger.debug(f"Connection reset during GET request: {self.path}")
            return
        except Exception as exc:
            logger.error(f"GET error: {exc}")
            try:
                self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except (BrokenPipeError, ConnectionResetError):
                return

    def handle_candles_request(self) -> None:
        try:
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)

            symbol = params.get('symbol', ['BTC'])[0].upper()
            granularity = int(params.get('granularity', ['3600'])[0])
            count = int(params.get('count', ['200'])[0])
            quote_currency = params.get('quote', ['GBP'])[0].upper()

            settings = self.bot.state.settings
            exchange = settings.get('exchange', 'coinbase')
            asset_class = settings.get('asset_class', 'crypto')

            candles = fetch_candles(
                exchange=exchange,
                symbol=symbol,
                quote_currency=quote_currency,
                granularity=granularity,
                candle_count=count,
                asset_class=asset_class
            )

            candle_data = [
                {
                    'time': c.time,
                    'open': c.open,
                    'high': c.high,
                    'low': c.low,
                    'close': c.close,
                    'volume': c.volume
                }
                for c in candles
            ]

            current_price = self.bot.state.last_price

            chart_row = next(
                (row for row in self.bot.state.scan_rows if row.get('symbol') == symbol),
                {}
            )

            self.send_json({
                'ok': True,
                'symbol': symbol,
                'granularity': granularity,
                'count': len(candle_data),
                'candles': candle_data,
                'current_price': current_price,
                'support': chart_row.get('support'),
                'resistance': chart_row.get('resistance'),
            })

        except BrokenPipeError:
            logger.debug("Client disconnected during candle request")
            return
        except Exception as e:
            logger.error(f"Error fetching candles: {e}")
            try:
                self.send_json({'ok': False, 'error': str(e)}, HTTPStatus.BAD_REQUEST)
            except BrokenPipeError:
                return

    def serve_static_file(self) -> None:
        try:
            path = self.path.lstrip('/')
            file_path = WEB_DIR / path

            if not file_path.exists() or not file_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            if path.endswith('.css'):
                content_type = 'text/css'
            elif path.endswith('.js'):
                content_type = 'application/javascript'
            elif path.endswith('.json'):
                content_type = 'application/json'
            elif path.endswith('.png'):
                content_type = 'image/png'
            elif path.endswith('.jpg') or path.endswith('.jpeg'):
                content_type = 'image/jpeg'
            elif path.endswith('.svg'):
                content_type = 'image/svg+xml'
            else:
                content_type = 'application/octet-stream'

            body = file_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        except BrokenPipeError:
            logger.debug("Client disconnected while sending static file")
            return
        except Exception as e:
            logger.error(f"Error serving static file: {e}")
            return

    def send_index(self) -> None:
        try:
            index_path = WEB_DIR / "index.html"
            if not index_path.exists():
                self.send_error(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"Dashboard missing at {index_path}",
                )
                return

            body = index_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        except BrokenPipeError:
            logger.debug("Client disconnected while sending index")
            return
        except Exception as e:
            logger.error(f"Error sending index: {e}")
            return

    def send_json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        try:
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

            try:
                self.wfile.write(body)
            except BrokenPipeError:
                logger.debug("Client disconnected while sending JSON response")
                return
            except ConnectionResetError:
                logger.debug("Connection reset while sending JSON response")
                return

        except BrokenPipeError:
            logger.debug("Client disconnected before response could be sent")
            return
        except ConnectionResetError:
            logger.debug("Connection reset before response could be sent")
            return
        except Exception as e:
            logger.error(f"Error sending JSON: {e}")
            return

    def do_POST(self) -> None:
        if not self._check_auth():
            return
        try:
            if self.path == "/api/start":
                self.bot.start()
                self.send_json({"ok": True})
                return

            if self.path == "/api/stop":
                self.bot.stop()
                self.send_json({"ok": True})
                return

            if self.path == "/api/reset":
                self.bot.reset()
                self.send_json({"ok": True})
                return

            if self.path == "/api/verify-oanda-positions":
                self.send_json(self.bot.verify_oanda_positions())
                return

            if self.path == "/api/sync-live-balance":
                self.send_json(self.bot.sync_live_balance_from_coinbase())
                return

            if self.path == "/api/sync-oanda-balance":
                self.send_json(self.bot.sync_paper_balance_from_oanda())
                return

            if self.path == "/api/coinbase-auth-check":
                self.send_json(coinbase_auth_check())
                return

            if self.path == "/api/oanda-auth-check":
                self.send_json(oanda_auth_check())
                return

            if self.path == "/api/settings":
                self.bot.update_settings(parse_json_body(self))
                self.send_json({"ok": True})
                return

            if self.path == "/api/backtest":
                payload = parse_json_body(self)
                settings = {**self.bot.snapshot()["settings"], **payload}
                self.send_json(run_backtest(settings))
                return

            if self.path == "/api/optimize":
                payload = parse_json_body(self)
                settings = {**self.bot.snapshot()["settings"], **payload}
                self.send_json(run_optimizer(settings))
                return

            if self.path == "/api/walk-forward":
                payload = parse_json_body(self)
                settings = {**self.bot.snapshot()["settings"], **payload}
                self.send_json(run_walk_forward(settings))
                return

            if self.path == "/api/strategy-switch-validation":
                payload = parse_json_body(self)
                settings = {**self.bot.snapshot()["settings"], **payload}
                self.send_json(run_strategy_switch_validation(settings))
                return

            if self.path == "/api/monte-carlo":
                payload = parse_json_body(self)
                settings = {**self.bot.snapshot()["settings"], **payload}
                self.send_json(run_monte_carlo(settings))
                return

            if self.path == "/api/validation":
                payload = parse_json_body(self)
                settings = {**self.bot.snapshot()["settings"], **payload}
                try:
                    from validation_engine import run_validation_suite
                    self.send_json(run_validation_suite(settings))
                except Exception as exc:
                    logger.exception("Validation suite failed")
                    self.send_json({"ok": False, "error": str(exc)})
                return

            if self.path == "/api/close-position":
                payload = parse_json_body(self)
                self.send_json(
                    self.bot.close_position_manual(
                        symbol=str(payload.get("symbol", "")),
                        mode=str(payload.get("mode", "profit_only")),
                    )
                )
                return

            if self.path == "/api/quote-comparison":
                payload = parse_json_body(self)
                settings = {**self.bot.snapshot()["settings"], **payload}
                self.send_json(coinbase_quote_comparison(settings))
                return

            if self.path == "/api/export-db":
                path = BASE_DIR / f"export_{today_key()}.json"
                self.bot.db.export_json(path)
                self.send_json({"ok": True, "path": str(path)})
                return

            if self.path == "/api/evolve-strategies":
                generations = int(parse_json_body(self).get('generations', 50))
                result = self.bot.evolve_strategies(generations)
                self.send_json(result)
                return

            if self.path == "/api/apply-strategy":
                payload = parse_json_body(self)
                candles = payload.get('candles', [])
                result = self.bot.apply_strategy_signal(candles)
                self.send_json(result)
                return

            if self.path == "/api/backfill-tpsl":
                self.bot.backfill_tpsl_from_positions()
                self.send_json({"ok": True, "message": "TP/SL backfill completed"})
                return

            if self.path == "/api/train-xgboost":
                self.bot.self_learning_trader.train_xgboost(force=True)
                self.send_json({"ok": True})
                return

            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            logger.error(f"POST error: {exc}")
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:
        if self.path.startswith("/api/status"):
            return
        super().log_message(format, *args)

def diagnostics() -> dict[str, Any]:
    accounts_path = "/api/v3/brokerage/accounts"
    return {
        "ok": True,
        "server": "Auxo",
        "dotenv_file_present": ENV_FILE.exists(),
        "dotenv_loaded_keys": sorted(DOTENV_LOADED_KEYS),
        "audit_log_file": str(AUDIT_LOG_FILE),
        "audit_log_present": AUDIT_LOG_FILE.exists(),
        "db_file": str(DB_FILE),
        "db_present": DB_FILE.exists(),
        "coinbase_key_name_present": bool(os.environ.get("COINBASE_API_KEY_NAME", "").strip()),
        "coinbase_private_key_present": bool(os.environ.get("COINBASE_API_PRIVATE_KEY", "").strip()),
        "coinbase_private_key_file_present": bool(os.environ.get("COINBASE_API_PRIVATE_KEY_FILE", "").strip()),
        "coinbase_private_key_file_readable": (
            resolve_local_path(os.environ.get("COINBASE_API_PRIVATE_KEY_FILE", "")).is_file()
            if os.environ.get("COINBASE_API_PRIVATE_KEY_FILE", "").strip()
            else False
        ),
        "coinbase_private_key_source": coinbase_private_key_source(),
        "oanda_api_base": oanda_api_base(),
        "oanda_account_id_present": bool(oanda_account_id()),
        "oanda_api_token_present": bool(oanda_api_token()),
        "oanda_demo_trading_confirm_present": (
            os.environ.get("OANDA_DEMO_TRADING_ENABLED", "").strip().lower() == "true"
        ),
        "oanda_demo_orders_armed": oanda_demo_orders_armed(),
        "live_confirm_present": (
            os.environ.get("LIVE_TRADING_CONFIRM", "")
            == "I_UNDERSTAND_THIS_PLACES_REAL_ORDERS"
        ),
        "cryptography_available": CRYPTOGRAPHY_AVAILABLE,
        "websocket_client_available": WEBSOCKET_AVAILABLE,
        "requests_available": REQUESTS_AVAILABLE,
        "live_status": coinbase_live_status_message(),
        "coinbase_signed_uri_example": f"GET api.coinbase.com{accounts_path}",
    }

def coinbase_auth_check() -> dict[str, Any]:
    data = coinbase_api_request("GET", "/api/v3/brokerage/accounts?limit=1")
    return {
        "ok": True,
        "accounts_visible": len(data.get("accounts", [])),
        "has_next": bool(data.get("has_next")),
    }

def oanda_auth_check() -> dict[str, Any]:
    configured = oanda_is_configured()
    result: dict[str, Any] = {
        "ok": False,
        "configured": configured,
        "api_base": oanda_api_base(),
        "account_id_present": bool(oanda_account_id()),
        "token_present": bool(oanda_api_token()),
        "practice_url": oanda_is_practice(),
        "demo_trading_armed": oanda_demo_orders_armed(),
    }
    if not configured:
        result["error"] = "Missing OANDA_ACCOUNT_ID or OANDA_API_TOKEN in .env"
        return result

    summary = oanda_account_summary()
    account = summary.get("account", {})
    result.update({
        "ok": True,
        "account_id": account.get("id") or oanda_account_id(),
        "currency": account.get("currency"),
        "balance": account.get("balance"),
    })
    return result

def coinbase_private_key_source() -> str:
    if os.environ.get("COINBASE_API_PRIVATE_KEY", "").strip():
        return "COINBASE_API_PRIVATE_KEY"
    if os.environ.get("COINBASE_API_PRIVATE_KEY_FILE", "").strip():
        return "COINBASE_API_PRIVATE_KEY_FILE"
    return ""

# ─── MAIN ────────────────────────────────────────────────────────────

def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    bot = PaperBot()
    BotRequestHandler.bot = bot

    server = ThreadingHTTPServer(("0.0.0.0", port), BotRequestHandler)
    logger.info(f"Auxo running at http://localhost:{port}")
    print(f"Auxo running at http://localhost:{port}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        bot.stop()
        logger.info("Auxo stopped")
        print("\nStopped.")

if __name__ == "__main__":
    main()
