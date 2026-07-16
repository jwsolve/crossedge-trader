# #!/usr/bin/env python3
"""
Auxo trading bot – multi‑asset paper/live trading with Opening Range strategy.
Supports both long (BUY) and short (SELL) positions.
"""

from __future__ import annotations

import json
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
from typing import Any

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
STATE_FILE = BASE_DIR / "bot_state.json"
ENV_FILE = BASE_DIR / ".env"
AUDIT_LOG_FILE = BASE_DIR / "bot_audit.jsonl"


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


DEFAULT_SETTINGS = {
    "asset_class": "crypto",
    "exchange": "coinbase",
    "symbol": "BTC",
    "watchlist": "BTC,ETH,SOL,XRP,DOGE,LINK,AVAX",
    "quote_currency": "GBP",
    "chart_mode": "line",
    "strategy": "opening_range",
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
    "max_daily_live_loss_gbp": 2.0,
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
    # News guard settings
    "news_guard_enabled": False,
    "news_guard_before_minutes": 30,
    "news_guard_after_minutes": 30,
    "news_guard_block_high": True,
    "news_guard_block_medium": False,
    "news_guard_block_low": False,
    # Opening Range Strategy settings
    "opening_range_minutes": 15,
    "opening_range_atr_period": 14,
    "opening_range_manipulation_threshold": 0.20,
    "opening_range_stop_loss_atr_multiplier": 1.5,
    "opening_range_take_profit_atr_multiplier": 2.5,
    "oanda_account_type": "standard",
    # Max drawdown
    "max_drawdown_pct": 20.0,
    # Short selling
    "allow_short_selling": False,
    # Telegram alerts
    "telegram_enabled": False,
    "telegram_bot_token": "",
    "telegram_chat_id": "",
    "telegram_alert_on_buy": True,
    "telegram_alert_on_sell": True,
    "telegram_alert_on_error": True,
    "telegram_alert_on_daily_summary": True,
    "telegram_alert_on_drawdown": True,
    "telegram_drawdown_alert_pct": 10.0,
    # ─── EMA Golden Cross settings ───
    "ema_short": 50,
    "ema_long": 200,
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


@dataclass
class Trade:
    time: str
    side: str
    symbol: str
    price: float
    quantity: float
    cash_after: float
    coin_after: float
    reason: str
    fee_paid: float
    exchange_order_id: str | None = None
    exchange_order_status: str | None = None
    exchange_average_filled_price: float | None = None
    exchange_filled_size: float | None = None


@dataclass
class JournalEntry:
    time: str
    symbol: str
    event: str
    message: str
    price: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SetupRecord:
    id: str
    time: str
    symbol: str
    strategy: str
    settings_key: str
    entry_price: float
    entry_quantity: float
    entry_cost: float
    entry_fee: float
    entry_reason: str
    entry_score: float
    base_score: float
    edge_score: float
    regime: str
    support_distance_pct: float | None = None
    resistance_distance_pct: float | None = None
    sr_range_pct: float | None = None
    reward_risk: float | None = None
    status: str = "OPEN"
    closed_quantity: float = 0.0
    realized_pnl: float = 0.0
    exit_fees: float = 0.0
    exit_time: str | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    pnl_pct: float | None = None


@dataclass
class ManagedOrder:
    order_id: str
    symbol: str
    product_id: str
    side: str
    role: str
    order_type: str
    status: str
    created_at: str
    updated_at: str
    expires_at: float
    retry_count: int = 0
    local_applied: bool = False
    price: float | None = None
    base_size: float | None = None
    quote_size: float | None = None
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Candle:
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class BotState:
    running: bool = False
    settings: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_SETTINGS))
    cash: float = DEFAULT_SETTINGS["starting_cash"]
    coin: float = 0.0  # Positive = long, Negative = short
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
    # News guard state
    news_events: list[dict[str, Any]] = field(default_factory=list)
    news_last_update: str = ""
    news_guard_status: str = "idle"
    # Opening Range state
    opening_range_analysis: dict[str, Any] = field(default_factory=dict)
    # Peak equity for drawdown tracking
    peak_equity: float = DEFAULT_SETTINGS["starting_cash"]
    stop_price: float | None = None
    target_price: float | None = None
    exit_mode: str = "fixed"
    is_short: bool = False


class PaperBot:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.state = self.load_state()
        self.thread: threading.Thread | None = None
        self.websocket_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.websocket_stop_event = threading.Event()
        self.feed_prices: dict[str, float] = {}
        self.news_guard_thread: threading.Thread | None = None
        self.news_guard_stop_event = threading.Event()
        self.restart_delay = 60  # seconds to wait before restarting on error
        self.shutdown_requested = False
        logger.info("Auxo bot initialised")

    def load_state(self) -> BotState:
        if not STATE_FILE.exists():
            state = BotState()
            state.day_start_date = today_key()
            state.peak_equity = float(state.settings["starting_cash"])
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
            )
            # After loading state, ensure oanda_account_type has a value
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
            return state

    def save_state(self) -> None:
        data = asdict(self.state)
        # Don't override running - preserve the current state
        # data["running"] = False
        STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug("State saved")

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
            self.thread = threading.Thread(target=self.run_loop, daemon=True)
            self.thread.start()
            self.start_websocket_feed_if_needed()
            self.start_news_guard_thread()
            self.sync_oanda_positions()
            logger.info("Bot started")

    def stop(self) -> None:
        with self.lock:
            self.state.running = False
            self.stop_event.set()
            self.websocket_stop_event.set()
            self.news_guard_stop_event.set()
            self.shutdown_requested = True
            self.save_state()
            logger.info("Bot stopped")

    # ─── OANDA Position Sync ─────────────────────────────────────────

    def sync_oanda_positions(self) -> None:
        """Sync open positions from OANDA into the bot's state."""
        # Don't check should_oanda_demo_trade() - we want to sync positions even if trading is disabled
        if self.state.settings.get("asset_class") != "forex" or self.state.settings.get("exchange") != "oanda_demo":
            return

        try:
            account_id = urllib.parse.quote(oanda_account_id())
            data = oanda_request(f"/v3/accounts/{account_id}/openPositions")
            positions = data.get("positions", [])

            with self.lock:
                # Clear existing positions that aren't on OANDA
                synced_symbols = set()

                for position in positions:
                    instrument = position.get("instrument", "")
                    symbol = instrument.replace("_", "")
                    synced_symbols.add(symbol)

                    units = int(position.get("short", {}).get("units", 0))
                    if units > 0:
                        # It's a short position
                        price = float(position.get("short", {}).get("averagePrice", 0.0))
                        if price <= 0:
                            price = self.state.last_price or 0.0
                        self.state.positions[symbol] = {
                            "quantity": -units,
                            "entry_price": price,
                            "highest_price": price,
                            "opened_at": now_iso(),
                            "trade_id": position.get("tradeID"),
                            "is_short": True,
                        }
                        self.state.active_symbol = symbol
                        self.state.entry_price = price
                        self.state.coin = -units
                        self.state.is_short = True
                        self.journal(symbol, "INFO", f"Synced OANDA SHORT position: {symbol} {units} @ {price}", price)
                    else:
                        # Check if it's a long position
                        units = int(position.get("long", {}).get("units", 0))
                        if units > 0:
                            price = float(position.get("long", {}).get("averagePrice", 0.0))
                            if price <= 0:
                                price = self.state.last_price or 0.0
                            self.state.positions[symbol] = {
                                "quantity": units,
                                "entry_price": price,
                                "highest_price": price,
                                "opened_at": now_iso(),
                                "trade_id": position.get("tradeID"),
                                "is_short": False,
                            }
                            self.state.active_symbol = symbol
                            self.state.entry_price = price
                            self.state.coin = units
                            self.state.is_short = False
                            self.journal(symbol, "INFO", f"Synced OANDA BUY position: {symbol} {units} @ {price}", price)

                # Remove positions that are no longer open on OANDA
                for symbol in list(self.state.positions.keys()):
                    if symbol not in synced_symbols:
                        self.journal(symbol, "INFO", f"Removing closed position: {symbol}", self.state.last_price)
                        del self.state.positions[symbol]

                if positions:
                    logger.info(f"Synced {len(positions)} positions from OANDA")
                    self.save_state()
                else:
                    # No positions - clear state
                    if self.state.positions:
                        self.state.positions = {}
                        self.state.active_symbol = None
                        self.state.entry_price = None
                        self.state.coin = 0.0
                        self.state.is_short = False
                        self.save_state()
                    logger.info("No open positions on OANDA")

        except Exception as exc:
            logger.warning(f"Failed to sync OANDA positions: {exc}")
            self.state.last_error = f"Position sync failed: {exc}"

    def build_ema_golden_cross_scan_rows(
        self,
        watchlist: list[str],
        candles_by_symbol: dict[str, list[Candle]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build scan rows using EMA Golden Cross strategy."""
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

            # Calculate EMAs
            ema_short_value = ema(history, ema_short)
            ema_long_value = ema(history, ema_long)
            ema_short_prev = ema(history[:-1], ema_short)
            ema_long_prev = ema(history[:-1], ema_long)

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

            # Determine signal
            signal = "HOLD"
            base_score = ((ema_short_value - ema_long_value) / ema_long_value) * 100

            # Golden Cross: short EMA crosses above long EMA
            if ema_short_prev <= ema_long_prev and ema_short_value > ema_long_value:
                signal = "BUY"
            # Death Cross: short EMA crosses below long EMA
            elif ema_short_prev >= ema_long_prev and ema_short_value < ema_long_value:
                signal = "SELL"
            elif ema_short_value > ema_long_value:
                signal = "WATCH uptrend"
            elif ema_short_value < ema_long_value:
                signal = "WATCH downtrend"

            # Check regimes and filters
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

    # ─── Opening Range Strategy ─────────────────────────────────────

    def calculate_atr(self, candles: list[Candle], period: int) -> float:
        """Calculate Average True Range."""
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

    def fetch_daily_opening_candle(self, symbol: str, candles: list[Candle]) -> dict[str, Any]:
        """Analyse the first candle of the day to determine bias."""
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

    def opening_range_signal(self, symbol: str, candles: list[Candle]) -> dict[str, Any]:
        """Generate BUY/SELL/HOLD signal based on the opening range strategy.
        Handles both bullish (BUY) and bearish (SELL) manipulation candles.
        """
        analysis = self.fetch_daily_opening_candle(symbol, candles)
        self.state.opening_range_analysis = analysis

        if analysis.get("bias") is None:
            return {"signal": "HOLD", "reason": "No opening candle found", "analysis": analysis}

        current_price = candles[-1].close if candles else 0
        trigger = analysis["trigger_level"]
        atr = analysis["atr"]

        stop_loss_mult = float(self.state.settings.get("opening_range_stop_loss_atr_multiplier", 1.5))
        take_profit_mult = float(self.state.settings.get("opening_range_take_profit_atr_multiplier", 2.5))

        # ─── Check if we have an open position for this symbol ───
        has_position = (
            (symbol in self.state.positions and abs(self.state.positions[symbol].get("quantity", 0)) > 0) or
            (self.state.active_symbol == symbol and abs(self.state.coin) > 0)
        )

        # ─── If we have a position, check for exit ───
        if has_position:
            position = self.state.positions.get(symbol, {})
            entry_price = float(position.get("entry_price") or self.state.entry_price or 0.0)
            is_short = position.get("is_short", False) or self.state.is_short
            quantity = float(position.get("quantity", 0)) or self.state.coin

            if entry_price > 0:
                if not is_short:
                    # ─── LONG POSITION – Check for SELL exit ───
                    stop_price = entry_price - (atr * stop_loss_mult)
                    target_price = entry_price + (atr * take_profit_mult)

                    # Update highest price for trailing stop
                    self.state.highest_price = max(self.state.highest_price or current_price, current_price)

                    # Check trailing stop
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
                    # ─── SHORT POSITION – Check for BUY exit ───
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

        # ─── No position – check for entry ───

        # ─── BULLISH MANIPULATION → BUY ───
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

        # ─── BEARISH MANIPULATION → SELL (Short) ───
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

        # ─── BLOW-OFF CANDLE ───
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

    # ─── WebSocket ──────────────────────────────────────────────────

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

    # ─── News Guard ─────────────────────────────────────────────────

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

    # ─── Journal & Audit ───────────────────────────────────────────

    def journal(
        self,
        symbol: str,
        event: str,
        message: str,
        price: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.state.journal.append(
            JournalEntry(
                time=now_iso(),
                symbol=symbol,
                event=event,
                message=message,
                price=price,
                details=details or {},
            )
        )
        self.state.journal = self.state.journal[-500:]
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

    # ─── Telegram Alerts ────────────────────────────────────────────

    def send_telegram_alert(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send a message via Telegram bot."""
        settings = self.state.settings

        # ─── Case-insensitive check ───
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
        """Format a trade alert message."""
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
        """Format an error alert message."""
        return f"""
⚠️ <b>AUXO ALERT</b>

<b>Error:</b> {error}
🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    def format_alert_drawdown(self, drawdown_pct: float, equity: float, peak_equity: float) -> str:
        """Format a drawdown alert message."""
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
        """Format a daily summary message."""
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
            self.save_state()
            logger.info("Bot state reset")

    # ─── Balance Sync ───────────────────────────────────────────────

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
            self.state.settings["starting_cash"] = available_cash
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
            current_coin = self.state.coin

        if settings.get("asset_class") != "forex" or settings.get("exchange") != "oanda_demo":
            raise RuntimeError("OANDA balance sync requires Asset Class = Forex and Exchange = OANDA demo.")
        if current_coin != 0 or self.state.positions:
            raise RuntimeError(
                "Refusing to sync OANDA paper balance while the bot has open paper/OANDA positions. "
                "Sell or reset first."
            )

        summary = oanda_account_summary()
        account = summary.get("account", {})
        balance = float(account.get("balance", 0.0))
        currency = str(account.get("currency") or settings.get("quote_currency", "USD")).upper()

        with self.lock:
            self.state.settings["starting_cash"] = balance
            self.state.settings["quote_currency"] = currency
            self.state.cash = balance
            self.state.coin = 0.0
            self.state.active_symbol = None
            self.state.entry_price = None
            self.state.active_stop_order_id = None
            self.state.day_start_equity = balance
            self.state.day_start_date = today_key()
            self.state.peak_equity = balance
            self.state.last_signal = f"Synced {currency} paper balance from OANDA demo"
            self.journal("", "INFO", self.state.last_signal, self.state.last_price)
            self.save_state()

        logger.info(f"Synced OANDA balance: {balance:.2f} {currency}")
        return {
            "ok": True,
            "quote_currency": currency,
            "available_cash": round(balance, 8),
        }

    # ─── Position Management ───────────────────────────────────────

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

        # Calculate P/L based on position direction
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

        # Close the position
        if is_short:
            # Closing a short = BUY
            self.paper_buy(symbol, price, reason, None, is_short=True)
        else:
            # Closing a long = SELL
            self.paper_sell(symbol, price, reason, None)

        with self.lock:
            self.state.last_signal = f"{reason}: {symbol}"
            self.save_state()

        logger.info(f"Manual close: {symbol} {mode} at {price:.6f}")
        return {
            "ok": True,
            "symbol": symbol,
            "mode": mode,
            "price": price,
            "estimated_pnl": pnl,
            "message": self.state.last_signal,
        }

    # ─── Settings Update ───────────────────────────────────────────

    def update_settings(self, updates: dict[str, Any]) -> None:
        numeric_fields = {
            "starting_cash",
            "trade_fee",
            "poll_seconds",
            "live_granularity",
            "live_candle_count",
            "short_window",
            "long_window",
            "base_nb_candles_buy",
            "base_nb_candles_sell",
            "low_offset",
            "low_offset_2",
            "high_offset",
            "high_offset_2",
            "ewo_high",
            "ewo_high_2",
            "ewo_low",
            "rsi_buy",
            "max_position_pct",
            "risk_per_trade_pct",
            "min_order_value",
            "stop_loss_pct",
            "take_profit_pct",
            "daily_loss_limit_pct",
            "cooldown_seconds",
            "sr_lookback_candles",
            "near_support_pct",
            "min_resistance_distance_pct",
            "min_sr_range_pct",
            "min_reward_risk",
            "support_stop_buffer_pct",
            "resistance_target_buffer_pct",
            "partial_take_profit_pct",
            "partial_take_profit_at_target_pct",
            "trailing_stop_pct",
            "trailing_activation_pct",
            "live_limit_offset_pct",
            "max_live_order_gbp",
            "max_daily_live_loss_gbp",
            "max_live_spread_pct",
            "min_live_quote_volume",
            "backtest_slippage_pct",
            "sr_zone_tolerance_pct",
            "sr_min_touches",
            "weak_pair_min_trades",
            "weak_pair_expectancy_limit_pct",
            "weak_pair_win_rate_limit_pct",
            "order_expiry_seconds",
            "order_retry_limit",
            "max_oanda_open_trades",
            "news_guard_before_minutes",
            "news_guard_after_minutes",
            "opening_range_minutes",
            "opening_range_atr_period",
            "opening_range_manipulation_threshold",
            "opening_range_stop_loss_atr_multiplier",
            "opening_range_take_profit_atr_multiplier",
            "max_drawdown_pct",
            "telegram_drawdown_alert_pct",
            "ema_short",
            "ema_long",
        }
        bool_fields = {
            "live_trading_enabled",
            "use_sr_filter",
            "use_dynamic_sr_exits",
            "partial_take_profit_enabled",
            "trailing_stop_enabled",
            "native_stop_enabled",
            "auto_disable_weak_pairs",
            "regime_filter_enabled",
            "allow_trending_regime",
            "allow_ranging_regime",
            "allow_volatile_regime",
            "allow_dead_regime",
            "order_replace_enabled",
            "websocket_enabled",
            "closed_candle_only",
            "oanda_demo_trading_enabled",
            "news_guard_enabled",
            "news_guard_block_high",
            "news_guard_block_medium",
            "news_guard_block_low",
            "telegram_enabled",
            "telegram_alert_on_buy",
            "telegram_alert_on_sell",
            "telegram_alert_on_error",
            "telegram_alert_on_daily_summary",
            "telegram_alert_on_drawdown",
            "allow_short_selling",
        }
        text_fields = {
            "asset_class",
            "exchange",
            "symbol",
            "quote_currency",
            "strategy",
            "position_sizing_mode",
            "live_order_type",
        }
        sensitive_fields = {
            "telegram_bot_token",
            "telegram_chat_id",
            "oanda_account_type",
        }
        lower_text_fields = {"chart_mode"}
        list_fields = {"watchlist"}

        with self.lock:
            for key, value in updates.items():
                if key in numeric_fields:
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

            self.state.settings["exchange"] = self.state.settings["exchange"].lower()
            self.state.settings["asset_class"] = self.state.settings.get("asset_class", "crypto").lower()
            if self.state.settings["asset_class"] not in {"crypto", "forex"}:
                self.state.settings["asset_class"] = "crypto"
            if self.state.settings["asset_class"] == "forex":
                if self.state.settings["exchange"] not in {"forex_demo", "oanda_demo"}:
                    self.state.settings["exchange"] = "forex_demo"
                self.state.settings["live_trading_enabled"] = False
                self.state.settings["websocket_enabled"] = False
                if self.state.settings["exchange"] != "oanda_demo":
                    self.state.settings["oanda_demo_trading_enabled"] = False
            self.state.settings["strategy"] = self.state.settings.get(
                "strategy", "opening_range"
            ).lower()
            if self.state.settings["strategy"] not in {"sma_cross", "ewo_offset", "opening_range", "ema_golden_cross"}:
                self.state.settings["strategy"] = "opening_range"
            self.state.settings["position_sizing_mode"] = self.state.settings.get(
                "position_sizing_mode", "balance_fraction"
            ).lower()
            if self.state.settings["position_sizing_mode"] not in {"balance_fraction", "risk_based"}:
                self.state.settings["position_sizing_mode"] = "balance_fraction"
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
            self.state.settings["short_window"] = max(
                2, int(self.state.settings["short_window"])
            )
            self.state.settings["long_window"] = max(
                self.state.settings["short_window"] + 1,
                int(self.state.settings["long_window"]),
            )
            self.state.settings["base_nb_candles_buy"] = max(
                2, int(self.state.settings["base_nb_candles_buy"])
            )
            self.state.settings["base_nb_candles_sell"] = max(
                2, int(self.state.settings["base_nb_candles_sell"])
            )
            self.state.settings["poll_seconds"] = max(
                5, int(self.state.settings["poll_seconds"])
            )
            self.state.settings["live_granularity"] = normalize_granularity(
                self.state.settings["live_granularity"]
            )
            # Allow lower candle counts for faster timeframes
            min_candles = strategy_minimum_candles(self.state.settings)
            # For 5m timeframe, we can use fewer candles
            if self.state.settings.get("live_granularity", 3600) <= 300:
                min_candles = min(min_candles, 80)  # Allow 80 candles for 5m
            self.state.settings["live_candle_count"] = max(
                min_candles,
                min(300, int(self.state.settings["live_candle_count"])),
            )
            self.state.settings["sr_lookback_candles"] = max(
                5, min(300, int(self.state.settings["sr_lookback_candles"]))
            )
            self.state.settings["near_support_pct"] = max(
                0.0, float(self.state.settings["near_support_pct"])
            )
            self.state.settings["min_resistance_distance_pct"] = max(
                0.0, float(self.state.settings["min_resistance_distance_pct"])
            )
            self.state.settings["risk_per_trade_pct"] = max(
                0.01, float(self.state.settings["risk_per_trade_pct"])
            )
            self.state.settings["min_order_value"] = max(
                0.0, float(self.state.settings["min_order_value"])
            )
            self.state.settings["min_sr_range_pct"] = max(
                0.0, float(self.state.settings["min_sr_range_pct"])
            )
            self.state.settings["min_reward_risk"] = max(
                0.0, float(self.state.settings["min_reward_risk"])
            )
            self.state.settings["support_stop_buffer_pct"] = max(
                0.0, float(self.state.settings["support_stop_buffer_pct"])
            )
            self.state.settings["resistance_target_buffer_pct"] = max(
                0.0, float(self.state.settings["resistance_target_buffer_pct"])
            )
            self.state.settings["partial_take_profit_pct"] = min(
                95.0, max(1.0, float(self.state.settings["partial_take_profit_pct"]))
            )
            self.state.settings["partial_take_profit_at_target_pct"] = min(
                99.0, max(1.0, float(self.state.settings["partial_take_profit_at_target_pct"]))
            )
            self.state.settings["trailing_stop_pct"] = max(
                0.1, float(self.state.settings["trailing_stop_pct"])
            )
            self.state.settings["trailing_activation_pct"] = max(
                0.0, float(self.state.settings["trailing_activation_pct"])
            )
            self.state.settings["max_live_order_gbp"] = max(
                1, float(self.state.settings["max_live_order_gbp"])
            )
            self.state.settings["max_daily_live_loss_gbp"] = max(
                1, float(self.state.settings["max_daily_live_loss_gbp"])
            )
            self.state.settings["max_live_spread_pct"] = max(
                0.01, float(self.state.settings["max_live_spread_pct"])
            )
            self.state.settings["min_live_quote_volume"] = max(
                0.0, float(self.state.settings["min_live_quote_volume"])
            )
            self.state.settings["live_limit_offset_pct"] = max(
                0.0, float(self.state.settings["live_limit_offset_pct"])
            )
            self.state.settings["backtest_slippage_pct"] = max(
                0.0, float(self.state.settings["backtest_slippage_pct"])
            )
            self.state.settings["sr_zone_tolerance_pct"] = max(
                0.0, float(self.state.settings["sr_zone_tolerance_pct"])
            )
            self.state.settings["sr_min_touches"] = max(
                1, int(self.state.settings["sr_min_touches"])
            )
            self.state.settings["weak_pair_min_trades"] = max(
                1, int(self.state.settings["weak_pair_min_trades"])
            )
            self.state.settings["weak_pair_win_rate_limit_pct"] = min(
                100.0, max(0.0, float(self.state.settings["weak_pair_win_rate_limit_pct"]))
            )
            self.state.settings["order_expiry_seconds"] = max(
                30, int(self.state.settings["order_expiry_seconds"])
            )
            self.state.settings["order_retry_limit"] = max(
                0, int(self.state.settings["order_retry_limit"])
            )
            self.state.settings["max_oanda_open_trades"] = max(
                1, int(self.state.settings.get("max_oanda_open_trades", 3))
            )
            self.state.settings["opening_range_minutes"] = max(
                1, int(self.state.settings["opening_range_minutes"])
            )
            self.state.settings["opening_range_atr_period"] = max(
                2, int(self.state.settings["opening_range_atr_period"])
            )
            self.state.settings["opening_range_manipulation_threshold"] = max(
                0.01, min(1.0, float(self.state.settings["opening_range_manipulation_threshold"]))
            )
            self.state.settings["opening_range_stop_loss_atr_multiplier"] = max(
                0.1, float(self.state.settings["opening_range_stop_loss_atr_multiplier"])
            )
            self.state.settings["opening_range_take_profit_atr_multiplier"] = max(
                0.1, float(self.state.settings["opening_range_take_profit_atr_multiplier"])
            )
            self.state.settings["max_drawdown_pct"] = max(
                1.0, float(self.state.settings.get("max_drawdown_pct", 20.0))
            )
            self.save_state()
            logger.info("Settings updated")

    # ─── Snapshot ──────────────────────────────────────────────────

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
            equity = self.equity(price)
            day_pnl = equity - self.state.day_start_equity
            total_pnl = equity - float(self.state.settings["starting_cash"])
            granularity = int(self.state.settings.get("live_granularity", 3600))

            # ─── Account type label ───
            account_type = self.state.settings.get("oanda_account_type", "standard")
            if not account_type:
                account_type = "standard"
                self.state.settings["oanda_account_type"] = "standard"

            account_type_label = "Spread Bet" if account_type == "spreadbet" else "CFD/Forex"

            return {
                "running": self.state.running,
                "settings": self.state.settings,
                "cash": round(self.state.cash, 8),
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
                "equity": round(equity, 8),
                "total_pnl": round(total_pnl, 8),
                "total_pnl_pct": pct(total_pnl, float(self.state.settings["starting_cash"])),
                "day_pnl": round(day_pnl, 8),
                "day_pnl_pct": pct(day_pnl, self.state.day_start_equity),
                "last_signal": self.state.last_signal,
                "last_error": self.state.last_error,
                "price_count": len(chart_prices),
                "short_sma": sma(chart_prices, int(self.state.settings["short_window"])),
                "long_sma": sma(chart_prices, int(self.state.settings["long_window"])),
                "support": chart_row.get("support"),
                "resistance": chart_row.get("resistance"),
                "chart_levels": chart_levels,
                "chart_trades": [
                    asdict(item)
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
            }

    # ─── Main Loop ──────────────────────────────────────────────────

    def run_loop(self) -> None:
        """Main bot loop with auto-restart on failure."""
        last_summary_date = ""
        summary_sent_today = False

        while not self.stop_event.is_set() and not self.shutdown_requested:
            try:
                self.tick()

                # ─── Send daily summary at 23:59 UTC ───
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

    def tick(self) -> None:
        with self.lock:
            settings = dict(self.state.settings)

        watchlist = parse_watchlist(settings.get("watchlist", settings["symbol"]))

        fetched_prices: dict[str, float] = {}
        fetched_candles: dict[str, list[Candle]] = {}
        errors: list[str] = []
        granularity = int(settings.get("live_granularity", 3600))
        candle_count = int(settings.get("live_candle_count", 300))

        # ─── Process each pair with logging ───
        for idx, symbol in enumerate(watchlist):
            try:
                logger.debug(f"Fetching {symbol}...")
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
                logger.debug(f"✓ {symbol} fetched: {len(candles)} candles, price {fetched_prices[symbol]}")
            except Exception as exc:
                logger.warning(f"✗ {symbol} failed: {exc}")
                errors.append(f"{symbol}: {exc}")
                # ─── Use fallback data ───
                history = self.state.price_history.get(symbol, [])
                if history:
                    fallback_price = history[-1]
                    fallback_candles = closes_to_candles(history[-40:])
                    logger.debug(f"Using fallback data for {symbol}: {fallback_price}")
                else:
                    fallback_price = 1.0
                    fallback_candles = [
                        Candle(time=int(time.time()) - i, open=1.0, high=1.0, low=1.0, close=1.0, volume=0.0)
                        for i in range(40, 0, -1)
                    ]
                    logger.debug(f"Using default data for {symbol}")
                fetched_candles[symbol] = fallback_candles
                fetched_prices[symbol] = fallback_price

        if not fetched_prices:
            logger.error("; ".join(errors) or "No candle data returned")
            # Use cached data instead of failing
            for symbol in watchlist:
                if symbol in self.state.price_history:
                    history = self.state.price_history[symbol]
                    fetched_prices[symbol] = history[-1] if history else 1.0
                    fetched_candles[symbol] = closes_to_candles(history[-40:]) if history else []

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

            if self.should_live_trade():
                self.manage_open_orders()

            decision = self.decide(fetched_prices, watchlist, fetched_candles)
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
                    self.live_buy(symbol, fetched_prices[symbol], decision, candles)
                else:
                    self.paper_buy(symbol, fetched_prices[symbol], decision, candles, is_short=is_short)
            elif decision.startswith("SELL"):
                parts = decision.split()
                symbol = parts[1] if len(parts) > 1 else self.state.active_symbol or settings["symbol"]
                sell_quantity = None
                if " partial " in f" {decision} ":
                    if self.wants_oanda_demo_trade() and symbol in self.state.positions:
                        sell_quantity = float(self.state.positions[symbol].get("quantity", 0.0)) * (
                            float(settings.get("partial_take_profit_pct", 50.0)) / 100
                        )
                    else:
                        sell_quantity = self.state.coin * (
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

            self.save_state()
            logger.debug("TICK: State saved")

    def decide(
        self,
        fetched_prices: dict[str, float],
        watchlist: list[str],
        candles_by_symbol: dict[str, list[Candle]] | None = None,
    ) -> str:
        settings = self.state.settings
        strategy = settings.get("strategy", "opening_range")
        candles_by_symbol = candles_by_symbol or {}

        if time.time() - self.state.last_action_time < float(settings["cooldown_seconds"]):
            return "Cooldown active"

        # ─── NEWS GUARD CHECK ───
        if settings.get("news_guard_enabled", False):
            for symbol in watchlist:
                blocked, reason = self.is_news_blocked(symbol, settings)
                if blocked:
                    return f"BLOCK {symbol} {reason}"

        active_price = self.price_for_active_position(fetched_prices)
        equity = self.equity(active_price)

        # ─── DAILY LOSS CHECK ───
        daily_loss_limit = float(settings["daily_loss_limit_pct"]) / 100
        if equity <= self.state.day_start_equity * (1 - daily_loss_limit):
            return "Daily loss limit reached"

        # ─── MAX DRAWDOWN CHECK ───
        max_drawdown_pct = float(settings.get("max_drawdown_pct", 20.0))
        drawdown_alert_pct = float(settings.get("telegram_drawdown_alert_pct", 10.0))
        if self.state.peak_equity > 0:
            current_drawdown = ((self.state.peak_equity - equity) / self.state.peak_equity) * 100
            if current_drawdown > max_drawdown_pct:
                return f"STOP Max drawdown {current_drawdown:.1f}% > {max_drawdown_pct}%"
            if (current_drawdown > drawdown_alert_pct and
                settings.get("telegram_alert_on_drawdown", True)):
                self.send_telegram_alert(
                    self.format_alert_drawdown(current_drawdown, equity, self.state.peak_equity)
                )

        # Update peak equity
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity

        # ─── EMA GOLDEN CROSS STRATEGY ───
        if strategy == "ema_golden_cross":
            scan_rows = self.build_ema_golden_cross_scan_rows(watchlist, candles_by_symbol)

            # ─── Only keep rows for symbols in the watchlist ───
            watchlist_set = set(watchlist)
            filtered_rows = []
            for row in scan_rows:
                if row.get("symbol") in watchlist_set:
                    filtered_rows.append(row)
            self.state.scan_rows = filtered_rows

            # Check for BUY signals
            candidates = [
                row for row in self.state.scan_rows
                if row["signal"] == "BUY" and row["price"] is not None
            ]
            if candidates:
                best = max(candidates, key=lambda row: row["score"])
                return f"BUY {best['symbol']} EMA Golden Cross score {best['score']:.3f}"

            # Check for SELL signals (shorting)
            if settings.get("allow_short_selling", False):
                sell_candidates = [
                    row for row in self.state.scan_rows
                    if row["signal"] == "SELL" and row["price"] is not None
                ]
                if sell_candidates:
                    best = max(sell_candidates, key=lambda row: row["score"])
                    return f"BUY {best['symbol']} SHORT EMA Death Cross"

            waiting = [row for row in self.state.scan_rows if row["signal"].startswith("WAIT")]
            if len(waiting) == len(self.state.scan_rows):
                return "Waiting for enough price data"
            return "HOLD no qualifying entry"

        # ─── OPENING RANGE STRATEGY (Modified to loop through all watchlist) ───
        if strategy == "opening_range":
            scan_rows = []

            for symbol in watchlist:
                candles = candles_by_symbol.get(symbol, [])

                # Check if we have enough data
                if len(candles) < max(20, int(settings.get("opening_range_atr_period", 14)) + 1):
                    # Not enough data - add a placeholder row
                    scan_rows.append({
                        "symbol": symbol,
                        "price": fetched_prices.get(symbol),
                        "signal": "WAIT data",
                        "score": 0,
                        "regime": "opening_range",
                        "support": None,
                        "resistance": None,
                        "entry": None,
                        "reason": "Not enough data for Opening Range",
                        "opening_bias": None,
                        "opening_range_ratio": None,
                        "opening_manipulation": None,
                        "opening_blowoff": None,
                        "opening_high": None,
                        "opening_low": None,
                        "is_short": False,
                    })
                    continue

                # Get the signal for this symbol
                signal = self.opening_range_signal(symbol, candles)
                analysis = signal.get("analysis", {})

                # Build scan row for each symbol
                scan_row = {
                    "symbol": symbol,
                    "price": candles[-1].close if candles else None,
                    "signal": signal["signal"],
                    "score": 0,
                    "regime": "opening_range",
                    "support": signal.get("stop"),
                    "resistance": signal.get("target"),
                    "entry": signal.get("entry"),
                    "reason": signal["reason"],
                    "opening_bias": analysis.get("bias"),
                    "opening_range_ratio": analysis.get("range_ratio"),
                    "opening_manipulation": analysis.get("manipulation"),
                    "opening_blowoff": analysis.get("blowoff"),
                    "opening_high": analysis.get("high"),
                    "opening_low": analysis.get("low"),
                    "is_short": signal.get("is_short", False),
                }
                scan_rows.append(scan_row)

            # Update all scan rows
            self.state.scan_rows = scan_rows

            # ─── Find the best BUY signal (for trading) ───
            buy_candidates = [
                row for row in scan_rows
                if row["signal"] == "BUY" and row["price"] is not None
            ]

            # Also check for SELL signals (short entries) if shorting is enabled
            sell_candidates = []
            if settings.get("allow_short_selling", False):
                sell_candidates = [
                    row for row in scan_rows
                    if row["signal"] == "SELL" and row["price"] is not None
                ]

            # Check if we have an open position to manage
            if abs(self.state.coin) > 0 and self.state.entry_price and self.state.active_symbol:
                symbol = self.state.active_symbol
                price = fetched_prices.get(symbol, active_price)
                history = self.state.price_history.get(symbol, [])
                candles = candles_by_symbol.get(symbol) or closes_to_candles(history)
                signal_candle_set = signal_candles(candles, settings)
                signal_history = [candle.close for candle in signal_candle_set] or history
                self.state.highest_price = max(self.state.highest_price or price, price)
                stop_price, target_price, exit_mode = exit_prices(
                    entry_price=self.state.entry_price,
                    candles=candles,
                    settings=settings,
                )

                if partial_take_profit_ready(
                    price=price,
                    entry_price=self.state.entry_price,
                    target_price=target_price,
                    settings=settings,
                    already_done=self.state.partial_take_profit_done,
                ):
                    self.state.partial_take_profit_done = True
                    return f"SELL {symbol} partial {exit_mode} target"

                trailing_stop = trailing_stop_price(
                    entry_price=self.state.entry_price,
                    highest_price=self.state.highest_price,
                    settings=settings,
                )
                if trailing_stop and price <= trailing_stop:
                    return f"SELL {symbol} trailing stop"

                if price <= stop_price:
                    return f"SELL {symbol} {exit_mode} stop"
                if price >= target_price:
                    return f"SELL {symbol} {exit_mode} target"

                return f"HOLD {symbol} position open"

            # ─── Decide on the best trade to take ───
            # Prefer BUY signals over SELL signals for long-only mode
            if buy_candidates:
                best = max(buy_candidates, key=lambda row: row.get("score", 0))
                symbol = best["symbol"]
                is_short = best.get("is_short", False)
                if is_short:
                    return f"BUY {symbol} SHORT {best['reason']}"
                else:
                    return f"BUY {symbol} {best['reason']}"

            # Check for SELL signals (short entries)
            if sell_candidates and settings.get("allow_short_selling", False):
                best = max(sell_candidates, key=lambda row: row.get("score", 0))
                return f"BUY {best['symbol']} SHORT {best['reason']}"

            # Check if we have any WAIT signals
            waiting = [row for row in scan_rows if row["signal"].startswith("WAIT")]
            if waiting:
                return f"HOLD {waiting[0]['reason']}"

            return "HOLD no qualifying entry"

        # ─── EXISTING STRATEGIES (SMA Cross, EWO Offset) ───
        signal_candles_by_symbol = {
            symbol: signal_candles(candles, settings)
            for symbol, candles in candles_by_symbol.items()
        }

        if self.wants_oanda_demo_trade():
            return self.decide_oanda_multi(
                fetched_prices,
                watchlist,
                candles_by_symbol,
                signal_candles_by_symbol,
                self.state.scan_rows,
                active_price,
            )

        if abs(self.state.coin) > 0 and self.state.entry_price and self.state.active_symbol:
            symbol = self.state.active_symbol
            price = fetched_prices.get(symbol, active_price)
            history = self.state.price_history.get(symbol, [])
            candles = candles_by_symbol.get(symbol) or closes_to_candles(history)
            signal_candle_set = signal_candles_by_symbol.get(symbol) or signal_candles(closes_to_candles(history), settings)
            signal_history = [candle.close for candle in signal_candle_set] or history
            self.state.highest_price = max(self.state.highest_price or price, price)
            stop_price, target_price, exit_mode = exit_prices(
                entry_price=self.state.entry_price,
                candles=candles,
                settings=settings,
            )
            if partial_take_profit_ready(
                price=price,
                entry_price=self.state.entry_price,
                target_price=target_price,
                settings=settings,
                already_done=self.state.partial_take_profit_done,
            ):
                self.state.partial_take_profit_done = True
                return f"SELL {symbol} partial {exit_mode} target"

            trailing_stop = trailing_stop_price(
                entry_price=self.state.entry_price,
                highest_price=self.state.highest_price,
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

        # ─── Build scan rows for SMA Cross or EWO Offset ───
        if strategy == "ewo_offset":
            scan_rows = self.build_ewo_scan_rows(watchlist, signal_candles_by_symbol)
        else:
            scan_rows = self.build_scan_rows(watchlist, signal_candles_by_symbol)

        # ─── Only keep rows for symbols in the watchlist ───
        watchlist_set = set(watchlist)
        filtered_rows = []
        for row in scan_rows:
            if row.get("symbol") in watchlist_set:
                filtered_rows.append(row)
            else:
                logger.debug(f"Removing {row.get('symbol')} from scan rows (not in watchlist)")
        self.state.scan_rows = filtered_rows

        candidates = [
            row for row in self.state.scan_rows
            if row["signal"] == "BUY" and row["price"] is not None
        ]
        if candidates:
            best = max(candidates, key=lambda row: row["score"])
            return f"BUY {best['symbol']} strongest trend score {best['score']:.3f}"

        waiting = [row for row in self.state.scan_rows if row["signal"].startswith("WAIT")]
        if len(waiting) == len(self.state.scan_rows):
            return "Waiting for enough price data"

        return "HOLD no qualifying entry"

    # ─── Strategy Helpers ──────────────────────────────────────────

    def decide_oanda_multi(
        self,
        fetched_prices: dict[str, float],
        watchlist: list[str],
        candles_by_symbol: dict[str, list[Candle]],
        signal_candles_by_symbol: dict[str, list[Candle]],
        scan_rows: list[dict[str, Any]],
        active_price: float,
    ) -> str:
        settings = self.state.settings

        for symbol, position in list(self.state.positions.items()):
            price = fetched_prices.get(symbol)
            if price is None:
                continue
            history = self.state.price_history.get(symbol, [])
            candles = candles_by_symbol.get(symbol) or closes_to_candles(history)
            signal_candle_set = signal_candles_by_symbol.get(symbol) or signal_candles(closes_to_candles(history), settings)
            signal_history = [candle.close for candle in signal_candle_set] or history
            entry_price = float(position.get("entry_price", 0.0))
            if entry_price <= 0:
                continue

            highest_price = max(float(position.get("highest_price", entry_price)), price)
            position["highest_price"] = highest_price
            self.state.positions[symbol] = position
            stop_price, target_price, exit_mode = exit_prices(
                entry_price=entry_price,
                candles=candles,
                settings=settings,
            )

            if partial_take_profit_ready(
                price=price,
                entry_price=entry_price,
                target_price=target_price,
                settings=settings,
                already_done=bool(position.get("partial_take_profit_done", False)),
            ):
                position["partial_take_profit_done"] = True
                self.state.positions[symbol] = position
                return f"SELL {symbol} partial {exit_mode} target"

            trailing_stop = trailing_stop_price(
                entry_price=entry_price,
                highest_price=highest_price,
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

        max_positions = int(settings.get("max_oanda_open_trades", 3))
        if len(self.state.positions) >= max_positions:
            return f"HOLD max OANDA trades open ({len(self.state.positions)}/{max_positions})"

        candidates = [
            row for row in scan_rows
            if row["signal"] == "BUY"
            and row["price"] is not None
            and row["symbol"] not in self.state.positions
        ]
        if candidates:
            best = max(candidates, key=lambda row: row["score"])
            return f"BUY {best['symbol']} strongest trend score {best['score']:.3f}"

        if self.state.positions:
            return f"HOLD {len(self.state.positions)} OANDA positions open"

        waiting = [row for row in scan_rows if row["signal"].startswith("WAIT")]
        if len(waiting) == len(scan_rows):
            return "Waiting for enough price data"
        return "HOLD no qualifying entry"

    def build_scan_rows(
        self,
        watchlist: list[str],
        candles_by_symbol: dict[str, list[Candle]] | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        candles_by_symbol = candles_by_symbol or {}

        # ─── EWO Offset Strategy ───
        if self.state.settings.get("strategy") == "ewo_offset":
            return self.build_ewo_scan_rows(watchlist, candles_by_symbol)

        # ─── EMA Golden Cross Strategy ───
        if self.state.settings.get("strategy") == "ema_golden_cross":
            return self.build_ema_golden_cross_scan_rows(watchlist, candles_by_symbol)

        # ─── SMA Cross Strategy ───
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

    def active_scan_row(self, symbol: str) -> dict[str, Any]:
        return next(
            (row for row in self.state.scan_rows if row.get("symbol") == symbol),
            {},
        )

    def record_setup_buy(
        self,
        symbol: str,
        price: float,
        quantity: float,
        entry_cost: float,
        entry_fee: float,
        reason: str,
    ) -> None:
        row = self.active_scan_row(symbol)
        setup_id = str(uuid.uuid4())
        record = SetupRecord(
            id=setup_id,
            time=now_iso(),
            symbol=symbol,
            strategy=str(self.state.settings.get("strategy", "sma_cross")),
            settings_key=setup_settings_key(self.state.settings),
            entry_price=price,
            entry_quantity=quantity,
            entry_cost=entry_cost,
            entry_fee=entry_fee,
            entry_reason=reason,
            entry_score=float(row.get("score") or 0.0),
            base_score=float(row.get("base_score") or row.get("score") or 0.0),
            edge_score=float(row.get("edge_score") or 0.0),
            regime=str(row.get("regime") or "unknown"),
            support_distance_pct=row.get("support_distance_pct"),
            resistance_distance_pct=row.get("resistance_distance_pct"),
            sr_range_pct=row.get("sr_range_pct"),
            reward_risk=row.get("reward_risk"),
        )
        self.state.setup_records.append(record)
        self.state.setup_records = self.state.setup_records[-500:]
        self.state.active_setup_id = setup_id
        self.state.active_setup_ids[symbol] = setup_id

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
            record.pnl_pct = pct(record.realized_pnl, record.entry_cost)
            self.state.active_setup_ids.pop(symbol, None)
            if self.state.active_setup_id == setup_id:
                self.state.active_setup_id = None

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
        """Place a BUY (long) or SELL (short) order."""
        settings = self.state.settings
        trade_fee = float(settings["trade_fee"])
        spend_reason = "manual override"

        if spend_override is not None:
            spend = spend_override
        else:
            spend, spend_reason = position_spend(
                cash=self.state.cash,
                entry_price=price,
                candles=candles or closes_to_candles(self.state.price_history.get(symbol, [])),
                settings=settings,
            )
        spend = min(spend, self.state.cash)

        if spend < float(settings.get("min_order_value", 1.0)):
            self.state.last_signal = f"{'SHORT' if is_short else 'BUY'} blocked: order below minimum {settings['quote_currency']} {settings.get('min_order_value', 1.0)}"
            self.journal(symbol, "BLOCK", self.state.last_signal, price, {"spend": spend})
            return

        fee_paid = fee_override if fee_override is not None else spend * trade_fee
        coin_bought = quantity_override if quantity_override is not None else (spend - fee_paid) / price

        # Calculate stop/target – use overrides if provided
        if stop_override is not None and target_override is not None:
            stop_price = stop_override
            target_price = target_override
            exit_mode = "opening_range"
        else:
            stop_price, target_price, exit_mode = exit_prices(
                entry_price=price,
                candles=candles or closes_to_candles(self.state.price_history.get(symbol, [])),
                settings=settings,
            )

        if is_short:
            # ─── SHORT SELL ───
            # For shorts, we use negative quantity
            self.state.coin -= coin_bought  # Negative = short position
            self.state.cash += spend  # We receive cash for shorting
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
                "quantity": -coin_bought,  # Negative = short
                "entry_price": price,
                "highest_price": price,
                "stop_price": stop_price,
                "target_price": target_price,
                "exit_mode": exit_mode,
                "partial_take_profit_done": False,
                "entry_cost": spend,
                "opened_at": now_iso(),
                "is_short": True,
            }

            self.state.trades.append(
                Trade(
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
                )
            )
            self.journal(symbol, "SHORT", reason, price, {"spend": spend, "quantity": coin_bought, "stop": stop_price, "target": target_price})
            logger.info(f"SHORT {symbol}: {coin_bought:.6f} @ {price:.6f} | {reason}")
        else:
            # ─── LONG BUY ───
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
            }

            self.state.trades.append(
                Trade(
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
                )
            )
            self.record_setup_buy(symbol, price, coin_bought, spend, fee_paid, reason)
            self.journal(symbol, "BUY", reason, price, {"spend": spend, "quantity": coin_bought, "stop": stop_price, "target": target_price})
            logger.info(f"BUY {symbol}: {coin_bought:.6f} @ {price:.6f} | {reason}")

        # Send Telegram alert
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
        settings = self.state.settings
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

        # Determine if we're closing a short or a long
        is_short = position.get("is_short", False) or self.state.is_short
        sold_quantity = min(abs(coin_available), abs(quantity_override or coin_available))
        gross = sold_quantity * price
        fee_paid = fee_override if fee_override is not None else gross * trade_fee
        cash_received = gross - fee_paid

        # Update cash and position
        self.state.cash += cash_received

        if is_short:
            # Closing a short – we buy back
            self.state.coin += sold_quantity  # Reduce the negative
            if symbol in self.state.positions:
                remaining = position_quantity + sold_quantity  # Moving toward zero
                if remaining >= 0:
                    self.state.positions.pop(symbol, None)
                else:
                    position["quantity"] = remaining
                    self.state.positions[symbol] = position
        else:
            # Closing a long – we sell
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

        self.state.last_price = price
        self.state.last_action_time = time.time()
        side = "SELL" if not is_short else "BUY"
        self.state.trades.append(
            Trade(
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
            )
        )
        self.record_setup_sell(
            symbol,
            price,
            sold_quantity,
            cash_received,
            fee_paid,
            reason,
            position_closed,
        )
        self.journal(symbol, side, reason, price, {"quantity": sold_quantity})
        logger.info(f"{side} {symbol}: {sold_quantity:.6f} @ {price:.6f} | {reason}")

        # Send Telegram alert
        if settings.get("telegram_alert_on_sell", True):
            trade = self.state.trades[-1]
            # Calculate P/L
            buy_trade = next(
                (t for t in reversed(self.state.trades[:-1]) if t.symbol == symbol and t.side in ["BUY", "SHORT"]),
                None
            )
            pnl = None
            if buy_trade:
                if is_short:
                    pnl = (price - buy_trade.price) * sold_quantity
                else:
                    pnl = (price - buy_trade.price) * sold_quantity
            self.send_telegram_alert(self.format_alert_trade(trade, pnl))

    # ─── OANDA Trading ─────────────────────────────────────────────

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

    def oanda_demo_buy(
        self,
        symbol: str,
        price: float,
        reason: str,
        candles: list[Candle] | None = None,
        is_short: bool = False,
    ) -> None:
        settings = self.state.settings
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

        spend, spend_reason = position_spend(
            cash=self.state.cash,
            entry_price=price,
            candles=candles or closes_to_candles(self.state.price_history.get(symbol, [])),
            settings=settings,
        )
        spend = min(spend, self.state.cash)
        if spend < float(settings.get("min_order_value", 1.0)):
            self.state.last_signal = f"OANDA {'SHORT' if is_short else 'BUY'} blocked: order below {settings['quote_currency']} {settings.get('min_order_value', 1.0)}"
            self.journal(symbol, "BLOCK", self.state.last_signal, price, {"spend": spend})
            return

        units = int(max(1, spend / price))
        if is_short:
            units = -units  # Negative units for short

        stop_price, target_price, exit_mode = exit_prices(
            entry_price=price,
            candles=candles or closes_to_candles(self.state.price_history.get(symbol, [])),
            settings=settings,
        )
        response = oanda_market_order(symbol, units, stop_price, target_price)
        fill = oanda_order_fill(response)
        fill_price = fill["price"] or price
        filled_units = fill["units"] or units
        cost = abs(filled_units * fill_price)
        fee = fill["commission"]

        if is_short:
            # Short position
            self.state.coin -= abs(filled_units)
            self.state.cash += cost
            self.state.is_short = True
            self.state.positions[symbol] = {
                "quantity": -abs(filled_units),
                "entry_price": fill_price,
                "highest_price": fill_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "stop": stop_price,
                "target": target_price,
                "exit_mode": exit_mode,
                "partial_take_profit_done": False,
                "entry_cost": cost,
                "opened_at": now_iso(),
                "trade_id": fill.get("trade_id"),
                "is_short": True,
            }
        else:
            # Long position
            self.state.cash -= cost + fee
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
                "entry_cost": cost + fee,
                "opened_at": now_iso(),
                "trade_id": fill.get("trade_id"),
                "is_short": False,
            }

        self.state.active_symbol = symbol
        self.state.entry_price = fill_price
        self.state.highest_price = fill_price
        self.state.stop_price = stop_price
        self.state.target_price = target_price
        self.state.last_price = fill_price
        self.state.last_action_time = time.time()

        side = "SHORT" if is_short else "BUY"
        trade_reason = f"{reason} | OANDA demo order | size {spend_reason} | {exit_mode} stop/target"
        self.state.trades.append(
            Trade(
                time=now_iso(),
                side=side,
                symbol=symbol,
                price=fill_price,
                quantity=abs(filled_units) if is_short else abs(filled_units),
                cash_after=self.state.cash,
                coin_after=self.state.coin,
                reason=trade_reason,
                fee_paid=fee,
                exchange_order_id=fill["order_id"],
                exchange_order_status=fill["status"],
                exchange_average_filled_price=fill_price,
                exchange_filled_size=abs(filled_units),
            )
        )
        self.record_setup_buy(symbol, fill_price, abs(filled_units), cost + fee, fee, trade_reason)
        self.journal(symbol, side, trade_reason, fill_price, {"spend": cost + fee, "quantity": abs(filled_units)})
        self.journal(symbol, "INFO", f"OANDA demo {side} filled", fill_price, fill)
        logger.info(f"OANDA {side} {symbol}: {abs(filled_units)} units @ {fill_price:.6f}")

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
        units = -int(max(1, round(quantity))) if not is_short else int(max(1, round(quantity)))

        response = oanda_market_order(symbol, units)
        fill = oanda_order_fill(response)
        fill_price = fill["price"] or price
        filled_units = min(abs(current_quantity), fill["units"] or abs(units))
        fee = fill["commission"]
        gross = filled_units * fill_price
        cash_received = gross - fee
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
        trade_reason = f"{reason} | OANDA demo order"
        self.state.trades.append(
            Trade(
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
            )
        )
        self.record_setup_sell(symbol, fill_price, filled_units, cash_received, fee, trade_reason, position_closed)
        self.journal(symbol, side, trade_reason, fill_price, {"quantity": filled_units})
        self.journal(symbol, "INFO", f"OANDA demo {side} filled", fill_price, fill)
        logger.info(f"OANDA {side} {symbol}: {filled_units} units @ {fill_price:.6f}")

    # ─── Live Trading ──────────────────────────────────────────────

    def should_live_trade(self) -> bool:
        settings = self.state.settings
        return (
            bool(settings.get("live_trading_enabled"))
            and settings.get("asset_class", "crypto") == "crypto"
            and settings.get("exchange") == "coinbase"
            and coinbase_live_is_armed()
        )

    def live_status(self) -> dict[str, Any]:
        account_type = self.state.settings.get("oanda_account_type", "standard")
        account_type_label = "Spread Bet" if account_type == "spreadbet" else "CFD/Forex"

        # ─── Check if we're in forex mode ───
        if self.state.settings.get("asset_class", "crypto") == "forex":
            exchange = self.state.settings.get("exchange")
            demo_orders_enabled = bool(self.state.settings.get("oanda_demo_trading_enabled"))
            demo_orders_armed = exchange == "oanda_demo" and demo_orders_enabled and oanda_demo_orders_armed()

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
                "websocket_enabled": False,
                "websocket_available": WEBSOCKET_AVAILABLE,
                "websocket_status": "forex paper only",
                "websocket_last_seen": "",
                "message": message,
                "news_guard_enabled": bool(self.state.settings.get("news_guard_enabled", False)),
                "news_guard_status": self.state.news_guard_status,
                "account_type": account_type,
                "account_type_label": account_type_label,
            }

        # ─── Crypto mode ───
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

    def live_buy(
        self,
        symbol: str,
        price: float,
        reason: str,
        candles: list[Candle] | None = None,
        is_short: bool = False,
    ) -> None:
        self.roll_live_daily_spend_if_needed()
        settings = self.state.settings
        max_order = float(settings["max_live_order_gbp"])
        max_daily = float(settings["max_daily_live_loss_gbp"])
        paper_spend, spend_reason = position_spend(
            cash=self.state.cash,
            entry_price=price,
            candles=candles or closes_to_candles(self.state.price_history.get(symbol, [])),
            settings=settings,
        )
        quote_size = round(min(max_order, paper_spend), 2)

        minimum_order = max(1.0, float(settings.get("min_order_value", 1.0)))
        if quote_size < minimum_order:
            self.state.last_signal = f"LIVE {'SHORT' if is_short else 'BUY'} blocked: order below {settings['quote_currency']} {minimum_order:.2f}"
            self.journal(symbol, "BLOCK", self.state.last_signal, price, {"quote_size": quote_size})
            return

        if self.state.live_daily_spend + quote_size > max_daily:
            self.state.last_signal = "LIVE BUY blocked: daily live cap reached"
            self.journal(symbol, "BLOCK", self.state.last_signal, price, {"quote_size": quote_size})
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
            self.state.last_signal = f"LIVE BUY blocked: {guard['reason']}"
            self.journal(symbol, "BLOCK", self.state.last_signal, price, guard)
            return

        gbp_available = coinbase_available_balance(settings["quote_currency"])
        if gbp_available < quote_size:
            self.state.last_signal = (
                f"LIVE BUY blocked: only {settings['quote_currency']} {gbp_available:.2f} available"
            )
            self.journal(symbol, "BLOCK", self.state.last_signal, price, {"available": gbp_available, "quote_size": quote_size})
            return

        product_id = f"{symbol}-{settings['quote_currency']}"
        order_type = str(settings.get("live_order_type", "market"))
        limit_offset = float(settings.get("live_limit_offset_pct", 0.05)) / 100
        limit_price = price * (1 + limit_offset)
        base_size = quote_size / limit_price if limit_price > 0 else 0.0
        stop_price, target_price, exit_mode = exit_prices(
            entry_price=price,
            candles=candles or closes_to_candles(self.state.price_history.get(symbol, [])),
            settings=settings,
        )

        if order_type in {"limit", "bracket", "native_stop_scaffold"}:
            order = coinbase_limit_order(
                product_id=product_id,
                side="BUY" if not is_short else "SELL",
                base_size=base_size,
                limit_price=limit_price,
            )
        else:
            order = coinbase_market_order(
                product_id=product_id,
                side="BUY" if not is_short else "SELL",
                quote_size=quote_size,
            )

        order_id = coinbase_order_id(order)
        managed = self.track_order(
            order_id,
            symbol,
            product_id,
            "BUY" if not is_short else "SELL",
            "ENTRY",
            order_type,
            price=limit_price if order_type != "market" else price,
            base_size=base_size if order_type != "market" else None,
            quote_size=quote_size,
            reason=f"{reason} | size {spend_reason}",
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
            self.state.last_signal = f"LIVE {'SHORT' if is_short else 'BUY'} pending/unfilled: {order_id}"
            self.journal(symbol, "INFO", self.state.last_signal, price, {"order": order, "fill": fill})

    def live_sell(
        self,
        symbol: str,
        price: float,
        reason: str,
        quantity_override: float | None = None,
        is_short: bool = False,
    ) -> None:
        settings = self.state.settings
        base_available = coinbase_available_balance(symbol)
        desired_size = quantity_override or self.state.coin
        base_size = min(base_available, desired_size)

        if base_size <= 0:
            self.state.last_signal = f"LIVE SELL blocked: no {symbol} balance available"
            self.journal(symbol, "BLOCK", self.state.last_signal, price)
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
                self.state.last_signal = f"LIVE SELL blocked: could not cancel native stop {stop_order_id}: {exc}"
                self.journal(symbol, "BLOCK", self.state.last_signal, price)
                return

        product_id = f"{symbol}-{settings['quote_currency']}"
        order_type = str(settings.get("live_order_type", "market"))
        if order_type == "limit":
            limit_offset = float(settings.get("live_limit_offset_pct", 0.05)) / 100
            order = coinbase_limit_order(
                product_id=product_id,
                side="SELL" if not is_short else "BUY",
                base_size=base_size,
                limit_price=price * (1 - limit_offset),
            )
        else:
            order = coinbase_market_order(
                product_id=product_id,
                side="SELL" if not is_short else "BUY",
                base_size=base_size,
            )
        order_id = coinbase_order_id(order)
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
            self.state.last_signal = f"LIVE SELL pending/unfilled: {order_id}"
            self.journal(symbol, "INFO", self.state.last_signal, price, {"order": order, "fill": fill})

    # ─── Equity & Helpers ──────────────────────────────────────────

    def equity(self, price: float | None) -> float:
        if self.state.positions:
            total = self.state.cash
            for symbol, position in self.state.positions.items():
                history = self.state.price_history.get(symbol, [])
                current_price = history[-1] if history else price or float(position.get("entry_price", 0.0))
                quantity = float(position.get("quantity", 0.0))
                total += quantity * current_price
            return total
        if not price:
            return self.state.cash
        return self.state.cash + (self.state.coin * price)

    def price_for_active_position(self, fetched_prices: dict[str, float]) -> float:
        if self.state.active_symbol and self.state.active_symbol in fetched_prices:
            return fetched_prices[self.state.active_symbol]
        first_price = next(iter(fetched_prices.values()))
        return first_price

    def roll_daily_equity_if_needed(self, price: float) -> None:
        current_day = today_key()
        if self.state.day_start_date != current_day:
            self.state.day_start_date = current_day
            self.state.day_start_equity = self.equity(price)
            self.state.peak_equity = self.state.day_start_equity

    def roll_live_daily_spend_if_needed(self) -> None:
        current_day = today_key()
        if self.state.live_day_start_date != current_day:
            self.state.live_day_start_date = current_day
            self.state.live_daily_spend = 0.0

    # ─── Order Management ──────────────────────────────────────────

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
            )
            new_order.retry_count = order.retry_count + 1
            self.audit("ORDER_REPLACED", old_order_id=order.order_id, new_order_id=replacement_id)
            logger.info(f"Order replaced: {order.order_id} → {replacement_id}")
        except Exception as exc:
            self.audit("ORDER_REPLACE_FAILED", order_id=order.order_id, error=str(exc))
            logger.warning(f"Order replace failed: {exc}")

    def submit_native_stop_for_position(self, entry_order: ManagedOrder, entry_price: float) -> None:
        if abs(self.state.coin) <= 0:
            return
        stop_price = float(entry_order.details.get("stop_price") or 0.0)
        exit_mode = str(entry_order.details.get("exit_mode") or "fixed")
        if stop_price <= 0:
            candles = closes_to_candles(self.state.price_history.get(entry_order.symbol, []))
            stop_price, _, exit_mode = exit_prices(entry_price, candles, self.state.settings)
        stop_order = coinbase_stop_limit_order(
            product_id=entry_order.product_id,
            side="SELL" if not self.state.is_short else "BUY",
            base_size=abs(self.state.coin),
            stop_price=stop_price,
            limit_price=stop_price * 0.995,
        )
        stop_order_id = coinbase_order_id(stop_order)
        self.state.active_stop_order_id = stop_order_id
        self.track_order(
            stop_order_id,
            entry_order.symbol,
            entry_order.product_id,
            "SELL" if not self.state.is_short else "BUY",
            "STOP",
            "stop_limit",
            price=stop_price,
            base_size=abs(self.state.coin),
            reason=f"{exit_mode} native stop",
            details={"entry_order_id": entry_order.order_id},
        )
        self.journal(
            entry_order.symbol,
            "INFO",
            f"Native stop-limit submitted {stop_order_id} via {exit_mode} stop",
            stop_price,
            {"entry_order_id": entry_order.order_id, "stop_order": stop_order},
        )
        logger.info(f"Native stop submitted: {stop_order_id} at {stop_price:.6f}")

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


# ─── Helper functions ────────────────────────────────────────────

def fetch_price(exchange: str, symbol: str, quote_currency: str) -> float:
    exchange = exchange.lower()
    symbol = symbol.upper()
    quote_currency = quote_currency.upper()

    if exchange == "coinbase":
        product = f"{symbol}-{quote_currency}"
        data = fetch_json(f"https://api.exchange.coinbase.com/products/{product}/ticker")
        return float(data["price"])

    if exchange == "kraken":
        kraken_symbol_map = {"BTC": "XBT", "DOGE": "XDG"}
        pair = f"{kraken_symbol_map.get(symbol, symbol)}{quote_currency}"
        data = fetch_json(f"https://api.kraken.com/0/public/Ticker?pair={pair}")
        if data.get("error"):
            raise RuntimeError("; ".join(data["error"]))
        result = next(iter(data["result"].values()))
        return float(result["c"][0])

    raise RuntimeError("Exchange must be coinbase or kraken")


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
            oanda_symbol = symbol.replace("_", "")  # Remove underscores
            return fetch_oanda_demo_candles(symbol, granularity, candle_count)
        return fetch_forex_demo_candles(symbol, granularity, candle_count)

    if exchange == "coinbase":
        return fetch_coinbase_candles(symbol, quote_currency, granularity, candle_count)

    if exchange == "kraken":
        return fetch_kraken_candles(symbol, quote_currency, granularity, candle_count)

    raise RuntimeError("Exchange must be coinbase or kraken")


def forex_pip_size(symbol: str) -> float:
    symbol = normalize_forex_symbol(symbol)
    return 0.01 if symbol.endswith("JPY") else 0.0001


def normalize_forex_symbol(symbol: str) -> str:
    """Convert OANDA instrument names to standard format."""
    # Remove underscores and slashes
    normalized = symbol.upper().replace("/", "").replace("-", "").replace("_", "").strip()
    return normalized


# ─── Simplified OANDA Configuration ──────────────────────────────

def oanda_account_id() -> str:
    """Get OANDA account ID from environment."""
    return os.environ.get("OANDA_ACCOUNT_ID", "").strip()


def oanda_api_token() -> str:
    """Get OANDA API token from environment."""
    return os.environ.get("OANDA_API_TOKEN", "").strip()


def oanda_is_configured() -> bool:
    """Check if OANDA credentials are configured."""
    return bool(oanda_account_id() and oanda_api_token())


def oanda_is_practice() -> bool:
    """Check if we're using the practice environment."""
    env = os.environ.get("OANDA_ENV", "practice").strip().lower()
    return env == "practice"


def oanda_api_base() -> str:
    """Get the OANDA API base URL."""
    # Check if explicitly set
    if os.environ.get("OANDA_API_BASE", "").strip():
        return os.environ["OANDA_API_BASE"].strip().rstrip("/")

    # Check the environment
    env = os.environ.get("OANDA_ENV", "practice").strip().lower()

    if env == "live":
        return "https://api-fxtrade.oanda.com"
    return "https://api-fxpractice.oanda.com"


def oanda_demo_orders_armed() -> bool:
    """
    Check if OANDA order placement is enabled.
    """
    # Check 1: Is the API configured?
    if not oanda_is_configured():
        return False

    # Check 2: Is order placement enabled?
    if os.environ.get("OANDA_DEMO_TRADING_ENABLED", "").strip().lower() != "true":
        return False

    return True


def oanda_demo_status_message() -> str:
    """Get a status message about OANDA configuration."""
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


def oanda_instrument(symbol: str) -> str:
    """Convert standard symbol to OANDA instrument format."""
    symbol = normalize_forex_symbol(symbol)
    if len(symbol) != 6:
        raise RuntimeError("OANDA forex pairs must be six-letter symbols like EURUSD")
    return f"{symbol[:3]}_{symbol[3:]}"


def oanda_symbol(instrument: str) -> str:
    return instrument.upper().replace("_", "")


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


def oanda_account_summary() -> dict[str, Any]:
    account_id = urllib.parse.quote(oanda_account_id())
    return oanda_request(f"/v3/accounts/{account_id}/summary")


def oanda_pricing(symbols: list[str]) -> dict[str, Any]:
    account_id = urllib.parse.quote(oanda_account_id())
    instruments = ",".join(oanda_instrument(symbol) for symbol in symbols)
    return oanda_request(
        f"/v3/accounts/{account_id}/pricing",
        {"instruments": instruments},
    )


def oanda_decimal(value: float, symbol: str) -> str:
    places = 3 if normalize_forex_symbol(symbol).endswith("JPY") else 5
    return f"{value:.{places}f}"


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


def oanda_order_fill(response: dict[str, Any]) -> dict[str, Any]:
    fill = response.get("orderFillTransaction")
    cancel = response.get("orderCancelTransaction") or response.get("orderCreateTransaction")
    if not fill:
        reason = cancel.get("reason") if isinstance(cancel, dict) else "not filled"
        raise RuntimeError(f"OANDA demo order was not filled: {reason}")

    units = abs(float(fill.get("units", 0.0)))
    price = float(fill.get("price", 0.0))
    pl = float(fill.get("pl", 0.0))
    financing = float(fill.get("financing", 0.0))
    commission = abs(float(fill.get("commission", 0.0)))
    trade_id = fill.get("tradeOpened", {}).get("tradeID") or fill.get("tradesClosed", [{}])[0].get("tradeID")
    return {
        "order_id": str(fill.get("id") or response.get("lastTransactionID") or ""),
        "trade_id": str(trade_id or ""),
        "status": "FILLED",
        "units": units,
        "price": price,
        "pl": pl,
        "financing": financing,
        "commission": commission,
        "raw": response,
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


def normalize_granularity(value: Any) -> int:
    granularity = int(value)
    allowed = [60, 300, 900, 3600, 21600, 86400]
    if granularity in allowed:
        return granularity
    return min(allowed, key=lambda item: abs(item - granularity))


def fetch_coinbase_candles(
    symbol: str,
    quote_currency: str,
    granularity: int,
    candle_count: int,
) -> list[Candle]:
    candle_count = max(20, min(300, int(candle_count)))
    end = datetime.now(timezone.utc)
    start = end - timedelta(seconds=granularity * candle_count)
    product = f"{symbol}-{quote_currency}"
    query = urllib.parse.urlencode({
        "granularity": int(granularity),
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": end.isoformat().replace("+00:00", "Z"),
    })
    data = fetch_json(f"https://api.exchange.coinbase.com/products/{product}/candles?{query}")
    candles = [
        Candle(
            time=int(item[0]),
            low=float(item[1]),
            high=float(item[2]),
            open=float(item[3]),
            close=float(item[4]),
            volume=float(item[5]),
        )
        for item in data
    ]
    return sorted(candles, key=lambda item: item.time)[-candle_count:]


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

        # ─── CLOSE EXISTING POSITION AT END OF DAY ───
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
            # ─── BULLISH MANIPULATION ───
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

                    # Check stop/target within the day
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

            # ─── BEARISH MANIPULATION (SHORT) ───
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

                    # Check stop/target within the day
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
            # Blow-off candles – wait for pullback
            if is_green and allow_short:
                # Bullish blow-off – wait for pullback to entry
                for candle in day_candles[1:]:
                    # Look for pullback to at least 50% of the move
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


def calculate_atr_from_candles(candles: list[Candle], period: int) -> float:
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

def run_ema_golden_cross_backtest(
    symbol: str,
    candles: list[Candle],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Backtest the EMA Golden Cross strategy."""
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

        ema_short_value = ema(closes, ema_short)
        ema_long_value = ema(closes, ema_long)
        ema_short_prev = ema(closes[:-1], ema_short)
        ema_long_prev = ema(closes[:-1], ema_long)

        if None in (ema_short_value, ema_long_value, ema_short_prev, ema_long_prev):
            continue

        can_trade = candle.time >= trade_start_time

        # Check for exit (Death Cross)
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

            # Also check stop loss and take profit
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

        # Check for entry (Golden Cross)
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


def optimizer_score(result: dict[str, Any]) -> float:
    return result["total_pnl_pct"] + (result["max_drawdown_pct"] * 0.75)


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


def run_walk_forward(settings: dict[str, Any]) -> dict[str, Any]:
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
            split_index = int(len(candles) * train_pct)
            train_candles = candles[:split_index]
            test_candles = candles[split_index:]

            if len(train_candles) < 61 or len(test_candles) < 20:
                raise RuntimeError("Not enough candles for train/test split")

            best_train: dict[str, Any] | None = None
            best_settings: dict[str, Any] | None = None

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


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def coinbase_private_key_configured() -> bool:
    if os.environ.get("COINBASE_API_PRIVATE_KEY", "").strip():
        return True
    key_file = os.environ.get("COINBASE_API_PRIVATE_KEY_FILE", "").strip()
    return bool(key_file and resolve_local_path(key_file).is_file())


def coinbase_private_key_source() -> str:
    if os.environ.get("COINBASE_API_PRIVATE_KEY", "").strip():
        return "COINBASE_API_PRIVATE_KEY"
    if os.environ.get("COINBASE_API_PRIVATE_KEY_FILE", "").strip():
        return "COINBASE_API_PRIVATE_KEY_FILE"
    return ""


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
        b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        + "."
        + b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
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


def coinbase_available_balance(currency: str) -> float:
    cursor = ""
    currency = currency.upper()

    while True:
        query = "?limit=250"
        if cursor:
            query += "&cursor=" + urllib.parse.quote(cursor)
        data = coinbase_api_request("GET", f"/api/v3/brokerage/accounts{query}")

        for account in data.get("accounts", []):
            if account.get("currency") == currency:
                balance = account.get("available_balance", {})
                return float(balance.get("value", 0.0))

        if not data.get("has_next"):
            return 0.0
        cursor = data.get("cursor", "")


def coinbase_auth_check() -> dict[str, Any]:
    data = coinbase_api_request("GET", "/api/v3/brokerage/accounts?limit=1")
    return {
        "ok": True,
        "accounts_visible": len(data.get("accounts", [])),
        "has_next": bool(data.get("has_next")),
    }


def diagnostics() -> dict[str, Any]:
    accounts_path = "/api/v3/brokerage/accounts"
    return {
        "ok": True,
        "server": "Auxo",
        "dotenv_file_present": ENV_FILE.exists(),
        "dotenv_loaded_keys": sorted(DOTENV_LOADED_KEYS),
        "audit_log_file": str(AUDIT_LOG_FILE),
        "audit_log_present": AUDIT_LOG_FILE.exists(),
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
        "stop_limit_stop_gtc": {
            "base_size": decimal_text(base_size, 10),
            "limit_price": decimal_text(limit_price, 8),
            "stop_price": decimal_text(stop_price, 8),
            "stop_direction": stop_direction,
        }
    }
    return coinbase_create_order(product_id, side, order_configuration)


def coinbase_bracket_order(
    product_id: str,
    side: str,
    base_size: float,
    limit_price: float,
    stop_trigger_price: float,
) -> dict[str, Any]:
    if base_size <= 0 or limit_price <= 0 or stop_trigger_price <= 0:
        raise RuntimeError("Bracket order requires positive size, limit, and stop trigger")
    order_configuration = {
        "trigger_bracket_gtc": {
            "base_size": decimal_text(base_size, 10),
            "limit_price": decimal_text(limit_price, 8),
            "stop_trigger_price": decimal_text(stop_trigger_price, 8),
        }
    }
    return coinbase_create_order(product_id, side, order_configuration)


def coinbase_create_order(
    product_id: str,
    side: str,
    order_configuration: dict[str, Any],
) -> dict[str, Any]:
    side = side.upper()
    if side not in {"BUY", "SELL"}:
        raise RuntimeError("Coinbase order side must be BUY or SELL")

    body = {
        "client_order_id": str(uuid.uuid4()),
        "product_id": product_id,
        "side": side,
        "order_configuration": order_configuration,
    }
    return coinbase_api_request("POST", "/api/v3/brokerage/orders", body)


def coinbase_get_order(order_id: str) -> dict[str, Any]:
    return coinbase_api_request("GET", f"/api/v3/brokerage/orders/historical/{urllib.parse.quote(order_id)}")


def coinbase_list_fills(order_id: str) -> dict[str, Any]:
    query = "?order_id=" + urllib.parse.quote(order_id)
    return coinbase_api_request("GET", f"/api/v3/brokerage/orders/historical/fills{query}")


def coinbase_cancel_orders(order_ids: list[str]) -> dict[str, Any]:
    return coinbase_api_request("POST", "/api/v3/brokerage/orders/batch_cancel", {"order_ids": order_ids})


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


def coinbase_order_id(response: dict[str, Any]) -> str:
    if response.get("order_id"):
        return str(response["order_id"])
    if response.get("success_response", {}).get("order_id"):
        return str(response["success_response"]["order_id"])
    if response.get("order", {}).get("order_id"):
        return str(response["order"]["order_id"])
    raise RuntimeError(f"Coinbase order response did not include an order id: {response}")


def decimal_text(value: float, places: int) -> str:
    return f"{value:.{places}f}".rstrip("0").rstrip(".")

def ema(values: list[float], window: int) -> float | None:
    """Calculate Exponential Moving Average."""
    if len(values) < window:
        return None
    multiplier = 2 / (window + 1)
    ema_value = sum(values[:window]) / window
    for price in values[window:]:
        ema_value = (price - ema_value) * multiplier + ema_value
    return ema_value


def ema_series(values: list[float], window: int) -> list[float | None]:
    """Calculate Exponential Moving Average for entire series."""
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


def sma(values: list[float], window: int) -> float | None:
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def ema_series(values: list[float], window: int) -> list[float | None]:
    if window <= 0:
        return [None for _ in values]
    result: list[float | None] = []
    multiplier = 2 / (window + 1)
    ema_value: float | None = None

    for index, value in enumerate(values):
        if index + 1 < window:
            result.append(None)
            continue
        if ema_value is None:
            ema_value = sum(values[index + 1 - window:index + 1]) / window
        else:
            ema_value = (value - ema_value) * multiplier + ema_value
        result.append(ema_value)

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


def closes_to_candles(closes: list[float]) -> list[Candle]:
    return [
        Candle(time=index, open=price, high=price, low=price, close=price, volume=1.0)
        for index, price in enumerate(closes)
    ]


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


def exit_prices(
    entry_price: float,
    candles: list[Candle],
    settings: dict[str, Any],
) -> tuple[float, float, str]:
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
        # Short position
        risk_per_unit = float(stop) - entry if stop else 0.0
        target_per_unit = entry - float(target) if target else 0.0
        current_per_unit = entry - float(price)
    else:
        # Long position
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
    rows: list[dict[str, Any]] = []
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

        rows.append({
            "symbol": symbol,
            "quantity": quantity,
            "entry_price": entry,
            "current_price": current,
            "highest_price": position.get("highest_price"),
            "stop_price": stop,
            "target_price": target,
            "stop": stop,
            "target": target,
            "unrealized_pnl": round(unrealized, 8),
            "unrealized_pnl_pct": pct(current - entry, entry) if entry else 0.0,
            "opened_at": position.get("opened_at"),
            "trade_id": position.get("trade_id"),
            "partial_take_profit_done": bool(position.get("partial_take_profit_done", False)),
            "is_short": is_short,
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
    return (
        f"sma {int(settings.get('short_window', 5))}/"
        f"{int(settings.get('long_window', 20))}"
    )


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
                # Closing a short
                pnl = (buy.price - trade.price) * abs(trade.quantity)
            else:
                # Closing a long
                pnl = (trade.price - buy.price) * abs(trade.quantity)
            stats[symbol]["closed_pnl"] += pnl
            if pnl >= 0:
                stats[symbol]["wins"] += 1
            else:
                stats[symbol]["losses"] += 1
        elif trade.side == "BUY" and trade.quantity < 0:
            # This is a short entry (recorded as BUY with negative quantity)
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


def parse_watchlist(value: str) -> list[str]:
    symbols: list[str] = []
    for item in value.replace("\n", ",").split(","):
        symbol = normalize_forex_symbol(item)
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def pct(value: float, base: float) -> float:
    if base == 0:
        return 0.0
    return round((value / base) * 100, 4)


def apply_slippage(price: float, side: str, slippage_fraction: float) -> float:
    if side.upper() == "BUY":
        return price * (1 + slippage_fraction)
    return price * (1 - slippage_fraction)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def today_key() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")


# ─── Currency mapping helpers for news guard ─────────────────────

def country_to_currency(country_code: str) -> str:
    mapping = {
        'USA': 'USD',
        'GBR': 'GBP',
        'JPN': 'JPY',
        'EUR': 'EUR',
        'AUS': 'AUD',
        'CAN': 'CAD',
        'CHE': 'CHF',
        'NZL': 'NZD',
        'CHN': 'CNY',
        'IND': 'INR',
        'BRA': 'BRL',
        'ZAF': 'ZAR',
        'RUS': 'RUB',
        'KOR': 'KRW',
        'MEX': 'MXN',
        'SGP': 'SGD',
        'HKG': 'HKD',
        'TWN': 'TWD',
        'IDN': 'IDR',
    }
    return mapping.get(country_code.upper(), country_code)


def symbol_to_currency(symbol: str, asset_class: str) -> str:
    symbol = symbol.upper()
    if asset_class == "forex":
        if len(symbol) == 6 and symbol.isalpha():
            return symbol[:3]
        return symbol
    return symbol


# ─── HTTP request handling ────────────────────────────────────────

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

    def do_GET(self) -> None:
        try:
            # ─── API endpoints ───
            if self.path == "/api/status":
                self.send_json(self.bot.snapshot())
                return

            if self.path == "/api/diagnostics":
                self.send_json(diagnostics())
                return

            if self.path == "/api/coinbase-auth-check":
                self.send_json(coinbase_auth_check())
                return

            if self.path == "/api/oanda-auth-check":
                self.send_json(oanda_auth_check())
                return

            if self.path == "/api/coinbase-gbp-products":
                self.send_json(coinbase_products_for_quote("GBP"))
                return

            if self.path == "/api/scan-rows-debug":
                self.send_json({
                    "scan_rows_count": len(self.bot.state.scan_rows),
                    "scan_rows": self.bot.state.scan_rows,
                    "watchlist": parse_watchlist(self.bot.state.settings.get("watchlist", ""))
                })
                return

            if self.path == "/api/oanda-test-pairs":
                settings = self.bot.state.settings
                watchlist = parse_watchlist(settings.get("watchlist", "EURUSD"))
                granularity = int(settings.get("live_granularity", 300))
                candle_count = 20  # Small count for testing

                results = []
                for symbol in watchlist:
                    try:
                        candles = fetch_candles(
                            exchange=settings["exchange"],
                            symbol=symbol,
                            quote_currency=settings["quote_currency"],
                            granularity=granularity,
                            candle_count=candle_count,
                            asset_class=str(settings.get("asset_class", "forex")),
                        )
                        results.append({
                            "symbol": symbol,
                            "success": True,
                            "count": len(candles),
                            "last_price": candles[-1].close if candles else None
                        })
                    except Exception as e:
                        results.append({
                            "symbol": symbol,
                            "success": False,
                            "error": str(e)[:100]
                        })

                self.send_json({
                    "total": len(watchlist),
                    "successful": len([r for r in results if r["success"]]),
                    "failed": len([r for r in results if not r["success"]]),
                    "results": results
                })
                return

            if self.path.startswith("/api/coinbase-products"):
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                quote = query.get("quote", ["GBP"])[0]
                self.send_json(coinbase_products_for_quote(quote))
                return

            # ─── Force price update (debug) ───
            if self.path == "/api/force-price":
                payload = parse_json_body(self)
                symbol = payload.get("symbol", "GBPCHF")
                price = payload.get("price", 1.08)
                with self.bot.lock:
                    if symbol in self.bot.state.price_history:
                        self.bot.state.price_history[symbol][-1] = price
                    else:
                        self.bot.state.price_history[symbol] = [price]
                    self.bot.state.last_price = price
                    self.bot.save_state()
                self.send_json({"ok": True, "symbol": symbol, "price": price})
                return

            if self.path == "/api/oanda-direct-test":
                import os
                account_id = os.environ.get("OANDA_ACCOUNT_ID", "").strip()
                api_token = os.environ.get("OANDA_API_TOKEN", "").strip()
                api_base = os.environ.get("OANDA_API_BASE", "https://api-fxtrade.oanda.com").strip()

                result = {
                    "account_id": account_id,
                    "api_token": api_token[:20] + "..." if api_token else "not set",
                    "api_base": api_base,
                    "env": os.environ.get("OANDA_ENV", "not set"),
                }

                try:
                    url = f"{api_base}/v3/accounts/{account_id}/summary"
                    req = urllib.request.Request(
                        url,
                        headers={
                            "Authorization": f"Bearer {api_token}",
                            "Accept": "application/json",
                        }
                    )
                    with urllib.request.urlopen(req, timeout=10) as response:
                        data = json.loads(response.read().decode())
                        result["success"] = True
                        result["balance"] = data.get("account", {}).get("balance")
                        result["currency"] = data.get("account", {}).get("currency")
                except urllib.error.HTTPError as e:
                    result["success"] = False
                    result["status"] = e.code
                    result["reason"] = e.reason
                    result["body"] = e.read().decode()
                except Exception as e:
                    result["success"] = False
                    result["error"] = str(e)

                self.send_json(result)
                return

            if self.path == "/api/oanda-instruments":
                try:
                    account_id = urllib.parse.quote(oanda_account_id())
                    data = oanda_request(f"/v3/accounts/{account_id}/instruments")
                    instruments = [inst.get("name") for inst in data.get("instruments", [])]
                    self.send_json({
                        "total": len(instruments),
                        "instruments": instruments[:50]  # Show first 50
                    })
                except Exception as e:
                    self.send_json({"ok": False, "error": str(e)})
                return

            if self.path == "/api/oanda-debug":
                import os
                account_id = oanda_account_id()
                api_token = oanda_api_token()
                api_base = oanda_api_base()

                self.send_json({
                    "account_id": account_id,
                    "api_token": api_token[:20] + "..." if api_token else "not set",
                    "api_base": api_base,
                    "is_configured": oanda_is_configured(),
                    "env_vars": {
                        "OANDA_ACCOUNT_ID": os.environ.get("OANDA_ACCOUNT_ID", "not set"),
                        "OANDA_API_TOKEN": os.environ.get("OANDA_API_TOKEN", "not set")[:20] + "..." if os.environ.get("OANDA_API_TOKEN") else "not set",
                        "OANDA_ENV": os.environ.get("OANDA_ENV", "not set"),
                        "OANDA_API_BASE": os.environ.get("OANDA_API_BASE", "not set"),
                        "OANDA_DEMO_TRADING_ENABLED": os.environ.get("OANDA_DEMO_TRADING_ENABLED", "not set"),
                    }
                })
                return

            # In BotRequestHandler.do_GET()
            if self.path == "/api/oanda-debug-fetch":
                import time
                settings = self.bot.state.settings
                watchlist = parse_watchlist(settings.get("watchlist", "EURUSD,GBPUSD,USDJPY"))
                granularity = int(settings.get("live_granularity", 300))
                candle_count = 20

                results = []
                for symbol in watchlist:
                    try:
                        start = time.time()
                        candles = fetch_candles(
                            exchange=settings["exchange"],
                            symbol=symbol,
                            quote_currency=settings["quote_currency"],
                            granularity=granularity,
                            candle_count=candle_count,
                            asset_class=str(settings.get("asset_class", "forex")),
                        )
                        elapsed = time.time() - start
                        results.append({
                            "symbol": symbol,
                            "success": True,
                            "count": len(candles),
                            " last_price": candles[-1].close if candles else None,
                            "elapsed": round(elapsed, 3)
                        })
                    except Exception as e:
                        results.append({
                            "symbol": symbol,
                            "success": False,
                            "error": str(e)[:200],
                            "error_type": type(e).__name__
                        })

                self.send_json({
                    "total": len(watchlist),
                    "successful": len([r for r in results if r["success"]]),
                    "failed": len([r for r in results if not r["success"]]),
                    "results": results
                })
                return

            # ─── Static files ───
            if self.path.endswith('.css') or self.path.endswith('.js') or self.path.endswith('.json') or self.path.endswith('.png') or self.path.endswith('.jpg') or self.path.endswith('.svg'):
                self.serve_static_file()
                return

            # ─── Default: serve index.html ───
            if not self.path.startswith("/api/"):
                self.send_index()
                return

            super().do_GET()
        except Exception as exc:
            logger.error(f"GET error: {exc}")
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def serve_static_file(self) -> None:
        """Serve static files with correct MIME type."""
        path = self.path.lstrip('/')
        file_path = WEB_DIR / path

        if not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        # Set correct MIME type
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

    def do_POST(self) -> None:
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

            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            logger.error(f"POST error: {exc}")
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def send_json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_index(self) -> None:
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

    def log_message(self, format: str, *args: Any) -> None:
        if self.path.startswith("/api/status"):
            return
        super().log_message(format, *args)


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
