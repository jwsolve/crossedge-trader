# bot_state.py
"""
BotState dataclass for the Auxo trading bot.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from database import Trade, JournalEntry, SetupRecord, ManagedOrder
from regime_detector import RegimeResult
from constants import DEFAULT_SETTINGS   # <--- ADD THIS

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
