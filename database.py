#!/usr/bin/env python3
"""
BotDatabase – SQLite storage for the Auxo trading bot.
Extracted from bot_server.py for better modularity.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auxo.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('auxo')

BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = BASE_DIR / "bot_state.json"
DB_FILE = BASE_DIR / "trades.db"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def today_key() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")


def pct(value: float, base: float) -> float:
    if base == 0:
        return 0.0
    return round((value / base) * 100, 4)


# ─── DATA CLASSES ──────────────────────────────────────────────────

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
    pnl: float = 0.0
    pnl_pct: float = 0.0
    entry_price: float | None = None
    exit_price: float | None = None
    trade_id: str | None = None
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    exit_mode: str | None = None
    exit_reason: str | None = None
    regime: str | None = None
    learning_context: dict[str, Any] = field(default_factory=dict)
    user_id: int = 1
    account_id: int = 1
    exchange: str | None = None
    engine_version: str | None = None


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
    signal_types: list[str] = field(default_factory=list)
    signal_scores: list[float] = field(default_factory=list)
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    exit_mode: str | None = None


@dataclass(frozen=True)
class Candle:
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


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
    client_order_id: str | None = None
    price: float | None = None
    base_size: float | None = None
    quote_size: float | None = None
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalHistory:
    signal_type: str
    total_signals: int = 0
    successful_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    last_updated: str = ""


# ─── MIGRATION FUNCTION ────────────────────────────────────────────

def migrate_to_database():
    """Migrate existing data from bot_state.json to the database."""
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

    # Migrate trades
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

    # Migrate journal
    for entry_data in data.get('journal', []):
        try:
            entry = JournalEntry(**entry_data)
            db.save_journal(entry)
        except Exception as e:
            logger.warning(f"Failed to migrate journal entry: {e}")

    # Migrate setup records
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

    # Migrate signal history
    for signal_type, history_data in data.get('signal_history', {}).items():
        try:
            db.save_signal_history(signal_type, history_data)
        except Exception as e:
            logger.warning(f"Failed to migrate signal history for {signal_type}: {e}")

    logger.info(f"Migration complete: {migrated} trades migrated")


# ─── DATABASE CLASS ──────────────────────────────────────────────────

class BotDatabase:
    """SQLite database for bot data storage with TP/SL tracking."""

    def __init__(self, db_path: Path = DB_FILE):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize database tables with TP/SL columns."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Multi-user foundation; user/account 1 preserve the current installation.
            cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE, display_name TEXT,
                password_hash TEXT, status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS trading_accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
                exchange TEXT NOT NULL DEFAULT 'coinbase', account_label TEXT NOT NULL DEFAULT 'Primary',
                encrypted_credentials TEXT, enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS user_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, account_id INTEGER NOT NULL,
                settings_json TEXT NOT NULL DEFAULT '{}', updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, account_id)
            )''')
            cursor.execute("INSERT OR IGNORE INTO users (id, display_name, status) VALUES (1, 'Auxo Owner', 'active')")
            cursor.execute("INSERT OR IGNORE INTO trading_accounts (id, user_id, exchange, account_label, enabled) VALUES (1, 1, 'coinbase', 'Primary Coinbase', 1)")
            cursor.execute("""CREATE TABLE IF NOT EXISTS auth_sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, session_token_hash TEXT UNIQUE NOT NULL, user_id INTEGER NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP, expires_at TEXT NOT NULL, last_seen_at TEXT DEFAULT CURRENT_TIMESTAMP, revoked_at TEXT)""")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_auth_sessions_user ON auth_sessions(user_id)")

            cursor.execute("""CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                account_id INTEGER,
                action TEXT NOT NULL,
                detail TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_user_time ON audit_log(user_id, created_at DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_account_time ON audit_log(account_id, created_at DESC)")

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE NOT NULL,
                    time TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    cash_after REAL NOT NULL,
                    coin_after REAL NOT NULL,
                    reason TEXT,
                    fee_paid REAL DEFAULT 0,
                    pnl REAL DEFAULT 0,
                    pnl_pct REAL DEFAULT 0,
                    entry_price REAL,
                    exit_price REAL,
                    exchange_order_id TEXT,
                    exchange_order_status TEXT,
                    stop_loss_price REAL,
                    take_profit_price REAL,
                    exit_mode TEXT,
                    exit_reason TEXT,
                    regime TEXT,
                    learning_context TEXT,
                    user_id INTEGER NOT NULL DEFAULT 1,
                    account_id INTEGER NOT NULL DEFAULT 1,
                    exchange TEXT,
                    engine_version TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute("PRAGMA table_info(trades)")
            columns = [col[1] for col in cursor.fetchall()]
            for col, col_type in [
                ('regime', 'TEXT'),
                ('exit_reason', 'TEXT'),
                ('stop_loss_price', 'REAL'),
                ('take_profit_price', 'REAL'),
                ('exit_mode', 'TEXT'),
                ('learning_context', 'TEXT'),
                ('user_id', 'INTEGER NOT NULL DEFAULT 1'),
                ('account_id', 'INTEGER NOT NULL DEFAULT 1'),
                ('exchange', 'TEXT'),
                ('engine_version', 'TEXT'),
            ]:
                if col not in columns:
                    cursor.execute(f'ALTER TABLE trades ADD COLUMN {col} {col_type}')
                    logger.info(f"Added {col} column to trades table")

            cursor.execute("UPDATE trades SET user_id=1 WHERE user_id IS NULL")
            cursor.execute("UPDATE trades SET account_id=1 WHERE account_id IS NULL")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_user_account_time ON trades(user_id, account_id, time DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_accounts_user ON trading_accounts(user_id)")

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    event TEXT NOT NULL,
                    message TEXT NOT NULL,
                    price REAL,
                    details_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS setup_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    setup_id TEXT UNIQUE NOT NULL,
                    time TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    settings_key TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_quantity REAL NOT NULL,
                    entry_cost REAL NOT NULL,
                    entry_fee REAL NOT NULL,
                    entry_reason TEXT,
                    entry_score REAL DEFAULT 0,
                    base_score REAL DEFAULT 0,
                    edge_score REAL DEFAULT 0,
                    regime TEXT,
                    support_distance_pct REAL,
                    resistance_distance_pct REAL,
                    sr_range_pct REAL,
                    reward_risk REAL,
                    status TEXT DEFAULT 'OPEN',
                    closed_quantity REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    exit_fees REAL DEFAULT 0,
                    exit_time TEXT,
                    exit_price REAL,
                    exit_reason TEXT,
                    pnl_pct REAL,
                    signal_types_json TEXT,
                    signal_scores_json TEXT,
                    stop_loss_price REAL,
                    take_profit_price REAL,
                    exit_mode TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_type TEXT UNIQUE NOT NULL,
                    total_signals INTEGER DEFAULT 0,
                    successful_trades INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    avg_pnl REAL DEFAULT 0,
                    weight REAL DEFAULT 0.5,
                    last_updated TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS symbol_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    avg_pnl REAL DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    best_trade REAL DEFAULT 0,
                    worst_trade REAL DEFAULT 0,
                    last_trade_time TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT UNIQUE NOT NULL,
                    start_equity REAL NOT NULL,
                    end_equity REAL NOT NULL,
                    daily_pnl REAL DEFAULT 0,
                    daily_pnl_pct REAL DEFAULT 0,
                    trades_count INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    max_drawdown_pct REAL DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    iteration INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    weight REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    total_signals INTEGER NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    win_rate REAL DEFAULT 0,
                    avg_win REAL DEFAULT 0,
                    avg_loss REAL DEFAULT 0,
                    profit_factor REAL DEFAULT 0,
                    kelly_value REAL DEFAULT 0,
                    sharpe_ratio REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    total_trades INTEGER DEFAULT 0,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol)
                )
            ''')

            conn.commit()
            logger.info("Database initialized successfully with enhanced tracking")

    # ─── Trade Methods ────────────────────────────────────────────────

    def get_user(self, user_id: int) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT id,email,display_name,password_hash,status FROM users WHERE id=?", (int(user_id),)).fetchone()
            return dict(row) if row else None

    def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT id,email,display_name,password_hash,status FROM users WHERE lower(email)=lower(?)", (email.strip(),)).fetchone()
            return dict(row) if row else None

    def owner_auth_configured(self) -> bool:
        u=self.get_user(1); return bool(u and u.get("email") and u.get("password_hash"))

    def configure_owner_auth(self, email: str, password_hash: str, display_name: str="Auxo Owner") -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE users SET email=?,password_hash=?,display_name=?,status='active',updated_at=CURRENT_TIMESTAMP WHERE id=1", (email.strip().lower(),password_hash,display_name.strip() or "Auxo Owner")); conn.commit()

    def create_auth_session(self, token_hash: str, user_id: int, expires_at: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO auth_sessions(session_token_hash,user_id,expires_at) VALUES (?,?,?)", (token_hash,int(user_id),expires_at)); conn.commit()

    def get_auth_session(self, token_hash: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory=sqlite3.Row
            row=conn.execute("SELECT s.id session_id,s.user_id,u.email,u.display_name,u.status FROM auth_sessions s JOIN users u ON u.id=s.user_id WHERE s.session_token_hash=? AND s.revoked_at IS NULL AND datetime(s.expires_at)>datetime('now') AND u.status='active'",(token_hash,)).fetchone()
            if not row: return None
            conn.execute("UPDATE auth_sessions SET last_seen_at=CURRENT_TIMESTAMP WHERE id=?",(row['session_id'],)); conn.commit(); return dict(row)

    def revoke_auth_session(self, token_hash: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE auth_sessions SET revoked_at=CURRENT_TIMESTAMP WHERE session_token_hash=?",(token_hash,)); conn.commit()

    def trading_accounts_for_user(self, user_id: int) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory=sqlite3.Row
            return [dict(r) for r in conn.execute(
                """SELECT a.id,a.user_id,a.exchange,a.account_label,a.enabled,a.created_at,a.updated_at,
                          COUNT(t.id) AS trade_rows
                   FROM trading_accounts a
                   LEFT JOIN trades t ON t.account_id=a.id AND t.user_id=a.user_id
                   WHERE a.user_id=?
                   GROUP BY a.id
                   ORDER BY a.id""",(int(user_id),)
            ).fetchall()]

    def create_trading_account(self, user_id:int, exchange:str, account_label:str) -> int:
        exchange=str(exchange or "").strip().lower()
        if exchange not in {"coinbase","kraken"}:
            raise ValueError("Exchange must be coinbase or kraken")
        with sqlite3.connect(self.db_path) as conn:
            if not conn.execute("SELECT 1 FROM users WHERE id=?",(int(user_id),)).fetchone():
                raise ValueError("User not found")
            cur=conn.execute(
                "INSERT INTO trading_accounts(user_id,exchange,account_label,enabled) VALUES (?,?,?,1)",
                (int(user_id),exchange,str(account_label or f"{exchange.title()} Paper").strip())
            )
            conn.commit()
            return int(cur.lastrowid)

    def update_trading_account(self, account_id:int, account_label:str, enabled:bool) -> None:
        account_id=int(account_id)
        if account_id == 1 and not enabled:
            raise ValueError("Primary owner account cannot be disabled")
        with sqlite3.connect(self.db_path) as conn:
            cur=conn.execute(
                "UPDATE trading_accounts SET account_label=?,enabled=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (str(account_label or "Trading Account").strip(),1 if enabled else 0,account_id)
            )
            if cur.rowcount != 1: raise ValueError("Trading account not found")
            conn.commit()

    def delete_trading_account(self, account_id:int) -> dict[str,Any]:
        account_id=int(account_id)
        if account_id == 1: raise ValueError("Primary owner account cannot be deleted")
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory=sqlite3.Row
            row=conn.execute("SELECT id,user_id FROM trading_accounts WHERE id=?",(account_id,)).fetchone()
            if not row: raise ValueError("Trading account not found")
            conn.execute("DELETE FROM user_settings WHERE account_id=?",(account_id,))
            conn.execute("DELETE FROM trading_accounts WHERE id=?",(account_id,))
            conn.commit()
            return dict(row)

    def write_audit(self, user_id:int|None, account_id:int|None, action:str, detail:Any=None) -> None:
        payload = json.dumps(detail, separators=(",",":"), default=str) if detail is not None else None
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO audit_log(user_id,account_id,action,detail) VALUES (?,?,?,?)",
                         (user_id,account_id,str(action),payload))
            conn.commit()

    def audit_entries(self, limit:int=200) -> list[dict[str,Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory=sqlite3.Row
            return [dict(r) for r in conn.execute(
                """SELECT l.id,l.user_id,l.account_id,l.action,l.detail,l.created_at,
                          u.email,u.display_name,a.account_label
                   FROM audit_log l
                   LEFT JOIN users u ON u.id=l.user_id
                   LEFT JOIN trading_accounts a ON a.id=l.account_id
                   ORDER BY l.id DESC LIMIT ?""",(max(1,min(int(limit),1000)),)
            ).fetchall()]

    # ─── D6 OWNER USER ADMINISTRATION ────────────────────────────────
    def admin_list_users(self) -> list[dict[str, Any]]:
        """Owner-facing user list with account and trade counts; never exposes password hashes."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT u.id, u.email, u.display_name, u.status, u.created_at, u.updated_at,
                       a.id AS account_id, a.exchange, a.account_label, a.enabled AS account_enabled,
                       COUNT(t.id) AS trade_rows
                FROM users u
                LEFT JOIN trading_accounts a ON a.id = (
                    SELECT a2.id FROM trading_accounts a2
                    WHERE a2.user_id=u.id ORDER BY a2.id LIMIT 1
                )
                LEFT JOIN trades t ON t.user_id=u.id AND (a.id IS NULL OR t.account_id=a.id)
                GROUP BY u.id, a.id
                ORDER BY u.id
            """).fetchall()
            return [dict(r) for r in rows]

    def admin_update_user(self, user_id: int, email: str, display_name: str, status: str) -> None:
        user_id = int(user_id)
        email = str(email or "").strip().lower()
        display_name = str(display_name or "").strip() or "Auxo User"
        status = str(status or "active").strip().lower()
        if status not in {"active", "disabled"}:
            raise ValueError("Status must be active or disabled")
        if user_id == 1 and status != "active":
            raise ValueError("The Auxo owner cannot be disabled")
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE users SET email=?,display_name=?,status=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (email, display_name, status, user_id),
            )
            if cur.rowcount != 1:
                raise ValueError("User not found")
            if status != "active":
                conn.execute(
                    "UPDATE auth_sessions SET revoked_at=CURRENT_TIMESTAMP WHERE user_id=? AND revoked_at IS NULL",
                    (user_id,),
                )
            conn.commit()

    def admin_set_password(self, user_id: int, password_hash: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE users SET password_hash=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (password_hash, int(user_id)),
            )
            if cur.rowcount != 1:
                raise ValueError("User not found")
            # Force all existing browser sessions for that user to log in again.
            conn.execute(
                "UPDATE auth_sessions SET revoked_at=CURRENT_TIMESTAMP WHERE user_id=? AND revoked_at IS NULL",
                (int(user_id),),
            )
            conn.commit()

    def admin_delete_user(self, user_id: int) -> dict[str, Any]:
        """Delete a non-owner login/account configuration while retaining historical trades."""
        user_id = int(user_id)
        if user_id == 1:
            raise ValueError("The Auxo owner cannot be deleted")
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            accounts = [int(r["id"]) for r in conn.execute(
                "SELECT id FROM trading_accounts WHERE user_id=? ORDER BY id", (user_id,)
            ).fetchall()]
            exists = conn.execute("SELECT 1 FROM users WHERE id=?", (user_id,)).fetchone()
            if not exists:
                raise ValueError("User not found")
            conn.execute("UPDATE auth_sessions SET revoked_at=CURRENT_TIMESTAMP WHERE user_id=? AND revoked_at IS NULL", (user_id,))
            conn.execute("DELETE FROM auth_sessions WHERE user_id=?", (user_id,))
            conn.execute("DELETE FROM user_settings WHERE user_id=?", (user_id,))
            conn.execute("DELETE FROM trading_accounts WHERE user_id=?", (user_id,))
            conn.execute("DELETE FROM users WHERE id=?", (user_id,))
            conn.commit()
        return {"account_ids": accounts}

    # ─── MILESTONE C: USER / ACCOUNT ISOLATION ───────────────────────

    def get_trading_account(self, account_id: int) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT id,user_id,exchange,account_label,enabled,created_at,updated_at "
                "FROM trading_accounts WHERE id=?",
                (int(account_id),),
            ).fetchone()
            return dict(row) if row else None

    def account_belongs_to_user(self, user_id: int, account_id: int) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM trading_accounts WHERE id=? AND user_id=? AND enabled=1",
                (int(account_id), int(user_id)),
            ).fetchone()
            return bool(row)

    def default_account_for_user(self, user_id: int) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT id,user_id,exchange,account_label,enabled,created_at,updated_at "
                "FROM trading_accounts WHERE user_id=? AND enabled=1 ORDER BY id LIMIT 1",
                (int(user_id),),
            ).fetchone()
            return dict(row) if row else None

    def get_user_settings(self, user_id: int, account_id: int) -> dict[str, Any]:
        if not self.account_belongs_to_user(user_id, account_id):
            raise PermissionError("Trading account does not belong to authenticated user")
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT settings_json FROM user_settings WHERE user_id=? AND account_id=?",
                (int(user_id), int(account_id)),
            ).fetchone()
        if not row or not row[0]:
            return {}
        try:
            value = json.loads(row[0])
            return value if isinstance(value, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}

    def save_user_settings(self, user_id: int, account_id: int, settings: dict[str, Any]) -> None:
        if not self.account_belongs_to_user(user_id, account_id):
            raise PermissionError("Trading account does not belong to authenticated user")
        payload = json.dumps(settings or {}, separators=(",", ":"), default=str)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO user_settings(user_id,account_id,settings_json,updated_at)
                   VALUES (?,?,?,CURRENT_TIMESTAMP)
                   ON CONFLICT(user_id,account_id) DO UPDATE SET
                     settings_json=excluded.settings_json,
                     updated_at=CURRENT_TIMESTAMP""",
                (int(user_id), int(account_id), payload),
            )
            conn.commit()

    def ensure_user_settings(self, user_id: int, account_id: int, settings: dict[str, Any]) -> None:
        if not self.account_belongs_to_user(user_id, account_id):
            raise PermissionError("Trading account does not belong to authenticated user")
        with sqlite3.connect(self.db_path) as conn:
            exists = conn.execute(
                "SELECT 1 FROM user_settings WHERE user_id=? AND account_id=?",
                (int(user_id), int(account_id)),
            ).fetchone()
        if not exists:
            self.save_user_settings(user_id, account_id, settings)

    def create_user_with_account(self, email: str, display_name: str, password_hash: str, exchange: str = "kraken", account_label: str = "Kraken Paper") -> dict[str, int]:
        """Create an isolated Auxo user and first trading account atomically."""
        email = str(email or "").strip().lower()
        exchange = str(exchange or "kraken").strip().lower()
        if exchange not in {"coinbase", "kraken"}:
            raise ValueError("Exchange must be coinbase or kraken")
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO users(email,display_name,password_hash,status) VALUES (?,?,?,'active')",
                        (email, str(display_name or "Auxo User").strip(), password_hash))
            user_id = int(cur.lastrowid)
            cur.execute("INSERT INTO trading_accounts(user_id,exchange,account_label,enabled) VALUES (?,?,?,1)",
                        (user_id, exchange, str(account_label or "Kraken Paper").strip()))
            account_id = int(cur.lastrowid)
            conn.commit()
        return {"user_id": user_id, "account_id": account_id}

    def save_trade(self, trade) -> int:
        trade_id = getattr(trade, 'trade_id', None) or str(uuid.uuid4())

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO trades (
                    trade_id, time, symbol, side, price, quantity,
                    cash_after, coin_after, reason, fee_paid,
                    pnl, pnl_pct, entry_price, exit_price,
                    exchange_order_id, exchange_order_status,
                    stop_loss_price, take_profit_price, exit_mode, exit_reason, regime, learning_context,
                    user_id, account_id, exchange, engine_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_id,
                trade.time,
                trade.symbol,
                trade.side,
                trade.price,
                trade.quantity,
                trade.cash_after,
                trade.coin_after,
                trade.reason,
                trade.fee_paid,
                getattr(trade, 'pnl', 0.0),
                getattr(trade, 'pnl_pct', 0.0),
                getattr(trade, 'entry_price', None),
                getattr(trade, 'exit_price', None),
                getattr(trade, 'exchange_order_id', None),
                getattr(trade, 'exchange_order_status', None),
                getattr(trade, 'stop_loss_price', None),
                getattr(trade, 'take_profit_price', None),
                getattr(trade, 'exit_mode', None),
                getattr(trade, 'exit_reason', None),
                getattr(trade, 'regime', None),
                json.dumps(getattr(trade, 'learning_context', {}) or {}, separators=(',', ':'), default=str),
                int(getattr(trade, 'user_id', 1) or 1),
                int(getattr(trade, 'account_id', 1) or 1),
                getattr(trade, 'exchange', None),
                getattr(trade, 'engine_version', None),
            ))

            return cursor.lastrowid

    def get_trades(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
        user_id: Optional[int] = None,
        account_id: Optional[int] = None,
    ) -> list[dict]:
        """Read trades, optionally scoped to an authenticated user/account.

        Existing engine callers that omit user_id/account_id retain their current
        behaviour. HTTP/user-facing callers should always supply both.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(int(user_id))
        if account_id is not None:
            clauses.append("account_id = ?")
            params.append(int(account_id))

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(int(limit))

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = [
                dict(row)
                for row in conn.execute(
                    f"SELECT * FROM trades{where} ORDER BY time DESC LIMIT ?",
                    tuple(params),
                ).fetchall()
            ]

        for item in rows:
            raw_context = item.get("learning_context")
            if isinstance(raw_context, str) and raw_context:
                try:
                    item["learning_context"] = json.loads(raw_context)
                except (TypeError, ValueError, json.JSONDecodeError):
                    item["learning_context"] = {}
            elif not isinstance(raw_context, dict):
                item["learning_context"] = {}
        return rows

    def get_trade_stats(
        self,
        symbol: Optional[str] = None,
        user_id: Optional[int] = None,
        account_id: Optional[int] = None,
        exchange: Optional[str] = None,
    ) -> dict:
        clauses = []
        params: list[Any] = []
        if symbol:
            clauses.append("symbol = ?"); params.append(symbol)
        if user_id is not None:
            clauses.append("user_id = ?"); params.append(int(user_id))
        if account_id is not None:
            clauses.append("account_id = ?"); params.append(int(account_id))
        if exchange:
            clauses.append("LOWER(COALESCE(exchange, '')) = ?"); params.append(str(exchange).lower())

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"""
            SELECT COUNT(*),
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END),
                   SUM(pnl), AVG(pnl),
                   AVG(CASE WHEN pnl > 0 THEN pnl ELSE NULL END),
                   AVG(CASE WHEN pnl <= 0 THEN pnl ELSE NULL END)
            FROM trades {where}
        """
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(sql, params).fetchone()
        if not result:
            return {}
        total=result[0] or 0; wins=result[1] or 0; losses=result[2] or 0
        avg_win=result[5] or 0.0; avg_loss=result[6] or 0.0
        return {
            "total_trades": total, "winning_trades": wins, "losing_trades": losses,
            "total_pnl": result[3] or 0.0, "avg_pnl": result[4] or 0.0,
            "avg_win": avg_win, "avg_loss": avg_loss,
            "win_rate": (wins/total*100) if total else 0.0,
            "profit_factor": abs(avg_win/avg_loss) if avg_loss != 0 else 0.0,
        }

    def update_performance_metrics(self, symbol: Optional[str] = None) -> None:
        stats = self.get_trade_stats(symbol)

        if stats.get('total_trades', 0) < 20:
            return

        win_rate = stats.get('win_rate', 0) / 100
        avg_win = stats.get('avg_win', 0)
        avg_loss = stats.get('avg_loss', 0)

        if avg_loss == 0:
            return

        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        kelly = win_rate - ((1 - win_rate) / profit_factor) if profit_factor > 0 else 0

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO performance_metrics (
                    symbol, win_rate, avg_win, avg_loss, profit_factor, kelly_value,
                    total_trades, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                symbol or 'ALL',
                stats.get('win_rate', 0),
                avg_win,
                avg_loss,
                profit_factor,
                max(0, kelly),
                stats.get('total_trades', 0),
            ))
            conn.commit()

    def get_kelly_metrics(self, symbol: Optional[str] = None) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM performance_metrics
                WHERE symbol = ? OR (symbol = 'ALL' AND ? IS NULL)
                ORDER BY updated_at DESC
                LIMIT 1
            ''', (symbol or 'ALL', symbol))

            result = cursor.fetchone()
            if result:
                return dict(result)
            return {}

    def backfill_tpsl_from_positions(self, state) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for symbol, position in state.positions.items():
                stop_price = position.get('stop_price')
                target_price = position.get('target_price')
                exit_mode = position.get('exit_mode')

                if stop_price or target_price:
                    cursor.execute('''
                        UPDATE trades
                        SET stop_loss_price = ?,
                            take_profit_price = ?,
                            exit_mode = ?
                        WHERE symbol = ?
                        AND exit_price IS NULL
                        AND side IN ('BUY', 'SHORT')
                    ''', (stop_price, target_price, exit_mode, symbol))

            conn.commit()
            logger.info("Backfilled TP/SL from positions")

    # ─── Journal Methods ─────────────────────────────────────────────

    def save_journal(self, entry) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO journal (time, symbol, event, message, price, details_json)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                entry.time,
                entry.symbol,
                entry.event,
                entry.message,
                entry.price,
                json.dumps(entry.details) if entry.details else None
            ))
            return cursor.lastrowid

    def get_journal(self, limit: int = 100, event_type: Optional[str] = None) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if event_type:
                cursor.execute('''
                    SELECT * FROM journal
                    WHERE event = ?
                    ORDER BY time DESC
                    LIMIT ?
                ''', (event_type, limit))
            else:
                cursor.execute('''
                    SELECT * FROM journal
                    ORDER BY time DESC
                    LIMIT ?
                ''', (limit,))

            rows = cursor.fetchall()
            result = []
            for row in rows:
                row_dict = dict(row)
                if row_dict.get('details_json'):
                    row_dict['details'] = json.loads(row_dict['details_json'])
                result.append(row_dict)
            return result

    # ─── Setup Record Methods ────────────────────────────────────────

    def save_setup_record(self, record) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO setup_records (
                    setup_id, time, symbol, strategy, settings_key,
                    entry_price, entry_quantity, entry_cost, entry_fee,
                    entry_reason, entry_score, base_score, edge_score,
                    regime, support_distance_pct, resistance_distance_pct,
                    sr_range_pct, reward_risk, status, closed_quantity,
                    realized_pnl, exit_fees, exit_time, exit_price,
                    exit_reason, pnl_pct, signal_types_json, signal_scores_json,
                    stop_loss_price, take_profit_price, exit_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.id,
                record.time,
                record.symbol,
                record.strategy,
                record.settings_key,
                record.entry_price,
                record.entry_quantity,
                record.entry_cost,
                record.entry_fee,
                record.entry_reason,
                record.entry_score,
                record.base_score,
                record.edge_score,
                record.regime,
                record.support_distance_pct,
                record.resistance_distance_pct,
                record.sr_range_pct,
                record.reward_risk,
                record.status,
                record.closed_quantity,
                record.realized_pnl,
                record.exit_fees,
                record.exit_time,
                record.exit_price,
                record.exit_reason,
                record.pnl_pct,
                json.dumps(record.signal_types) if record.signal_types else None,
                json.dumps(record.signal_scores) if record.signal_scores else None,
                getattr(record, 'stop_loss_price', None),
                getattr(record, 'take_profit_price', None),
                getattr(record, 'exit_mode', None),
            ))
            return cursor.lastrowid

    def get_setup_records(self, symbol: Optional[str] = None, limit: int = 100) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if symbol:
                cursor.execute('''
                    SELECT * FROM setup_records
                    WHERE symbol = ?
                    ORDER BY time DESC
                    LIMIT ?
                ''', (symbol, limit))
            else:
                cursor.execute('''
                    SELECT * FROM setup_records
                    ORDER BY time DESC
                    LIMIT ?
                ''', (limit,))

            rows = cursor.fetchall()
            result = []
            for row in rows:
                row_dict = dict(row)
                if row_dict.get('signal_types_json'):
                    row_dict['signal_types'] = json.loads(row_dict['signal_types_json'])
                if row_dict.get('signal_scores_json'):
                    row_dict['signal_scores'] = json.loads(row_dict['signal_scores_json'])
                result.append(row_dict)
            return result

    def get_setup_performance(self, symbol: Optional[str] = None) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if symbol:
                cursor.execute('''
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN status = 'CLOSED' THEN 1 ELSE 0 END) as closed,
                        SUM(CASE WHEN status = 'CLOSED' AND realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN status = 'CLOSED' AND realized_pnl <= 0 THEN 1 ELSE 0 END) as losses,
                        SUM(realized_pnl) as total_pnl,
                        AVG(CASE WHEN status = 'CLOSED' THEN pnl_pct ELSE NULL END) as avg_pnl_pct,
                        AVG(CASE WHEN status = 'CLOSED' AND realized_pnl > 0 THEN pnl_pct ELSE NULL END) as avg_win_pct,
                        AVG(CASE WHEN status = 'CLOSED' AND realized_pnl <= 0 THEN pnl_pct ELSE NULL END) as avg_loss_pct
                    FROM setup_records
                    WHERE symbol = ?
                ''', (symbol,))
            else:
                cursor.execute('''
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN status = 'CLOSED' THEN 1 ELSE 0 END) as closed,
                        SUM(CASE WHEN status = 'CLOSED' AND realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN status = 'CLOSED' AND realized_pnl <= 0 THEN 1 ELSE 0 END) as losses,
                        SUM(realized_pnl) as total_pnl,
                        AVG(CASE WHEN status = 'CLOSED' THEN pnl_pct ELSE NULL END) as avg_pnl_pct,
                        AVG(CASE WHEN status = 'CLOSED' AND realized_pnl > 0 THEN pnl_pct ELSE NULL END) as avg_win_pct,
                        AVG(CASE WHEN status = 'CLOSED' AND realized_pnl <= 0 THEN pnl_pct ELSE NULL END) as avg_loss_pct
                    FROM setup_records
                ''')

            row = cursor.fetchone()
            if row:
                closed = row[1] or 0
                wins = row[2] or 0
                losses = row[3] or 0
                return {
                    'total_setups': row[0] or 0,
                    'closed_setups': closed,
                    'winning_setups': wins,
                    'losing_setups': losses,
                    'total_pnl': row[4] or 0.0,
                    'avg_pnl_pct': row[5] or 0.0,
                    'avg_win_pct': row[6] or 0.0,
                    'avg_loss_pct': row[7] or 0.0,
                    'win_rate': (wins / closed * 100) if closed > 0 else 0.0,
                }
            return {}

    # ─── Signal History Methods ──────────────────────────────────────

    def save_signal_history(self, signal_type: str, history_data: dict) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO signal_history (
                    signal_type, total_signals, successful_trades, total_pnl,
                    win_rate, avg_pnl, weight, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_type,
                history_data.get('total_signals', 0),
                history_data.get('successful_trades', 0),
                history_data.get('total_pnl', 0.0),
                history_data.get('win_rate', 0.0),
                history_data.get('avg_pnl', 0.0),
                history_data.get('weight', 0.5),
                history_data.get('last_updated', now_iso()),
            ))

    def get_signal_history(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM signal_history')

            result = {}
            for row in cursor.fetchall():
                result[row['signal_type']] = dict(row)
            return result

    # ─── Symbol Performance Methods ──────────────────────────────────

    def update_symbol_performance(self, symbol: str) -> None:
        stats = self.get_trade_stats(symbol)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO symbol_performance (
                    symbol, total_trades, winning_trades, losing_trades,
                    total_pnl, avg_pnl, win_rate, last_trade_time, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                symbol,
                stats.get('total_trades', 0),
                stats.get('winning_trades', 0),
                stats.get('losing_trades', 0),
                stats.get('total_pnl', 0.0),
                stats.get('avg_pnl', 0.0),
                stats.get('win_rate', 0.0),
                now_iso(),
            ))

    def get_symbol_performance(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM symbol_performance ORDER BY total_pnl DESC')
            return [dict(row) for row in cursor.fetchall()]

    # ─── Daily Performance Methods ───────────────────────────────────

    def save_daily_performance(self, date: str, data: dict) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO daily_performance (
                    date, start_equity, end_equity, daily_pnl, daily_pnl_pct,
                    trades_count, winning_trades, losing_trades, max_drawdown_pct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                date,
                data.get('start_equity', 0.0),
                data.get('end_equity', 0.0),
                data.get('daily_pnl', 0.0),
                data.get('daily_pnl_pct', 0.0),
                data.get('trades_count', 0),
                data.get('winning_trades', 0),
                data.get('losing_trades', 0),
                data.get('max_drawdown_pct', 0.0),
            ))

    def get_daily_performance(self, days: int = 30) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM daily_performance
                ORDER BY date DESC
                LIMIT ?
            ''', (days,))
            return [dict(row) for row in cursor.fetchall()]

    # ─── Learning History Methods ────────────────────────────────────

    def save_learning_history(self, iteration: int, signal_type: str, weight: float, win_rate: float, total_signals: int) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO learning_history (
                    iteration, timestamp, signal_type, weight, win_rate, total_signals
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                iteration,
                now_iso(),
                signal_type,
                weight,
                win_rate,
                total_signals,
            ))

    def get_learning_history(self, signal_type: Optional[str] = None, limit: int = 100) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if signal_type:
                cursor.execute('''
                    SELECT * FROM learning_history
                    WHERE signal_type = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (signal_type, limit))
            else:
                cursor.execute('''
                    SELECT * FROM learning_history
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit,))

            return [dict(row) for row in cursor.fetchall()]

    # ─── Cleanup Methods ─────────────────────────────────────────────

    def cleanup_old_data(self, days: int = 30) -> None:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM journal WHERE time < ?', (cutoff,))
            cursor.execute('DELETE FROM learning_history WHERE timestamp < ?', (cutoff,))
            conn.commit()

    def vacuum(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('VACUUM')

    def export_json(self, path: Path) -> None:
        data = {
            'trades': self.get_trades(limit=999999),
            'journal': self.get_journal(limit=999999),
            'setup_records': self.get_setup_records(limit=999999),
            'signal_history': self.get_signal_history(),
            'symbol_performance': self.get_symbol_performance(),
            'performance_metrics': self.get_kelly_metrics(),
        }
        path.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')
        logger.info(f"Exported {len(data['trades'])} trades to {path}")

    def import_json(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding='utf-8'))

        for trade_data in data.get('trades', []):
            trade = Trade(**trade_data)
            self.save_trade(trade)

        for entry_data in data.get('journal', []):
            entry = JournalEntry(**entry_data)
            self.save_journal(entry)

        for record_data in data.get('setup_records', []):
            record = SetupRecord(**record_data)
            self.save_setup_record(record)

        for signal_type, history_data in data.get('signal_history', {}).items():
            self.save_signal_history(signal_type, history_data)

        logger.info(f"Imported {len(data.get('trades', []))} trades from {path}")
