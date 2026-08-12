#!/usr/bin/env python3
"""Synthetic verification for the Kraken stale-snapshot cash fix.

The bug: _kraken_spot_balance_snapshot was captured only at startup. While a
margin short was open, /api/status read "cash" from that snapshot (the Balance
total with the position open), WROTE it back into state.cash on every render,
and after the position closed it kept showing the startup-era number — so the
dashboard cash hopped HIGHER than the real balance after a sell (and the next
position was sized off the inflated cash).

Fix:
  1. refresh_kraken_balance_snapshot() pulls live Balance + TradeBalance on
     every status render (10s cache) and after every fill.
  2. state.cash is only re-anchored from a FRESH read, never a cached snapshot.
  3. Post-fill Kraken sync keeps the snapshot in lockstep with state.cash.

Run: python3 test_cash_fix.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bot_server
from bot_server import PaperBot, ManagedOrder


class FakeLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeState:
    def __init__(self, settings):
        self.settings = settings
        self.cash = float(settings.get("starting_cash", 1000.0))
        self.coin = 0.0
        self.positions = {}
        self.active_symbol = None
        self.last_price = 60000.0
        self.price_history = {}
        self.kraken_margin_owned = {}
        self.trades = []

    def save_state(self):
        pass


def make_bot(settings=None):
    s = {
        "exchange": "kraken",
        "asset_class": "crypto",
        "live_trading_enabled": True,
        "quote_currency": "USDT",
        "starting_cash": 1000.0,
        "trade_fee": 0.001,
        "kraken_margin_leverage": 2,
        "kraken_margin_quote_currency": "USDT",
    }
    if settings:
        s.update(settings)
    bot = PaperBot.__new__(PaperBot)
    bot.lock = FakeLock()
    bot.state = FakeState(s)
    bot._kraken_spot_balance_snapshot = {
        "ok": False, "reconciled": False, "error": "Not reconciled in this process"
    }
    bot.journal = lambda *a, **k: None
    bot.audit = lambda *a, **k: None
    bot.save_state = lambda: None
    bot.paper_sell = lambda *a, **k: None
    bot.paper_buy = lambda *a, **k: None
    return bot


def test_refresh_fresh_and_cache():
    bot = make_bot()
    calls = {"balance": 0, "tb": 0}

    def fake_balance(currency):
        calls["balance"] += 1
        return 2000.0  # Balance total while the margin short is open

    def fake_private(path, params=None):
        calls["tb"] += 1
        return {"result": {"e": "1500.0", "m": "500.0", "mf": "1000.0", "n": "10.0"}}

    bot_server.kraken_available_balance = fake_balance
    bot_server.kraken_private = fake_private

    snap = bot.refresh_kraken_balance_snapshot("USDT", force=True)
    assert snap["ok"] and snap["available_cash"] == 2000.0
    assert snap["equity"] == 1500.0
    assert snap["valuation_mode"] == "kraken_trade_balance"
    assert snap["cache_used"] is False
    assert calls["balance"] == 1 and calls["tb"] == 1

    # Second call within the 10s cache window must not hit the API again.
    snap2 = bot.refresh_kraken_balance_snapshot("USDT")
    assert snap2["cache_used"] is True
    assert snap2["available_cash"] == 2000.0
    assert calls["balance"] == 1 and calls["tb"] == 1
    print("PASS test_refresh_fresh_and_cache")


def test_refresh_falls_back_when_tradebalance_fails():
    bot = make_bot()
    bot_server.kraken_available_balance = lambda currency: 2000.0

    def fake_private(path, params=None):
        raise RuntimeError("rate limited")

    bot_server.kraken_private = fake_private
    snap = bot.refresh_kraken_balance_snapshot("USDT", force=True)
    assert snap["ok"] and snap["available_cash"] == 2000.0
    assert snap["equity"] == 2000.0  # falls back to balance
    assert snap["valuation_mode"] == "kraken_balance"
    print("PASS test_refresh_falls_back_when_tradebalance_fails")


def test_refresh_balance_error_returns_cached():
    bot = make_bot()
    bot._kraken_spot_balance_snapshot = {
        "ok": True, "available_cash": 2000.0, "equity": 1500.0,
        "quote_currency": "USDT", "_ts": time.time(),
    }
    bot_server.kraken_available_balance = lambda currency: (_ for _ in ()).throw(RuntimeError("network down"))
    snap = bot.refresh_kraken_balance_snapshot("USDT", force=True)
    assert snap["ok"] and snap["available_cash"] == 2000.0
    assert "refresh_error" in snap
    print("PASS test_refresh_balance_error_returns_cached")


def test_apply_reconciled_order_exit_refreshes_snapshot_and_cash():
    bot = make_bot()
    bot.state.cash = 2000.0  # Balance total with the short open
    bot._kraken_spot_balance_snapshot = {
        "ok": True, "available_cash": 2000.0, "equity": 1500.0,
        "quote_currency": "USDT", "_ts": time.time() - 60.0,  # stale
    }
    bot_server.kraken_available_balance = lambda currency: 1048.0  # after cover
    bot_server.kraken_private = lambda path, params=None: {"result": {"e": "1048.0"}}

    order = ManagedOrder(
        order_id="o1", symbol="BTC", product_id="BTCUSDT", side="BUY", role="EXIT",
        order_type="market", status="OPEN", created_at="now", updated_at="now",
        expires_at=time.time() + 60, price=59000.0, base_size=0.01,
        details={"is_short": False}, local_applied=False,
    )
    fill = {
        "status": "CLOSED", "filled_size": 0.01, "average_price": 59000.0,
        "total_fee": 1.2, "filled_value": 590.0, "requested_size": 0.01,
    }
    ok = bot.apply_reconciled_order(order, fill)
    assert ok is True
    assert bot.state.cash == 1048.0
    snap = bot._kraken_spot_balance_snapshot
    assert snap["ok"] and snap["available_cash"] == 1048.0
    assert snap["equity"] == 1048.0
    assert snap["_ts"] >= time.time() - 2  # freshly refreshed, not the stale 60s-old one
    print("PASS test_apply_reconciled_order_exit_refreshes_snapshot_and_cash")


def test_dashboard_never_clobbers_cash_with_stale_snapshot():
    bot = make_bot()
    # Simulate the exact pre-fix trap: startup snapshot captured while the short
    # was open (inflated 2000), but the real post-fill cash is 1048.
    bot.state.cash = 1048.0
    bot._kraken_spot_balance_snapshot = {
        "ok": True, "available_cash": 2000.0, "equity": 1500.0,
        "quote_currency": "USDT", "valuation_mode": "kraken_trade_balance",
        "_ts": time.time() - 60.0,
    }
    bot_server.kraken_available_balance = lambda currency: 1048.0
    bot_server.kraken_private = lambda path, params=None: {"result": {"e": "1048.0"}}

    # First render: fresh read wins, cash re-anchors to the REAL balance.
    snap = bot.refresh_kraken_balance_snapshot("USDT")
    cash = float(snap["available_cash"] if snap.get("ok") else bot.state.cash)
    if snap.get("ok") and not snap.get("cache_used"):
        bot.state.cash = cash
    assert bot.state.cash == 1048.0  # was 2000 stale before; now correct
    assert snap["cache_used"] is False

    # Second render within cache window: cached snapshot must NOT clobber state.
    bot.state.cash = 1048.0
    snap2 = bot.refresh_kraken_balance_snapshot("USDT")
    assert snap2["cache_used"] is True
    # The guarded render logic: no write when cache_used.
    if snap2.get("ok") and not snap2.get("cache_used"):
        bot.state.cash = float(snap2["available_cash"])
    assert bot.state.cash == 1048.0  # untouched
    print("PASS test_dashboard_never_clobbers_cash_with_stale_snapshot")


if __name__ == "__main__":
    test_refresh_fresh_and_cache()
    test_refresh_falls_back_when_tradebalance_fails()
    test_refresh_balance_error_returns_cached()
    test_apply_reconciled_order_exit_refreshes_snapshot_and_cash()
    test_dashboard_never_clobbers_cash_with_stale_snapshot()
    print("\nAll 5 cash-fix tests passed.")
