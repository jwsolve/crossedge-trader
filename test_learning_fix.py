#!/usr/bin/env python3
"""Synthetic verification for the expectancy weights + champion/challenger gate.

Run: python3 test_learning_fix.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

# Test against the real modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

import database as db_mod

# ─── 1. Expectancy weights ──────────────────────────────────────────
from bot_server import SelfLearningTrader

class FakeState:
    learning_iterations = 0
    last_learning_update = ""
    trades = []
    candle_history = {}
    last_price = None
    def save_state(self): pass

class FakeBot:
    def __init__(self):
        self.state = FakeState()
    def save_state(self):
        pass

class FakeDB:
    def __init__(self):
        self.history = {}
        self.learning = []
    def get_signal_history(self):
        return self.history
    def save_signal_history(self, signal_type, data):
        self.history[signal_type] = data
    def save_learning_history(self, **kw):
        self.learning.append(kw)

def build_trader():
    return SelfLearningTrader(FakeBot(), FakeDB())

def test_asymmetric_winner_gets_strong_weight():
    """45% win rate at 2.5R avg win / 1R avg loss must weight ABOVE 1.0.

    Old code: weight = 0.5 + 2*(0.45-0.5) = 0.4 (down-weighted a profitable edge).
    """
    t = build_trader()
    # 100 trades: 45 wins of +2.5R, 55 losses of -1R -> expectancy +0.575R
    for i in range(100):
        win = (i % 100) < 45
        r = 2.5 if win else -1.0
        t.record_signal_outcome(['trend_up'], pnl=100.0 if win else -40.0, success=win, r_multiple=r)
    w = t.get_signal_weight('trend_up')
    assert w >= 1.0, f"asymmetric winner should be strong, got {w:.3f}"
    # Unshrunk expectancy +0.575R -> 0.5 + 1.5*0.575 = 1.36; shrink(100) = 0.833 -> ~1.22
    assert 1.0 <= w <= 1.5, f"weight out of expected band: {w:.3f}"
    print(f"PASS asymmetric winner: 45% WR @ 2.5R/1R -> weight {w:.3f} (old code: 0.400)")

def test_frequent_small_winner_no_longer_dominates():
    """70% win rate with tiny wins (avg +0.3R) vs 1R losses is expectancy-negative
    (-0.09R). Old code gave it ~1.1 (strong). New code must keep it weak."""
    t = build_trader()
    for i in range(100):
        win = (i % 100) < 70
        r = 0.3 if win else -1.0
        t.record_signal_outcome(['macd_buy'], pnl=3.0 if win else -10.0, success=win, r_multiple=r)
    w = t.get_signal_weight('macd_buy')
    assert w < 1.0, f"negative-expectancy signal must stay weak, got {w:.3f}"
    print(f"PASS frequent small winner: 70% WR @ 0.3R/1R -> weight {w:.3f} (old code: ~1.100)")

def test_legacy_fallback_still_works():
    """Rows recorded before R tracking (no r_multiple passed) use the old mapping."""
    t = build_trader()
    for i in range(30):
        win = (i % 30) < 21  # 70% win rate, no R data
        t.record_signal_outcome(['rsi_oversold'], pnl=1.0 if win else -1.0, success=win)
    w = t.get_signal_weight('rsi_oversold')
    # old formula: posterior (21+2)/(30+4) = 0.676 -> 0.5 + 2*0.176 = 0.853
    assert abs(w - 0.853) < 0.01, f"legacy fallback weight off: {w:.3f}"
    print(f"PASS legacy fallback: no-R row still maps win rate -> weight {w:.3f}")

def test_small_sample_shrink():
    """10 samples can't earn a maxed weight; 200 samples of same edge get closer to full."""
    def weight_for(n, wins_pct, win_r, loss_r):
        t = build_trader()
        for i in range(n):
            win = (i % n) < n * wins_pct
            r = win_r if win else -loss_r
            t.record_signal_outcome(['volume_spike'], pnl=1.0 if win else -1.0, success=win, r_multiple=r)
        return t.get_signal_weight('volume_spike')
    w10 = weight_for(10, 0.5, 2.0, 1.0)   # expectancy +0.5R but tiny sample
    w200 = weight_for(200, 0.5, 2.0, 1.0)
    assert w10 < w200, f"smaller sample should shrink harder: {w10:.3f} vs {w200:.3f}"
    assert w200 > 1.0
    print(f"PASS shrinkage: 10 samples -> {w10:.3f}, 200 samples -> {w200:.3f}")

# ─── 2. Champion / challenger gate ─────────────────────────────────
import numpy as np
import xgboost as xgb
from types import SimpleNamespace
from datetime import datetime, timezone

T0 = 1_700_000_000  # epoch seconds, even -> parity of T0 + 61*i == parity of i
STEP = 61            # odd step so entry_ts % 2 == i % 2

def build_X(n):
    """Features exactly as train_xgboost builds them via the monkeypatched
    _extract_features: [entry_ts, entry_ts % 2, zeros...]."""
    return np.array([[T0 + i * STEP, (T0 + i * STEP) % 2] + [0] * 11 for i in range(n)])

def make_trade(i, win):
    """Trade whose label is determined by exit_reason (like the real path)."""
    ts = datetime.fromtimestamp(T0 + i * STEP, tz=timezone.utc)
    return SimpleNamespace(
        time=ts.isoformat(),
        symbol='BTC-USD',
        entry_price=100.0,
        stop_loss_price=90.0,
        take_profit_price=125.0,
        exit_price=125.0 if win else 90.0,
        exit_reason='take profit hit' if win else 'stop loss hit',
    )

def setup_gate_env(t, n=80):
    """Feed train_xgboost a synthetic bot state: labels = parity of trade index."""
    t.bot.state.trades = [make_trade(i, i % 2 == 0) for i in range(n)]
    t.bot.state.candle_history = {
        'BTC-USD': [{'time': 1_700_000_000 + j * 60, 'open': 100, 'high': 101,
                     'low': 99, 'close': 100 + j * 0.01, 'volume': 1000}
                    for j in range(80)],
    }
    tmp = Path(tempfile.mkdtemp())
    t.model_path = tmp / "model.pkl"
    t.features_path = tmp / "features.pkl"
    t.model_meta_path = tmp / "meta.json"
    t._extract_features = lambda candles, entry_time: [
        entry_time, entry_time % 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

REAL_XGB = xgb.XGBClassifier  # captured before any monkeypatching

class WeakChallenger:
    """XGBClassifier stand-in whose fit() shuffles labels, so the challenger
    stays ~0.5 even after train_xgboost refits it."""
    def __init__(self, **kw):
        self._model = None
    def fit(self, X, y):
        rng = np.random.default_rng(0)
        self._model = REAL_XGB(n_estimators=30, max_depth=4,
                               use_label_encoder=False,
                               eval_metric='logloss', random_state=999)
        self._model.fit(X, rng.permutation(y))
        return self
    def predict(self, X):
        return self._model.predict(X)

def test_gate_rejects_worse_challenger():
    """Challenger that underperforms the champion on the holdout must NOT deploy."""
    import bot_server
    t = build_trader()
    setup_gate_env(t)

    # champion: trained on the REAL labels -> ~1.0 holdout accuracy
    X, y = build_X(80), np.array([i % 2 for i in range(80)])
    champ = xgb.XGBClassifier(n_estimators=30, max_depth=4, use_label_encoder=False,
                              eval_metric='logloss', random_state=1)
    champ.fit(X[:56], y[:56])
    champ_acc = (champ.predict(X[56:]) == y[56:]).mean()
    assert champ_acc > 0.9, f"test setup broken: champion acc {champ_acc:.3f}"
    t.xgb_model = champ

    weak_acc = (WeakChallenger().fit(X[:56], y[:56]).predict(X[56:]) == y[56:]).mean()
    assert weak_acc < 0.6, f"test setup broken: weak acc {weak_acc:.3f}"

    orig = bot_server.xgb.XGBClassifier
    bot_server.xgb.XGBClassifier = WeakChallenger
    t.last_train_time = 0
    t.train_xgboost(force=True)
    bot_server.xgb.XGBClassifier = orig

    assert t.xgb_model is champ, "weak challenger must NOT replace champion"
    assert t._load_model_meta() is None, "no promotion meta for rejected model"
    print(f"PASS gate rejects: champ {champ_acc:.3f} vs weak {weak_acc:.3f}, champion kept")

def test_gate_promotes_better_challenger():
    """Challenger beating the champion by margin on the SAME holdout gets promoted."""
    import bot_server
    t = build_trader()
    setup_gate_env(t)

    # deliberately bad champion (trained on shuffled labels, ~0.5)
    rng = np.random.default_rng(3)
    X, y = build_X(80), np.array([i % 2 for i in range(80)])
    champ = xgb.XGBClassifier(n_estimators=30, max_depth=4, use_label_encoder=False,
                              eval_metric='logloss', random_state=1)
    champ.fit(X[:56], rng.permutation(y[:56]))
    champ_acc = (champ.predict(X[56:]) == y[56:]).mean()
    assert champ_acc < 0.65, f"test setup broken: noise champ acc {champ_acc:.3f}"
    t.xgb_model = champ

    good = xgb.XGBClassifier(n_estimators=30, max_depth=4, use_label_encoder=False,
                             eval_metric='logloss', random_state=2)
    # train_xgboost will fit this on real Xtr/ytr -> ~1.0 holdout
    orig = bot_server.xgb.XGBClassifier
    bot_server.xgb.XGBClassifier = lambda **kw: good

    t.last_train_time = 0
    t.train_xgboost(force=True)
    bot_server.xgb.XGBClassifier = orig

    assert t.xgb_model is good, "better challenger should be promoted"
    meta = t._load_model_meta()
    assert meta is not None and meta['holdout_acc'] > 0.9, f"meta wrong: {meta}"
    print(f"PASS gate promotes: challenger {meta['holdout_acc']:.3f} beat champion {champ_acc:.3f} on same holdout")

def test_gate_refuses_first_model_below_floor():
    """No champion yet + holdout below floor -> no deploy, model stays None."""
    import bot_server
    t = build_trader()
    setup_gate_env(t)
    t.xgb_model = None

    orig = bot_server.xgb.XGBClassifier
    bot_server.xgb.XGBClassifier = WeakChallenger
    t.last_train_time = 0
    t.train_xgboost(force=True)
    bot_server.xgb.XGBClassifier = orig

    assert t.xgb_model is None, "sub-floor first model must not deploy"
    assert t._load_model_meta() is None
    print("PASS first-model floor: holdout ~0.5 < 0.55, refused deploy")

# ─── 3. DB migration ───────────────────────────────────────────────
def test_db_migration():
    """Old-schema DB gets the new columns via ALTER; round-trip persists R data."""
    tmp = Path(tempfile.mkdtemp()) / "old.db"
    conn = db_mod.sqlite3.connect(tmp)
    conn.execute('''
        CREATE TABLE signal_history (
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
    conn.commit()
    conn.close()

    d = db_mod.BotDatabase(db_path=tmp)
    d.save_signal_history('trend_up', {
        'total_signals': 12, 'successful_trades': 6, 'total_pnl': 3.0,
        'win_rate': 50.0, 'avg_pnl': 0.25, 'weight': 1.0,
        'total_r': 2.4, 'total_win_r': 3.6, 'total_loss_r': 1.2,
        'last_updated': 'now',
    })
    got = d.get_signal_history()['trend_up']
    assert got['total_r'] == 2.4 and got['total_win_r'] == 3.6 and got['total_loss_r'] == 1.2, got
    print("PASS DB migration: old schema upgraded, R fields round-trip")

if __name__ == '__main__':
    test_asymmetric_winner_gets_strong_weight()
    test_frequent_small_winner_no_longer_dominates()
    test_legacy_fallback_still_works()
    test_small_sample_shrink()
    test_gate_rejects_worse_challenger()
    test_gate_promotes_better_challenger()
    test_gate_refuses_first_model_below_floor()
    test_db_migration()
    print("\nALL TESTS PASSED")
