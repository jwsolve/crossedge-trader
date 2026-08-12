#!/usr/bin/env python3
"""Synthetic verification for the self_learning_enabled toggle fix.

The trap being fixed: the toggle was a kill switch for the WHOLE self-learning
lane. Flipping it off meant:
  - no new ML entries (intended), but ALSO
  - no reversal-signal exits on held positions (positions left to rot),
  - retraining kept running anyway (the "learning" never actually stopped),
  - and it did nothing at all when the strategy dropdown wasn't self_learning.

New semantics: the toggle gates ML ENTRIES and retraining only. Reversal SELL
signals on held positions always flow; hard TP/SL/trailing (protective_exit_decision)
were already strategy-independent and untouched.

Run: python3 test_toggle_fix.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bot_server import PaperBot, SelfLearningTrader


class FakeLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeState:
    def __init__(self, settings):
        self.settings = settings
        self.last_action_time = 0.0
        self.active_symbol = ""
        self.coin = 0.0
        self.positions = {}
        self.day_start_equity = 10000.0
        self.peak_equity = 10000.0
        self.day_start_date = ""
        self.learning_iterations = 0
        self.last_learning_update = ""
        self.trades = []
        self.candle_history = {}
        self.last_price = None

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


class FakeBot:
    def __init__(self, settings, positions=None):
        base = {
            'cooldown_seconds': 0,
            'daily_loss_limit_pct': 50.0,
            'max_drawdown_pct': 20.0,
            'news_guard_enabled': False,
            'strategy': 'self_learning',
        }
        base.update(settings)
        self.lock = FakeLock()
        self.state = FakeState(base)
        self.state.positions = positions or {}
        self._kraken_live_balance = {}
        self.self_learning_trader = SelfLearningTrader(self, FakeDB())

    def price_for_active_position(self, fetched_prices):
        return fetched_prices.get(self.state.active_symbol, 100.0)

    def equity(self, price):
        return self.state.day_start_equity

    def live_exchange(self):
        return "paper"

    def is_news_blocked(self, symbol, settings):
        return False, ""

    def save_state(self):
        pass


def make_analysis(direction, weight=1.2, strength=0.8):
    """A signal analysis that clears every should_enter_trade gate."""
    signed = strength if direction == 'BUY' else -strength
    return {
        'signals': [{'weight': weight, 'strength': strength, 'type': 'trend_up', 'direction': direction}],
        'composite_score': signed,
        'confidence': min(1.0, weight / 3.0),
        'direction': direction,
        'signal_count': 1,
        'signal_types': ['trend_up'],
        'signal_scores': [weight * strength],
    }


def decide(bot, analysis, symbol="BTC-USD"):
    """Run decide_self_learning with one watchlist symbol returning `analysis`."""
    bot.self_learning_trader.analyze_candles_with_indicators = (
        lambda candles, settings: analysis
    )
    # Bind the REAL decide_self_learning implementation onto the fake bot.
    bound = PaperBot.decide_self_learning.__get__(bot, FakeBot)
    return bound(
        fetched_prices={symbol: 100.0},
        watchlist=[symbol],
        candles_by_symbol={symbol: [{}] * 60},  # >= 50 candles required
    )


def test_toggle_off_blocks_new_entries():
    """Toggle OFF + no position + strong BUY signal -> no entry, clear HOLD."""
    bot = FakeBot({'self_learning_enabled': False})
    decision = decide(bot, make_analysis('BUY'))
    assert not decision.startswith('BUY'), f"should NOT enter, got: {decision}"
    assert "disabled" in decision, f"expected disabled HOLD, got: {decision}"


def test_toggle_off_keeps_reversal_exits():
    """Toggle OFF + held position + SELL reversal -> exit STILL fires.

    This is the core of the fix: pausing the ML must never orphan a position.
    """
    bot = FakeBot(
        {'self_learning_enabled': False},
        positions={"BTC-USD": {"quantity": 1.0, "entry_price": 90.0, "is_short": False}},
    )
    decision = decide(bot, make_analysis('SELL'))
    assert decision.startswith('SELL'), f"exit must fire while disabled, got: {decision}"
    assert "BTC-USD" in decision, f"exit must name the held symbol, got: {decision}"


def test_toggle_off_no_signal_clear_hold():
    """Toggle OFF + no exit signal -> HOLD message says entries off, exits armed."""
    bot = FakeBot({'self_learning_enabled': False})
    decision = decide(bot, make_analysis('BUY'))
    assert "entries off" in decision and "exits armed" in decision, f"got: {decision}"


def test_toggle_on_entries_unchanged():
    """Toggle ON -> normal BUY entry, exactly as before the fix."""
    bot = FakeBot({'self_learning_enabled': True})
    decision = decide(bot, make_analysis('BUY'))
    assert decision.startswith('BUY'), f"entry should fire, got: {decision}"


def test_toggle_on_exits_unchanged():
    """Toggle ON + held position + SELL reversal -> exit fires as before."""
    bot = FakeBot(
        {'self_learning_enabled': True},
        positions={"BTC-USD": {"quantity": 1.0, "entry_price": 90.0, "is_short": False}},
    )
    decision = decide(bot, make_analysis('SELL'))
    assert decision.startswith('SELL'), f"exit should fire, got: {decision}"


def test_should_enter_trade_no_longer_kills_signals():
    """should_enter_trade must evaluate signals even with toggle OFF.

    It serves both entry and exit paths; the toggle gate lives in
    decide_self_learning, which decides what to do with the signal.
    """
    trader = SelfLearningTrader(FakeBot({'self_learning_enabled': False}), FakeDB())
    ok, direction, score, types = trader.should_enter_trade(
        make_analysis('BUY'), {'self_learning_enabled': False}
    )
    assert ok and direction == 'BUY', f"signal must survive toggle-off, got: {ok}, {direction}"


def test_default_settings_learning_on():
    """Absent toggle -> learning enabled (backwards compatible)."""
    bot = FakeBot({})
    decision = decide(bot, make_analysis('BUY'))
    assert decision.startswith('BUY'), f"default must allow entries, got: {decision}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nAll {len(tests)} toggle tests passed.")
