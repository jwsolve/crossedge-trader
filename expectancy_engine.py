#!/usr/bin/env python3
"""
Expectancy Engine – advanced performance analytics for trading.

Computes:
- Win rate, average win/loss, profit factor, expectancy
- R‑multiple distribution
- Max consecutive losses, recovery factor
- Sharpe ratio (simplified)
- Composite scores per symbol / strategy / regime
- Rolling performance metrics
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import json

# ─── Try to use pandas for efficient analysis (optional) ──────────
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class TradeStats:
    """Aggregated statistics for a set of trades."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    break_even: int = 0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    win_rate: float = 0.0
    loss_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    expectancy_per_risk: float = 0.0  # R-multiple expectancy
    avg_r_multiple: float = 0.0
    sharpe_ratio: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    recovery_factor: float = 0.0
    total_fees: float = 0.0
    fee_pct_of_pnl: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0


class ExpectancyEngine:
    """
    Computes trading performance metrics from a list of Trade objects.
    Can filter by symbol, strategy, regime, date range, etc.
    """

    def __init__(self, trades: Optional[List] = None, db=None):
        """
        trades: list of Trade objects (from bot_server)
        db: BotDatabase instance (optional, to load trades on demand)
        """
        self.db = db
        self._trades = trades or []
        self._metrics_cache = {}
        self._r_multiple_cache = {}

    # ─── Load / update trades ──────────────────────────────────────

    def load_trades_from_db(self, symbol: Optional[str] = None,
                            strategy: Optional[str] = None,
                            regime: Optional[str] = None,
                            days: Optional[int] = None,
                            limit: int = 10000) -> None:
        """Load trades from the database with optional filters."""
        if not self.db:
            raise RuntimeError("No database connection provided.")
        # We'll get all trades and filter in-memory to keep it simple
        raw = self.db.get_trades(limit=limit)
        trades = []
        for t in raw:
            # Convert dict to Trade object (if needed)
            # Assuming Trade dataclass is available; we can import it or reconstruct
            # We'll use a helper to convert
            trade = self._dict_to_trade(t)
            if symbol and trade.symbol != symbol:
                continue
            if strategy and trade.reason and strategy not in trade.reason:
                continue
            if regime and trade.regime and trade.regime != regime:
                continue
            if days:
                try:
                    dt = datetime.fromisoformat(trade.time)
                    if (datetime.now() - dt).days > days:
                        continue
                except:
                    pass
            trades.append(trade)
        self._trades = trades
        self._metrics_cache.clear()
        self._r_multiple_cache.clear()

    def _dict_to_trade(self, d):
        """Convert a dict (from DB) to a Trade object."""
        # We need to import Trade from bot_server, but to avoid circular import,
        # we can create a simple placeholder or use a dict as a Trade-like object.
        # We'll use a simple class for internal use.
        class SimpleTrade:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        return SimpleTrade(**d)

    def set_trades(self, trades: List) -> None:
        """Directly set the trade list (e.g., from bot.state.trades)."""
        self._trades = trades
        self._metrics_cache.clear()
        self._r_multiple_cache.clear()

    # ─── Core computations ─────────────────────────────────────────

    @staticmethod
    def _is_completed_trade(t: Any) -> bool:
        """Only include realised/completed trades in performance statistics.

        A cancelled/rejected/expired order with no execution is an order event,
        not a trade. Open/pending entries are also excluded until they have an
        exit. Partially filled entries remain eligible as executions but do not
        become a closed trade until an exit/PnL record exists.
        """
        status = str(
            getattr(t, "exchange_order_status", None)
            or getattr(t, "status", None)
            or ""
        ).upper().strip()
        filled = getattr(t, "exchange_filled_size", None)
        if filled is None:
            filled = getattr(t, "filled_size", None)
        filled = abs(float(filled or 0.0))

        if status in {"CANCELLED", "CANCELED", "REJECTED", "FAILED", "EXPIRED",
                       "PENDING", "UNFILLED"} and filled <= 1e-12:
            return False

        # A trade contributes to win/loss/expectancy only after it has an exit
        # or an explicit completed status. This prevents open entries with the
        # default pnl=0 from becoming break-even trades.
        exit_price = getattr(t, "exit_price", None)
        if exit_price is None and status not in {"FILLED", "CLOSED", "EXECUTED"}:
            return False
        return hasattr(t, "pnl")

    def compute_stats(self, trades: Optional[List] = None) -> TradeStats:
        """Compute statistics from completed/executed trades only."""
        if trades is None:
            trades = self._trades
        if not trades:
            return TradeStats()

        trades = [t for t in trades if self._is_completed_trade(t)]
        if not trades:
            return TradeStats()

        # Extract PnL and fees
        pnls = [t.pnl for t in trades if hasattr(t, 'pnl')]
        fees = [t.fee_paid for t in trades if hasattr(t, 'fee_paid')]
        total_fees = sum(fees) if fees else 0.0

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        break_even = [p for p in pnls if p == 0]

        total = len(pnls)
        win_count = len(wins)
        loss_count = len(losses)
        be_count = len(break_even)

        total_pnl = sum(pnls) if pnls else 0.0
        avg_pnl = total_pnl / total if total > 0 else 0.0
        avg_win = sum(wins) / win_count if win_count > 0 else 0.0
        avg_loss = abs(sum(losses) / loss_count) if loss_count > 0 else 0.0
        max_win = max(wins) if wins else 0.0
        max_loss = min(losses) if losses else 0.0

        win_rate = win_count / total if total > 0 else 0.0
        loss_rate = loss_count / total if total > 0 else 0.0

        profit_factor = sum(wins) / abs(sum(losses)) if losses else (float('inf') if sum(wins) > 0 else 0.0)

        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss) if avg_loss is not None else 0.0

        # R-multiple expectancy: avg win / avg loss ratio
        if avg_loss > 0:
            avg_r_win = avg_win / avg_loss
            avg_r_loss = 1.0
            avg_r_multiple = (win_rate * avg_r_win) - (loss_rate * avg_r_loss)
        else:
            avg_r_multiple = 0.0

        # Consecutive wins/losses
        max_cons_wins = 0
        max_cons_losses = 0
        curr_wins = 0
        curr_losses = 0
        for p in pnls:
            if p > 0:
                curr_wins += 1
                curr_losses = 0
                max_cons_wins = max(max_cons_wins, curr_wins)
            elif p < 0:
                curr_losses += 1
                curr_wins = 0
                max_cons_losses = max(max_cons_losses, curr_losses)
            else:
                curr_wins = 0
                curr_losses = 0

        # Sharpe ratio (simplified using daily returns if we have dates)
        # For simplicity, we can use PnL standard deviation.
        if len(pnls) > 1:
            mean_pnl = sum(pnls) / len(pnls)
            std_pnl = math.sqrt(sum((p - mean_pnl)**2 for p in pnls) / (len(pnls) - 1))
            sharpe = (mean_pnl / std_pnl) * math.sqrt(252) if std_pnl > 0 else 0.0
        else:
            sharpe = 0.0

        # Recovery factor: total pnl / max drawdown
        # We need cumulative sum to compute max drawdown
        cumulative = []
        running = 0.0
        for p in pnls:
            running += p
            cumulative.append(running)
        peak = 0.0
        drawdowns = []
        for val in cumulative:
            if val > peak:
                peak = val
            drawdown = peak - val
            drawdowns.append(drawdown)
        max_drawdown = max(drawdowns) if drawdowns else 0.0
        recovery_factor = total_pnl / max_drawdown if max_drawdown > 0 else 0.0

        # Fee impact
        fee_pct = total_fees / abs(total_pnl) if total_pnl != 0 else 0.0

        stats = TradeStats(
            total_trades=total,
            wins=win_count,
            losses=loss_count,
            break_even=be_count,
            total_pnl=total_pnl,
            avg_pnl=avg_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_win=max_win,
            max_loss=max_loss,
            win_rate=win_rate,
            loss_rate=loss_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            expectancy_per_risk=avg_r_multiple,
            avg_r_multiple=avg_r_multiple,
            sharpe_ratio=sharpe,
            max_consecutive_wins=max_cons_wins,
            max_consecutive_losses=max_cons_losses,
            recovery_factor=recovery_factor,
            total_fees=total_fees,
            fee_pct_of_pnl=fee_pct,
            best_trade=max_win,
            worst_trade=max_loss,
        )
        return stats

    # ─── R-multiple distribution ──────────────────────────────────

    def get_r_multiple_distribution(self, r_value: float = 1.0,
                                    trades: Optional[List] = None) -> Dict[str, Any]:
        """
        Returns distribution of R-multiples (e.g., number of trades that achieved 1R, 2R, etc.)
        R = risk per trade (we estimate based on entry and stop loss if available).
        For simplicity, we use avg_loss as a proxy for risk.
        """
        if trades is None:
            trades = self._trades
        if not trades:
            return {}

        # We need risk per trade. Use stop_loss_price if available, else use entry_price * stop_loss_pct.
        r_multipliers = []
        for t in trades:
            # Estimate risk as (entry - stop) if we have it, else use avg_loss as fallback
            risk = None
            if hasattr(t, 'stop_loss_price') and t.stop_loss_price and t.entry_price:
                risk = abs(t.entry_price - t.stop_loss_price) * abs(t.quantity)
            elif hasattr(t, 'entry_price') and t.entry_price and hasattr(t, 'exit_price') and t.exit_price:
                # use volatility approximation: average loss of the set
                # fallback: use global avg_loss from compute_stats
                pass
            # If we can't compute per-trade risk, we skip.
            if risk is not None and risk > 0:
                r = t.pnl / risk
                r_multipliers.append(r)

        if not r_multipliers:
            return {}

        # Build distribution bins
        bins = [-float('inf'), -5, -3, -2, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2, 3, 5, float('inf')]
        labels = ['<-5', '-5 to -3', '-3 to -2', '-2 to -1', '-1 to -0.5', '-0.5 to -0.1', '-0.1 to 0',
                  '0 to 0.1', '0.1 to 0.5', '0.5 to 1', '1 to 2', '2 to 3', '3 to 5', '>5']

        counts = {label: 0 for label in labels}
        for r in r_multipliers:
            for i, bin_edge in enumerate(bins[:-1]):
                if bin_edge <= r < bins[i+1]:
                    counts[labels[i]] += 1
                    break

        # Also compute win rate by R threshold
        thresholds = [0, 0.5, 1, 2, 3, 5]
        threshold_stats = {}
        for thresh in thresholds:
            count = sum(1 for r in r_multipliers if r >= thresh)
            threshold_stats[f'>= {thresh}R'] = {
                'count': count,
                'pct': count / len(r_multipliers) if r_multipliers else 0
            }

        return {
            'distribution': counts,
            'threshold_stats': threshold_stats,
            'total_trades': len(r_multipliers),
            'avg_r': sum(r_multipliers) / len(r_multipliers) if r_multipliers else 0,
            'max_r': max(r_multipliers) if r_multipliers else 0,
            'min_r': min(r_multipliers) if r_multipliers else 0,
        }

    # ─── Scores and summaries ──────────────────────────────────────

    def get_symbol_scores(self) -> Dict[str, float]:
        """Compute a composite score per symbol based on expectancy and consistency."""
        symbols = set(t.symbol for t in self._trades if hasattr(t, 'symbol'))
        scores = {}
        for sym in symbols:
            sym_trades = [t for t in self._trades if t.symbol == sym]
            stats = self.compute_stats(sym_trades)
            if stats.total_trades < 5:
                continue
            # Composite: expectancy * profit_factor / (1 + max_consecutive_losses)
            score = stats.expectancy * stats.profit_factor / (1 + stats.max_consecutive_losses)
            scores[sym] = round(score, 4)
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def get_strategy_scores(self) -> Dict[str, float]:
        """Score each strategy (inferred from trade.reason)."""
        strategies = {}
        for t in self._trades:
            # Extract strategy from reason (e.g., "EWO offset sell" -> "EWO")
            if hasattr(t, 'reason') and t.reason:
                # Simple heuristic: take first word before " " or "|"
                parts = t.reason.split()
                if parts:
                    strat = parts[0].upper()
                    strategies.setdefault(strat, []).append(t)
        scores = {}
        for strat, trades in strategies.items():
            stats = self.compute_stats(trades)
            if stats.total_trades < 3:
                continue
            # Score: expectancy * win_rate
            score = stats.expectancy * stats.win_rate
            scores[strat] = round(score, 4)
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def get_regime_scores(self) -> Dict[str, float]:
        """Score by regime if trade has regime attribute."""
        regimes = defaultdict(list)
        for t in self._trades:
            if hasattr(t, 'regime') and t.regime:
                regimes[t.regime].append(t)
        scores = {}
        for regime, trades in regimes.items():
            stats = self.compute_stats(trades)
            if stats.total_trades < 3:
                continue
            score = stats.expectancy * stats.win_rate
            scores[regime] = round(score, 4)
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def summary(self) -> Dict[str, Any]:
        """Get a summary of overall performance and per-symbol scores."""
        overall = self.compute_stats()
        return {
            'overall': {
                'total_trades': overall.total_trades,
                'win_rate': round(overall.win_rate * 100, 2),
                'avg_pnl': round(overall.avg_pnl, 4),
                'expectancy': round(overall.expectancy, 4),
                'profit_factor': round(overall.profit_factor, 2),
                'sharpe_ratio': round(overall.sharpe_ratio, 2),
                'max_consecutive_losses': overall.max_consecutive_losses,
                'recovery_factor': round(overall.recovery_factor, 2),
                'avg_r_multiple': round(overall.avg_r_multiple, 2),
            },
            'symbol_scores': self.get_symbol_scores(),
            'strategy_scores': self.get_strategy_scores(),
            'regime_scores': self.get_regime_scores(),
            'r_distribution': self.get_r_multiple_distribution(),
        }

    # ─── Refresh / cache management ──────────────────────────────

    def refresh(self, trades: Optional[List] = None) -> None:
        """Force recomputation of all metrics."""
        if trades is not None:
            self.set_trades(trades)
        self._metrics_cache.clear()
        self._r_multiple_cache.clear()

    # ─── For UI / dashboard ───────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary suitable for JSON serialization."""
        summary = self.summary()
        # Convert datetime objects if any
        return summary
