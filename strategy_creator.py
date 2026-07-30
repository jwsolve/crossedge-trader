#!/usr/bin/env python3
"""
Strategy Creator - Genetic Algorithm for Trading Strategy Evolution.
FIXED: Uses cached candles, no OANDA calls during evolution.
"""

import random
import json
import logging
import threading
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

STRATEGY_CREATOR_AVAILABLE = True

logger = logging.getLogger('auxo')

# ─── DATA CLASSES ──────────────────────────────────────────────────

@dataclass
class TradingStrategy:
    """A complete trading strategy with entry/exit rules and parameters."""
    id: str
    name: str
    entry_rules: List[str]
    exit_rule: str
    filters: List[str]
    parameters: Dict[str, float]
    fitness_score: float = 0.0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    trades_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation: int = 0


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0


# ─── REAL BACKTEST ENGINE ──────────────────────────────────────────

class BacktestEngine:
    """Real backtesting engine - NO OANDA CALLS!"""

    def __init__(self):
        self.cached_candles = None

    def set_candles(self, candles: list):
        """Set candles once for all evaluations."""
        self.cached_candles = candles

    def evaluate_strategy(self, strategy: TradingStrategy) -> StrategyPerformance:
        """
        Evaluate a strategy on cached candles.
        NO OANDA CALLS - just local calculations.
        """
        performance = StrategyPerformance()
        candles = self.cached_candles

        if not candles or len(candles) < 50:
            return performance

        # ─── SIMULATE TRADING ──────────────────────────────────────
        # This uses the actual candle data to simulate trades
        # based on the strategy's entry/exit rules

        entry_price = None
        entry_index = 0
        trades = []
        equity = 1000.0  # Starting equity for backtest

        for i in range(50, len(candles)):
            candle = candles[i]
            price = candle.close

            # Check entry rules
            should_enter = self.check_entry_rules(strategy, candles, i)

            if should_enter and entry_price is None:
                # Enter trade
                entry_price = price
                entry_index = i
                continue

            # Check exit rules
            if entry_price is not None:
                should_exit = self.check_exit_rules(strategy, candles, i, entry_price)

                if should_exit:
                    # Exit trade
                    exit_price = price
                    pnl = (exit_price - entry_price) / entry_price * 100
                    trades.append({
                        'entry': entry_price,
                        'exit': exit_price,
                        'pnl_pct': pnl,
                        'win': pnl > 0
                    })
                    entry_price = None

        # Calculate performance
        if trades:
            performance.total_trades = len(trades)
            performance.wins = sum(1 for t in trades if t['win'])
            performance.losses = performance.total_trades - performance.wins
            performance.win_rate = (performance.wins / performance.total_trades) * 100

            pnls = [t['pnl_pct'] for t in trades]
            winning_pnls = [t['pnl_pct'] for t in trades if t['win']]
            losing_pnls = [-t['pnl_pct'] for t in trades if not t['win']]

            performance.total_pnl = sum(pnls)
            performance.avg_win = sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0
            performance.avg_loss = sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0

            # Max drawdown
            peak = 0
            drawdown = 0
            for t in trades:
                peak = max(peak, t['pnl_pct'])
                drawdown = min(drawdown, t['pnl_pct'] - peak)
            performance.max_drawdown = abs(drawdown)

            # Sharpe ratio
            if pnls:
                avg = sum(pnls) / len(pnls)
                std = (sum((p - avg) ** 2 for p in pnls) / len(pnls)) ** 0.5
                performance.sharpe_ratio = avg / std if std > 0 else 0

            # Profit factor
            gross_profit = sum(winning_pnls) if winning_pnls else 0
            gross_loss = abs(sum(losing_pnls)) if losing_pnls else 0
            performance.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        return performance

    def check_entry_rules(self, strategy: TradingStrategy, candles: list, index: int) -> bool:
        """Check if entry rules are triggered."""
        if index < 20:
            return False

        price = candles[index].close
        price_prev = candles[index-1].close

        # SMA crossover
        if 'sma_crossover_bullish' in strategy.entry_rules:
            fast = self.sma(candles, index, 5)
            slow = self.sma(candles, index, 20)
            fast_prev = self.sma(candles, index-1, 5)
            slow_prev = self.sma(candles, index-1, 20)
            if fast_prev <= slow_prev and fast > slow:
                return True

        if 'sma_crossover_bearish' in strategy.entry_rules:
            fast = self.sma(candles, index, 5)
            slow = self.sma(candles, index, 20)
            fast_prev = self.sma(candles, index-1, 5)
            slow_prev = self.sma(candles, index-1, 20)
            if fast_prev >= slow_prev and fast < slow:
                return True

        # RSI oversold/overbought
        if 'rsi_oversold' in strategy.entry_rules:
            rsi = self.rsi(candles, index)
            if rsi is not None and rsi < 30:
                return True

        if 'rsi_overbought' in strategy.entry_rules:
            rsi = self.rsi(candles, index)
            if rsi is not None and rsi > 70:
                return True

        # Price breakout
        if 'breakout_high' in strategy.entry_rules:
            high_20 = max(c.high for c in candles[max(0, index-20):index])
            if price > high_20:
                return True

        if 'breakout_low' in strategy.entry_rules:
            low_20 = min(c.low for c in candles[max(0, index-20):index])
            if price < low_20:
                return True

        # Volume spike
        if 'volume_spike' in strategy.entry_rules:
            avg_vol = sum(c.volume for c in candles[max(0, index-20):index]) / 20
            if candles[index].volume > avg_vol * 2:
                return True

        return False

    def check_exit_rules(self, strategy: TradingStrategy, candles: list, index: int, entry_price: float) -> bool:
        """Check if exit rules are triggered."""
        price = candles[index].close
        pnl_pct = (price - entry_price) / entry_price * 100

        # Fixed stop loss
        if 'fixed_stop' in strategy.exit_rule:
            stop_pct = strategy.parameters.get('stop_loss_pct', 2.0)
            if pnl_pct < -stop_pct:
                return True

        # Fixed target
        if 'fixed_target' in strategy.exit_rule:
            target_pct = strategy.parameters.get('take_profit_pct', 3.0)
            if pnl_pct > target_pct:
                return True

        # Trailing stop
        if 'trailing_stop' in strategy.exit_rule:
            trail_pct = strategy.parameters.get('trailing_stop_pct', 2.0)
            activation = strategy.parameters.get('trailing_activation_pct', 3.0)
            if pnl_pct > activation:
                # Track highest price since entry
                highest = max(c.high for c in candles[max(0, index-50):index+1])
                if (highest - price) / highest * 100 > trail_pct:
                    return True

        # RSI reversal
        if 'rsi_reversal' in strategy.exit_rule:
            rsi = self.rsi(candles, index)
            rsi_prev = self.rsi(candles, index-1)
            if rsi is not None and rsi_prev is not None:
                if pnl_pct > 0 and rsi > 70 and rsi < rsi_prev:
                    return True

        # Breakeven
        if 'breakeven' in strategy.exit_rule:
            if pnl_pct > 0.5:
                return False  # Move stop to breakeven is handled elsewhere
            if pnl_pct < -0.1:
                return True

        return False

    def sma(self, candles: list, index: int, period: int) -> float:
        """Calculate Simple Moving Average."""
        if index < period:
            return candles[index].close
        values = [c.close for c in candles[index-period:index]]
        return sum(values) / len(values) if values else 0

    def rsi(self, candles: list, index: int, period: int = 14) -> Optional[float]:
        """Calculate RSI."""
        if index < period + 1:
            return None

        gains = []
        losses = []
        for i in range(index - period, index):
            change = candles[i].close - candles[i-1].close
            gains.append(max(change, 0))
            losses.append(max(-change, 0))

        avg_gain = sum(gains) / period if gains else 0
        avg_loss = sum(losses) / period if losses else 0

        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


# ─── STRATEGY GENERATOR ──────────────────────────────────────────

class StrategyGenerator:
    """Generate new trading strategies."""

    def __init__(self, seed: int = None):
        if seed:
            random.seed(seed)

        self.entry_rules = [
            'sma_crossover_bullish',
            'sma_crossover_bearish',
            'rsi_oversold',
            'rsi_overbought',
            'breakout_high',
            'breakout_low',
            'volume_spike',
            'engulfing_bullish',   # ADD THESE
            'engulfing_bearish',   # ADD THESE
            'support_bounce',      # ADD THESE
            'resistance_rejection' # ADD THESE
        ]

        self.exit_rules = [
            'fixed_stop',
            'fixed_target',
            'trailing_stop',
            'rsi_reversal',
            'breakeven',
        ]

        self.param_ranges = {
            'sma_fast': (5, 15, 1),
            'sma_slow': (20, 50, 5),
            'stop_loss_pct': (0.5, 3.0, 0.1),
            'take_profit_pct': (1.0, 5.0, 0.1),
            'trailing_stop_pct': (0.5, 2.0, 0.1),
            'trailing_activation_pct': (1.0, 4.0, 0.1),
        }

    def generate_random_strategy(self) -> TradingStrategy:
        """Generate a random trading strategy."""
        num_entries = random.randint(1, 2)
        entry_rules = random.sample(self.entry_rules, num_entries)
        exit_rule = random.choice(self.exit_rules)

        parameters = {}
        for param_name, (min_val, max_val, step) in self.param_ranges.items():
            if random.random() < 0.5:
                if isinstance(min_val, int):
                    parameters[param_name] = random.randint(min_val, max_val)
                else:
                    parameters[param_name] = round(random.uniform(min_val, max_val) / step) * step
                    parameters[param_name] = round(parameters[param_name], 2)

        return TradingStrategy(
            id=str(random.randint(10000, 99999)),
            name=f"strategy_{random.randint(100, 999)}",
            entry_rules=entry_rules,
            exit_rule=exit_rule,
            filters=[],
            parameters=parameters,
            generation=0
        )

    def mutate_strategy(self, strategy: TradingStrategy, mutation_rate: float = 0.2) -> TradingStrategy:
        """Mutate a strategy."""
        mutated = TradingStrategy(
            id=str(random.randint(10000, 99999)),
            name=strategy.name + '_mut',
            entry_rules=strategy.entry_rules.copy(),
            exit_rule=strategy.exit_rule,
            filters=strategy.filters.copy(),
            parameters=strategy.parameters.copy(),
            generation=strategy.generation + 1
        )

        if random.random() < mutation_rate:
            if random.random() < 0.5:
                available = [r for r in self.entry_rules if r not in mutated.entry_rules]
                if available:
                    mutated.entry_rules.append(random.choice(available))
            elif len(mutated.entry_rules) > 1:
                mutated.entry_rules.pop(random.randint(0, len(mutated.entry_rules) - 1))

        return mutated

    def crossover(self, parent1: TradingStrategy, parent2: TradingStrategy) -> TradingStrategy:
        """Combine two strategies."""
        combined_entries = list(set(parent1.entry_rules + parent2.entry_rules))
        if len(combined_entries) > 2:
            combined_entries = random.sample(combined_entries, random.randint(1, 2))

        exit_rule = random.choice([parent1.exit_rule, parent2.exit_rule])

        parameters = {}
        all_params = set(parent1.parameters.keys()) | set(parent2.parameters.keys())
        for param in all_params:
            if param in parent1.parameters and param in parent2.parameters:
                if isinstance(parent1.parameters[param], int):
                    parameters[param] = int((parent1.parameters[param] + parent2.parameters[param]) / 2)
                else:
                    parameters[param] = round((parent1.parameters[param] + parent2.parameters[param]) / 2, 2)
            elif param in parent1.parameters:
                parameters[param] = parent1.parameters[param]
            else:
                parameters[param] = parent2.parameters[param]

        return TradingStrategy(
            id=str(random.randint(10000, 99999)),
            name=f"cross_{parent1.name}_{parent2.name}",
            entry_rules=combined_entries,
            exit_rule=exit_rule,
            filters=[],
            parameters=parameters,
            generation=max(parent1.generation, parent2.generation) + 1
        )


# ─── GENETIC ALGORITHM ─────────────────────────────────────────────

class GeneticStrategyOptimizer:
    """Genetic Algorithm for evolving trading strategies."""

    def __init__(self, population_size: int = 50, elite_percent: float = 0.1):
        self.population_size = population_size
        self.elite_percent = elite_percent
        self.population: List[TradingStrategy] = []
        self.generation = 0
        self.generator = StrategyGenerator()
        self.backtester = BacktestEngine()
        self.best_strategy: Optional[TradingStrategy] = None
        self.best_fitness = -float('inf')
        self.logger = logging.getLogger('auxo')

    def set_candles(self, candles: list):
        """Set candles for backtesting."""
        self.backtester.set_candles(candles)

    def initialize_population(self, size: int = None):
        """Initialize the population with random strategies."""
        size = size or self.population_size
        self.population = [self.generator.generate_random_strategy() for _ in range(size)]
        self.generation = 0
        self.logger.info(f"Initialized population with {size} strategies")

    def evaluate_all(self):
        """Evaluate all strategies using the cached candles."""
        for strategy in self.population:
            performance = self.backtester.evaluate_strategy(strategy)
            strategy.fitness_score = self.calculate_fitness(performance)
            strategy.win_rate = performance.win_rate
            strategy.total_pnl = performance.total_pnl
            strategy.trades_count = performance.total_trades

    def calculate_fitness(self, performance: StrategyPerformance) -> float:
        """Calculate fitness score for a strategy."""
        if performance.total_trades < 1:
            return -1000

        score = 0.0
        score += performance.win_rate * 0.4
        score += performance.sharpe_ratio * 10
        score -= performance.max_drawdown * 2

        if performance.total_trades > 10:
            score += 5

        return round(score, 2)

    def select_top(self):
        """Select the top performing strategies."""
        sorted_pop = sorted(self.population, key=lambda s: s.fitness_score, reverse=True)
        keep_count = max(2, int(len(sorted_pop) * self.elite_percent))
        self.population = sorted_pop[:keep_count]

        if self.population and self.population[0].fitness_score > self.best_fitness:
            self.best_fitness = self.population[0].fitness_score
            self.best_strategy = self.population[0]
            self.logger.info(f"New best fitness: {self.best_fitness:.2f}")

    def evolve(self, mutation_rate: float = 0.3):
        """Evolve to the next generation."""
        self.generation += 1

        elites = self.population[:max(2, int(len(self.population) * 0.2))]
        offspring = []

        while len(offspring) < self.population_size - len(elites):
            parent1 = random.choice(self.population)
            parent2 = random.choice(self.population)
            child = self.generator.crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child = self.generator.mutate_strategy(child)
            offspring.append(child)

        self.population = elites + offspring
        self.logger.debug(f"Generation {self.generation}: {len(self.population)} strategies")

    def run_evolution(self, generations: int = 20) -> Optional[TradingStrategy]:
        """Run the full evolution process."""
        self.initialize_population()

        for gen in range(generations):
            self.evaluate_all()

            # ─── DON'T THROW AWAY ALL STRATEGIES ───────────────────────
            # Keep at least some strategies even if they're not great
            if len([s for s in self.population if s.fitness_score > -1000]) < 5:
                # If we don't have enough good strategies, keep some bad ones
                self.population = self.population[:10] + [self.generator.generate_random_strategy() for _ in range(10)]

            self.select_top()
            self.evolve()

            if gen % 5 == 0:
                avg_fitness = sum(s.fitness_score for s in self.population) / len(self.population)
                self.logger.info(f"Gen {gen+1}/{generations}: best={self.best_fitness:.2f}, avg={avg_fitness:.2f}")

        # ─── FALLBACK: Return the best even if it's not great ────────
        if not self.best_strategy and self.population:
            self.best_strategy = max(self.population, key=lambda s: s.fitness_score)
            self.logger.info(f"Using best available: {self.best_strategy.name} (fitness: {self.best_strategy.fitness_score:.2f})")

        return self.best_strategy


# ─── STRATEGY MANAGER ──────────────────────────────────────────────

class StrategyManager:
    """Manages trading strategies for the bot."""

    def __init__(self, bot):
        self.bot = bot
        self.optimizer = GeneticStrategyOptimizer()
        self.active_strategies: List[TradingStrategy] = []
        self.current_strategy_id: Optional[str] = None
        self.evolution_running = False
        self.logger = logging.getLogger('auxo')

    def evolve_strategies(self, candles: list, generations: int = 20) -> Optional[TradingStrategy]:
        """
        Evolve new strategies using cached candles.
        NO OANDA CALLS - uses cached data only.
        """
        if self.evolution_running:
            self.logger.warning("Evolution already running")
            return None

        if not candles or len(candles) < 50:
            self.logger.warning("Not enough candles for evolution")
            return None

        self.evolution_running = True

        try:
            # Set candles once
            self.optimizer.set_candles(candles)

            # Run evolution
            best = self.optimizer.run_evolution(generations)

            if best:
                self.active_strategies.append(best)
                self.current_strategy_id = best.id
                self.logger.info(f"Evolved: {best.name} ({best.win_rate:.1f}% win rate)")
            else:
                self.logger.warning("No strategy evolved")

            return best

        except Exception as e:
            self.logger.error(f"Evolution error: {e}")
            return None
        finally:
            self.evolution_running = False

    def select_strategy(self) -> Optional[TradingStrategy]:
        """Select the best strategy."""
        if not self.active_strategies:
            return None
        return max(self.active_strategies, key=lambda s: s.fitness_score)

    def get_performance_summary(self) -> dict:
        """Get summary of all strategies."""
        return {
            'active_strategies': len(self.active_strategies),
            'current_strategy_id': self.current_strategy_id,
            'strategy_details': [
                {
                    'id': s.id,
                    'name': s.name,
                    'fitness': s.fitness_score,
                    'win_rate': s.win_rate,
                    'trades': s.trades_count
                }
                for s in self.active_strategies[-5:]
            ]
        }
