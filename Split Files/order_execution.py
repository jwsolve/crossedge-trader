# order_execution.py
"""
Order execution: paper, live, OANDA.
All functions take a bot instance as first argument.
"""

import time
import uuid
from typing import Any, Optional, List

from utils import now_iso, normalize_forex_symbol, parse_watchlist, pct
from indicators import exit_prices, support_resistance, sr_buy_allowed, position_spend, partial_take_profit_ready, trailing_stop_price, closes_to_candles, signal_candles, ewo_offset_signal
from risk_manager import calculate_position_size, get_regime_adaptations
from coinbase_api import (
    coinbase_live_is_armed, coinbase_live_status_message,
    coinbase_available_balance, coinbase_market_order, coinbase_limit_order,
    coinbase_stop_limit_order, coinbase_order_id, coinbase_reconcile_order,
    coinbase_cancel_orders, coinbase_round_price, coinbase_round_size,
    live_market_guard, coinbase_api_request
)
from oanda_api import (
    oanda_market_order, oanda_order_fill, oanda_demo_orders_armed,
    oanda_demo_status_message, oanda_is_configured, oanda_request,
    oanda_account_id, oanda_account_summary, oanda_instrument,
    oanda_stream_pricing, oanda_decimal
)
from database import Trade, ManagedOrder, SetupRecord

# ─── Paper Trading ───────────────────────────────────────────────

def paper_buy(
    bot,
    symbol: str,
    price: float,
    reason: str,
    candles: list | None = None,
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
    with bot.lock:
        settings = dict(bot.state.settings)
        # We'll update state later

    trade_fee = float(settings["trade_fee"])
    spend_reason = "manual override"

    # ─── Get regime-adaptive stop/target ─────────────────────────
    regime_adapt = get_regime_adaptations(bot.state)
    stop_mult = regime_adapt.get('stop_multiplier', 1.0)
    tp_mult = regime_adapt.get('take_profit_multiplier', 1.0)

    if spend_override is not None:
        spend = spend_override
    else:
        if candles is None:
            with bot.lock:
                candles = closes_to_candles(bot.state.price_history.get(symbol, []))

        # Temporarily modify stop/target percentages for this trade
        original_stop = settings.get('stop_loss_pct', 2.0)
        original_tp = settings.get('take_profit_pct', 3.0)
        settings['stop_loss_pct'] = original_stop * stop_mult
        settings['take_profit_pct'] = original_tp * tp_mult

        try:
            with bot.lock:
                cash = bot.state.cash
            spend, spend_reason = calculate_position_size(
                cash=cash,
                entry_price=price,
                candles=candles,
                symbol=symbol,
                settings=settings,
                db=bot.db,
                state=bot.state,
                position_side="SHORT" if is_short else "LONG"
            )
        finally:
            # Restore original values
            settings['stop_loss_pct'] = original_stop
            settings['take_profit_pct'] = original_tp

    with bot.lock:
        if spend > bot.state.cash:
            spend = bot.state.cash

        if spend < float(settings.get("min_order_value", 1.0)):
            bot.state.last_signal = f"{'SHORT' if is_short else 'BUY'} blocked: order below minimum {settings['quote_currency']} {settings.get('min_order_value', 1.0)}"
            bot.journal(symbol, "BLOCK", bot.state.last_signal, price, {"spend": spend})
            return

        fee_paid = fee_override if fee_override is not None else spend * trade_fee
        coin_bought = quantity_override if quantity_override is not None else (spend - fee_paid) / price

        if stop_override is not None and target_override is not None:
            stop_price = stop_override
            target_price = target_override
            exit_mode = "opening_range"
        else:
            stop_price, target_price, exit_mode = exit_prices(
                entry_price=price,
                candles=candles or closes_to_candles(bot.state.price_history.get(symbol, [])),
                settings=settings,
            )

        # Apply regime multipliers to stop/target if not overridden
        if stop_override is None and target_override is None:
            # Already applied via settings adjustment above, but ensure consistency
            pass

        if is_short:
            bot.state.coin -= coin_bought
            bot.state.cash += spend
            bot.state.active_symbol = symbol
            bot.state.entry_price = price
            bot.state.highest_price = price
            bot.state.stop_price = stop_price
            bot.state.target_price = target_price
            bot.state.exit_mode = exit_mode
            bot.state.active_stop_order_id = None
            bot.state.partial_take_profit_done = False
            bot.state.last_price = price
            bot.state.last_action_time = time.time()
            bot.state.is_short = True

            bot.state.positions[symbol] = {
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
                cash_after=bot.state.cash,
                coin_after=bot.state.coin,
                reason=f"{reason} | size {spend_reason} | {exit_mode} stop/target",
                fee_paid=fee_paid,
                exchange_order_id=exchange_order_id,
                exchange_order_status=exchange_order_status,
                exchange_average_filled_price=exchange_average_filled_price,
                exchange_filled_size=coin_bought if exchange_order_id else None,
                stop_loss_price=stop_price,
                take_profit_price=target_price,
                exit_mode=exit_mode,
                regime=bot.state.current_regime.regime if bot.state.current_regime else None,
            )
            bot.record_trade(trade)
            bot.journal(symbol, "SHORT", reason, price, {"spend": spend, "quantity": coin_bought, "stop": stop_price, "target": target_price})
        else:
            bot.state.cash -= spend
            bot.state.coin += coin_bought
            bot.state.active_symbol = symbol
            bot.state.entry_price = price
            bot.state.highest_price = price
            bot.state.stop_price = stop_price
            bot.state.target_price = target_price
            bot.state.exit_mode = exit_mode
            bot.state.active_stop_order_id = None
            bot.state.partial_take_profit_done = False
            bot.state.last_price = price
            bot.state.last_action_time = time.time()
            bot.state.is_short = False

            bot.state.positions[symbol] = {
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
                cash_after=bot.state.cash,
                coin_after=bot.state.coin,
                reason=f"{reason} | size {spend_reason} | {exit_mode} stop/target",
                fee_paid=fee_paid,
                exchange_order_id=exchange_order_id,
                exchange_order_status=exchange_order_status,
                exchange_average_filled_price=exchange_average_filled_price,
                exchange_filled_size=coin_bought if exchange_order_id else None,
                stop_loss_price=stop_price,
                take_profit_price=target_price,
                exit_mode=exit_mode,
                regime=bot.state.current_regime.regime if bot.state.current_regime else None,
            )
            bot.record_trade(trade)
            bot.record_setup_buy(symbol, price, coin_bought, spend, fee_paid, reason, stop_price, target_price, exit_mode)
            bot.journal(symbol, "BUY", reason, price, {"spend": spend, "quantity": coin_bought, "stop": stop_price, "target": target_price})

        if settings.get("telegram_alert_on_buy", True):
            bot.send_telegram_alert(bot.format_alert_trade(bot.state.trades[-1]))

def paper_sell(
    bot,
    symbol: str,
    price: float,
    reason: str,
    quantity_override: float | None = None,
    fee_override: float | None = None,
    exchange_order_id: str | None = None,
    exchange_order_status: str | None = None,
    exchange_average_filled_price: float | None = None,
) -> None:
    with bot.lock:
        settings = dict(bot.state.settings)
        trade_fee = float(settings["trade_fee"])

        if abs(bot.state.coin) <= 0 and symbol not in bot.state.positions:
            bot.state.last_signal = "SELL blocked: no position"
            bot.journal(symbol, "BLOCK", "SELL blocked: no position", price)
            return

        position = bot.state.positions.get(symbol, {})
        position_quantity = float(position.get("quantity", 0.0))
        coin_available = bot.state.coin if bot.state.active_symbol == symbol else position_quantity

        if abs(coin_available) <= 0:
            bot.state.last_signal = f"SELL blocked: no {symbol} position"
            bot.journal(symbol, "BLOCK", f"SELL blocked: no {symbol} position", price)
            return

        is_short = position.get("is_short", False) or bot.state.is_short
        entry_price = float(position.get("entry_price") or bot.state.entry_price or 0.0)
        sold_quantity = min(abs(coin_available), abs(quantity_override or coin_available))

        if is_short:
            pnl = (entry_price - price) * sold_quantity
        else:
            pnl = (price - entry_price) * sold_quantity

        gross = sold_quantity * price
        fee_paid = fee_override if fee_override is not None else gross * trade_fee
        cash_received = gross - fee_paid

        bot.state.cash += cash_received

        if is_short:
            bot.state.coin += sold_quantity
            if symbol in bot.state.positions:
                remaining = position_quantity + sold_quantity
                if remaining >= 0:
                    bot.state.positions.pop(symbol, None)
                else:
                    position["quantity"] = remaining
                    bot.state.positions[symbol] = position
        else:
            bot.state.coin -= sold_quantity
            if symbol in bot.state.positions:
                remaining = position_quantity - sold_quantity
                if remaining <= 0:
                    bot.state.positions.pop(symbol, None)
                else:
                    position["quantity"] = remaining
                    bot.state.positions[symbol] = position

        position_closed = abs(bot.state.coin) <= 0.0000000001 and len(bot.state.positions) == 0
        if position_closed:
            bot.state.coin = 0.0
            bot.state.active_symbol = None
            bot.state.entry_price = None
            bot.state.highest_price = None
            bot.state.stop_price = None
            bot.state.target_price = None
            bot.state.active_stop_order_id = None
            bot.state.partial_take_profit_done = False
            bot.state.is_short = False
            bot.state.positions = {}

        bot.state.last_price = price
        bot.state.last_action_time = time.time()
        side = "SELL" if not is_short else "BUY"

        trade = Trade(
            time=now_iso(),
            side=side,
            symbol=symbol,
            price=price,
            quantity=-sold_quantity if is_short else sold_quantity,
            cash_after=bot.state.cash,
            coin_after=bot.state.coin,
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
            regime=bot.state.current_regime.regime if bot.state.current_regime else None,
        )
        bot.record_trade(trade)

        bot.record_setup_sell(
            symbol,
            price,
            sold_quantity,
            cash_received,
            fee_paid,
            reason,
            position_closed,
        )

        if hasattr(bot, 'self_learning_trader') and position_closed:
            setup_record = next(
                (r for r in reversed(bot.state.setup_records)
                 if r.symbol == symbol and r.status == "CLOSED"),
                None
            )
            if setup_record and setup_record.signal_types:
                success = pnl > 0
                bot.self_learning_trader.record_signal_outcome(
                    setup_record.signal_types,
                    pnl,
                    success
                )

        bot.journal(symbol, side, reason, price, {"quantity": sold_quantity, "pnl": pnl})

        if settings.get("telegram_alert_on_sell", True):
            bot.send_telegram_alert(bot.format_alert_trade(trade, pnl))

# ─── Live Trading ────────────────────────────────────────────────

def should_live_trade(state) -> bool:
    settings = state.settings
    return (
        bool(settings.get("live_trading_enabled"))
        and settings.get("asset_class", "crypto") == "crypto"
        and settings.get("exchange") == "coinbase"
        and coinbase_live_is_armed()
    )

def live_buy(
    bot,
    symbol: str,
    price: float,
    reason: str,
    candles: list | None = None,
    is_short: bool = False,
) -> None:
    bot.roll_live_daily_spend_if_needed()
    with bot.lock:
        settings = dict(bot.state.settings)
        cash = bot.state.cash
        live_daily_spend = bot.state.live_daily_spend
        positions = dict(bot.state.positions)

    max_order = float(settings["max_live_order_gbp"])
    max_daily = float(settings["max_daily_live_loss_gbp"])
    max_coinbase_positions = int(settings.get("max_coinbase_open_trades", 3))

    if candles is None:
        with bot.lock:
            candles = closes_to_candles(bot.state.price_history.get(symbol, []))

    # ─── Get regime-adaptive stop/target ─────────────────────────
    regime_adapt = get_regime_adaptations(bot.state)
    stop_mult = regime_adapt.get('stop_multiplier', 1.0)
    tp_mult = regime_adapt.get('take_profit_multiplier', 1.0)

    original_stop = settings.get('stop_loss_pct', 2.0)
    original_tp = settings.get('take_profit_pct', 3.0)
    settings['stop_loss_pct'] = original_stop * stop_mult
    settings['take_profit_pct'] = original_tp * tp_mult

    try:
        paper_spend, spend_reason = calculate_position_size(
            cash=cash,
            entry_price=price,
            candles=candles,
            symbol=symbol,
            settings=settings,
            db=bot.db,
            state=bot.state,
            position_side="SHORT" if is_short else "LONG"
        )
    finally:
        settings['stop_loss_pct'] = original_stop
        settings['take_profit_pct'] = original_tp

    quote_size = round(min(max_order, paper_spend), 2)

    minimum_order = max(1.0, float(settings.get("min_order_value", 1.0)))
    if quote_size < minimum_order:
        with bot.lock:
            bot.state.last_signal = f"LIVE {'SHORT' if is_short else 'BUY'} blocked: order below {settings['quote_currency']} {minimum_order:.2f}"
            bot.journal(symbol, "BLOCK", bot.state.last_signal, price, {"quote_size": quote_size})
        return

    if len(positions) >= max_coinbase_positions:
        with bot.lock:
            bot.state.last_signal = "LIVE BUY blocked: max coinbase trades reached"
            bot.journal(symbol, "BLOCK", bot.state.last_signal, price, {"quote_size": quote_size})
        return

    if live_daily_spend + quote_size > max_daily:
        with bot.lock:
            bot.state.last_signal = "LIVE BUY blocked: daily live cap reached"
            bot.journal(symbol, "BLOCK", bot.state.last_signal, price, {"quote_size": quote_size})
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
        with bot.lock:
            bot.state.last_signal = f"LIVE BUY blocked: {guard['reason']}"
            bot.journal(symbol, "BLOCK", bot.state.last_signal, price, guard)
        return

    gbp_available = coinbase_available_balance(settings["quote_currency"])
    if gbp_available < quote_size:
        with bot.lock:
            bot.state.last_signal = (
                f"LIVE BUY blocked: only {settings['quote_currency']} {gbp_available:.2f} available"
            )
            bot.journal(symbol, "BLOCK", bot.state.last_signal, price, {"available": gbp_available, "quote_size": quote_size})
        return

    product_id = f"{symbol}-{settings['quote_currency']}"
    order_type = str(settings.get("live_order_type", "market"))
    limit_offset = float(settings.get("live_limit_offset_pct", 0.05)) / 100

    # ─── Use price aggregator to get best price and exchange ──────────
    active_exchange = settings.get("active_exchange", "coinbase")
    if active_exchange not in bot.connectors:
        logger.warning(f"Active exchange {active_exchange} not available, falling back to coinbase")
        active_exchange = "coinbase"

    connector = bot.connectors.get(active_exchange)
    if not connector:
        raise RuntimeError(f"No connector available for {active_exchange}")

    try:
        best_price, best_exchange, _ = bot.price_aggregator.get_best_price(symbol, side="BUY")
        limit_price = coinbase_round_price(best_price * (1 + limit_offset), product_id)

        ok, volume, recommended = bot.price_aggregator.check_liquidity(symbol, "BUY", quote_size)
        if not ok:
            with bot.lock:
                bot.state.last_signal = f"LIVE BUY blocked: insufficient liquidity on {active_exchange} (need {quote_size}, have {volume})"
                bot.journal(symbol, "BLOCK", bot.state.last_signal, price)
            return
    except Exception as e:
        with bot.lock:
            bot.state.last_signal = f"LIVE BUY blocked: price/liquidity check failed: {e}"
            bot.journal(symbol, "BLOCK", bot.state.last_signal, price)
        return

    base_size = quote_size / limit_price if limit_price > 0 else 0.0
    base_size = coinbase_round_size(base_size, product_id)

    stop_price, target_price, exit_mode = exit_prices(
        entry_price=price,
        candles=candles or closes_to_candles(bot.state.price_history.get(symbol, [])),
        settings=settings,
    )

    # ─── Place order via the selected connector ──────────────────────
    if order_type in {"limit", "bracket", "native_stop_scaffold"}:
        order = coinbase_limit_order(
            product_id=product_id,
            side="BUY" if not is_short else "SELL",
            base_size=base_size,
            limit_price=limit_price,
        )
    else:
        order = connector.market_buy(symbol, quote_size)
        try:
            order_id = order.get("order_id") or order.get("orderId") or str(order.get("id"))
        except:
            order_id = str(uuid.uuid4())
        order = {"order_id": order_id, "raw": order}

    order_id = coinbase_order_id(order)
    managed = track_order(
        bot,
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
    if apply_reconciled_order(bot, managed, fill):
        return
    if fill["filled_size"] <= 0:
        with bot.lock:
            bot.state.last_signal = f"LIVE {'SHORT' if is_short else 'BUY'} pending/unfilled: {order_id}"
            bot.journal(symbol, "INFO", bot.state.last_signal, price, {"order": order, "fill": fill})

def live_sell(
    bot,
    symbol: str,
    price: float,
    reason: str,
    quantity_override: float | None = None,
    is_short: bool = False,
) -> None:
    settings = dict(bot.state.settings)  # copy outside lock
    base_available = coinbase_available_balance(symbol)
    desired_size = quantity_override or bot.state.coin
    base_size = min(base_available, desired_size)

    if base_size <= 0:
        with bot.lock:
            bot.state.last_signal = f"LIVE SELL blocked: no {symbol} balance available"
            bot.journal(symbol, "BLOCK", bot.state.last_signal, price)
        return

    if bot.state.active_stop_order_id:
        stop_order_id = bot.state.active_stop_order_id
        try:
            cancel_response = coinbase_cancel_orders([stop_order_id])
            bot.journal(
                symbol,
                "INFO",
                f"Cancelled native stop before live sell: {stop_order_id}",
                price,
                {"cancel_response": cancel_response},
            )
            bot.state.active_stop_order_id = None
        except Exception as exc:
            with bot.lock:
                bot.state.last_signal = f"LIVE SELL blocked: could not cancel native stop {stop_order_id}: {exc}"
                bot.journal(symbol, "BLOCK", bot.state.last_signal, price)
            return

    product_id = f"{symbol}-{settings['quote_currency']}"
    order_type = str(settings.get("live_order_type", "market"))

    if order_type == "limit":
        limit_offset = float(settings.get("live_limit_offset_pct", 0.05)) / 100
        limit_price = coinbase_round_price(price * (1 - limit_offset), product_id)
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
    order_id = coinbase_order_id(order)
    managed = track_order(
        bot,
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
    if apply_reconciled_order(bot, managed, fill):
        if abs(bot.state.coin) > 0 and bool(settings.get("native_stop_enabled")) and bot.state.entry_price:
            submit_native_stop_for_position(
                bot,
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
                bot.state.entry_price,
            )
        return
    if fill["filled_size"] <= 0:
        with bot.lock:
            bot.state.last_signal = f"LIVE SELL pending/unfilled: {order_id}"
            bot.journal(symbol, "INFO", bot.state.last_signal, price, {"order": order, "fill": fill})

# ─── OANDA Demo Trading ──────────────────────────────────────────

def should_oanda_demo_trade(state) -> bool:
    settings = state.settings
    return (
        bool(settings.get("oanda_demo_trading_enabled"))
        and settings.get("asset_class") == "forex"
        and settings.get("exchange") == "oanda_demo"
        and oanda_demo_orders_armed()
    )

def wants_oanda_demo_trade(state) -> bool:
    settings = state.settings
    return (
        bool(settings.get("oanda_demo_trading_enabled"))
        and settings.get("asset_class") == "forex"
        and settings.get("exchange") == "oanda_demo"
    )

def oanda_demo_buy(
    bot,
    symbol: str,
    price: float,
    reason: str,
    candles: list | None = None,
    is_short: bool = False,
) -> None:
    settings = bot.state.settings
    quote_currency = settings.get("quote_currency", "GBP")

    if symbol in bot.state.positions:
        bot.state.last_signal = f"OANDA {'SHORT' if is_short else 'BUY'} blocked: {symbol} already has an open position"
        bot.journal(symbol, "BLOCK", bot.state.last_signal, price)
        return

    max_positions = int(settings.get("max_oanda_open_trades", 3))
    if len(bot.state.positions) >= max_positions:
        bot.state.last_signal = f"OANDA {'SHORT' if is_short else 'BUY'} blocked: max open trades reached ({max_positions})"
        bot.journal(symbol, "BLOCK", bot.state.last_signal, price)
        return

    if is_short and not settings.get("allow_short_selling", False):
        bot.state.last_signal = f"OANDA SHORT blocked: short selling disabled"
        bot.journal(symbol, "BLOCK", bot.state.last_signal, price)
        return

    cash = bot.state.cash

    risk_pct = float(settings.get("risk_per_trade_pct", 1.0)) / 100
    risk_cash = cash * risk_pct

    max_pct = float(settings.get("max_position_pct", 0.25))
    max_position_cash = cash * max_pct

    stop_pct = float(settings.get("stop_loss_pct", 2.0)) / 100

    position_value_gbp = min(risk_cash / stop_pct, max_position_cash)
    position_value_gbp = max(position_value_gbp, float(settings.get("min_order_value", 1.0)))
    position_value_gbp = min(position_value_gbp, cash)

    min_order = float(settings.get("min_order_value", 1.0))
    if position_value_gbp < min_order:
        bot.state.last_signal = f"OANDA {'SHORT' if is_short else 'BUY'} blocked: position {position_value_gbp:.2f} below minimum {min_order:.2f} {quote_currency}"
        bot.journal(symbol, "BLOCK", bot.state.last_signal, price, {"position_value": position_value_gbp, "min_order": min_order})
        return

    exchange_rate = bot.get_exchange_rate(symbol)

    position_value_gbp = min(risk_cash / stop_pct, max_position_cash)

    price_in_gbp = price / exchange_rate
    units = int(position_value_gbp / price_in_gbp)

    units = max(1, units)

    stop_price, target_price, exit_mode = exit_prices(
        entry_price=price,
        candles=candles or closes_to_candles(bot.state.price_history.get(symbol, [])),
        settings=settings,
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

    except Exception as e:
        bot.state.last_signal = f"OANDA order failed: {e}"
        bot.journal(symbol, "ERROR", bot.state.last_signal, price)
        return

    if is_short:
        bot.state.cash += position_value_actual
        bot.state.coin -= abs(filled_units)
        bot.state.is_short = True
        bot.state.positions[symbol] = {
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
        bot.state.cash -= position_value_actual
        bot.state.coin += abs(filled_units)
        bot.state.is_short = False
        bot.state.positions[symbol] = {
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

    bot.state.active_symbol = symbol
    bot.state.entry_price = fill_price
    bot.state.highest_price = fill_price
    bot.state.stop_price = stop_price
    bot.state.target_price = target_price
    bot.state.last_price = fill_price
    bot.state.last_action_time = time.time()
    bot.state.last_signal = f"OANDA {'SHORT' if is_short else 'BUY'} {symbol} @ {fill_price:.6f} | Cash: {bot.state.cash:.2f}"

    side = "SHORT" if is_short else "BUY"
    trade_reason = f"{reason} | OANDA demo order | {exit_mode} stop/target"

    trade = Trade(
        time=now_iso(),
        side=side,
        symbol=symbol,
        price=fill_price,
        quantity=abs(filled_units),
        cash_after=bot.state.cash,
        coin_after=bot.state.coin,
        reason=trade_reason,
        fee_paid=fee,
        exchange_order_id=fill["order_id"],
        exchange_order_status=fill["status"],
        exchange_average_filled_price=fill_price,
        exchange_filled_size=abs(filled_units),
        stop_loss_price=stop_price,
        take_profit_price=target_price,
        exit_mode=exit_mode,
        regime=bot.state.current_regime.regime if bot.state.current_regime else None,
    )
    bot.record_trade(trade)
    bot.record_setup_buy(symbol, fill_price, abs(filled_units), position_value_actual, fee, trade_reason, stop_price, target_price, exit_mode)
    bot.journal(symbol, side, trade_reason, fill_price, {"spend": position_value_actual, "quantity": abs(filled_units)})

def oanda_demo_sell(
    bot,
    symbol: str,
    price: float,
    reason: str,
    quantity_override: float | None = None,
) -> None:
    position = bot.state.positions.get(symbol)
    if not position:
        bot.state.last_signal = "OANDA SELL blocked: no position"
        bot.journal(symbol, "BLOCK", bot.state.last_signal, price)
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

    exchange_rate = bot.get_exchange_rate(symbol)

    if is_short:
        pnl_quote = (entry_price - fill_price) * filled_units
    else:
        pnl_quote = (fill_price - entry_price) * filled_units

    pnl = pnl_quote / exchange_rate

    gross = filled_units * fill_price
    cash_received = gross / exchange_rate - fee

    bot.state.cash += cash_received

    remaining = abs(current_quantity) - filled_units
    position_closed = remaining <= 0.0000000001

    if position_closed:
        bot.state.positions.pop(symbol, None)
    else:
        position["quantity"] = -remaining if is_short else remaining
        bot.state.positions[symbol] = position

    bot.state.active_symbol = next(iter(bot.state.positions), None)
    active_position = bot.state.positions.get(bot.state.active_symbol or "", {})
    bot.state.entry_price = active_position.get("entry_price")
    bot.state.highest_price = active_position.get("highest_price")
    bot.state.stop_price = active_position.get("stop_price")
    bot.state.target_price = active_position.get("target_price")
    bot.state.is_short = active_position.get("is_short", False)
    bot.state.last_price = fill_price
    bot.state.last_action_time = time.time()

    side = "SELL" if not is_short else "BUY"
    trade_reason = f"{reason} | OANDA demo order | PnL: {pnl:.2f} GBP"

    trade = Trade(
        time=now_iso(),
        side=side,
        symbol=symbol,
        price=fill_price,
        quantity=filled_units,
        cash_after=bot.state.cash,
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
        regime=bot.state.current_regime.regime if bot.state.current_regime else None,
    )
    bot.record_trade(trade)
    bot.record_setup_sell(symbol, fill_price, filled_units, cash_received, fee, trade_reason, position_closed)
    bot.journal(symbol, side, trade_reason, fill_price, {"quantity": filled_units, "pnl": pnl})
    bot.journal(symbol, "INFO", f"OANDA demo {side} filled", fill_price, fill)

# ─── Order Management ─────────────────────────────────────────────

def track_order(
    bot,
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
        expires_at=time.time() + int(bot.state.settings.get("order_expiry_seconds", 180)),
        price=price,
        base_size=base_size,
        quote_size=quote_size,
        reason=reason,
        details=details or {},
    )
    bot.state.open_orders.append(order)
    bot.state.open_orders = bot.state.open_orders[-120:]
    bot.audit("ORDER_TRACKED", order=asdict(order))
    return order

def managed_order(bot, order_id: str) -> ManagedOrder | None:
    return next((item for item in bot.state.open_orders if item.order_id == order_id), None)

def manage_open_orders(bot) -> None:
    for order in list(bot.state.open_orders):
        if order.status in {"FILLED", "CANCELLED", "FAILED", "EXPIRED"}:
            continue

        try:
            fill = coinbase_reconcile_order(order.order_id)
        except Exception as exc:
            order.updated_at = now_iso()
            order.status = "RECONCILE_ERROR"
            bot.audit("ORDER_RECONCILE_ERROR", order_id=order.order_id, error=str(exc))
            continue

        apply_reconciled_order(bot, order, fill)
        if order.status == "FILLED":
            continue

        if time.time() >= order.expires_at:
            expire_order(bot, order)

def apply_reconciled_order(bot, order: ManagedOrder, fill: dict[str, Any]) -> bool:
    order.updated_at = now_iso()
    order.status = fill.get("status", "UNKNOWN")
    if fill["filled_size"] <= 0 or order.local_applied:
        return False

    filled_price = fill["average_price"] or order.price or bot.state.last_price or 0.0
    is_short = order.details.get("is_short", False)

    if order.role == "ENTRY":
        filled_quote = (fill["filled_value"] or order.quote_size or 0.0) + fill["total_fee"]
        bot.state.live_daily_spend += min(order.quote_size or filled_quote, filled_quote)
        paper_buy(
            bot,
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
            submit_native_stop_for_position(bot, order, filled_price)
    elif order.role in {"EXIT", "STOP"}:
        paper_sell(
            bot,
            order.symbol,
            filled_price,
            f"LIVE {order.role} filled {order.order_id} | {order.reason}",
            quantity_override=fill["filled_size"],
            fee_override=fill["total_fee"],
            exchange_order_id=order.order_id,
            exchange_order_status=order.status,
            exchange_average_filled_price=filled_price,
        )
        if order.role == "STOP" and abs(bot.state.coin) <= 0 and not bot.state.positions:
            bot.state.active_stop_order_id = None

    order.local_applied = True
    order.status = "FILLED"
    order.updated_at = now_iso()
    bot.audit("ORDER_FILLED_APPLIED", order=asdict(order), fill=fill)
    return True

def expire_order(bot, order: ManagedOrder) -> None:
    try:
        cancel_response = coinbase_cancel_orders([order.order_id])
        order.status = "EXPIRED"
        order.updated_at = now_iso()
        if bot.state.active_stop_order_id == order.order_id:
            bot.state.active_stop_order_id = None
        bot.audit("ORDER_EXPIRED_CANCELLED", order=asdict(order), cancel_response=cancel_response)
    except Exception as exc:
        order.status = "CANCEL_FAILED"
        order.updated_at = now_iso()
        bot.audit("ORDER_EXPIRE_CANCEL_FAILED", order=asdict(order), error=str(exc))
        return

    if (
        bool(bot.state.settings.get("order_replace_enabled"))
        and order.retry_count < int(bot.state.settings.get("order_retry_limit", 1))
        and order.role in {"ENTRY", "EXIT"}
    ):
        replace_order(bot, order)

def replace_order(bot, order: ManagedOrder) -> None:
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
        new_order = track_order(
            bot,
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
        bot.audit("ORDER_REPLACED", old_order_id=order.order_id, new_order_id=replacement_id)
    except Exception as exc:
        bot.audit("ORDER_REPLACE_FAILED", order_id=order.order_id, error=str(exc))

def submit_native_stop_for_position(bot, entry_order: ManagedOrder, entry_price: float) -> None:
    if abs(bot.state.coin) <= 0:
        return

    if not bot.state.settings.get("native_stop_enabled", False):
        return

    stop_price = float(entry_order.details.get("stop_price") or 0.0)
    exit_mode = str(entry_order.details.get("exit_mode") or "fixed")
    if stop_price <= 0:
        candles = closes_to_candles(bot.state.price_history.get(entry_order.symbol, []))
        stop_price, _, exit_mode = exit_prices(entry_price, candles, bot.state.settings)

    product_id = entry_order.product_id
    limit_price = coinbase_round_price(stop_price * 0.995, product_id)
    stop_price = coinbase_round_price(stop_price, product_id)
    base_size = coinbase_round_size(abs(bot.state.coin), product_id)

    try:
        stop_order = coinbase_stop_limit_order(
            product_id=product_id,
            side="SELL" if not bot.state.is_short else "BUY",
            base_size=base_size,
            stop_price=stop_price,
            limit_price=limit_price,
        )
        stop_order_id = coinbase_order_id(stop_order)
        bot.state.active_stop_order_id = stop_order_id
        track_order(
            bot,
            stop_order_id,
            entry_order.symbol,
            product_id,
            "SELL" if not bot.state.is_short else "BUY",
            "STOP",
            "stop_limit",
            price=stop_price,
            base_size=base_size,
            reason=f"{exit_mode} native stop",
            details={"entry_order_id": entry_order.order_id},
        )
        bot.journal(
            entry_order.symbol,
            "INFO",
            f"Native stop-limit submitted {stop_order_id} via {exit_mode} stop",
            stop_price,
            {"entry_order_id": entry_order.order_id, "stop_order": stop_order},
        )
    except Exception as e:
        bot.state.stop_price = stop_price
        bot.journal(
            entry_order.symbol,
            "WARNING",
            f"Native stop failed, using simulated stop at {stop_price:.6f}",
            stop_price,
            {"error": str(e)},
        )

def sync_native_stop_fill(bot) -> None:
    if not bot.state.active_stop_order_id or not bot.state.active_symbol:
        return
    stop_order_id = bot.state.active_stop_order_id
    fill = coinbase_reconcile_order(stop_order_id)
    if fill["filled_size"] <= 0:
        return

    symbol = bot.state.active_symbol
    filled_price = fill["average_price"] or bot.state.last_price or 0.0
    paper_sell(
        bot,
        symbol,
        filled_price,
        f"NATIVE STOP filled {stop_order_id}",
        quantity_override=fill["filled_size"],
        fee_override=fill["total_fee"],
        exchange_order_id=stop_order_id,
        exchange_order_status=fill["status"],
        exchange_average_filled_price=filled_price,
    )
    if abs(bot.state.coin) <= 0 and not bot.state.positions:
        bot.state.active_stop_order_id = None

# ─── Sync functions ───────────────────────────────────────────────

def sync_oanda_positions(bot) -> None:
    if not should_oanda_demo_trade(bot.state):
        with bot.lock:
            if bot.state.positions:
                bot.state.positions = {}
                bot.state.active_symbol = None
                bot.state.coin = 0.0
                bot.state.is_short = False
                bot.save_state()
        return

    try:
        account_id = urllib.parse.quote(oanda_account_id())
        data = oanda_request(f"/v3/accounts/{account_id}/openPositions")
        positions = data.get("positions", [])

        with bot.lock:
            bot.state.positions = {}
            bot.state.coin = 0.0
            bot.state.active_symbol = None
            bot.state.is_short = False

        for position in positions:
            instrument = position.get("instrument", "")
            symbol = instrument.replace("_", "")
            units = int(position.get("short", {}).get("units", 0))
            if units > 0:
                price = float(position.get("short", {}).get("averagePrice", 0.0))
                if price <= 0:
                    price = bot.state.last_price or 0.0
                with bot.lock:
                    bot.state.positions[symbol] = {
                        "quantity": -units,
                        "entry_price": price,
                        "highest_price": price,
                        "opened_at": now_iso(),
                        "trade_id": position.get("tradeID"),
                        "is_short": True,
                        "entry_time": time.time(),
                    }
                    bot.state.active_symbol = symbol
                    bot.state.entry_price = price
                    bot.state.coin = -units
                    bot.state.is_short = True
                bot.journal(symbol, "INFO", f"Synced OANDA SHORT position: {symbol} {units} @ {price}", price)
            else:
                units = int(position.get("long", {}).get("units", 0))
                if units > 0:
                    price = float(position.get("long", {}).get("averagePrice", 0.0))
                    if price <= 0:
                        price = bot.state.last_price or 0.0
                    with bot.lock:
                        bot.state.positions[symbol] = {
                            "quantity": units,
                            "entry_price": price,
                            "highest_price": price,
                            "opened_at": now_iso(),
                            "trade_id": position.get("tradeID"),
                            "is_short": False,
                            "entry_time": time.time(),
                        }
                        bot.state.active_symbol = symbol
                        bot.state.entry_price = price
                        bot.state.coin = units
                        bot.state.is_short = False
                    bot.journal(symbol, "INFO", f"Synced OANDA BUY position: {symbol} {units} @ {price}", price)

        if positions:
            logger.info(f"Synced {len(positions)} positions from OANDA")
        else:
            logger.info("No OANDA positions to sync")
        bot.save_state()
    except Exception as exc:
        logger.warning(f"Failed to sync OANDA positions: {exc}")

def get_oanda_account_summary(bot) -> dict[str, Any]:
    if not should_oanda_demo_trade(bot.state):
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
        return {"ok": False, "error": str(e)}

def sync_live_balance_always(bot) -> None:
    """Sync Coinbase balance even if there are open positions."""
    try:
        with bot.lock:
            settings = dict(bot.state.settings)

        if not settings.get("live_trading_enabled"):
            return

        quote_currency = settings.get("quote_currency", "GBP")
        actual_balance = coinbase_available_balance(quote_currency)

        if actual_balance <= 0:
            return

        with bot.lock:
            bot.state.cash = actual_balance
            bot.state.settings['starting_cash'] = actual_balance

            current_day = today_key()
            if bot.state.day_start_date != current_day:
                bot.state.day_start_equity = bot.equity(bot.state.last_price)
                bot.state.day_start_date = current_day
                bot.state.peak_equity = bot.state.day_start_equity

            bot.save_state()
    except Exception as e:
        pass

def sync_live_balance_from_coinbase(bot) -> dict[str, Any]:
    with bot.lock:
        settings = dict(bot.state.settings)
        current_coin = bot.state.coin

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

    with bot.lock:
        bot.state.settings["starting_cash"] = available_cash
        bot.state.cash = available_cash
        bot.state.coin = 0.0
        bot.state.active_symbol = None
        bot.state.entry_price = None
        bot.state.active_stop_order_id = None
        bot.state.day_start_equity = available_cash
        bot.state.day_start_date = today_key()
        bot.state.peak_equity = available_cash
        bot.state.last_signal = f"Synced {quote_currency} balance from Coinbase"
        bot.save_state()

    return {
        "ok": True,
        "quote_currency": quote_currency,
        "available_cash": round(available_cash, 8),
    }

def sync_paper_balance_from_oanda(bot) -> dict[str, Any]:
    with bot.lock:
        settings = dict(bot.state.settings)

    if settings.get("asset_class") != "forex" or settings.get("exchange") != "oanda_demo":
        raise RuntimeError("OANDA balance sync requires Asset Class = Forex and Exchange = OANDA demo.")

    summary = oanda_account_summary()
    account = summary.get("account", {})
    balance = float(account.get("balance", 0.0))
    currency = str(account.get("currency") or settings.get("quote_currency", "USD")).upper()

    account_id = urllib.parse.quote(oanda_account_id())
    positions_data = oanda_request(f"/v3/accounts/{account_id}/openPositions")
    oanda_positions = positions_data.get("positions", [])

    with bot.lock:
        bot.state.positions = {}
        bot.state.coin = 0.0
        bot.state.active_symbol = None
        bot.state.is_short = False
        bot.state.entry_price = None
        bot.state.highest_price = None
        bot.state.stop_price = None
        bot.state.target_price = None
        bot.state.active_stop_order_id = None
        bot.state.partial_take_profit_done = False

        bot.state.cash = balance
        bot.state.settings["starting_cash"] = balance
        bot.state.settings["quote_currency"] = currency
        bot.state.day_start_equity = balance
        bot.state.peak_equity = balance

        for position in oanda_positions:
            instrument = position.get("instrument", "")
            symbol = instrument.replace("_", "")

            short_units = int(position.get("short", {}).get("units", 0))
            long_units = int(position.get("long", {}).get("units", 0))

            if short_units > 0:
                avg_price = float(position.get("short", {}).get("averagePrice", 0.0))
                bot.state.positions[symbol] = {
                    "quantity": -short_units,
                    "entry_price": avg_price,
                    "highest_price": avg_price,
                    "is_short": True,
                    "opened_at": now_iso(),
                    "entry_time": time.time(),
                }
                bot.state.coin = -short_units
                bot.state.active_symbol = symbol
                bot.state.is_short = True
                bot.state.entry_price = avg_price

            elif long_units > 0:
                avg_price = float(position.get("long", {}).get("averagePrice", 0.0))
                bot.state.positions[symbol] = {
                    "quantity": long_units,
                    "entry_price": avg_price,
                    "highest_price": avg_price,
                    "is_short": False,
                    "opened_at": now_iso(),
                    "entry_time": time.time(),
                }
                bot.state.coin = long_units
                bot.state.active_symbol = symbol
                bot.state.is_short = False
                bot.state.entry_price = avg_price

        bot.state.last_signal = f"Synced {currency} balance from OANDA: {balance:.2f}"
        bot.journal("", "INFO", bot.state.last_signal, bot.state.last_price)
        bot.save_state()

    return {
        "ok": True,
        "quote_currency": currency,
        "available_cash": round(balance, 8),
        "positions": len(oanda_positions),
        "balance": balance
    }

def close_position_manual(bot, symbol: str, mode: str = "profit_only") -> dict[str, Any]:
    symbol = normalize_forex_symbol(symbol or "").upper()
    mode = str(mode or "profit_only").lower()

    if mode not in {"profit_only", "force"}:
        raise RuntimeError("Invalid close mode.")

    with bot.lock:
        settings = dict(bot.state.settings)
        position = dict((bot.state.positions or {}).get(symbol, {}))
        single_active = bot.state.active_symbol == symbol and abs(bot.state.coin or 0.0) > 0

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
        with bot.lock:
            entry_price = float(bot.state.entry_price or 0.0)
            quantity = float(bot.state.coin or 0.0)
            is_short = bot.state.is_short

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

    if should_oanda_demo_trade(bot.state):
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

            with bot.lock:
                bot.state.positions.pop(symbol, None)
                bot.state.active_symbol = None
                bot.state.coin = 0.0
                bot.state.is_short = False
                bot.state.entry_price = None
                bot.state.highest_price = None
                bot.state.stop_price = None
                bot.state.target_price = None
                bot.state.active_stop_order_id = None
                bot.state.partial_take_profit_done = False
                bot.state.last_price = fill_price
                bot.state.last_action_time = time.time()
                bot.state.last_signal = f"{reason}: {symbol}"
                bot.save_state()

            trade = Trade(
                time=now_iso(),
                side="SELL" if not is_short else "BUY",
                symbol=symbol,
                price=fill_price,
                quantity=filled_units,
                cash_after=bot.state.cash,
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
                regime=bot.state.current_regime.regime if bot.state.current_regime else None,
            )
            bot.record_trade(trade)

            bot.journal(symbol, "INFO", f"OANDA manual close: {reason} at {fill_price:.6f}", fill_price, {"pnl": pnl})

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

    else:
        if is_short:
            paper_buy(bot, symbol, price, reason, None, is_short=True)
        else:
            paper_sell(bot, symbol, price, reason, None)

        with bot.lock:
            bot.state.last_signal = f"{reason}: {symbol}"
            bot.save_state()

        return {
            "ok": True,
            "symbol": symbol,
            "mode": mode,
            "price": price,
            "estimated_pnl": pnl,
            "message": f"Paper position closed at {price:.6f}",
            "exchange": "Paper"
        }