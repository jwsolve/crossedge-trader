#!/usr/bin/env python3
"""
Coinbase Advanced Trade API integration.
"""

import os
import json
import base64
import math
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta, timezone

# ─── cryptography ────────────────────────────────────────────────
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, utils
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    hashes = serialization = ec = utils = None
    CRYPTOGRAPHY_AVAILABLE = False

# ─── utils ──────────────────────────────────────────────────────
from utils import fetch_json, BASE_DIR, DOTENV_LOADED_KEYS

# ─── Constants ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# ─── Product cache ─────────────────────────────────────────────
_product_cache: dict[str, dict] = {}

def get_product_details(product_id: str) -> dict:
    """Get product details with caching."""
    if product_id in _product_cache:
        return _product_cache[product_id]
    try:
        url = f"https://api.exchange.coinbase.com/products/{product_id}"
        data = fetch_json(url)
        _product_cache[product_id] = data
        return data
    except Exception as e:
        # Return empty dict on error
        return {}

def coinbase_price_precision(product_id: str) -> int:
    """Get the price precision for a Coinbase product."""
    details = get_product_details(product_id)
    quote_increment = details.get('quote_increment', '0.01')
    if '.' in quote_increment:
        precision = len(quote_increment.split('.')[1].rstrip('0'))
    else:
        precision = 0
    return min(max(precision, 2), 8)

def coinbase_size_precision(product_id: str) -> int:
    """Get the size precision for a Coinbase product."""
    details = get_product_details(product_id)
    base_increment = details.get('base_increment', '0.00000001')
    if '.' in base_increment:
        precision = len(base_increment.split('.')[1].rstrip('0'))
    else:
        precision = 0
    return min(max(precision, 2), 8)

def coinbase_min_order_size(product_id: str) -> float:
    """Get the minimum order size for a Coinbase product."""
    details = get_product_details(product_id)
    base_min_size = details.get('base_min_size', '0.00000001')
    try:
        return float(base_min_size)
    except:
        return 0.00000001

def coinbase_round_price(price: float, product_id: str) -> float:
    """Round price to the nearest multiple of quote_increment."""
    details = get_product_details(product_id)
    quote_increment = details.get('quote_increment', '0.01')
    try:
        increment = float(quote_increment)
    except:
        increment = 0.01
    if increment > 0:
        return math.floor(price / increment) * increment
    return price

def coinbase_round_size(size: float, product_id: str) -> float:
    """Round size down to the nearest multiple of base_increment."""
    details = get_product_details(product_id)
    base_increment = details.get('base_increment', '0.00000001')
    try:
        increment = float(base_increment)
    except:
        increment = 0.00000001
    if increment > 0:
        rounded = math.floor(size / increment) * increment
    else:
        rounded = size
    # Ensure minimum order size
    min_size = coinbase_min_order_size(product_id)
    if rounded < min_size:
        rounded = min_size
    return rounded

# ─── Key management ─────────────────────────────────────────────

def resolve_local_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path

def coinbase_private_key_configured() -> bool:
    if os.environ.get("COINBASE_API_PRIVATE_KEY", "").strip():
        return True
    key_file = os.environ.get("COINBASE_API_PRIVATE_KEY_FILE", "").strip()
    return bool(key_file and resolve_local_path(key_file).is_file())

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

# ─── JWT generation ─────────────────────────────────────────────

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

# ─── API request ────────────────────────────────────────────────

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

# ─── Balance ─────────────────────────────────────────────────────

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

# ─── Orders ──────────────────────────────────────────────────────

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
        "stop_limit_gtc": {
            "base_size": decimal_text(base_size, 10),
            "limit_price": decimal_text(limit_price, 8),
            "stop_price": decimal_text(stop_price, 8),
            "stop_direction": stop_direction,
        }
    }

    body = {
        "client_order_id": str(uuid.uuid4()),
        "product_id": product_id,
        "side": side.upper(),
        "order_configuration": order_configuration,
    }

    return coinbase_api_request("POST", "/api/v3/brokerage/orders", body)

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

def coinbase_order_id(response: dict[str, Any]) -> str:
    if response.get("order_id"):
        return str(response["order_id"])
    if response.get("success_response", {}).get("order_id"):
        return str(response["success_response"]["order_id"])
    if response.get("order", {}).get("order_id"):
        return str(response["order"]["order_id"])
    raise RuntimeError(f"Coinbase order response did not include an order id: {response}")

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

# ─── Candles & Ticker ────────────────────────────────────────────

def fetch_coinbase_candles(symbol: str, quote_currency: str, granularity: int, candle_count: int) -> list:
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
    from types import SimpleNamespace
    candles = [
        SimpleNamespace(
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
    from utils import parse_watchlist, strategy_minimum_candles
    from backtest import run_backtest_for_symbol

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

                candles = fetch_coinbase_candles(symbol, quote, granularity, candle_count)
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

# ─── Live Market Guard ───────────────────────────────────────────

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

    candles = fetch_coinbase_candles(symbol, quote_currency, granularity, candle_count)
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

def coinbase_auth_check() -> dict[str, Any]:
    data = coinbase_api_request("GET", "/api/v3/brokerage/accounts?limit=1")
    return {
        "ok": True,
        "accounts_visible": len(data.get("accounts", [])),
        "has_next": bool(data.get("has_next")),
    }
