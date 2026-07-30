#!/usr/bin/env python3
"""
OANDA API integration.
"""

import os
import json
import urllib.request
import urllib.parse
import threading
import time
from typing import Any, Optional, List, Dict, Callable

from utils import normalize_forex_symbol, fetch_json, FOREX_BASE_RATES

# ─── Authentication ──────────────────────────────────────────────

def oanda_account_id() -> str:
    return os.environ.get("OANDA_ACCOUNT_ID", "").strip()

def oanda_api_token() -> str:
    return os.environ.get("OANDA_API_TOKEN", "").strip()

def oanda_is_configured() -> bool:
    return bool(oanda_account_id() and oanda_api_token())

def oanda_is_practice() -> bool:
    env = os.environ.get("OANDA_ENV", "practice").strip().lower()
    return env == "practice"

def oanda_api_base() -> str:
    if os.environ.get("OANDA_API_BASE", "").strip():
        return os.environ["OANDA_API_BASE"].strip().rstrip("/")
    if oanda_is_practice():
        return "https://api-fxpractice.oanda.com"
    return "https://api-fxtrade.oanda.com"

def oanda_demo_orders_armed() -> bool:
    if not oanda_is_configured():
        return False
    if os.environ.get("OANDA_DEMO_TRADING_ENABLED", "").strip().lower() != "true":
        return False
    return True

def oanda_demo_status_message() -> str:
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

def oanda_instrument(symbol: str) -> str:
    symbol = normalize_forex_symbol(symbol)
    if len(symbol) != 6:
        raise RuntimeError("OANDA forex pairs must be six-letter symbols like EURUSD")
    return f"{symbol[:3]}_{symbol[3:]}"

# ─── API Request ─────────────────────────────────────────────────

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

def oanda_account_summary() -> dict[str, Any]:
    account_id = urllib.parse.quote(oanda_account_id())
    return oanda_request(f"/v3/accounts/{account_id}/summary")

# ─── Order filling ──────────────────────────────────────────────

def oanda_order_fill(response: dict[str, Any]) -> dict[str, Any]:
    fill = response.get("orderFillTransaction")
    cancel = response.get("orderCancelTransaction") or response.get("orderCreateTransaction")
    if not fill:
        reason = cancel.get("reason") if isinstance(cancel, dict) else "not filled"
        raise RuntimeError(f"OANDA demo order was not filled: {reason}")

    units = abs(float(fill.get("units", 0.0)))
    price = float(fill.get("price", 0.0))
    commission = abs(float(fill.get("commission", 0.0)))
    trade_id = fill.get("tradeOpened", {}).get("tradeID") or fill.get("tradesClosed", [{}])[0].get("tradeID")
    return {
        "order_id": str(fill.get("id") or response.get("lastTransactionID") or ""),
        "trade_id": str(trade_id or ""),
        "status": "FILLED",
        "units": units,
        "price": price,
        "commission": commission,
        "raw": response,
    }

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

def oanda_decimal(value: float, symbol: str) -> str:
    places = 3 if normalize_forex_symbol(symbol).endswith("JPY") else 5
    return f"{value:.{places}f}"

# ─── Candles ─────────────────────────────────────────────────────

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
    from datetime import datetime, timezone
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1]
    if "." in raw:
        head, fraction = raw.split(".", 1)
        raw = f"{head}.{fraction[:6].ljust(6, '0')}"
    parsed = datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())

def fetch_oanda_demo_candles(symbol: str, granularity: int, candle_count: int) -> list:
    from types import SimpleNamespace
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
    candles: list = []
    for item in data.get("candles", []):
        mid = item.get("mid") or {}
        if not mid:
            continue
        candles.append(
            SimpleNamespace(
                time=parse_oanda_time(str(item["time"])),
                open=float(mid["o"]),
                high=float(mid["h"]),
                low=float(mid["l"]),
                close=float(mid["c"]),
                volume=float(item.get("volume", 0.0)),
            )
        )
    return sorted(candles, key=lambda item: item.time)[-count:]

# ─── Streaming ───────────────────────────────────────────────────

def oanda_stream_pricing(
    symbols: list[str],
    on_price: Callable,
    on_error: Callable = None,
    stop_event: threading.Event = None,
) -> None:
    if not oanda_is_configured():
        error = "OANDA not configured - cannot start stream"
        if on_error:
            on_error(error)
        raise RuntimeError(error)

    account_id = urllib.parse.quote(oanda_account_id())
    instruments = ",".join(oanda_instrument(sym) for sym in symbols)

    base = oanda_api_base()
    stream_base = base.replace("api-", "stream-")
    url = f"{stream_base}/v3/accounts/{account_id}/pricing/stream"

    params = {
        "instruments": instruments,
        "snapshot": "True",
    }
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}"

    request = urllib.request.Request(
        full_url,
        headers={
            "Authorization": f"Bearer {oanda_api_token()}",
            "Accept": "application/json",
            "User-Agent": "auxo-trading-bot/1.0",
            "Connection": "keep-alive",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=None) as response:
            buffer = ""
            while True:
                if stop_event and stop_event.is_set():
                    break

                chunk = response.read(1024).decode("utf-8")
                if not chunk:
                    break

                buffer += chunk

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        if data.get("type") == "PRICE":
                            on_price(data)
                        elif data.get("type") == "HEARTBEAT":
                            pass
                        elif data.get("type") == "SNAPSHOT":
                            for price_data in data.get("prices", []):
                                on_price(price_data)
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        pass

    except Exception as e:
        if on_error:
            on_error(str(e))
        raise

# ─── Auth Check (for web_server) ─────────────────────────────────

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
