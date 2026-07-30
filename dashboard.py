#!/usr/bin/env python3
"""
Auxo Dashboard – Separate Flask app that reads directly from OANDA.
No trading logic – just a clean UI for monitoring.
"""

from flask import Flask, jsonify, request, send_from_directory, render_template_string
import os
import urllib.request
import urllib.parse
import json
import base64
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ─── LOGGING ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dashboard')

# ─── FLASK APP ──────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-me')

# ─── PATHS ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

# ─── OANDA CONFIG ──────────────────────────────────────────────────
def oanda_account_id() -> str:
    return os.environ.get("OANDA_ACCOUNT_ID", "").strip()

def oanda_api_token() -> str:
    return os.environ.get("OANDA_API_TOKEN", "").strip()

def oanda_is_configured() -> bool:
    return bool(oanda_account_id() and oanda_api_token())

def oanda_api_base() -> str:
    env = os.environ.get("OANDA_ENV", "practice").strip().lower()
    if env == "practice":
        return "https://api-fxpractice.oanda.com"
    return "https://api-fxtrade.oanda.com"

def oanda_instrument(symbol: str) -> str:
    """Convert symbol to OANDA instrument format."""
    symbol = symbol.upper().replace("/", "").replace("-", "").replace("_", "")
    if len(symbol) != 6:
        raise ValueError(f"OANDA forex pairs must be six-letter symbols like EURUSD, got {symbol}")
    return f"{symbol[:3]}_{symbol[3:]}"

def oanda_request(path: str, params: Optional[dict] = None, timeout: int = 10) -> dict:
    """Make a request to OANDA API."""
    if not oanda_is_configured():
        raise RuntimeError("OANDA not configured")

    query = urllib.parse.urlencode(params or {})
    url = f"{oanda_api_base()}{path}"
    if query:
        url = f"{url}?{query}"

    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {oanda_api_token()}",
            "Accept": "application/json",
            "User-Agent": "auxo-dashboard/1.0",
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OANDA API error {e.code}: {body}") from e

# ─── OANDA DATA FETCHING ──────────────────────────────────────────

def get_oanda_account_summary() -> dict[str, Any]:
    """Get complete account summary from OANDA."""
    if not oanda_is_configured():
        return {"ok": False, "error": "OANDA not configured"}

    try:
        account_id = urllib.parse.quote(oanda_account_id())

        # Get account summary
        summary = oanda_request(f"/v3/accounts/{account_id}/summary")
        account = summary.get("account", {})

        # Get open positions
        positions_data = oanda_request(f"/v3/accounts/{account_id}/openPositions")
        positions = positions_data.get("positions", [])

        # Get pending orders (for TP/SL)
        orders_data = oanda_request(f"/v3/accounts/{account_id}/orders")
        orders = orders_data.get("orders", [])

        # Build TP/SL map
        order_map = {}
        for order in orders:
            instrument = order.get("instrument", "")
            if not instrument:
                continue
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

        # Get current prices for positions
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

        # Build position details
        position_details = []
        for position in positions:
            instrument = position.get("instrument", "")
            symbol = instrument.replace("_", "")

            # Find current price
            current_price = None
            for price_data in prices:
                if price_data.get("instrument") == instrument:
                    bid = float(price_data.get("bids", [{}])[0].get("price", 0))
                    ask = float(price_data.get("asks", [{}])[0].get("price", 0))
                    if bid > 0 and ask > 0:
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
                    "current_price": current_price if current_price else long_avg,
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
                    "current_price": current_price if current_price else short_avg,
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
            "has_positions": len(position_details) > 0,
        }

    except Exception as e:
        logger.error(f"Failed to get OANDA account summary: {e}")
        return {"ok": False, "error": str(e)}

# ─── ROUTES ──────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the dashboard."""
    return send_from_directory(WEB_DIR, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files from web directory."""
    return send_from_directory(WEB_DIR, filename)

@app.route('/api/health')
def api_health():
    """Health check."""
    return jsonify({
        "status": "ok",
        "timestamp": str(datetime.now()),
        "server": "Auxo Dashboard",
        "oanda_configured": oanda_is_configured()
    })

@app.route('/api/status')
def api_status():
    """Get complete OANDA account status."""
    try:
        summary = get_oanda_account_summary()
        if not summary.get("ok"):
            return jsonify({"ok": False, "error": summary.get("error")}), 500

        # Format for the frontend
        positions = []
        for pos in summary.get("positions", []):
            is_short = pos.get("side") == "SHORT"
            entry = pos.get("average_price", 0)
            current = pos.get("current_price", entry)
            units = pos.get("units", 0)
            unrealized_pnl = pos.get("unrealized_pnl", 0)

            if is_short:
                pnl_pct = ((entry - current) / entry) * 100 if entry > 0 else 0
            else:
                pnl_pct = ((current - entry) / entry) * 100 if entry > 0 else 0

            positions.append({
                "symbol": pos.get("symbol", ""),
                "quantity": units,
                "entry_price": entry,
                "current_price": current,
                "stop_price": pos.get("stop_loss"),
                "target_price": pos.get("take_profit"),
                "stop": pos.get("stop_loss"),
                "target": pos.get("take_profit"),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "unrealized_pnl_pct": round(pnl_pct, 2),
                "is_short": is_short,
                "opened_at": datetime.now().isoformat(),
            })

        return jsonify({
            "ok": True,
            "running": True,  # Dashboard doesn't track running state
            "cash": summary.get("balance", 0),
            "equity": summary.get("equity", 0),
            "total_pnl": summary.get("total_pnl", 0),
            "currency": summary.get("currency", "GBP"),
            "positions": positions,
            "oanda_connected": True,
            "oanda_balance": summary.get("balance"),
            "oanda_equity": summary.get("equity"),
            "oanda_positions_count": summary.get("positions_count", 0),
        })

    except Exception as e:
        logger.error(f"Error in /api/status: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/api/trades', methods=['GET'])
def api_trades():
    """Get recent trades from OANDA."""
    try:
        # OANDA doesn't have a direct trades endpoint in the same way
        # You'd need to use the transactions endpoint
        # For simplicity, return empty or use your database
        return jsonify({
            "ok": True,
            "trades": [],
            "message": "Trade history from OANDA via dashboard"
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/api/candles')
def api_candles():
    """Get candles for a symbol from OANDA."""
    try:
        symbol = request.args.get('symbol', 'EURUSD').upper()
        granularity = request.args.get('granularity', 'H1')
        count = int(request.args.get('count', 200))

        # Map granularity
        granularity_map = {
            '60': 'M1', '300': 'M5', '900': 'M15',
            '3600': 'H1', '21600': 'H6', '86400': 'D'
        }
        if granularity in granularity_map:
            granularity = granularity_map[granularity]

        instrument = oanda_instrument(symbol)
        data = oanda_request(
            f"/v3/instruments/{urllib.parse.quote(instrument)}/candles",
            {
                "price": "M",
                "granularity": granularity,
                "count": count,
            }
        )

        candles = []
        for item in data.get("candles", []):
            mid = item.get("mid", {})
            candles.append({
                "time": int(datetime.fromisoformat(item["time"].replace("Z", "+00:00")).timestamp()),
                "open": float(mid.get("o", 0)),
                "high": float(mid.get("h", 0)),
                "low": float(mid.get("l", 0)),
                "close": float(mid.get("c", 0)),
                "volume": float(item.get("volume", 0)),
            })

        return jsonify({
            "ok": True,
            "symbol": symbol,
            "granularity": granularity,
            "count": len(candles),
            "candles": candles
        })

    except Exception as e:
        logger.error(f"Error fetching candles: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('DASHBOARD_PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
