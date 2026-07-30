# web_server.py
"""
HTTP server and request handlers for the Auxo bot dashboard.
"""

import json
import urllib.parse
import os
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from bot import PaperBot
from utils import now_iso, today_key
from backtest import run_backtest, run_optimizer, run_walk_forward
from coinbase_api import coinbase_products_for_quote, coinbase_quote_comparison, coinbase_auth_check
from oanda_api import oanda_auth_check
from strategy_creator import STRATEGY_CREATOR_AVAILABLE
from http.server import ThreadingHTTPServer

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

def parse_json_body(handler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8"))

class BotRequestHandler(SimpleHTTPRequestHandler):
    bot: PaperBot

    def __init__(self, *args, **kwargs):
        self.bot = BotRequestHandler.bot
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def _check_auth(self) -> bool:
        token = os.environ.get("BOT_API_TOKEN", "")
        if not token:
            return True
        header = self.headers.get("Authorization", "")
        expected = f"Bearer {token}"
        if header == expected:
            return True
        self.send_json({"ok": False, "error": "Unauthorized"}, HTTPStatus.UNAUTHORIZED)
        return False

    def do_GET(self):
        if self.path.startswith("/api/") and not self._check_auth():
            return
        try:
            if self.path == "/api/status":
                self.send_json(self.bot.snapshot())
                return
            if self.path == "/api/status-light":
                self.send_json({
                    "running": self.bot.state.running,
                    "equity": self.bot.equity(self.bot.state.last_price),
                    "cash": self.bot.state.cash,
                    "last_price": self.bot.state.last_price,
                    "positions": len(self.bot.state.positions),
                    "last_signal": self.bot.state.last_signal,
                    "last_error": self.bot.state.last_error,
                })
                return
            if self.path == "/api/diagnostics":
                self.send_json(diagnostics())
                return
            if self.path.startswith("/api/candles"):
                self.handle_candles_request()
                return
            if self.path == "/api/coinbase-auth-check":
                self.send_json(coinbase_auth_check())
                return
            if self.path == "/api/oanda-auth-check":
                self.send_json(oanda_auth_check())
                return
            if self.path == "/api/coinbase-gbp-products":
                self.send_json(coinbase_products_for_quote("GBP"))
                return
            if self.path.startswith("/api/coinbase-products"):
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                quote = query.get("quote", ["GBP"])[0]
                self.send_json(coinbase_products_for_quote(quote))
                return
            if self.path == "/api/strategy-dashboard":
                result = self.bot.get_strategy_dashboard()
                self.send_json(result)
                return
            if self.path == "/api/sync-oanda-balance":
                summary = self.bot.get_oanda_account_summary()
                if summary.get("ok"):
                    with self.bot.lock:
                        self.bot.state.cash = summary.get("balance", 0)
                        self.bot.state.positions = {}
                        self.bot.state.coin = 0.0
                        self.bot.state.active_symbol = None
                        for pos in summary.get("positions", []):
                            symbol = pos.get("symbol")
                            units = pos.get("units", 0)
                            avg_price = pos.get("average_price", 0)
                            is_short = pos.get("side") == "SHORT"
                            if units != 0:
                                self.bot.state.positions[symbol] = {
                                    "quantity": -units if is_short else units,
                                    "entry_price": avg_price,
                                    "highest_price": avg_price,
                                    "is_short": is_short,
                                    "opened_at": now_iso(),
                                    "entry_time": time.time(),
                                }
                        self.bot.save_state()
                    self.send_json({
                        "ok": True,
                        "balance": summary.get("balance"),
                        "equity": summary.get("equity"),
                        "positions": summary.get("positions_count"),
                        "pnl": summary.get("total_pnl")
                    })
                else:
                    self.send_json({"ok": False, "error": summary.get("error")})
                return
            # static files
            if self.path.endswith(('.css', '.js', '.json', '.png', '.jpg', '.svg')):
                self.serve_static_file()
                return
            if not self.path.startswith("/api/"):
                self.send_index()
                return
            self.send_error(HTTPStatus.NOT_FOUND, f"Endpoint not found: {self.path}")
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def handle_candles_request(self):
        try:
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            symbol = params.get('symbol', ['BTC'])[0].upper()
            granularity = int(params.get('granularity', ['3600'])[0])
            count = int(params.get('count', ['200'])[0])
            quote_currency = params.get('quote', ['GBP'])[0].upper()
            settings = self.bot.state.settings
            exchange = settings.get('exchange', 'coinbase')
            asset_class = settings.get('asset_class', 'crypto')
            candles = fetch_candles(
                exchange=exchange,
                symbol=symbol,
                quote_currency=quote_currency,
                granularity=granularity,
                candle_count=count,
                asset_class=asset_class
            )
            candle_data = [
                {
                    'time': c.time,
                    'open': c.open,
                    'high': c.high,
                    'low': c.low,
                    'close': c.close,
                    'volume': c.volume
                }
                for c in candles
            ]
            current_price = self.bot.state.last_price
            chart_row = next(
                (row for row in self.bot.state.scan_rows if row.get('symbol') == symbol),
                {}
            )
            self.send_json({
                'ok': True,
                'symbol': symbol,
                'granularity': granularity,
                'count': len(candle_data),
                'candles': candle_data,
                'current_price': current_price,
                'support': chart_row.get('support'),
                'resistance': chart_row.get('resistance'),
            })
        except Exception as e:
            self.send_json({'ok': False, 'error': str(e)}, HTTPStatus.BAD_REQUEST)

    def serve_static_file(self):
        path = self.path.lstrip('/')
        file_path = WEB_DIR / path
        if not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = {
            '.css': 'text/css', '.js': 'application/javascript',
            '.json': 'application/json', '.png': 'image/png',
            '.jpg': 'image/jpeg', '.svg': 'image/svg+xml'
        }.get(path.suffix, 'application/octet-stream')
        body = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def send_index(self):
        index_path = WEB_DIR / "index.html"
        if not index_path.exists():
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Dashboard missing at {index_path}")
            return
        body = index_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data, status=HTTPStatus.OK):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if not self._check_auth():
            return
        try:
            if self.path == "/api/start":
                self.bot.start()
                self.send_json({"ok": True})
                return
            if self.path == "/api/stop":
                self.bot.stop()
                self.send_json({"ok": True})
                return
            if self.path == "/api/reset":
                self.bot.reset()
                self.send_json({"ok": True})
                return
            if self.path == "/api/verify-oanda-positions":
                self.send_json(self.bot.verify_oanda_positions())
                return
            if self.path == "/api/sync-live-balance":
                self.send_json(self.bot.sync_live_balance_from_coinbase())
                return
            if self.path == "/api/sync-oanda-balance":
                self.send_json(self.bot.sync_paper_balance_from_oanda())
                return
            if self.path == "/api/settings":
                self.bot.update_settings(parse_json_body(self))
                self.send_json({"ok": True})
                return
            if self.path == "/api/backtest":
                payload = parse_json_body(self)
                settings = {**self.bot.snapshot()["settings"], **payload}
                self.send_json(run_backtest(settings))
                return
            if self.path == "/api/optimize":
                payload = parse_json_body(self)
                settings = {**self.bot.snapshot()["settings"], **payload}
                self.send_json(run_optimizer(settings))
                return
            if self.path == "/api/walk-forward":
                payload = parse_json_body(self)
                settings = {**self.bot.snapshot()["settings"], **payload}
                self.send_json(run_walk_forward(settings))
                return
            if self.path == "/api/close-position":
                payload = parse_json_body(self)
                self.send_json(
                    self.bot.close_position_manual(
                        symbol=str(payload.get("symbol", "")),
                        mode=str(payload.get("mode", "profit_only")),
                    )
                )
                return
            if self.path == "/api/quote-comparison":
                payload = parse_json_body(self)
                settings = {**self.bot.snapshot()["settings"], **payload}
                self.send_json(coinbase_quote_comparison(settings))
                return
            if self.path == "/api/export-db":
                path = BASE_DIR / f"export_{today_key()}.json"
                self.bot.db.export_json(path)
                self.send_json({"ok": True, "path": str(path)})
                return
            if self.path == "/api/evolve-strategies":
                generations = int(parse_json_body(self).get('generations', 50))
                result = self.bot.evolve_strategies(generations)
                self.send_json(result)
                return
            if self.path == "/api/apply-strategy":
                payload = parse_json_body(self)
                candles = payload.get('candles', [])
                result = self.bot.apply_strategy_signal(candles)
                self.send_json(result)
                return
            if self.path == "/api/backfill-tpsl":
                self.bot.backfill_tpsl_from_positions()
                self.send_json({"ok": True, "message": "TP/SL backfill completed"})
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
