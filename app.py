#!/usr/bin/env python3
"""
Auxo Bot - Flask Application for Namecheap Shared Hosting
Full trading bot with API endpoints and dashboard serving.
"""

from flask import Flask, jsonify, request, send_from_directory
import os
import sys
import json
import threading
import logging
from datetime import datetime
from pathlib import Path

# ─── LOGGER ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('auxo')

# ─── FLASK APP ──────────────────────────────────────────────────────
app = Flask(__name__)

# ─── CORS SUPPORT ──────────────────────────────────────────────────
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# ─── PATHS ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

# ─── DUMMY BOT CLASS (for when import fails) ──────────────────────
class DummyBot:
    """Dummy bot used when bot_server import fails."""
    def __init__(self):
        self.running = False
        self.state = type('obj', (object,), {'running': False, 'cash': 0, 'positions': {}, 'settings': {}})()
    def snapshot(self):
        return {"error": "Bot not available", "running": False}
    def start(self):
        pass
    def stop(self):
        pass
    def reset(self):
        pass
    def update_settings(self, data):
        pass
    def sync_paper_balance_from_oanda(self):
        return {"ok": False, "error": "Bot not available"}
    def close_position_manual(self, symbol, mode):
        return {"ok": False, "error": "Bot not available"}
    def verify_oanda_positions(self):
        return {"ok": False, "error": "Bot not available"}

# ─── IMPORT BOT SERVER ─────────────────────────────────────────────
try:
    from bot_server import PaperBot, logger as bot_logger
    from bot_server import run_backtest, run_optimizer, run_walk_forward
    from bot_server import coinbase_quote_comparison, coinbase_products_for_quote
    from bot_server import oanda_auth_check, fetch_candles
    logger.info("Bot server imported successfully")
    BOT_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import bot_server: {e}")
    BOT_AVAILABLE = False
    # Define fallback functions
    def run_backtest(settings): return {"ok": False, "error": "Bot not available"}
    def run_optimizer(settings): return {"ok": False, "error": "Bot not available"}
    def run_walk_forward(settings): return {"ok": False, "error": "Bot not available"}
    def coinbase_quote_comparison(settings): return {"ok": False, "error": "Bot not available"}
    def coinbase_products_for_quote(quote): return {"ok": False, "error": "Bot not available"}
    def oanda_auth_check(): return {"ok": False, "error": "Bot not available"}
    def fetch_candles(*args, **kwargs): return []
except Exception as e:
    logger.error(f"Error importing bot_server: {e}")
    BOT_AVAILABLE = False

# ─── CREATE BOT INSTANCE ──────────────────────────────────────────
if BOT_AVAILABLE:
    try:
        bot = PaperBot()
        logger.info("Bot instance created successfully")
    except Exception as e:
        logger.error(f"Failed to create bot: {e}")
        bot = DummyBot()
        BOT_AVAILABLE = False
else:
    bot = DummyBot()

# ─── AUTO-START BOT ────────────────────────────────────────────────
def start_bot_background():
    try:
        logger.info("Starting bot automatically...")
        if BOT_AVAILABLE:
            bot.start()
            logger.info("Bot started successfully")
        else:
            logger.info("Bot not available - skipping auto-start")
    except Exception as e:
        logger.error(f"Failed to auto-start bot: {e}")

# Only start if bot is available and not in test mode
if BOT_AVAILABLE and os.environ.get('AUXO_TEST') != 'true':
    thread = threading.Thread(target=start_bot_background, daemon=True)
    thread.start()
    logger.info("Bot background thread started")
else:
    logger.info("Bot auto-start disabled (bot not available or test mode)")

# ─── SERVE WEB FOLDER ──────────────────────────────────────────────
@app.route('/')
def index():
    """Serve the dashboard."""
    try:
        return send_from_directory(WEB_DIR, 'index.html')
    except Exception as e:
        logger.error(f"Error serving index: {e}")
        return "Auxo Bot - Dashboard not found", 404

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files from web directory."""
    try:
        return send_from_directory(WEB_DIR, filename)
    except Exception as e:
        logger.error(f"Error serving {filename}: {e}")
        return "File not found", 404

# ─── API ENDPOINTS ──────────────────────────────────────────────────

@app.route('/api/health')
def api_health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "timestamp": str(datetime.now()),
        "server": "Auxo Bot on Namecheap",
        "bot_available": BOT_AVAILABLE,
        "bot_running": bot.state.running if hasattr(bot, 'state') else False
    })

@app.route('/api/status')
def api_status():
    """Get complete bot status."""
    try:
        if not BOT_AVAILABLE:
            return jsonify({"ok": False, "error": "Bot not available", "running": False}), 503
        return jsonify(bot.snapshot())
    except Exception as e:
        logger.error(f"Error in /api/status: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/api/start', methods=['POST'])
def api_start():
    """Start the bot."""
    try:
        if not BOT_AVAILABLE:
            return jsonify({"ok": False, "error": "Bot not available"}), 503
        bot.start()
        return jsonify({"ok": True, "message": "Bot started"})
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stop the bot."""
    try:
        if not BOT_AVAILABLE:
            return jsonify({"ok": False, "error": "Bot not available"}), 503
        bot.stop()
        return jsonify({"ok": True, "message": "Bot stopped"})
    except Exception as e:
        logger.error(f"Error stopping bot: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def api_reset():
    """Reset the bot."""
    try:
        if not BOT_AVAILABLE:
            return jsonify({"ok": False, "error": "Bot not available"}), 503
        bot.reset()
        return jsonify({"ok": True, "message": "Bot reset"})
    except Exception as e:
        logger.error(f"Error resetting bot: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/api/settings', methods=['POST'])
def api_settings():
    """Update bot settings."""
    try:
        if not BOT_AVAILABLE:
            return jsonify({"ok": False, "error": "Bot not available"}), 503
        data = request.get_json()
        if not data:
            return jsonify({"ok": False, "error": "No data provided"}), 400
        bot.update_settings(data)
        return jsonify({"ok": True, "message": "Settings saved"})
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/api/sync-oanda-balance', methods=['POST'])
def api_sync_oanda():
    """Sync OANDA balance."""
    try:
        if not BOT_AVAILABLE:
            return jsonify({"ok": False, "error": "Bot not available"}), 503
        result = bot.sync_paper_balance_from_oanda()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error syncing OANDA: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/api/oanda-auth-check', methods=['POST'])
def api_oanda_auth():
    """Check OANDA authentication."""
    try:
        result = oanda_auth_check()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error checking OANDA auth: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/api/close-position', methods=['POST'])
def api_close_position():
    """Close a position manually."""
    try:
        if not BOT_AVAILABLE:
            return jsonify({"ok": False, "error": "Bot not available"}), 503
        data = request.get_json()
        result = bot.close_position_manual(
            symbol=data.get('symbol', ''),
            mode=data.get('mode', 'profit_only')
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/api/verify-oanda-positions', methods=['POST'])
def api_verify_oanda():
    """Verify OANDA positions."""
    try:
        if not BOT_AVAILABLE:
            return jsonify({"ok": False, "error": "Bot not available"}), 503
        result = bot.verify_oanda_positions()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error verifying OANDA positions: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    """Run backtest."""
    try:
        data = request.get_json()
        if BOT_AVAILABLE:
            settings = {**bot.snapshot()["settings"], **data}
        else:
            settings = data or {}
        result = run_backtest(settings)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/api/optimize', methods=['POST'])
def api_optimize():
    """Run optimizer."""
    try:
        data = request.get_json()
        if BOT_AVAILABLE:
            settings = {**bot.snapshot()["settings"], **data}
        else:
            settings = data or {}
        result = run_optimizer(settings)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error running optimizer: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/api/walk-forward', methods=['POST'])
def api_walk_forward():
    """Run walk-forward optimization."""
    try:
        data = request.get_json()
        if BOT_AVAILABLE:
            settings = {**bot.snapshot()["settings"], **data}
        else:
            settings = data or {}
        result = run_walk_forward(settings)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error running walk-forward: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/api/quote-comparison', methods=['POST'])
def api_quote_comparison():
    """Compare quotes."""
    try:
        data = request.get_json()
        if BOT_AVAILABLE:
            settings = {**bot.snapshot()["settings"], **data}
        else:
            settings = data or {}
        result = coinbase_quote_comparison(settings)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error comparing quotes: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/api/coinbase-products', methods=['GET'])
def api_coinbase_products():
    """Get Coinbase products."""
    try:
        quote = request.args.get('quote', 'GBP')
        result = coinbase_products_for_quote(quote)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting Coinbase products: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/api/candles')
def api_candles():
    """Get candles for chart."""
    try:
        symbol = request.args.get('symbol', 'EURUSD').upper()
        granularity = int(request.args.get('granularity', '3600'))
        count = int(request.args.get('count', '200'))
        quote = request.args.get('quote', 'GBP').upper()

        if BOT_AVAILABLE:
            settings = bot.snapshot().get('settings', {})
            exchange = settings.get('exchange', 'coinbase')
            asset_class = settings.get('asset_class', 'crypto')
        else:
            exchange = 'coinbase'
            asset_class = 'crypto'

        candles = fetch_candles(
            exchange=exchange,
            symbol=symbol,
            quote_currency=quote,
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

        return jsonify({
            'ok': True,
            'symbol': symbol,
            'granularity': granularity,
            'count': len(candle_data),
            'candles': candle_data
        })
    except Exception as e:
        logger.error(f"Error fetching candles: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/export-db', methods=['POST'])
def api_export_db():
    """Export database to JSON."""
    try:
        if not BOT_AVAILABLE:
            return jsonify({"ok": False, "error": "Bot not available"}), 503
        path = BASE_DIR / f"export_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        bot.db.export_json(path)
        return jsonify({"ok": True, "path": str(path)})
    except Exception as e:
        logger.error(f"Error exporting DB: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

# ─── ERROR HANDLING ────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found", "path": request.path}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

# ─── MAIN ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
