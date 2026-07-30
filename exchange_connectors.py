#!/usr/bin/env python3
"""
Exchange connectors for Coinbase, Binance, Kraken.
Provides price feeds, order book checks, and trading functions.
"""

import os
import time
import json
import hmac
import hashlib
import urllib.request
import urllib.parse
from typing import Optional, Dict, List, Tuple

# ─── Binance (using python-binance) ──────────────────────────────
try:
    from binance.client import Client as BinanceClient
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    BinanceClient = None

# ─── Kraken (using krakenex) ──────────────────────────────────────
try:
    import krakenex
    KRAKEN_AVAILABLE = True
except ImportError:
    KRAKEN_AVAILABLE = False
    krakenex = None

# ─── Helper: retry with exponential backoff ──────────────────────
def api_request_with_retry(func, *args, max_retries=3, backoff_factor=1.5, **kwargs):
    """
    Call API function with exponential backoff on failure.
    """
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = backoff_factor ** attempt
            time.sleep(wait)
            continue

# ─── Coinbase connector ────────────────────────────────────────────
class CoinbaseConnector:
    def __init__(self, api_key, private_key, quote_currency="GBP"):
        self.api_key = api_key
        self.private_key = private_key
        self.quote = quote_currency

    def get_price(self, symbol):
        """Get current price from Coinbase."""
        product = f"{symbol}-{self.quote}"
        try:
            url = f"https://api.exchange.coinbase.com/products/{product}/ticker"
            data = fetch_json(url)
            return float(data.get("price", 0))
        except:
            return 0.0

    def get_order_book(self, symbol, level=1, limit=None):
        product = f"{symbol}-{self.quote}"
        # Coinbase only accepts level=1,2,3
        if level not in [1, 2, 3]:
            level = 2   # default to top 50 for liquidity checks
        try:
            url = f"https://api.exchange.coinbase.com/products/{product}/book?level={level}"
            data = fetch_json(url)
            bids = [(float(b[0]), float(b[1])) for b in data.get("bids", [])]
            asks = [(float(a[0]), float(a[1])) for a in data.get("asks", [])]
            return {"bids": bids, "asks": asks}
        except Exception:
            return {"bids": [], "asks": []}

    def market_buy(self, symbol, quote_size):
        """Place market buy order (via bot's existing function)."""
        # This will be called from the bot's live_buy, so we'll just return the order response.
        # The actual placement is handled by the bot's coinbase_market_order.
        from bot_server import coinbase_market_order, coinbase_order_id, coinbase_reconcile_order
        product = f"{symbol}-{self.quote}"
        order = coinbase_market_order(product, "BUY", quote_size=quote_size)
        return order

    def market_sell(self, symbol, base_size):
        """Place market sell order."""
        from bot_server import coinbase_market_order
        product = f"{symbol}-{self.quote}"
        order = coinbase_market_order(product, "SELL", base_size=base_size)
        return order

# ─── Binance connector ─────────────────────────────────────────────
class BinanceConnector:
    def __init__(self, api_key, api_secret, quote_currency="GBP"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.quote = quote_currency.upper()
        if BINANCE_AVAILABLE:
            self.client = BinanceClient(api_key, api_secret)
        else:
            self.client = None

    def get_price(self, symbol):
        """Get current price from Binance."""
        if not self.client:
            return 0.0
        try:
            ticker = self.client.get_symbol_ticker(symbol=f"{symbol}{self.quote}")
            return float(ticker.get("price", 0))
        except:
            return 0.0

    def get_order_book(self, symbol, limit=5, level=None):
        """
        Get order book depth.
        - limit: number of price levels (default 5)
        - level: ignored (kept for compatibility with Coinbase connector)
        """
        if not self.client:
            return {"bids": [], "asks": []}
        try:
            depth = self.client.get_order_book(symbol=f"{symbol}{self.quote}", limit=limit)
            bids = [(float(b[0]), float(b[1])) for b in depth.get("bids", [])]
            asks = [(float(a[0]), float(a[1])) for a in depth.get("asks", [])]
            return {"bids": bids, "asks": asks}
        except:
            return {"bids": [], "asks": []}

    def market_buy(self, symbol, quote_size):
        """Place market buy order (using Binance)."""
        if not self.client:
            raise RuntimeError("Binance client not available.")
        order = self.client.order_market_buy(symbol=f"{symbol}{self.quote}", quoteOrderQty=quote_size)
        return order

    def market_sell(self, symbol, base_size):
        """Place market sell order."""
        if not self.client:
            raise RuntimeError("Binance client not available.")
        order = self.client.order_market_sell(symbol=f"{symbol}{self.quote}", quantity=base_size)
        return order

# ─── Kraken connector ──────────────────────────────────────────────
class KrakenConnector:
    def __init__(self, api_key, api_secret, quote_currency="GBP"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.quote = quote_currency.upper()
        if KRAKEN_AVAILABLE:
            self.client = krakenex.API()
            self.client.load_key(api_key, api_secret)
        else:
            self.client = None

    def get_price(self, symbol):
        """Get current price from Kraken."""
        if not self.client:
            return 0.0
        try:
            # Kraken pair is like XBTUSD, but we need mapping for symbols like BTC->XBT
            kraken_symbol = self._map_symbol(symbol)
            pair = f"{kraken_symbol}{self.quote}"
            ticker = self.client.query_public("Ticker", {"pair": pair})
            result = ticker.get("result", {})
            if not result:
                return 0.0
            # The result has a key like 'XBTUSD'
            data = result.get(pair, {})
            return float(data.get("c", ["0"])[0])  # last price
        except:
            return 0.0

    def get_order_book(self, symbol, limit=5, level=None):
        """
        Get order book depth.
        - limit: number of price levels (default 5)
        - level: ignored (kept for compatibility with Coinbase connector)
        """
        if not self.client:
            return {"bids": [], "asks": []}
        try:
            kraken_symbol = self._map_symbol(symbol)
            pair = f"{kraken_symbol}{self.quote}"
            depth = self.client.query_public("Depth", {"pair": pair, "count": limit})
            result = depth.get("result", {})
            if not result:
                return {"bids": [], "asks": []}
            data = result.get(pair, {})
            bids = [(float(b[0]), float(b[1])) for b in data.get("bids", [])]
            asks = [(float(a[0]), float(a[1])) for a in data.get("asks", [])]
            return {"bids": bids, "asks": asks}
        except:
            return {"bids": [], "asks": []}

    def _map_symbol(self, symbol):
        """Map common symbols to Kraken notation."""
        mapping = {"BTC": "XBT", "ETH": "ETH", "DOGE": "XDG"}
        return mapping.get(symbol.upper(), symbol.upper())

    def market_buy(self, symbol, quote_size):
        """Place market buy order (via client)."""
        if not self.client:
            raise RuntimeError("Kraken client not available.")
        kraken_symbol = self._map_symbol(symbol)
        pair = f"{kraken_symbol}{self.quote}"
        order = self.client.query_private("AddOrder", {
            "pair": pair,
            "type": "buy",
            "ordertype": "market",
            "quote": str(quote_size),
        })
        return order

    def market_sell(self, symbol, base_size):
        """Place market sell order."""
        if not self.client:
            raise RuntimeError("Kraken client not available.")
        kraken_symbol = self._map_symbol(symbol)
        pair = f"{kraken_symbol}{self.quote}"
        order = self.client.query_private("AddOrder", {
            "pair": pair,
            "type": "sell",
            "ordertype": "market",
            "volume": str(base_size),
        })
        return order

# ─── Price Aggregator ──────────────────────────────────────────────
class PriceAggregator:
    def __init__(self, connectors: Dict[str, any]):
        self.connectors = connectors

    def get_best_price(self, symbol, side="BUY"):
        """
        Get the best price across all enabled exchanges.
        For BUY, returns the lowest ask (min price).
        For SELL, returns the highest bid (max price).
        Returns (price, exchange_name, connector).
        """
        best_price = None
        best_exchange = None
        best_connector = None

        for name, conn in self.connectors.items():
            try:
                if side.upper() == "BUY":
                    book = conn.get_order_book(symbol, level=1)
                    if book and book.get("asks"):
                        price = book["asks"][0][0]  # best ask
                    else:
                        price = conn.get_price(symbol)
                else:
                    book = conn.get_order_book(symbol, level=1)
                    if book and book.get("bids"):
                        price = book["bids"][0][0]  # best bid
                    else:
                        price = conn.get_price(symbol)
                if price > 0:
                    if best_price is None or (side.upper() == "BUY" and price < best_price) or (side.upper() == "SELL" and price > best_price):
                        best_price = price
                        best_exchange = name
                        best_connector = conn
            except Exception as e:
                # Log error but continue
                print(f"Error getting price from {name}: {e}")
                continue

        if best_price is None:
            raise RuntimeError("No price available from any exchange.")
        return best_price, best_exchange, best_connector

    def check_liquidity(self, symbol, side, quote_size, min_volume_factor=1.5):
        """
        Check if there is enough liquidity for a desired order size.
        Returns (ok, available_volume, recommended_exchange).
        """
        best_volume = 0
        best_exchange = None
        best_connector = None

        for name, conn in self.connectors.items():
            try:
                book = conn.get_order_book(symbol, level=2)   # 2 gives top 50 orders
                total = 0
                if side.upper() == "BUY":
                    for price, qty in book.get("asks", []):
                        total += price * qty
                        if total >= quote_size * min_volume_factor:
                            break
                else:
                    for price, qty in book.get("bids", []):
                        total += price * qty
                        if total >= quote_size * min_volume_factor:
                            break
                if total > best_volume:
                    best_volume = total
                    best_exchange = name
                    best_connector = conn
            except:
                continue

        if best_volume >= quote_size * min_volume_factor:
            return True, best_volume, best_exchange
        else:
            return False, best_volume, best_exchange

# ─── Helper to fetch JSON (reuse from bot_server) ────────────────
def fetch_json(url, timeout=10):
    """Fetch JSON from URL (reused from bot_server)."""
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "local-paper-trading-bot/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            raise RuntimeError(
                "HTTP 403 Forbidden. The exchange may be blocking this network/IP."
            ) from exc
        raise

# ─── Factory to create connectors ─────────────────────────────────
def create_connectors(quote_currency):
    """Factory to create all enabled exchange connectors."""
    connectors = {}

    # Coinbase
    from bot_server import coinbase_live_is_armed, coinbase_private_key_value
    if coinbase_live_is_armed():
        api_key = os.environ.get("COINBASE_API_KEY_NAME", "")
        private_key = coinbase_private_key_value()
        connectors["coinbase"] = CoinbaseConnector(
            api_key=api_key,
            private_key=private_key,
            quote_currency=quote_currency
        )

    # Binance
    if BINANCE_AVAILABLE:
        api_key = os.environ.get("BINANCE_API_KEY", "")
        api_secret = os.environ.get("BINANCE_API_SECRET", "")
        if api_key and api_secret:
            connectors["binance"] = BinanceConnector(api_key, api_secret, quote_currency)

    # Kraken
    if KRAKEN_AVAILABLE:
        api_key = os.environ.get("KRAKEN_API_KEY", "")
        api_secret = os.environ.get("KRAKEN_API_SECRET", "")
        if api_key and api_secret:
            connectors["kraken"] = KrakenConnector(api_key, api_secret, quote_currency)

    return connectors
