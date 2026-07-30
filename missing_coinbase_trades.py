#!/usr/bin/env python3
"""
Sync Coinbase positions into bot_state.json.
Run this script when the bot is stopped.

It will:
- Fetch all balances from Coinbase.
- Treat any non‑zero balance (except the quote currency) as an open position.
- Compute average entry price from historical fills (BUY orders).
- Update the "positions", "cash", "coin", and "active_symbol" fields.
- Save the updated state.

Note: This is an approximation and may not be 100% accurate.
"""

import json
import time
from pathlib import Path
from bot_server import (
    load_dotenv,
    coinbase_api_request,
    fetch_json,
    BASE_DIR,
    STATE_FILE,
)

# ─── Configuration ────────────────────────────────────────────────
QUOTE_CURRENCY = "GBP"   # Change to your quote currency (e.g., USD, EUR)
# ──────────────────────────────────────────────────────────────────

def get_all_accounts():
    """Fetch all account balances from Coinbase."""
    accounts = []
    cursor = ""
    while True:
        query = "?limit=250"
        if cursor:
            query += f"&cursor={cursor}"
        data = coinbase_api_request("GET", f"/api/v3/brokerage/accounts{query}")
        accounts.extend(data.get("accounts", []))
        if not data.get("has_next"):
            break
        cursor = data.get("cursor", "")
    return accounts

def get_fills_for_product(product_id, limit=100):
    """Fetch fills for a specific product (e.g., 'BTC-GBP')."""
    fills = []
    cursor = ""
    while True:
        query = f"?limit={limit}&product_id={product_id}"
        if cursor:
            query += f"&cursor={cursor}"
        data = coinbase_api_request("GET", f"/api/v3/brokerage/orders/historical/fills{query}")
        fills.extend(data.get("fills", []))
        if not data.get("has_next"):
            break
        cursor = data.get("cursor", "")
    return fills

def get_ticker_price(product_id):
    """Get current price for a product."""
    try:
        url = f"https://api.exchange.coinbase.com/products/{product_id}/ticker"
        data = fetch_json(url)
        return float(data.get("price", 0))
    except Exception:
        return 0.0

def compute_avg_entry_price(fills):
    """
    Compute weighted average entry price from BUY fills.
    Only considers BUY orders (assuming long position).
    If no BUY fills, returns None.
    """
    total_value = 0.0
    total_size = 0.0
    for fill in fills:
        if fill.get("side", "").upper() == "BUY":
            size = float(fill.get("size", 0))
            price = float(fill.get("price", 0))
            total_value += size * price
            total_size += size
    if total_size > 0:
        return total_value / total_size
    return None

def main():
    # Load .env
    load_dotenv()
    print("📊 Fetching Coinbase accounts...")
    accounts = get_all_accounts()

    if not accounts:
        print("❌ No accounts returned from Coinbase. Check API keys.")
        return

    print(f"🔍 Found {len(accounts)} accounts. Inspecting first 3:")
    for i, acc in enumerate(accounts[:3]):
        print(f"  Account {i+1}: {json.dumps(acc, indent=2)}")

    # Separate quote currency balance and asset balances
    cash = 0.0
    assets = {}
    for acc in accounts:
        currency = acc.get("currency", "")
        # Try several possible fields for balance
        balance_obj = acc.get("balance", {})
        available_obj = acc.get("available_balance", {})
        hold_obj = acc.get("hold", {})
        # Prefer available_balance, then balance, then hold
        bal_value = float(available_obj.get("value", 0)) if available_obj else 0.0
        if bal_value == 0:
            bal_value = float(balance_obj.get("value", 0))
        if bal_value == 0:
            bal_value = float(hold_obj.get("value", 0))
        if bal_value == 0:
            # Some responses have 'value' directly under 'balance' – we already tried that
            continue

        if currency.upper() == QUOTE_CURRENCY.upper():
            cash = bal_value
        else:
            assets[currency] = bal_value

    print(f"💰 Cash ({QUOTE_CURRENCY}): {cash:.2f}")
    print(f"📦 Found {len(assets)} asset(s) with non‑zero balance:")
    for sym, qty in assets.items():
        print(f"    {sym}: {qty}")

    if not assets:
        print("⚠️ No assets with non‑zero balance found. Are you sure you have positions?")
        print("   If you do, check your quote currency setting (currently: {QUOTE_CURRENCY}).")
        return

    # Load existing state or create new
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
    else:
        state = {
            "positions": {},
            "cash": 0.0,
            "coin": 0.0,
            "active_symbol": None,
            "settings": {},
            "trades": [],
            "price_history": {},
        }

    # Update cash
    state["cash"] = cash

    # Process each asset
    positions = {}
    active_symbol = None
    largest_quantity = 0.0

    for symbol, quantity in assets.items():
        product_id = f"{symbol}-{QUOTE_CURRENCY}"
        print(f"  🔍 Processing {symbol} (qty: {quantity})...")
        # Fetch fills for this product
        fills = get_fills_for_product(product_id, limit=200)
        avg_price = compute_avg_entry_price(fills)

        if avg_price is None:
            print(f"     ⚠️ No BUY fills found for {symbol}, using current price as entry.")
            avg_price = get_ticker_price(product_id)
            if avg_price == 0:
                print(f"     ❌ Could not fetch price for {symbol}, skipping.")
                continue

        # Build position entry
        positions[symbol] = {
            "quantity": quantity,
            "entry_price": avg_price,
            "highest_price": avg_price,
            "is_short": False,
            "opened_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stop_price": None,
            "target_price": None,
            "exit_mode": "fixed",
            "entry_time": time.time(),
            "partial_take_profit_done": False,
            "entry_cost": quantity * avg_price,
        }

        # Track largest position to set as active
        if quantity > largest_quantity:
            largest_quantity = quantity
            active_symbol = symbol

    # Update state
    state["positions"] = positions
    state["coin"] = largest_quantity
    state["active_symbol"] = active_symbol

    # Save state
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

    print(f"\n✅ Sync complete!")
    print(f"   Cash: {cash:.2f} {QUOTE_CURRENCY}")
    print(f"   Positions: {len(positions)}")
    for sym, pos in positions.items():
        print(f"     {sym}: {pos['quantity']} @ {pos['entry_price']:.2f}")
    print(f"\n   Active symbol: {active_symbol}")
    print("\n⚠️  Remember to stop the bot before running this script.")
    print("   After restarting, the positions should appear in the UI.")

if __name__ == "__main__":
    main()
