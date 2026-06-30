# CrossEdge Trader

CrossEdge Trader is a local web-based trading dashboard for crypto and forex testing.

It can run as a paper trading bot, connect to Coinbase for crypto trading, and connect to an OANDA practice account for forex demo trading. It is designed for local use on your own machine.

Trading is disabled by default. API keys are only needed if you want Coinbase live trading or OANDA demo account trading.

## Quick Start

```bash
git clone https://github.com/yourusername/crossedge-trader.git
cd crossedge-trader
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the bot:

```bash
python bot_server.py
```

Open:

```text
http://localhost:8080
```

## Requirements

- Python 3.11 or newer recommended
- Windows, macOS, or Linux
- Internet connection
- Modern web browser

## Features

- Crypto paper trading
- Forex paper trading
- Coinbase crypto live trading when explicitly armed
- OANDA practice/demo forex trading
- Multiple OANDA demo positions
- Market scanner
- Candlestick and line charts
- Support and resistance zones
- Dynamic S/R exits
- Market regime detection
- Backtesting
- Optimisation
- Walk-forward validation
- Strategy learning
- Symbol performance tracking
- Decision journal

## Project Structure

```text
crossedge-trader/
├── README.md
├── requirements.txt
├── .env.example
├── bot_server.py
├── bot_state.json
├── bot_audit.jsonl
└── web/
    └── index.html
```

`bot_state.json` stores local settings, paper balances, positions, trades, and learning data.

`bot_audit.jsonl` stores an append-only activity log.

Do not commit `.env`, private keys, `bot_state.json`, or `bot_audit.jsonl`.

## Configuration

Create a `.env` file from the example file:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

API credentials are optional for paper trading.

## Coinbase Crypto Trading

Coinbase live trading is locked unless all safety conditions are met.

Required `.env` values:

```env
COINBASE_API_KEY_NAME=
COINBASE_API_PRIVATE_KEY_FILE=
LIVE_TRADING_CONFIRM=I_UNDERSTAND_THIS_PLACES_REAL_ORDERS
```

You can also use `COINBASE_API_PRIVATE_KEY`, but using `COINBASE_API_PRIVATE_KEY_FILE` is safer.

Live Coinbase trading also requires:

```text
Live Trading = Enabled
```

Start with very small limits.

## OANDA Demo Forex Trading

Required `.env` values:

```env
OANDA_ENV=practice
OANDA_API_BASE=https://api-fxpractice.oanda.com
OANDA_ACCOUNT_ID=your-demo-account-id
OANDA_API_TOKEN=your-personal-access-token
OANDA_DEMO_TRADING_ENABLED=true
```

Dashboard setup:

1. Click `Apply OANDA Demo Preset`
2. Save settings
3. Click `Check OANDA Demo`
4. Click `Sync OANDA Paper Balance`
5. Enable `OANDA Demo Orders` only when ready
6. Start the bot

OANDA demo trading should only use a practice account.

## Paper Trading

Paper trading works without API keys.

Recommended first setup:

```text
Asset Class: Crypto or Forex
Live Trading: Disabled
OANDA Demo Orders: Disabled
```

Use paper mode to test watchlists, strategies, position sizing, stops, targets, S/R filters, backtests, and walk-forward results.

## Dashboard Guide

Overview shows equity, cash, chart, position, and signal.

Scanner shows market scan rows, support/resistance, regime, score, and signal.

Trades shows local buy and sell records.

Strategy Learning shows setup records, open/closed status, realised P/L, and symbol performance.

Backtester runs historical tests using the current settings.

Optimiser tests strategy combinations and ranks results.

Walk-Forward Test splits candles into training and unseen test data.

Settings controls market mode, exchange, strategy, risk, S/R, exits, execution, and learning filters.

## Common Issues

### No candles returned

The exchange did not return candle data for that symbol. Try another symbol, quote currency, or timeframe.

### Not enough candle data

Increase `Live Candle Count`.

### OANDA invalid instrument

Use `GBPUSD`, not `GPBUSD`.

### Coinbase 401 Unauthorized

Check API key name, private key, permissions, and `.env` formatting.

### Unexpected JSON character

The frontend expected JSON but the server returned an error page. Restart the server and check the terminal output.

### Frontend does not update

Hard refresh with `Ctrl + F5`, then restart the server.

## Security Notes

- Never commit `.env`
- Never commit private keys
- Keep API keys restricted
- Start with small order sizes
- Use demo/paper mode first

## Disclaimer

CrossEdge Trader is experimental trading software for education, research, and personal testing.

Trading cryptocurrency and foreign exchange involves significant financial risk. Automated trading can lose money quickly. Test thoroughly in paper or demo mode before risking real capital. You are responsible for any trades placed through your own exchange or broker accounts.
