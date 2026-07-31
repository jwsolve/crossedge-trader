# CrossEdge Trader

CrossEdge Trader (Auxo) is a local web-based algorithmic trading and
research dashboard for cryptocurrency and forex.

It supports paper trading, Coinbase crypto trading when explicitly
armed, and OANDA practice-account forex trading. The project combines
market scanning, support/resistance analysis, regime detection, adaptive
risk management, strategy learning, backtesting, optimisation,
walk-forward testing, and a Validation Lab.

> **Status:** Experimental / research software. Auxo has a substantial
> trading and risk-management framework, but it has **not yet been
> proven to have a persistent profitable edge through statistically
> rigorous out-of-sample validation**. Do not interpret the
> sophistication of the software as evidence of profitability.

## Current Status

Auxo currently provides:

-   Crypto paper trading
-   Forex paper trading
-   Coinbase crypto live trading when explicitly armed
-   OANDA practice/demo forex trading
-   Market scanner
-   Candlestick and line charts
-   Support and resistance zones
-   Dynamic S/R exits
-   Market regime detection
-   Risk-based/adaptive position sizing
-   ATR-based risk controls
-   Kelly-style sizing support
-   Multiple exit mechanisms
-   Partial take-profit support
-   Trailing-stop support
-   Native Coinbase protective stop support
-   Backtesting
-   Strategy optimisation
-   Walk-forward testing
-   Strategy learning
-   Expectancy analysis
-   Symbol performance tracking
-   Decision journal
-   Validation Lab
-   Fee/slippage stress testing
-   Parameter sensitivity testing
-   Monte Carlo analysis when sufficient closed-trade data is available

## Current Performance Evidence

Auxo should currently be treated as a **research system rather than a
proven profitable trading strategy**.

The current local trading database contains historical trade/setup
information, but the genuinely closed setup sample is still too small to
make a statistically strong claim of a durable edge. The Validation Lab
is therefore intended to establish whether the apparent edge survives
unseen data and realistic execution assumptions.

The validation process should establish:

1.  Whether the core strategy is profitable without ML.
2.  Whether S/R and regime filtering improve the underlying strategy.
3.  Whether expectancy filtering improves results.
4.  Whether ML adds genuine out-of-sample value.
5.  Whether results survive realistic fees and slippage.
6.  Whether results remain robust when parameters are varied.
7.  Whether performance is consistent across symbols and market regimes.

**Do not optimise against a test period and then treat that same period
as proof of performance.**

## Account and P/L Model

Auxo keeps the following accounting values separate:

  -----------------------------------------------------------------------
  Value                               Meaning
  ----------------------------------- -----------------------------------
  **Starting Cash**                   Fixed baseline used for
                                      account-performance calculations

  **Cash**                            Currently available cash

  **Open Position Value**             Current market value of open
                                      positions

  **Realised P/L**                    P/L from completed trades

  **Unrealised P/L**                  Current P/L on open positions

  **Equity**                          Cash + open position value

  **Total P/L**                       Current equity compared with
                                      Starting Cash
  -----------------------------------------------------------------------

Example:

``` text
Starting Cash       £50.91
Available Cash      £30.74
Open Position Value £19.54
Equity              £50.28
Total P/L           -£0.63
```

**Starting Cash is a fixed baseline and should not automatically change
when exchange balances change.**

Exchange balance synchronisation updates the live cash balance, not the
starting baseline.

## Quick Start

``` bash
git clone https://github.com/jwsolve/crossedge-trader.git
cd crossedge-trader
python -m venv venv
```

### Windows

``` bash
venv\Scripts\activate
```

### macOS/Linux

``` bash
source venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Start the bot:

``` bash
python bot_server.py
```

Open:

``` text
http://localhost:8080
```

Auxo's default web server port is **8080**.

## Requirements

-   Python 3.11 or newer recommended
-   Windows, macOS, or Linux
-   Internet connection for exchange market data
-   Modern web browser

## Running Auxo Locally vs on a VPS

Auxo can run perfectly well **locally on a laptop or desktop**, which is ideal for development, testing, paper trading and occasional use.

However, if you want Auxo to keep running continuously without having to leave your laptop switched on, connected to the internet and running the bot process, a **VPS (Virtual Private Server)** is a better option.

A VPS gives Auxo a dedicated environment that can stay online 24/7, making it more suitable for:

- Continuous market monitoring
- Automated paper trading
- Long-running validation jobs
- Continuous strategy learning
- Live trading where appropriate
- Accessing the dashboard without keeping your personal computer running

### Budget VPS option

For a relatively inexpensive starting point, **Fasthosts** offers Linux VPS plans suitable for smaller applications. Their current VPS range includes a VPS 2 with **2 vCPU, 2GB RAM and 90GB NVMe storage**, currently advertised at **£3/month for the first 3 months** (regular price £4/month). Pricing and promotions can change, so check the provider before purchasing.

If you use Fasthosts, you can support the project author by using this referral link:

Fasthosts referral link https://www.fasthosts.co.uk/referral?referral=trcq42yhhk25g

**Important:** a VPS is not required to use Auxo. A local installation is perfectly suitable for development and testing. A VPS simply removes the requirement to leave your own laptop or desktop running continuously.

## Project Structure

``` text
crossedge-trader/
├── README.md
├── requirements.txt
├── .env.example
├── bot_server.py
├── database.py
├── exchange_connectors.py
├── expectancy_engine.py
├── regime_detector.py
├── strategy_creator.py
├── validation_engine.py
├── bot_state.json
├── bot_audit.jsonl
├── trades.db
└── web/
    └── index.html
```

The exact project structure may change as Auxo develops.

`bot_state.json` stores local settings and runtime state.

`bot_audit.jsonl` stores an append-only activity/audit log.

`trades.db` stores local trading and research data.

Do not commit secrets or private runtime state to a public repository
unless you intentionally want that data published.

## Configuration

Create a `.env` file from the example:

``` bash
cp .env.example .env
```

Windows PowerShell:

``` powershell
Copy-Item .env.example .env
```

API credentials are optional for paper trading.

# Trading Modes

## Paper Trading

Paper trading is the recommended starting point.

Use:

``` text
Live Trading = Disabled
OANDA Demo Orders = Disabled
```

Paper mode allows you to test watchlists, scanner signals, strategies,
position sizing, stops, targets, S/R, regime detection, backtesting,
optimisation, walk-forward testing and Validation Lab results.

# Coinbase Crypto Trading

Coinbase live trading is deliberately locked behind multiple safety
conditions.

Required environment values include:

``` env
COINBASE_API_KEY_NAME=
COINBASE_API_PRIVATE_KEY_FILE=
LIVE_TRADING_CONFIRM=I_UNDERSTAND_THIS_PLACES_REAL_ORDERS
```

You may also use `COINBASE_API_PRIVATE_KEY`, although storing the
private key in a protected file is preferable.

Live trading must also be explicitly enabled in the dashboard.

### Native Coinbase Stops

Auxo's current Coinbase native stop-limit configuration is:

``` text
stop_limit_stop_limit_gtc
```

The older/incorrect `stop_limit` configuration is rejected by the
Coinbase Advanced Trade API.

### Live-stop safety

Native protective-stop failure should be treated as a live safety event.
Paper mode can use a simulated stop fallback for testing, but live
trading should not silently continue as though an exchange-native
protective stop exists.

Start with very small limits and verify native stop behaviour before
increasing live capital.

# OANDA Demo Forex Trading

Required `.env` values:

``` env
OANDA_ENV=practice
OANDA_API_BASE=https://api-fxpractice.oanda.com
OANDA_ACCOUNT_ID=your-demo-account-id
OANDA_API_TOKEN=your-personal-access-token
OANDA_DEMO_TRADING_ENABLED=true
```

Dashboard setup:

1.  Click `Apply OANDA Demo Preset`
2.  Save settings
3.  Click `Check OANDA Demo`
4.  Click `Sync OANDA Paper Balance`
5.  Enable `OANDA Demo Orders` only when ready
6.  Start the bot

OANDA demo trading should only use a practice account.

# Trading Settings

Important settings include:

### Risk

-   Risk-based position sizing
-   Risk per trade
-   Maximum position allocation
-   Stop-loss settings
-   Daily loss protection
-   Maximum drawdown protection
-   ATR-based risk adjustments
-   Kelly-style sizing

### Strategy

-   Strategy selection
-   Signal thresholds
-   Support/resistance filtering
-   Minimum S/R range
-   Minimum reward/risk
-   Regime filtering
-   Dynamic S/R exits
-   Strategy learning

### Execution

-   Exchange selection
-   Paper/live mode
-   Native-stop configuration
-   Slippage assumptions

### Learning

-   Expectancy
-   Symbol performance
-   Weak-pair detection
-   Setup history
-   Decision journal
-   ML confidence where enabled

# Dashboard Guide

## Overview

Shows equity, cash, Starting Cash, P/L, chart, open position, current
signal and trading status.

## Scanner

Shows market opportunities including symbol, signal, score,
support/resistance, regime and execution information where available.

## Trades

Shows local trade activity and execution history.

## Strategy Learning

Shows setup history, open/closed status, realised performance and
symbol-level results.

## Backtester

Runs historical strategy tests using available candle data and current
strategy/risk settings.

## Optimiser

Tests combinations of strategy parameters and ranks candidates.
Optimisation results should be treated as candidate configurations, not
proof of future profitability.

## Walk-Forward Test

Splits historical candles into training and unseen test data. The unseen
period should remain separate from optimisation when it is being used as
a genuine validation period.

## Validation Lab

The Validation Lab is designed to answer:

> **Does the strategy remain robust when tested against data and
> conditions it was not optimised for?**

It is intended to cover:

-   Baseline strategy testing
-   S/R comparisons
-   Strategy comparisons
-   Fee stress
-   Slippage stress
-   Parameter sensitivity
-   Monte Carlo analysis when sufficient closed trades are available
-   Robustness scoring

Recommended process:

``` text
Core strategy
      ↓
Baseline backtest
      ↓
S/R comparison
      ↓
Regime comparison
      ↓
Expectancy filter
      ↓
ML comparison
      ↓
Fee/slippage stress
      ↓
Parameter sensitivity
      ↓
Monte Carlo
      ↓
Out-of-sample / walk-forward
      ↓
Final robustness assessment
```

Do not repeatedly optimise against the same test period.

# Trading Philosophy

Auxo is designed around several principles:

-   Prefer entries around meaningful support rather than chasing
    extended moves.
-   Respect resistance and protect profitable trades.
-   Require adequate reward/risk before entering.
-   Adapt strategy behaviour to the detected market regime.
-   Control risk before attempting to maximise returns.
-   Learn from closed trades without introducing look-ahead bias.
-   Treat exchange-native protection as preferable for live risk
    control.

# Validation Philosophy

A profitable backtest alone is insufficient.

A strategy should ideally survive:

-   Unseen data
-   Walk-forward testing
-   Parameter changes
-   Higher fees
-   Higher slippage
-   Different market regimes
-   Different symbols
-   Monte Carlo trade ordering
-   Realistic execution assumptions

A strategy that only works with one exact parameter combination should
be treated as potentially overfit.

# Common Issues

### No candles returned

The exchange did not return candle data for that symbol. Try another
symbol, quote currency or timeframe.

### Not enough candle data

Increase `Live Candle Count` or ensure the selected historical period
contains enough data.

### OANDA invalid instrument

Check the instrument spelling and use the broker's supported instrument
name, for example `GBPUSD`.

### Coinbase 401 Unauthorized

Check API key name, private key, API permissions and `.env` formatting.

### Coinbase native stop rejected

If you see:

``` text
unknown field "stop_limit"
```

check that the running server uses:

``` text
stop_limit_stop_limit_gtc
```

and that the latest code is installed.

### Unexpected JSON character

The frontend expected JSON but the server returned an error page.
Restart the server and check the terminal output.

### Frontend does not update

Hard refresh with `Ctrl + F5`, then restart the server.

### Starting Cash keeps changing

Starting Cash is intended to be a fixed performance baseline. Exchange
balance synchronisation should update available Cash, not overwrite
Starting Cash. If it changes unexpectedly, verify that the running
installation contains the latest account/P/L handling code.

# Security

Never commit:

-   `.env`
-   API private keys
-   Exchange secrets
-   Personal access tokens
-   Private certificates
-   Unnecessary account credentials

Keep API permissions restricted.

Use paper/demo trading first and small order sizes when live trading is
eventually enabled.

# Recommended Deployment Progression

``` text
1. Install locally
       ↓
2. Paper trade
       ↓
3. Verify scanner and signals
       ↓
4. Verify stops and exits
       ↓
5. Run backtests
       ↓
6. Run optimisation
       ↓
7. Run walk-forward tests
       ↓
8. Run Validation Lab
       ↓
9. Paper trade live market conditions
       ↓
10. Coinbase/OANDA demo
       ↓
11. Very small live capital
       ↓
12. Increase exposure only after evidence
```

Do not skip directly from a profitable backtest to significant live
capital.

# Development Status

CrossEdge Trader is under active development.

The project is moving beyond a simple indicator bot toward a research
and execution platform with:

-   Regime-aware strategy selection
-   Adaptive risk
-   Expectancy analysis
-   Strategy learning
-   Walk-forward testing
-   Optimisation
-   ML-assisted confidence
-   Validation and robustness testing
-   Exchange execution and reconciliation

The current priority is **proving that each component improves
out-of-sample performance without introducing overfitting or unrealistic
backtest assumptions**, rather than simply adding more indicators or AI
features.

# Disclaimer

CrossEdge Trader is experimental trading software for education,
research, and personal testing.

Trading cryptocurrency and foreign exchange involves significant
financial risk. Automated trading can lose money quickly. Backtested or
simulated performance does not guarantee future results.

The presence of machine learning, optimisation, regime detection, risk
management or other advanced features does **not** mean the strategy is
profitable.

Test thoroughly in paper or demo mode before risking real capital.

You are responsible for any trades placed through your own exchange or
broker accounts.
