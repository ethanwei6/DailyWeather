# DailyWeather

DailyWeather is a Python research system for pricing daily-high temperature prediction markets. It discovers live Polymarket weather contracts, maps market text into city/date/temperature buckets, builds a weather-model probability consensus, and runs a paper-only Kelly rebalancing ledger.

The project is intentionally built like a trading system, not a notebook. Live execution is isolated behind a disabled adapter while the strategy accumulates paper-trading and calibration evidence.

## What It Does

- Discovers active Polymarket weather markets through Gamma search and event endpoints.
- Parses binary YES/NO and bucketed temperature markets into normalized Fahrenheit intervals.
- Pulls weather forecasts from Open-Meteo forecast, GFS, ECMWF, GraphCast-through-GFS, and ensemble endpoints where available.
- Extracts weather features including precipitation, cloud cover, wind, humidity, dew point, pressure range, apparent temperature, and solar radiation.
- Builds multiple probability views, then aggregates them at the independent weather-source level to avoid overstating model agreement.
- Applies observation-aware same-day corrections so already-reached or already-impossible outcomes are handled explicitly.
- Uses price-aware Kelly edge, spread, price-band, source-agreement, and lead-time filters before paper-trading.
- Maintains a SQLite paper ledger with Kelly sizing, position updates, expired-position settlement, run history, and calibration tables.
- Exposes a disabled live execution boundary for future Polymarket API or MCP integration after paper validation.

## Why It Is Interesting

Weather markets look simple, but the implementation has several real trading-system problems:

- Market text is inconsistent and often binary rather than a clean multi-outcome chain.
- Polymarket discovery can be slow or flaky, so live runs need bounded request sizes, error isolation, and runtime budgets.
- Forecast models are highly correlated. Counting every transform of one weather source as an independent model creates false confidence.
- Same-day temperature markets change character after the local daily high is likely already reached.
- A paper strategy is useless unless expired positions settle and forecast scores become calibratable.

DailyWeather addresses those problems directly in code.

## Repository Layout

```text
weather_strategy/
  cli.py            Command-line workflows for live scans, paper runs, reports, calibration
  polymarket.py     Gamma discovery and CLOB quote normalization
  parser.py         Market text, city, date, and temperature-bucket parsing
  weather.py        Forecast ingestion and weather-feature extraction
  observations.py   Observed-high ingestion from NWS, METAR, and fallback proxy data
  forecast.py       Probability models, consensus, and same-day path adjustment
  signals.py        Fair-value edge scoring and entry filters
  paper.py          SQLite paper ledger, Kelly rebalancing, settlement, calibration
  execution.py      Disabled live-execution boundary
tests/              Unit tests and deterministic fixtures
docs/               Strategy and operations notes
```

## Quick Start

Run the test suite:

```bash
python3 -m unittest discover -s tests
```

Run a deterministic fixture-based paper pass:

```bash
python3 -m weather_strategy.cli paper-run \
  --fixture tests/fixtures/weather_markets.json \
  --ledger work/data/paper_trades.sqlite \
  --min-model-count 1
```

Run a bounded live paper pass:

```bash
python3 -m weather_strategy.cli paper-run \
  --ledger work/data/weather_kelly_paper.sqlite \
  --limit 150 \
  --discovery-request-limit 50 \
  --discovery-pages 1 \
  --max-runtime-seconds 90 \
  --progress-every 10 \
  --min-edge 0.08 \
  --min-model-agreement 0.65 \
  --high-confidence-price-threshold 0.75 \
  --high-confidence-min-kelly-edge 0.02 \
  --bankroll-usd 1000 \
  --kelly-fraction 0.25 \
  --max-position-usd 50 \
  --min-trade-usd 1 \
  --same-day-entry-start-hour 11 \
  --same-day-entry-cutoff-hour 17 \
  --min-lead-days 1 \
  --max-lead-days 2 \
  --weights-file work/data/model_weights.json \
  --run-log-dir work/logs/paper_runs
```

Summarize the paper ledger:

```bash
python3 -m weather_strategy.cli report \
  --ledger work/data/weather_kelly_paper.sqlite \
  --bankroll-usd 1000
```

Summarize calibration:

```bash
python3 -m weather_strategy.cli calibration \
  --ledger work/data/weather_kelly_paper.sqlite
```

Backtest recorded forecast snapshots and fit model/source weights:

```bash
python3 -m weather_strategy.cli backtest \
  --ledger work/data/weather_kelly_paper.sqlite \
  --bankroll-usd 1000 \
  --kelly-fraction 0.25 \
  --max-position-usd 50 \
  --min-trade-usd 1 \
  --min-edge 0.08 \
  --min-model-agreement 0.65 \
  --train-fraction 0.70 \
  --output-weights work/data/model_weights.json \
  --max-observation-lookups 200 \
  --run-log-dir work/logs/backtests
```

Detailed automation artifacts are written under `work/logs/backtests/` and `work/logs/paper_runs/`. They include full scored-outcome detail, weights, settings, skipped-market reasons, and replay diagnostics.

## Trading Guardrails

The default workflow is paper-only. Real orders are not sent.

Current entry controls include:

- Price-aware minimum buffered Kelly edge.
- Minimum independent-source model agreement.
- Minimum source count.
- Lower effective absolute edge threshold for high-probability markets.
- Higher effective absolute edge threshold for low-probability longshots.
- Maximum spread.
- Price-band filter to avoid dust/illiquid prices.
- One selected entry per city/date group to reduce correlated exposure.
- Tomorrow/following-day lead-time focus for new entries.
- Same-day entry windows only when observation-aware scoring is useful.
- Actual station observations via NWS/METAR before forecast proxies.
- Existing-position hold logic separated from new-entry timing gates.
- Expired-position settlement before live discovery.

## Latest Validation Snapshot

After the latest backtest and weighting pass:

```text
Tests: 44 passed
Backtest:
  resolved forecast rows: 2484
  train/test rows: 1738 / 746
  held-out Brier score, market vs calibrated FV: 0.078101 / 0.061802
  held-out Kelly replay, calibrated weights: 16 trades, 50.0% hit rate, +$3624.28 simulated PnL
Temporary weighted live paper smoke:
  markets discovered/scored/skipped: 242 / 5 / 237
  forecast rows inserted: 5
  settled expired positions: 0
  weather/quote/observation/settlement errors: 0 / 0 / 0 / 0
  signals: 0
  Kelly executions: 0
  runtime_limited: false
```

Backtest PnL is a research diagnostic, not a production claim: replay uses recorded forecast snapshots and recorded prices, not a full historical order book. The zero-signal live smoke is expected under the stricter model logic because low-probability apparent edges must clear a higher effective hurdle than high-probability contracts with the same absolute gap.

## Next Research Directions

- Add historical backfills so Brier score and log loss can populate faster than waiting for live paper runs.
- Add market-specific resolution station parsing instead of city-coordinate defaults.
- Add NOAA NBM, more explicit HRRR/NBM coverage, and market-specific station verification where available.
- Build reliability tables by city, source, lead time, and bucket type.
- Add a portfolio optimizer that can reason over mutually exclusive city/date buckets.
- Only after sustained calibration, implement a real execution adapter behind `weather_strategy.execution`.

## Disclaimer

This is a research and paper-trading project. It is not financial advice and does not place real trades by default.
