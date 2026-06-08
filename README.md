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
- Uses edge, spread, price-band, source-agreement, and lead-time filters before paper-trading.
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
  observations.py   Observed-high ingestion from NWS and Open-Meteo proxy data
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
  --bankroll-usd 1000 \
  --kelly-fraction 0.25 \
  --max-position-usd 50 \
  --min-trade-usd 1 \
  --same-day-entry-start-hour 11 \
  --same-day-entry-cutoff-hour 17 \
  --min-lead-days 1 \
  --max-lead-days 2
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

## Trading Guardrails

The default workflow is paper-only. Real orders are not sent.

Current entry controls include:

- Minimum edge after uncertainty buffer.
- Minimum independent-source model agreement.
- Minimum source count.
- Maximum spread.
- Price-band filter to avoid dust/illiquid prices.
- One selected entry per city/date group to reduce correlated exposure.
- Tomorrow/following-day lead-time focus for new entries.
- Same-day entry windows only when observation-aware scoring is useful.
- Expired-position settlement before live discovery.

## Latest Validation Snapshot

After the latest hardening pass:

```text
Tests: 35 passed
Live paper run:
  markets discovered/scored/skipped: 572 / 111 / 422
  forecast rows inserted: 111
  settled expired positions: 3
  weather/quote/observation/settlement errors: 0 / 0 / 0 / 0
  signals: 0
  Kelly executions: 0
  runtime_limited: true
```

The zero-signal result is expected under the stricter model logic: the top apparent edges were mostly ultra-low-price contracts below the tradeable price band.

## Next Research Directions

- Add historical backfills so Brier score and log loss can populate faster than waiting for live paper runs.
- Add market-specific resolution station parsing instead of city-coordinate defaults.
- Add NOAA NBM, HRRR, METAR, and station-specific verification where available.
- Build reliability tables by city, source, lead time, and bucket type.
- Add a portfolio optimizer that can reason over mutually exclusive city/date buckets.
- Only after sustained calibration, implement a real execution adapter behind `weather_strategy.execution`.

## Disclaimer

This is a research and paper-trading project. It is not financial advice and does not place real trades by default.
