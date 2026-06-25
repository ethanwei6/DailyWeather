# DailyWeather

DailyWeather is a Python research system for pricing daily-high temperature prediction markets. It discovers live Polymarket weather contracts, maps market text into city/date/temperature buckets, builds a weather-model probability consensus, and runs a paper-only Kelly rebalancing ledger.

The project is intentionally built like a trading system, not a notebook. Live execution is isolated behind a disabled adapter while the strategy accumulates paper-trading and calibration evidence.

## What It Does

- Discovers active Polymarket weather markets through Gamma search and event endpoints.
- Replays resolved historical markets through Telonex market datasets and tick-level quote Parquet.
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
  telonex.py        Telonex historical market and quote-data client
  parser.py         Market text, city, date, and temperature-bucket parsing
  weather.py        Forecast ingestion and weather-feature extraction
  observations.py   Observed-high ingestion from NWS, METAR, and fallback proxy data
  forecast.py       Probability models, consensus, and same-day path adjustment
  signals.py        Fair-value edge scoring and entry filters
  paper.py          SQLite paper ledger, Kelly rebalancing, settlement, calibration
  long_backtest.py  Historical Polymarket/weather replay with cached real API payloads
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
  --ledger work/data/weather_live_forward_100.sqlite \
  --strategy-profile live-forward-utc12-relaxed-no-tail-0.20-trim-holds \
  --limit 250 \
  --discovery-request-limit 50 \
  --discovery-pages 1 \
  --max-runtime-seconds 420 \
  --progress-every 10 \
  --same-day-entry-start-hour 11 \
  --same-day-entry-cutoff-hour 17 \
  --weights-file work/data/model_weights.json \
  --run-log-dir work/logs/live_forward_paper
```

The `live-forward-utc12-relaxed-no-tail-0.20-trim-holds` profile applies the Telonex-tested `$100` paper settings, including explicit NO-token entries, `75%` fractional Kelly, `25%` current-equity max-position cap, strict `10%` NO counter-event risk outside UTC noon, a relaxed `20%` NO counter-event cap only for the `12:00 UTC` window, and Kelly-target trimming for valid holds. Use `--strategy-profile live-forward-utc12-relaxed-no-tail-0.20` to keep the same entry gates while preserving valid holds at their current notional.

The live-forward automation runs this paper-only profile at `00:00`, `06:00`, `12:00`, and `18:00` UTC so global city markets are checked before the local weather day begins across Americas, Europe, and Asia-Pacific. It uses each market city's timezone for the local lead-day filter and writes detailed JSON logs with scored rows, `passes_signal_filter` / `signal_filter_reason`, signal-filter counts, skipped-market reasons, per-city coverage, bucket-shape cohorts, local lead-day timing, positions, equity, PnL, and the applied `strategy_profile`.

Summarize the paper ledger:

```bash
python3 -m weather_strategy.cli report \
  --ledger work/data/weather_live_forward_100.sqlite \
  --bankroll-usd 100
```

Summarize calibration:

```bash
python3 -m weather_strategy.cli calibration \
  --ledger work/data/weather_live_forward_100.sqlite
```

Backtest recorded forecast snapshots and fit model/source weights:

```bash
python3 -m weather_strategy.cli backtest \
  --ledger work/data/weather_kelly_paper.sqlite \
  --bankroll-usd 1000 \
  --kelly-fraction 0.25 \
  --max-position-usd 1000 \
  --max-position-fraction 0.25 \
  --edge-position-full-cap-edge 0.25 \
  --edge-position-min-multiplier 0.35 \
  --min-trade-usd 1 \
  --min-edge 0.08 \
  --min-model-agreement 1.0 \
  --hold-min-model-agreement 0.65 \
  --hold-min-fair-value 0.60 \
  --hold-market-confirmation-price 0.80 \
  --hold-market-confirmation-min-fair-value 0.50 \
  --min-signal-fair-value 0.70 \
  --min-price 0.125 \
  --yes-side-min-price 0.20 \
  --no-side-min-edge 0.10 \
  --no-side-high-confidence-min-edge 0.02 \
  --no-side-max-price 0.95 \
  --no-side-max-counter-event-probability 0.10 \
  --hold-no-side-max-counter-event-probability 0.15 \
  --train-fraction 0.70 \
  --output-weights work/data/model_weights.json \
  --max-observation-lookups 200 \
  --run-log-dir work/logs/backtests
```

Detailed automation artifacts are written under `work/logs/backtests/`, `work/logs/live_forward_paper/`, and `work/logs/paper_runs/`. They include full scored-outcome detail, weights, settings, skipped-market reasons, and replay diagnostics.

Run a long historical replay over resolved Polymarket weather markets:

```bash
python3 -m weather_strategy.cli long-backtest \
  --strategy-profile live-forward-utc12-relaxed-no-tail-0.20-trim-holds \
  --bankroll-usd 100 \
  --pages 20 \
  --limit-per-page 50 \
  --max-markets 8000 \
  --max-runtime-seconds 180 \
  --price-source telonex \
  --market-source telonex \
  --entry-hours-utc 0,12 \
  --min-lead-days 1 \
  --max-lead-days 2 \
  --max-price-staleness-minutes 90 \
  --historical-price-slippage 0.01 \
  --forecast-availability-lag-hours 6 \
  --run-log-dir work/logs/telonex_backtests \
  --cache-dir work/cache/telonex_long_backtest \
  --summary-only
```

Add `--allow-no-side-entries` to replay or paper-trade real NO-token entries for binary markets. This is useful for research and paper validation, but real execution remains disabled until there is more out-of-sample evidence and explicit live risk-control review. The default `--no-side-min-edge 0.10` applies an extra absolute-edge floor to normal NO-token entries because low-edge NO trades were the weakest cohort in the long replay. The default `--no-side-high-confidence-min-edge 0.02` allows smaller absolute edges only when the explicit NO token already trades at or above the high-confidence price threshold. The default `--no-side-max-price 0.95` now matches the global max-price gate after the latest replay showed a small PnL and calibration improvement versus the prior 93c cap; set it below `--max-price` to make NO entries stricter again. The default `--no-side-max-counter-event-probability 0.10` blocks new NO entries when any underlying model view still gives the opposite YES event more than a 10% chance. Existing NO positions use the wider default `--hold-no-side-max-counter-event-probability 0.15`, which reduced churn on high-FV holds in the latest replay without loosening new-entry standards.

Use `--disable-bounded-no-side-entries` to research a stricter NO profile that blocks exact/range NO buckets while keeping open-ended NO tails available. The latest Telonex replay did not promote that profile: it reduced PnL and increased concentration on the same real-data slice.

Long-backtest artifacts are written under `work/logs/long_backtests/` or the configured run-log directory. Current backtests default to Telonex for historical Polymarket market discovery and tick-level quote data. They include scored outcomes, execution detail, calibration buckets, bucket-shape trade cohorts, PnL concentration, skipped-market reasons, data-provenance counters, a strict `real_data_audit` pass/fail block, and a `strategy_recommendation_diagnostics` block that compares clean sizing/tail variants before promoting a next paper-test profile.

Telonex is optional at install time but required for current historical replays:

```bash
python3 -m pip install '.[telonex]'
cp .env.example .env
# then set TELONEX_API_KEY in .env; never commit .env
```

## Trading Guardrails

The default workflow is paper-only. Real orders are not sent.

Current entry controls include:

- Price-aware minimum buffered Kelly edge.
- Minimum independent-source model agreement.
- Minimum source count.
- Lower effective absolute edge threshold for high-probability markets.
- Higher effective absolute edge threshold for low-probability longshots.
- Maximum spread.
- Default `12.5c` general minimum price gate, plus a stricter `20c` YES-side entry floor. The split keeps low-price NO research entries available while blocking low-price YES longshots, which were weak in the expanded stress replay.
- Default `70%` minimum model fair-value gate for new entries, based on the long replay showing weak realized accuracy in mid-confidence apparent edges.
- Default full model-source agreement for new entries, with a looser `65%` agreement threshold for holding existing positions when the thesis remains positive.
- The current forward-paper profile trims valid holds back to the updated Kelly target. The preserved-hold profile remains available for comparison, but the latest Telonex slice showed better PnL, slightly lower drawdown, and slightly lower concentration with trimming.
- Bounded exact/range temperature buckets are allowed only through stricter price, fair-value, edge, and full-agreement gates. Bounded NO-side entries can be disabled separately with `--disable-bounded-no-side-entries` for research.
- One selected entry per city/date group to reduce correlated exposure.
- Tomorrow/following-day lead-time focus for new entries.
- Same-day entry windows only when observation-aware scoring is useful.
- Actual station observations via NWS/METAR before forecast proxies.
- Historical station METAR/ASOS cross-checks for resolved station markets.
- Experimental NO-token entries require an additional default `10c` absolute-edge floor because low-edge NO trades were historically weak.
- Experimental NO-token entries also require low model-tail risk: no individual model view may assign more than `10%` probability to the opposite YES event, and new NO entries are capped at the global `95c` max-price gate.
- Kelly targets can compound from current paper/replay equity after the strategy proves itself; this scales winning capital without loosening signal gates.
- Optional edge-scaled position caps keep marginal low-edge trades smaller while allowing the full cap once buffered edge reaches `25c`.
- Existing-position hold logic separated from new-entry timing gates.
- Expired-position settlement before live discovery.

## Latest Validation Snapshot

Current Telonex-backed replay:

```text
Profile: live-forward-utc12-relaxed-no-tail-0.20-trim-holds
Tests: 141 passed
Source: Telonex Polymarket market dataset + Telonex daily quote Parquet, Open-Meteo Single Runs forecasts, station METAR/ASOS cross-checks
Bankroll: $100
Raw markets discovered: 1000
Parsed markets: 100
Markets with Telonex price history: 100
Sessions: 20
Scored rows: 1856
Signals: 22
Trades: 12 completed tokens
Executions: 32
Buys / sells / settlements: 16 / 16 / 0
Ending equity: $294.76
PnL: +$194.76
Return: +194.76%
Max drawdown in selected replay: $7.02
Event hit rate: 75% on 12 traded tokens
Top-1 PnL share: 62.25%
Bucket-shape PnL: upper-tail +$132.06 on 3 trades; bounded +$62.71 on 9 trades
Unprofitable event winners: 1 token, -$3.05 realized PnL
Weather cross-check mismatches: 0
Traded weather-checked executions: 32 / 32
Future/stale price violations: 0 / 0
Unavailable forecast violations: 0
Forecast availability lag: 6h
Price-history errors: 0
NO-side price-history errors: 0
Runtime-limited: false
  real_data_audit: passed
```

Backtest PnL is a research diagnostic, not a production claim. The current Telonex-backed sample is clean but still small and concentrated: the top trade contributes 62.25% of total PnL. Disabling bounded NO-side exact/range entries reduced PnL and increased concentration, so it is not promoted. A global 20% NO-entry counter-event tail produced more trades and higher in-sample PnL (+$216.46 on 18 trades), but also much higher gross exposure and max drawdown ($24.16), so it remains high risk. The narrower UTC-12-only relaxed 20% NO-entry tail with Kelly-target hold trimming is the current forward-paper candidate because it increased trade count versus the strict baseline while preserving weather-validation cleanliness and keeping drawdown materially below the global-loose variants. The replay still shows one ultimately winning event that realized a loss after an intermediate sell; that is tracked explicitly as `unprofitable_event_winner_trades` for future trader-logic work. Older Gamma/CLOB artifacts showed much higher headline PnL over broader saved slices; those are now treated as legacy research until reproduced on Telonex tick-level quote data.

Historical quote replay requests are bounded to the strategy's scheduled decision and maintenance timestamps, rather than the full market lifetime. That keeps Telonex replays faster, avoids unnecessary future quote history, and matches the live strategy's available-information set more closely.

The current long-backtest code emits `real_data_audit`, which fails fast if a replay uses fixture forecasts, missing forecast/price timestamps, future or stale historical prices, unavailable forecast runs, synthetic NO pricing, unresolved traded tokens, ambiguous traded settlement, or weather/payout mismatches.

Scored-outcome JSON now separates `timing_entry_eligible` from `passes_signal_filter`, `signal_eligible`, and `trade_eligible`. This matters because some ultra-low-price rows can have large apparent fair-value gaps while still being rejected by price, fair-value, NO-tail, or exact-bucket gates; analysis should use `passes_signal_filter`/`signal_filter_reason` for tradability, not the timing-only legacy `entry_eligible` field.

## Next Research Directions

- Add historical backfills so Brier score and log loss can populate faster than waiting for live paper runs.
- Add market-specific resolution station parsing instead of city-coordinate defaults.
- Refresh missing historical forecast payloads with stable live network access and re-run the same replay without cache misses.
- Continue paper-validating explicit NO-token positions before adding any real execution adapter support.
- Continue testing the split `12.5c` general / `20c` YES-side price floors, split `10c` normal / `2c` high-confidence NO-side edge floor, `95c` max-price gate, and split `10%` NO-entry / `15%` NO-hold counter-event tail-risk gates out of sample.
- Continue testing stricter NO-side price/lead-time controls; the latest replay still shows that the highest-price NO buckets carry worse marginal reliability despite sometimes adding in-sample PnL.
- Add NOAA NBM, more explicit HRRR/NBM coverage, and market-specific station verification where available.
- Build reliability tables by city, source, lead time, and bucket type.
- Add a portfolio optimizer that can reason over mutually exclusive city/date buckets.
- Only after sustained calibration, implement a real execution adapter behind `weather_strategy.execution`.

## Disclaimer

This is a research and paper-trading project. It is not financial advice and does not place real trades by default.
