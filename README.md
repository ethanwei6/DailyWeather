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
  --limit 250 \
  --discovery-request-limit 50 \
  --discovery-pages 1 \
  --max-runtime-seconds 420 \
  --progress-every 10 \
  --min-edge 0.08 \
  --min-model-agreement 1.0 \
  --hold-min-model-agreement 0.65 \
  --hold-min-fair-value 0.60 \
  --hold-market-confirmation-price 0.80 \
  --hold-market-confirmation-min-fair-value 0.50 \
  --min-signal-fair-value 0.70 \
  --min-price 0.125 \
  --yes-side-min-price 0.20 \
  --allow-no-side-entries \
  --no-side-min-edge 0.10 \
  --no-side-high-confidence-min-edge 0.02 \
  --no-side-max-price 0.95 \
  --no-side-max-counter-event-probability 0.10 \
  --hold-no-side-max-counter-event-probability 0.15 \
  --high-confidence-price-threshold 0.75 \
  --high-confidence-min-kelly-edge 0.02 \
  --bankroll-usd 100 \
  --kelly-fraction 0.75 \
  --compound-kelly-sizing \
  --max-position-usd 100 \
  --max-position-fraction 0.25 \
  --edge-position-full-cap-edge 0.25 \
  --edge-position-min-multiplier 0.35 \
  --min-trade-usd 1 \
  --same-day-entry-start-hour 11 \
  --same-day-entry-cutoff-hour 17 \
  --min-lead-days 1 \
  --max-lead-days 2 \
  --weights-file work/data/model_weights.json \
  --run-log-dir work/logs/live_forward_paper
```

The live-forward automation runs this paper-only profile at `00:00`, `06:00`, `12:00`, and `18:00` UTC so global city markets are checked before the local weather day begins across Americas, Europe, and Asia-Pacific. It uses each market city's timezone for the local lead-day filter and writes detailed JSON logs with scored rows, `passes_signal_filter` / `signal_filter_reason`, signal-filter counts, skipped-market reasons, per-city coverage, local lead-day timing, positions, equity, and PnL.

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

Detailed automation artifacts are written under `work/logs/backtests/` and `work/logs/paper_runs/`. They include full scored-outcome detail, weights, settings, skipped-market reasons, and replay diagnostics.

Run a long historical replay over resolved Polymarket weather markets:

```bash
python3 -m weather_strategy.cli long-backtest \
  --bankroll-usd 100 \
  --pages 20 \
  --limit-per-page 50 \
  --max-markets 8000 \
  --max-runtime-seconds 180 \
  --entry-hours-utc 0,12 \
  --min-lead-days 1 \
  --max-lead-days 2 \
  --max-price-staleness-minutes 90 \
  --historical-price-slippage 0.01 \
  --forecast-availability-lag-hours 6 \
  --kelly-fraction 0.75 \
  --compound-kelly-sizing \
  --max-position-usd 100 \
  --max-position-fraction 0.25 \
  --edge-position-full-cap-edge 0.25 \
  --edge-position-min-multiplier 0.35 \
  --kelly-market-blend 0.0 \
  --min-trade-usd 1 \
  --min-edge 0.08 \
  --min-signal-fair-value 0.70 \
  --min-price 0.125 \
  --yes-side-min-price 0.20 \
  --min-model-agreement 1.0 \
  --hold-min-model-agreement 0.65 \
  --hold-min-fair-value 0.60 \
  --hold-market-confirmation-price 0.80 \
  --hold-market-confirmation-min-fair-value 0.50 \
  --run-log-dir work/logs/long_backtests \
  --cache-dir work/cache/long_backtest \
  --summary-only \
  --allow-no-side-entries \
  --no-side-min-edge 0.10 \
  --no-side-high-confidence-min-edge 0.02 \
  --no-side-max-price 0.95 \
  --no-side-max-counter-event-probability 0.10 \
  --hold-no-side-max-counter-event-probability 0.15
```

Add `--allow-no-side-entries` to replay or paper-trade real NO-token entries for binary markets. This is useful for research and paper validation, but real execution remains disabled until there is more out-of-sample evidence and explicit live risk-control review. The default `--no-side-min-edge 0.10` applies an extra absolute-edge floor to normal NO-token entries because low-edge NO trades were the weakest cohort in the long replay. The default `--no-side-high-confidence-min-edge 0.02` allows smaller absolute edges only when the explicit NO token already trades at or above the high-confidence price threshold. The default `--no-side-max-price 0.95` now matches the global max-price gate after the latest replay showed a small PnL and calibration improvement versus the prior 93c cap; set it below `--max-price` to make NO entries stricter again. The default `--no-side-max-counter-event-probability 0.10` blocks new NO entries when any underlying model view still gives the opposite YES event more than a 10% chance. Existing NO positions use the wider default `--hold-no-side-max-counter-event-probability 0.15`, which reduced churn on high-FV holds in the latest replay without loosening new-entry standards.

Long-backtest artifacts are written under `work/logs/long_backtests/`. They include scored outcomes, execution detail, calibration buckets, trade diagnostics, skipped-market reasons, data-provenance counters, a strict `real_data_audit` pass/fail block, and a `strategy_recommendation_diagnostics` block that compares clean sizing/tail variants before promoting a next paper-test profile.

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
- Existing positions are preserved instead of trimmed back to Kelly target when the hold thesis remains valid; this reduces churn after the market starts confirming a good entry.
- Bounded exact/range temperature buckets disabled by default for new entries because station-source rounding can make settlement cross-checks ambiguous.
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

After the latest long-backtest validation pass:

```text
Tests: 115 passed

YES-only reference replay:
  bankroll: $100
  raw markets discovered: 5405
  parsed markets: 3141
  markets with real CLOB price history: 3141
  scored rows: 8235
  trades: 5
  executions: 10
  buys / sells: 5 / 5
  ending equity: $158.30
  PnL: +$58.30
  traded weather cross-check mismatches: 0
  market-level weather cross-check mismatches: 0
  ambiguous exact/range station checks: 46
  price-history misses: 0
  forecast payload misses: 1119
  future/stale price violations: 0 / 0
  unavailable forecast violations: 0
  forecast availability lag: 6h
  counterfactual trim-valid-holds-to-Kelly replay: +$44.59
  counterfactual Kelly replay best tested variant: current defaults
  runtime_limited: false
  run log: work/logs/long_backtests/20260619T081433Z-long-backtest.json

Aggressive 25% paper-only YES+NO historical replay:
  bankroll: $100
  raw markets discovered: 10588
  parsed markets: 6804
  markets with usable price history: 3941
  scored rows: 20373
  trades: 38
  executions: 86
  buys / sells / settlements: 47 / 16 / 23
  ending equity: $640.93
  PnL: +$540.93
  NO-side PnL: +$379.47 across 30 trades
  YES-side PnL: +$161.46 across 8 trades
  traded weather cross-check mismatches: 0
  traded weather-checked tokens: 38 / 38
  traded weather-matched tokens: 38 / 38
  traded ambiguous / Polymarket-only / unresolved tokens: 0 / 0 / 0
  signal-eligible weather-checked rows: 64 / 64
  signal-eligible weather-matched rows: 64 / 64
  market-level weather cross-check mismatches: 0
  ambiguous exact/range station checks: 56
  YES price-history errors: 2686
  NO price-history misses/errors: 0
  forecast payload misses: 622
  future/stale price violations: 0 / 0
  unavailable forecast violations: 0
  forecast availability lag: 6h
  fractional Kelly: 0.75
  compound Kelly sizing: true
  minimum price: 0.125
  YES-side minimum price: 0.20
  NO-side normal minimum absolute edge: 0.10
  NO-side high-confidence minimum absolute edge: 0.02 at 75c+ NO price
  NO-side maximum entry price: 0.95
  NO-side maximum entry counter-event probability: 0.10
  NO-side maximum hold counter-event probability: 0.15
  current paper-test max position: 25% of current sizing equity, with absolute fail-safe set to starting bankroll
  previous 20% paper-test profile: +$384.84 with $13.94 max drawdown
  safer 15% baseline profile: +$234.56 with $7.83 max drawdown
  edge-scaled cap: applied after the percentage cap; 35% floor, full cap at 25c buffered edge
  current paper-test profile max drawdown: $20.40
  legacy 5% cap selected profile: +$51.52 with $1.41 max drawdown
  10% cap selected profile: +$127.51 with $3.87 max drawdown
  15% baseline cap selected profile: +$234.56 with $7.83 max drawdown
  legacy 9% entry-tail profile: +$376.37 across 35 trades
  legacy 9% hold-tail profile: +$363.11 across 38 trades
  prior 93c NO max-price profile: +$376.81 across 31 trades
  legacy 90c NO max-price profile: +$369.54 across 26 trades
  realized losing trades: 1, combined loss -$0.19
  realized trade hit rate: 97.37%
  event hit rate: 94.74% with 36 event wins and 2 event losses
  profitable event-loser trades: 2
  top 1 trade share of PnL: 14.82%
  top 5 trade share of PnL: 54.18%
  top 10 trade share of PnL: 81.19%
  runtime_limited: false
  selected-candidate calibration: 38 traded tokens, 94.74% actual rate, 97.33% average FV, 81.84% average market price
  selected-candidate Brier score, model vs market: 0.051043 / 0.082777
  fresh-bankroll first-half / second-half slice PnL: +$177.59 / +$153.10
  fresh-bankroll last-30% session slice PnL: +$117.48
  weakest positive monthly slices: April +$22.70, June +$7.35
  real_data_audit: passed
  run log: work/logs/long_backtests/20260620T053819Z-1781933899864632000-long-backtest.json
```

Backtest PnL is a research diagnostic, not a production claim. The long replay uses real Gamma market discovery, CLOB historical price bars with market-lifetime `startTs`/`endTs` bounds, Open-Meteo Single Runs historical forecast payloads where available, Polymarket settlement prices, and station METAR/ASOS weather cross-checks where available. The artifact records timestamp-quality diagnostics for every scored row and execution: there were no future-price, stale-price, or unavailable-forecast violations, every forecast used a six-hour availability lag, and the 90-minute price-staleness cap kept the maximum scored quote age to one hour in the latest completed replay. Settlement-quality diagnostics separate Polymarket-only, weather-checked, ambiguous, and unresolved rows; all 38 traded tokens were independently weather-checked and matched settlement. The broad fair-value model is still worse calibrated than market price overall, so the profitable result comes from a filtered subset rather than global model superiority. The useful filters are the `70%` model fair-value gate, full model-source agreement for new entries, disabled bounded exact/range bucket entries, valid-hold preservation, the split `10c` normal / `2c` high-confidence NO-side absolute-edge floor, the `95c` max-price gate, the split `10%` NO-side entry / `15%` NO-side hold counter-event tail-risk gates, the split `12.5c` general / `20c` YES-side price floor, and the corrected edge-scaled cap. The latest promoted paper profile uses a true `$100` bankroll with a `25%` current-equity position cap, an absolute fail-safe set to the starting bankroll, and ended at `$640.93`; the safer `15%` baseline made `+$234.56` with a smaller `$7.83` max drawdown. The `25%` cap is promoted for paper testing only because it kept the same 38 trades, event hit rate, and clean settlement checks while raising in-sample PnL to `+$540.93`; real execution remains disabled. Moving the NO-side entry-tail cap from 9% to 10% added three weather-matched event winners and improved PnL without admitting the weaker 12%+ tail cohort. Moving the NO-side hold-tail cap from 9% to 15% remains important because the legacy 9% hold-tail replay over-churns high-FV NO positions. Looser NO-entry tail caps made more in sample but are not promoted because they lowered event hit rate or introduced ambiguous weather validation. Exact/range station markets can disagree with settlement because of resolution-source rounding, so those are logged as ambiguous and are not eligible for new entries by default.

The current long-backtest code also emits `real_data_audit`, which fails fast if a replay uses fixture forecasts, missing forecast/price timestamps, future or stale CLOB prices, unavailable forecast runs, synthetic NO pricing, unresolved traded tokens, ambiguous traded settlement, or weather/payout mismatches.

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
