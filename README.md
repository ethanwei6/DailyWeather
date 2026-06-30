# DailyWeather

DailyWeather is a Python research system for pricing daily-high temperature prediction markets. It discovers live Polymarket weather contracts, maps market text into city/date/temperature buckets, builds a weather-model probability consensus, and tests Kelly-style trading rules against paper and guarded live-execution ledgers.

The project is intentionally built like a trading system, not a notebook. The core research loop is live-like historical replay: build one real-data scored-outcome artifact, then sweep strategy profiles without refetching Telonex or weather data.

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
- Exposes guarded live execution separately from research replay and paper validation.

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
  backtest_engine.py Comparison reports, breakdowns, and equity-curve artifacts
  execution.py      Guarded live-execution adapter
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

Run a performance-first live paper pass. Temperature markets resolve daily, so broader market discovery and complete scoring matter more than shaving minutes from runtime:

```bash
python3 -m weather_strategy.cli paper-run \
  --ledger work/data/weather_live_forward_100.sqlite \
  --strategy-profile live-forward-strict-no-tail-0.11-preserve-highconv-bounded-edge-0.10 \
  --limit 2000 \
  --discovery-request-limit 50 \
  --discovery-pages 20 \
  --max-runtime-seconds 3600 \
  --progress-every 25 \
  --same-day-entry-start-hour 11 \
  --same-day-entry-cutoff-hour 17 \
  --weights-file work/data/model_weights.json \
  --run-log-dir work/logs/live_forward_paper
```

Run the current `$50` window-bankroll live-forward candidate:

```bash
python3 -m weather_strategy.cli paper-run \
  --ledger work/data/weather_live_forward_50.sqlite \
  --strategy-profile live-forward-50-windowbank-0.25-kelly-0.50-poscap-0.25-strict-no-tail-0.14-bprice-0.70 \
  --limit 2000 \
  --discovery-request-limit 50 \
  --discovery-pages 20 \
  --max-runtime-seconds 3600 \
  --progress-every 25 \
  --same-day-entry-start-hour 11 \
  --same-day-entry-cutoff-hour 17 \
  --weights-file work/data/model_weights.json \
  --run-log-dir work/logs/live_forward_paper
```

The active live-money profile uses a `14%` NO counter-event cap, `50%` fractional Kelly, and a `25%` automation-window Kelly bankroll with a `25%` per-window-bankroll position cap. This favors diversification across more city/outcome bets over spending a whole run's exposure on one or two positions. Candidate details should be reviewed through generated live-like comparison artifacts rather than static tracked reports.

The `live-forward-strict-no-tail-0.11-preserve-highconv-bounded-edge-0.10` profile applies the Telonex-tested `$100` paper settings, including explicit NO-token entries, `75%` fractional Kelly, a `25%` current-equity max-position cap with a `$175` absolute cap, an `11%` NO counter-event risk cap at every entry hour, preservation of valid existing holds, a partial-exit rule that sells half of an invalid hold only when model FV remains at least `90%` and the quote is between `50c` and `65c`, a hold-only high-conviction exception that lets existing NO positions use a `20%` counter-event cap only when model FV is at least `98%` and buffered edge is at least `35c`, and a `10c` minimum edge for bounded exact/range buckets. Use `--strategy-profile live-forward-strict-no-tail-preserve-highconv-bounded-edge-0.10` for the stricter `10%` NO-tail comparison profile, `--strategy-profile live-forward-strict-no-tail-preserve-highconv-bounded-edge-0.15` for the stricter bounded-edge comparison profile, `--strategy-profile live-forward-strict-no-tail-trim-highconv-bounded-edge-0.15` for the lower-exposure trim-to-Kelly comparison profile, or `--strategy-profile live-forward-utc12-relaxed-no-tail-0.20-trim-highconv-bounded-edge-0.15` only as the higher-risk relaxed-tail comparison profile.

The live-forward automation runs this profile at `00:00`, `06:00`, `12:00`, and `18:00` UTC so global city markets are checked before the local weather day begins across Americas, Europe, and Asia-Pacific. Long historical backtests now default to the same four UTC windows and write `entry_hours_match_live_forward` into the run artifact. The live paper run uses each market city's timezone for the local lead-day filter and writes detailed JSON logs with scored rows, `passes_signal_filter` / `signal_filter_reason`, signal-filter counts, skipped-market reasons, per-city coverage, bucket-shape cohorts, local lead-day timing, positions, equity, PnL, and the applied `strategy_profile`.

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

Prepare local Polymarket API credentials without enabling live trading:

```bash
python3.12 -m venv .venv-live
.venv-live/bin/python -m pip install -r requirements-live.txt
.venv-live/bin/python -m weather_strategy.cli live-setup --create-wallet
.venv-live/bin/python -m weather_strategy.cli live-setup --derive-clob-creds
.venv-live/bin/python -m weather_strategy.cli live-setup --check-geoblock --clob-readonly-smoke
```

This writes a gitignored `.env.local` with `0600` permissions. The setup command prints only public addresses, boolean status, and redacted keys. It does not fund wallets, deploy a deposit wallet, approve tokens, place orders, or cancel orders. Keep `DAILYWEATHER_LIVE_TRADING=0` until there is explicit live-trading approval and a separate risk-control review. The geoblock check is a runtime sanity check for the current outbound network, not a substitute for legal or eligibility review.

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

Build a one-year live-like scored-outcome artifact over resolved Polymarket weather markets:

```bash
python3 -m weather_strategy.cli build-live-like-backtest \
  --strategy-profile live-forward-50-windowbank-0.25-kelly-0.50-poscap-0.25-strict-no-tail-0.14-bprice-0.70 \
  --bankroll-usd 50 \
  --lookback-days 365 \
  --max-markets 50000 \
  --max-runtime-seconds 0 \
  --price-source telonex \
  --market-source telonex \
  --entry-hours-utc 0,6,12,18 \
  --max-price-staleness-minutes 90 \
  --historical-price-slippage 0.01 \
  --forecast-availability-lag-hours 6 \
  --settlement-audit polymarket_only \
  --http-hard-timeout-seconds 300 \
  --run-log-dir work/logs/live_like_backtests \
  --cache-dir work/cache/live_like_backtest \
  --summary-only
```

Then compare candidate profiles from the saved scored outcomes without refetching forecasts or prices:

```bash
python3 -m weather_strategy.cli compare-live-like-strategies \
  --source-run-log work/logs/live_like_backtests/<artifact>-long-backtest.json \
  --run-log-dir work/logs/live_like_strategy_replays \
  --output-dir work/reports/live_like_strategy_comparison \
  --summary-only
```

The comparison output writes `summary.json`, `summary.md`, `trades.json`, and `equity_curves.svg`. It reports PnL, max drawdown, Sharpe, average monthly return, annualized return, trade count, side mix, city/region mix, target-month cohorts, entry-hour cohorts, win rates, and minimum cash. The current live profile is flagged explicitly in the comparison rows. See `docs/backtesting_engine.md` for the canonical workflow.

Add `--allow-no-side-entries` to replay or paper-trade real NO-token entries for binary markets. Live execution exists behind an explicit `--execution-mode live --confirm-live` path and should use the small configured bot bankroll plus the same profile tested in live-like replay. The default `--no-side-min-edge 0.10` applies an extra absolute-edge floor to normal NO-token entries because low-edge NO trades were the weakest cohort in the long replay. The default `--no-side-high-confidence-min-edge 0.02` allows smaller absolute edges only when the explicit NO token already trades at or above the high-confidence price threshold. The default `--no-side-max-price 0.95` now matches the global max-price gate after the latest replay showed a small PnL and calibration improvement versus the prior 93c cap; set it below `--max-price` to make NO entries stricter again. The default `--no-side-max-counter-event-probability 0.10` blocks new NO entries when any underlying model view still gives the opposite YES event more than a 10% chance. Existing NO positions use the wider default `--hold-no-side-max-counter-event-probability 0.15`, which reduced churn on high-FV holds in the latest replay without loosening new-entry standards.

Use `--disable-bounded-no-side-entries` to research a stricter NO profile that blocks exact/range NO buckets while keeping open-ended NO tails available. On the latest `$50` saved-data replay it removed the Dallas bounded-bucket loss cluster, but the promoted `$50` profile instead uses a bounded-confirmed rule: bounded exact/range buckets must have at least `98%` model FV and a `75c` market price. That keeps only market-confirmed, near-certain bounded trades while still allowing the strategy to learn from this cohort in forward paper.

Long-backtest artifacts are written under `work/logs/long_backtests/` or the configured run-log directory. Current backtests default to Telonex for historical Polymarket market discovery and tick-level quote data. They include scored outcomes, execution detail, calibration buckets, bucket-shape trade cohorts, PnL concentration, skipped-market reasons, data-provenance counters, a strict `real_data_audit` pass/fail block, and a `strategy_recommendation_diagnostics` block that compares clean sizing/tail variants before promoting a next paper-test profile.

Telonex is optional at install time but required for current historical replays:

```bash
python3 -m pip install '.[telonex]'
cp .env.example .env
# then set TELONEX_API_KEY in .env; never commit .env
```

## Trading Guardrails

Backtesting and paper runs are the default workflows. Real orders require the explicit live execution flags, local secret configuration, and the small bot bankroll configured outside the repository.

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
- The current forward-paper profile preserves valid existing holds after entry and partially exits only invalid holds that still have very high model FV. The trim-to-Kelly profile remains available for comparison, but the latest strict-entry Telonex replay showed better PnL in every chronological slice when valid holds were preserved.
- Bounded exact/range temperature buckets are allowed only through stricter price, fair-value, edge, and full-agreement gates. Bounded NO-side entries can be disabled separately with `--disable-bounded-no-side-entries` for research.
- One selected entry per city/date group to reduce correlated exposure.
- Tomorrow/following-day lead-time focus for new entries.
- Same-day entry windows only when observation-aware scoring is useful.
- Actual station observations via NWS/METAR before forecast proxies.
- Historical station METAR/ASOS cross-checks for resolved station markets.
- Experimental NO-token entries require an additional default `10c` absolute-edge floor because low-edge NO trades were historically weak.
- Experimental NO-token entries also require low model-tail risk: no individual model view may assign more than `10%` probability to the opposite YES event, and new NO entries are capped at the global `95c` max-price gate.
- Kelly targets compound from current paper/replay equity, with a `$175` absolute cap and unchanged `25%` current-equity cap. This lets paper capital scale after wins without loosening signal gates or changing early `$100` bankroll exposure.
- Optional edge-scaled position caps keep marginal low-edge trades smaller while allowing the full cap once buffered edge reaches `25c`.
- Existing-position hold logic separated from new-entry timing gates.
- Expired-position settlement before live discovery.

## Validation Policy

Backtest PnL is a research diagnostic, not a production claim. Performance should be reported from generated artifacts under `work/reports/` and `work/logs/`, not hard-coded into tracked docs. The current validation standard is:

- use Telonex market datasets and tick-level quote data,
- use historical forecast payloads with the same availability lag the live trader would have had,
- use the same four UTC automation windows as the live trader,
- use Polymarket resolved payouts for broad settlement, with station weather cross-checks available as an audit mode,
- compare the current live profile against candidate profiles from the same scored-outcome artifact,
- report PnL, drawdown, Sharpe, monthly/annualized return, trade count, hit rate, side/region/city mix, target-month cohorts, and entry-hour cohorts.

Model diagnostics should report calibration by full model key, forecast source, and model family for both all resolved rows and selected signal rows. Source-level Brier improvements must improve the end-to-end replay, not just the probability table.

Historical quote replay requests are bounded to the strategy's scheduled decision and maintenance timestamps, rather than the full market lifetime. That keeps Telonex replays faster, avoids unnecessary future quote history, and matches the live strategy's available-information set more closely.

Replay diagnostics now include completed-trade cohorts by city, side, bucket shape, entry hour, lead day, first-buy price, first-buy fair-value bucket, edge bucket, model agreement, weather cross-check status, and NO-side counter-event probability. This is the main review surface for deciding whether future changes are real improvements or just headline-PnL overfit.

The current long-backtest code emits `real_data_audit`, which fails fast if a replay uses fixture forecasts, missing forecast/price timestamps, future or stale historical prices, unavailable forecast runs, synthetic NO pricing, unresolved traded tokens, ambiguous traded settlement, or weather/payout mismatches.

Scored-outcome JSON now separates `timing_entry_eligible` from `passes_signal_filter`, `signal_eligible`, and `trade_eligible`. This matters because some ultra-low-price rows can have large apparent fair-value gaps while still being rejected by price, fair-value, NO-tail, or exact-bucket gates; analysis should use `passes_signal_filter`/`signal_filter_reason` for tradability, not the timing-only legacy `entry_eligible` field.

## Next Research Directions

- Add historical backfills so Brier score and log loss can populate faster than waiting for live paper runs.
- Add market-specific resolution station parsing instead of city-coordinate defaults.
- Refresh missing historical forecast payloads with stable live network access and re-run the same replay without cache misses.
- Continue validating explicit NO-token positions with the live-like replay and small live-money bankroll before increasing capital.
- Continue testing the split `12.5c` general / `20c` YES-side price floors, split `10c` normal / `2c` high-confidence NO-side edge floor, `95c` max-price gate, split `10%` NO-entry / `15%` default NO-hold counter-event gates, and the `20%` high-conviction NO-hold exception out of sample.
- Continue testing stricter NO-side price/lead-time controls; the latest replay still shows that the highest-price NO buckets carry worse marginal reliability despite sometimes adding in-sample PnL.
- Use the source/model-family calibration diagnostics before changing forecast weights; source-level Brier improvements must improve the end-to-end replay, not just the probability table.
- Add NOAA NBM, more explicit HRRR/NBM coverage, and market-specific station verification where available.
- Build reliability tables by city, source, lead time, and bucket type.
- Add a portfolio optimizer that can reason over mutually exclusive city/date buckets.
- Add resumable/batched Telonex quote preparation so a full one-year, all-city artifact can be built without serial per-token ingestion.

## Disclaimer

This is a research and paper-trading project. It is not financial advice and does not place real trades by default.
