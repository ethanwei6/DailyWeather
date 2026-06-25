# Operations Runbook

This runbook describes how to operate DailyWeather as a paper-trading system.

## Test Gate

Always run tests before a live paper run:

```bash
python3 -m unittest discover -s tests
```

## Live Paper Run

The recommended live run is bounded so API issues cannot hang the workflow indefinitely:

```bash
python3 -m weather_strategy.cli paper-run \
  --ledger work/data/weather_live_forward_100.sqlite \
  --strategy-profile live-forward-strict-no-tail-trim-highconv-bounded-edge-0.15 \
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

The named profile applies the Telonex-tested `$100` paper settings: explicit NO-token entries, `75%` fractional Kelly, current-equity compounding, a `25%` current-equity max-position cap, `70%` minimum fair value, `100%` model-source agreement for new entries, a strict `10%` NO counter-event cap at every entry hour, Kelly-target trimming for valid holds, a hold-only high-conviction exception for existing NO positions, and a `15c` minimum edge for bounded exact/range buckets. Use `--strategy-profile live-forward-utc12-relaxed-no-tail-0.20-trim-highconv-bounded-edge-0.15` only as the higher-risk relaxed-tail comparison profile.

## Reporting

```bash
python3 -m weather_strategy.cli report \
  --ledger work/data/weather_live_forward_100.sqlite \
  --bankroll-usd 100
```

```bash
python3 -m weather_strategy.cli calibration \
  --ledger work/data/weather_live_forward_100.sqlite
```

## Backtest And Model Weights

Backtest recorded forecast snapshots, resolve final highs from historical weather archives, and write shrunk accuracy weights:

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

The live paper runner loads `work/data/model_weights.json` by default. These weights adjust consensus fair value and therefore Kelly target sizing. Runtime outputs under `work/` remain local and are intentionally not committed.

## Detailed Logs

Each automated cycle writes analysis artifacts:

- `work/logs/backtests/*.json` contains resolved-row counts, train/test accuracy, learned weights, and Kelly replay diagnostics.
- `work/logs/live_forward_paper/*.json` contains loaded weights, settings, all scored outcomes, signal-filter counts, skipped-market reason counts, per-city and per-target-date coverage, local lead-day timing, signals, post-run positions, error counts, equity, and PnL.
- `work/logs/paper_runs/*.json` contains the same paper-run schema when using the default log directory.
- `work/logs/long_backtests/*.json` contains real historical Gamma/CLOB/Open-Meteo replay artifacts, including a strict `real_data_audit`, data-quality checks, PnL concentration, city/month/hour cohorts, sizing sensitivity, entry-hour-only counterfactuals, fresh-bankroll robustness slices, cap-fraction slices, and strategy recommendation diagnostics.

The automation reports the exact `run_log_path` values after each run. Use those files for performance analysis instead of relying only on the chat summary. Treat `entry_eligible` as timing-only; use `passes_signal_filter` and `signal_filter_reason` when deciding whether a row was tradable.

## Cadence

The intended cadence is a small number of broad next-day runs, not hourly churn. For global city coverage, the live forward paper automation runs four times per UTC day:

- `00:00 UTC`: Americas evening and Asia-Pacific morning coverage.
- `06:00 UTC`: Europe morning and Asia-Pacific afternoon refresh.
- `12:00 UTC`: Europe afternoon and Asia-Pacific evening pre-local-day coverage.
- `18:00 UTC`: Americas afternoon/evening and Europe late-day coverage.

The market filter uses each market city's timezone to calculate the local lead window, so these UTC windows keep scanning tomorrow and following-day markets as different regions approach their local day boundary. Existing positions are still maintained on every run.

## Trading Gates

The live paper trader only opens new positions when all of these pass:

- fair-value edge passes a price-aware Kelly gate,
- source-level model agreement is `100%` for new entries,
- existing positions can be held at `65%` agreement when the fair-value thesis remains valid,
- model fair value is at least `70%` for new entries,
- high-probability markets above `75%` can trade with a `2%` minimum buffered Kelly edge,
- lower-probability markets use the normal `8%` minimum buffered Kelly edge,
- Kelly targets can be sized from current paper equity with `--compound-kelly-sizing` once the strict gates pass,
- edge-scaled position caps are applied after the absolute/percentage cap is chosen, so marginal low-edge target sizes stay smaller even when percentage-of-equity sizing compounds,
- price is inside the `12.5%` to `95%` tradeable band,
- YES-token entries must trade at `20%` or higher; NO research replays use the general `12.5%` floor,
- experimental NO-token entries need a `10%` normal absolute edge, can use a `2%` high-confidence edge at `75c+`, must be priced no higher than the global `95c` max-price gate, and must pass the `10%` opposite-event model-tail cap; existing NO positions can be held through a wider `15%` opposite-event tail if the rest of the hold thesis remains valid,
- CLOB spread is no more than `10%`,
- lead-time and same-day entry windows allow a new entry.

Existing positions are not liquidated merely because new same-day entries are time-blocked. They are held, reduced, or closed based on updated fair value, agreement, dispersion, and price.

## Long Historical Replay

Use the long replay when changing trading gates or sizing:

```bash
python3 -m weather_strategy.cli long-backtest \
  --strategy-profile live-forward-strict-no-tail-trim-highconv-bounded-edge-0.15 \
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
  --run-log-dir work/logs/long_backtests \
  --summary-only
```

Do not promote a parameter just because headline PnL improves. Check the long-run artifact for `real_data_audit.passed`, concentration, weak months/cities, drawdown, settlement-quality diagnostics, weather cross-check status, timestamp-quality diagnostics, selected-candidate calibration, source/model-family calibration, realized trade hit rate, event hit rate, profitable event-loser trades, unprofitable event-winner trades, exit-management decision value, fresh-bankroll robustness slices, cap-fraction robustness slices, and whether the improvement survives the counterfactual variants. Settlement quality should show that traded tokens are independently weather-checked and matched, not merely Polymarket-only or ambiguous. The current forward-paper candidate is `live-forward-strict-no-tail-trim-highconv-bounded-edge-0.15`: it keeps the strict `10%` NO-entry counter-event cap at every entry hour, trims valid holds to the updated Kelly target, only widens existing NO-hold counter-event tolerance to `20%` when FV is at least `98%` and buffered edge is at least `35c`, and requires at least `15c` edge for bounded exact/range buckets. On the saved real Telonex/Open-Meteo replay, removing the UTC-12 relaxed `20%` NO-tail exception improved PnL, event hit rate, drawdown, and concentration across the full sample and every chronological slice. Source-level diagnostics currently favor `single_run_ecmwf_ifs025`, but direct ECMWF overweighting and `hourly_curve_max` downweighting reduced end-to-end replay PnL, so do not promote model-weight changes from source Brier alone. Market-blended Kelly sizing is available through `--kelly-market-blend`, but keep it at `0.0` unless future out-of-sample evidence says otherwise. Keep `--max-runtime-seconds` set on live refresh runs so missing historical forecast payloads cannot hang the workflow. It is paper-testable with explicit NO-token settlement, but real execution should remain disabled.

## Failure Handling

If Gamma or CLOB calls fail:

- the discovery layer keeps request sizes bounded,
- failed search queries are skipped,
- per-market quote/weather errors are counted,
- progress is emitted to stderr,
- the run stops scoring new markets once the runtime budget is reached.

If a live run fails entirely, use `report` and `calibration` to inspect the persisted ledger state. Do not infer fresh trading metrics from an incomplete run.

## Local Artifacts

Runtime outputs and SQLite ledgers belong under `work/` and are intentionally ignored by git. The source code, tests, docs, and deterministic fixtures are the reproducible project artifacts.
