# Operations Runbook

This runbook describes how to operate DailyWeather as a paper-trading system and, only after explicit approval, as a guarded live Polymarket strategy.

## Test Gate

Always run tests before a live paper run:

```bash
python3 -m unittest discover -s tests
```

## Live Paper Run

The live run should optimize expected trading performance and market coverage over wall-clock speed. Highest-temperature markets resolve daily, so a run that takes tens of minutes is acceptable if it discovers and scores more real opportunities. Keep `--max-runtime-seconds` as a hang guard, not as a speed target:

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

The named profile applies the Telonex-tested `$100` paper settings: explicit NO-token entries, `75%` fractional Kelly, current-equity compounding, a `25%` current-equity max-position cap with a `$175` absolute cap, `70%` minimum fair value, `100%` model-source agreement for new entries, an `11%` NO counter-event cap at every entry hour, preservation of valid existing holds, partial exits for invalid holds only when FV remains at least `90%` and quote is between `50c` and `65c`, a hold-only high-conviction exception for existing NO positions, and a `10c` minimum edge for bounded exact/range buckets. Use `--strategy-profile live-forward-strict-no-tail-preserve-highconv-bounded-edge-0.10` to compare the stricter `10%` NO-tail profile, `--strategy-profile live-forward-strict-no-tail-preserve-highconv-bounded-edge-0.15` to compare the stricter bounded-edge profile, `--strategy-profile live-forward-strict-no-tail-trim-highconv-bounded-edge-0.15` to compare the lower-exposure trim-to-Kelly hold variant, or `--strategy-profile live-forward-utc12-relaxed-no-tail-0.20-trim-highconv-bounded-edge-0.15` only as the higher-risk relaxed-tail comparison profile.

The current `$50` forward-paper/live-money candidate is performance-first and uses a window-bankroll sizing layer:

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

It keeps the strict `14%` NO counter-event cap and bounded exact/range bucket confirmation at `98%` model FV and a `70c` market price, but sizes each automation window from `25%` of current sizing equity. The per-position cap is `25%` of that window bankroll, so the bot can spread one global window across more independent city/outcome bets while still allowing the full bankroll to be deployed over a full day if enough opportunities appear. Keep reviewing the `$50` live/paper ledger and generated live-like comparison artifacts for event hit rate, realized PnL, concentration, drawdown, cash exhaustion, and real-data quality out of sample.

## Live Money Run

Live execution is opt-in and fail-closed. It requires:

- `.env.local` with `0600` permissions,
- `DAILYWEATHER_LIVE_TRADING=1`,
- funded Polymarket collateral/pUSD in the deposit wallet,
- CLOB API credentials for the bot signer,
- `--execution-mode live --confirm-live` on the command line.

For the current `$50` profile, use:

```bash
.venv-live/bin/python -m weather_strategy.cli paper-run \
  --ledger work/data/weather_live_money_50.sqlite \
  --strategy-profile live-forward-50-windowbank-0.25-kelly-0.50-poscap-0.25-strict-no-tail-0.14-bprice-0.70 \
  --limit 2000 \
  --discovery-request-limit 50 \
  --discovery-pages 20 \
  --max-runtime-seconds 3600 \
  --progress-every 25 \
  --same-day-entry-start-hour 11 \
  --same-day-entry-cutoff-hour 17 \
  --weights-file work/data/model_weights.json \
  --run-log-dir work/logs/live_money \
  --execution-mode live \
  --confirm-live \
  --live-env-file .env.local
```

The live adapter submits FOK CLOB market orders only after the same weather model, timing, signal, and Kelly gates pass. It records successful live fills in the SQLite ledger with CLOB order IDs, transaction hashes, filled shares, filled notional, and average fill price. Failed live orders raise an error before the local ledger records a fill.

Risk caps are read from `.env.local`: `DAILYWEATHER_MAX_BANKROLL_USD`, `DAILYWEATHER_MAX_ORDER_USD`, and `DAILYWEATHER_MAX_DAILY_LOSS_USD`. The active profile does not use a hard cash reserve; instead, each run sizes Kelly from a `25%` automation-window bankroll and caps new exposure to `25%` of current sizing equity. Keep the command on the four global windows used by the live-forward automation: `00:00`, `06:00`, `12:00`, and `18:00` UTC.

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

For the `$50` candidate, use `work/data/weather_live_forward_50.sqlite` and `--bankroll-usd 50`.

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
- `work/logs/long_backtests/*.json` contains real historical Gamma/CLOB/Open-Meteo replay artifacts, including a strict `real_data_audit`, data-quality checks, PnL concentration, city/month/hour cohorts, sizing sensitivity, entry-hour-only counterfactuals, fresh-bankroll robustness slices, cap-fraction slices, and strategy recommendation diagnostics. Current long replays default to the same `00:00`, `06:00`, `12:00`, and `18:00` UTC cadence as the live automation and record `entry_hours_match_live_forward`.

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
  --strategy-profile live-forward-strict-no-tail-0.11-preserve-highconv-bounded-edge-0.10 \
  --bankroll-usd 100 \
  --pages 20 \
  --limit-per-page 50 \
  --max-markets 8000 \
  --max-runtime-seconds 180 \
  --entry-hours-utc 0,6,12,18 \
  --min-lead-days 1 \
  --max-lead-days 2 \
  --max-price-staleness-minutes 90 \
  --historical-price-slippage 0.01 \
  --forecast-availability-lag-hours 6 \
  --run-log-dir work/logs/long_backtests \
  --summary-only
```

Once a long replay has produced a real-data artifact with `scored_outcomes_detail`, use cached scored-outcome replay for fast strategy sweeps without refetching forecasts, prices, observations, or market metadata:

```bash
python3 -m weather_strategy.cli replay-scored-outcomes \
  --source-run-log work/logs/long_backtests/<artifact>-long-backtest.json \
  --strategy-profiles live-forward-50-reserve-0.25-kelly-0.50-cap-0.20-strict-no-tail-0.14-bprice-0.70,live-forward-50-reserve-0.25-kelly-0.50-cap-0.20-strict-no-tail-0.14-bprice-0.70-bounded-fv95-edge08-stdev03 \
  --run-log-dir work/logs/scored_replays \
  --summary-only
```

The replay command writes one detailed `*-scored-replay.json` artifact per profile under `work/logs/scored_replays/`. The summary output is a compact scorecard; the detailed artifacts keep `executions_detail`, top trades, cached scored rows used for comparison, inherited real-data audit status, and proof that forecasts/prices/observations were not refetched. Use `--strategy-profiles all` to sweep every named preset except `manual`.

New long-backtest and live-forward score rows include both calibrated `model_probabilities` and uncalibrated `raw_model_probabilities`. To test a new probability calibration without refetching weather or prices, replay a raw-aware artifact with:

```bash
python3 -m weather_strategy.cli replay-scored-outcomes \
  --source-run-log work/logs/long_backtests/<raw-aware-artifact>-long-backtest.json \
  --strategy-profile live-forward-50-reserve-0.25-kelly-0.50-cap-0.20-strict-no-tail-0.14-bprice-0.70-bounded-fv95-edge08-stdev03 \
  --recompute-from-raw-model-probabilities \
  --weights-file work/data/model_weights.json \
  --summary-only
```

The compact replay summary reports `raw_recalibration.rows_with_raw_model_probabilities` and `raw_recalibration.rows_recomputed`. Older artifacts that predate raw logging will show zero recomputed rows and are still valid for frozen-FV strategy replay, but they cannot validate new probability calibration exactly.

Do not promote a parameter just because headline PnL improves. Use the live-like backtesting flow in `docs/backtesting_engine.md`, then check `real_data_audit.passed`, concentration, weak months/cities, drawdown, settlement-quality diagnostics, timestamp-quality diagnostics, selected-candidate calibration, source/model-family calibration, realized trade hit rate, event hit rate, profitable event-loser trades, unprofitable event-winner trades, exit-management decision value, fresh-bankroll robustness slices, cap-fraction robustness slices, region/side/city splits, and whether the improvement survives direct candidate-profile comparison. Market-blended Kelly sizing is available through `--kelly-market-blend`, but keep it at `0.0` unless future out-of-sample evidence says otherwise. Keep `--max-runtime-seconds` set on live refresh runs so missing forecast payloads cannot hang the workflow.

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
