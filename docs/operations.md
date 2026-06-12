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

## Reporting

```bash
python3 -m weather_strategy.cli report \
  --ledger work/data/weather_kelly_paper.sqlite \
  --bankroll-usd 1000
```

```bash
python3 -m weather_strategy.cli calibration \
  --ledger work/data/weather_kelly_paper.sqlite
```

## Backtest And Model Weights

Backtest recorded forecast snapshots, resolve final highs from historical weather archives, and write shrunk accuracy weights:

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

The live paper runner loads `work/data/model_weights.json` by default. These weights adjust consensus fair value and therefore Kelly target sizing. Runtime outputs under `work/` remain local and are intentionally not committed.

## Detailed Logs

Each automated cycle writes analysis artifacts:

- `work/logs/backtests/*.json` contains resolved-row counts, train/test accuracy, learned weights, and Kelly replay diagnostics.
- `work/logs/paper_runs/*.json` contains loaded weights, settings, all scored outcomes, signals, skipped markets, post-run positions, error counts, equity, and PnL.

The automation reports the exact `run_log_path` values after each run. Use those files for performance analysis instead of relying only on the chat summary.

## Cadence

The intended cadence is a small number of broad next-day runs, not hourly churn. The preferred schedule is:

- Morning: scan tomorrow and following-day markets.
- Evening: refresh next-day pricing and settle expired positions where final observations are available.

## Trading Gates

The live paper trader only opens new positions when all of these pass:

- fair-value edge passes a price-aware Kelly gate,
- source-level model agreement is at least `65%`,
- high-probability markets above `75%` can trade with a `2%` minimum buffered Kelly edge,
- lower-probability markets use the normal `8%` minimum buffered Kelly edge,
- price is inside the `5%` to `95%` tradeable band,
- CLOB spread is no more than `10%`,
- lead-time and same-day entry windows allow a new entry.

Existing positions are not liquidated merely because new same-day entries are time-blocked. They are held, reduced, or closed based on updated fair value, agreement, dispersion, and price.

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
