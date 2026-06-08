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
  --bankroll-usd 1000 \
  --kelly-fraction 0.25 \
  --max-position-usd 50 \
  --min-trade-usd 1 \
  --same-day-entry-start-hour 11 \
  --same-day-entry-cutoff-hour 17 \
  --min-lead-days 1 \
  --max-lead-days 2
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

## Cadence

The intended cadence is a small number of broad next-day runs, not hourly churn. The preferred schedule is:

- Morning: scan tomorrow and following-day markets.
- Evening: refresh next-day pricing and settle expired positions where final observations are available.

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
