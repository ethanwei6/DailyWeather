# Live-Like Backtesting Engine

DailyWeather backtests should imitate the live trader. The current workflow is:

1. Build one expensive scored-outcome artifact from real Telonex Polymarket data and historical weather forecasts.
2. Replay many strategy profiles from that artifact without refetching prices or forecasts.
3. Compare strategies with the same timing cadence, sizing semantics, and signal filters used by the live automation.

This avoids mixing old ad-hoc historical runs with live-forward behavior.

## Build A One-Year Artifact

The build command uses the live automation cadence by default: `00:00`, `06:00`, `12:00`, and `18:00` UTC. It also defaults to Telonex market discovery, Telonex tick-level quote history, and Polymarket resolved payouts for settlement. Station weather cross-checks can still be enabled, but they are intentionally not on the critical path for broad year-scale replays.

```bash
python3 -m weather_strategy.cli build-live-like-backtest \
  --strategy-profile live-forward-50-windowbank-0.25-kelly-0.50-poscap-0.25-strict-no-tail-0.14-bprice-0.70 \
  --bankroll-usd 50 \
  --lookback-days 365 \
  --max-markets 50000 \
  --max-runtime-seconds 0 \
  --progress-every 500 \
  --settlement-audit polymarket_only \
  --http-hard-timeout-seconds 300 \
  --weights-file work/data/model_weights.json \
  --cache-dir work/cache/live_like_backtest \
  --run-log-dir work/logs/live_like_backtests \
  --summary-only
```

Use `--min-end-date` and `--max-end-date` when a fixed evaluation window is needed. If omitted, `--lookback-days 365` means the 365 calendar days ending yesterday.

The output artifact contains:

- raw and parsed market counts,
- usable Telonex price-history counts,
- live-cadence sessions,
- scored outcomes,
- signal/filter diagnostics,
- trade executions,
- equity curve,
- real-data audit,
- source settings, including whether entry hours match the live-forward cadence.

## Replay Strategy Profiles

After the source artifact exists, sweep strategies without network refetches:

```bash
python3 -m weather_strategy.cli compare-live-like-strategies \
  --source-run-log work/logs/live_like_backtests/<artifact>-long-backtest.json \
  --run-log-dir work/logs/live_like_strategy_replays \
  --output-dir work/reports/live_like_strategy_comparison \
  --summary-only
```

By default this compares the current live profile against important candidates:

- `live-forward-50-windowbank-0.25-kelly-0.50-poscap-0.25-strict-no-tail-0.14-bprice-0.70` flagged as current live,
- the same window-bankroll profile with 50% and 20% position caps,
- paced two-slot and four-slot variants,
- the prior high-win bounded-confirmed candidate,
- the reserve-based lower-sizing candidate.

Pass `--strategy-profiles a,b,c` to choose a different sweep.

## Report Outputs

`compare-live-like-strategies` writes:

- `summary.json` for machine-readable performance,
- `summary.md` for quick review,
- `equity_curves.svg` for visual equity comparison,
- `trades.json` for full execution rows by strategy.

The report includes PnL, return, average monthly return, annualized return, 365-day Sharpe, max drawdown, trade count, buy/sell/settlement counts, hit rates, buy notional, minimum cash, region splits, side splits, city splits, target-month splits, and entry-hour splits.

## Data Policy

Keep raw reusable data under `work/cache/`. It is intentionally gitignored and can be large. Do not commit Telonex downloads, weather payload caches, `.env`, `.env.local`, ledgers, private keys, API keys, or generated run logs.

Tracked docs should describe how to reproduce the analysis. Generated results should live under `work/logs/` or `work/reports/`.
