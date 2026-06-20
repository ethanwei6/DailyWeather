# Long Backtest Report

Date: 2026-06-20 Singapore time

This report summarizes the live-compatible historical replay used to decide whether DailyWeather is worth advancing toward live Polymarket execution. The replay uses real Gamma market discovery, real CLOB historical price bars, Open-Meteo Single Runs historical forecasts with a six-hour availability lag, Polymarket settlement prices, and METAR/ASOS weather cross-checks where available.

## Command

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
  --max-price 0.95 \
  --weights-file work/data/model_weights.json \
  --run-log-dir work/logs/long_backtests \
  --cache-dir work/cache/long_backtest \
  --progress-every 250 \
  --summary-only
```

Completed paper-test run log: `work/logs/long_backtests/20260620T053819Z-1781933899864632000-long-backtest.json`

The completed artifact was generated without hitting the runtime budget. It preserves price-history failures, forecast misses, and data-quality counters explicitly rather than silently filling missing historical payloads.

## Completed Replay

| Metric | Value |
| --- | ---: |
| Starting bankroll | $100.00 |
| Ending equity | $640.93 |
| PnL | +$540.93 |
| Return | +540.93% |
| Raw markets discovered | 10,588 |
| Parsed resolved markets | 6,804 |
| Markets with usable price history | 3,941 |
| Scored outcomes | 20,373 |
| Signal-eligible outcomes | 64 |
| Trades | 38 |
| Executions | 86 |
| Buy / sell / settlement executions | 47 / 16 / 23 |
| Open positions | 0 |
| Runtime limited | false |

Settlement quality:

| Scope | Rows/Tokens | Weather checked | Weather matched | Weather mismatches | Ambiguous | Polymarket-only | Unresolved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Signal-eligible rows | 64 | 64 | 64 | 0 | 0 | 0 | 0 |
| Traded tokens | 38 | 38 | 38 | 0 | 0 | 0 | 0 |

Side attribution:

| Side | Trades | Buy notional | Realized PnL | Return on buy notional |
| --- | ---: | ---: | ---: | ---: |
| YES | 8 | $329.80 | +$161.46 | +48.96% |
| NO | 30 | $1,360.05 | +$379.47 | +27.90% |

The YES-only reference replay on an earlier 10-page market set ended at `$158.30`, or `+$58.30`, with 5 trades. Adding explicit NO-token replay broadened the opportunity set, but the current paper-test result should be judged on its own 38-token sample and the `15%` baseline should stay visible as the lower-drawdown reference.

## Real-Data Audit

The long replay now emits a `real_data_audit` block. A run is promotion-eligible only if that audit passes. The audit verifies that scored rows have historical CLOB price timestamps, historical forecast run times, Open-Meteo Single Runs forecast sources, no future/stale price usage, no unavailable forecast runs, explicit NO-token attribution, signal-eligible weather matches, and traded-token weather matches.

For the latest completed replay, `real_data_audit.passed=true`: no future/stale price violations, no unavailable-forecast violations, every scored forecast observed a six-hour availability lag, all signal-eligible rows were weather-checked and matched, all explicit NO rows used real NO tokens, and all 38 traded tokens were independently weather-checked and matched settlement.

## Current Gates

The accepted research profile currently requires:

- full source-level agreement for new entries,
- model fair value of at least `70%`,
- a general `12.5c` minimum price and stricter `20c` YES-side minimum price,
- disabled bounded exact/range bucket entries,
- explicit NO-token prices, not synthetic `1 - YES` approximations,
- a `10c` normal NO-side absolute edge,
- a `2c` high-confidence NO-side edge only when the explicit NO token is already at or above `75c`,
- a `95c` max-entry price,
- a `10%` NO-side entry counter-event cap, rejecting NO entries when any model view still gives the original YES event more than 10% probability,
- a wider `15%` NO-side hold counter-event cap, avoiding forced exits when an existing high-FV position has only a moderate tail-risk wobble,
- `0.75` fractional Kelly with current-equity compounding,
- a `25%` current-equity paper-test position cap with the absolute fail-safe set to the starting bankroll,
- an edge-scaled cap applied after that percentage cap, flooring marginal edges at `35%` and reaching the full cap at a `25c` buffered edge,
- raw model fair value for Kelly sizing; optional market-blended sizing is implemented for diagnostics but is not the selected default.

## Sizing Sensitivity

The earlier `$175` / `75%` max-position run on the same `$100` starting bankroll is classified as an aggressive sizing stress test, not the live-compatible headline.

| Variant | PnL | Trades | Trade hit | Event hit | Max drawdown | Role |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Current `25%` cap, 25c full-size edge, 10% entry NO-tail, 15% hold NO-tail, 95c cap | +$540.93 | 38 | 97.37% | 94.74% | $20.40 | Aggressive paper-test replay |
| Same selected profile with 20% position cap | +$384.84 | 38 | 97.37% | 94.74% | $13.94 | Prior aggressive paper-test replay |
| Same selected profile with 15% position cap | +$234.56 | 38 | 97.37% | 94.74% | $7.83 | Safer baseline replay |
| Same selected profile with 10% position cap | +$127.51 | 38 | 97.37% | 94.74% | $3.87 | Lower-risk reference cap |
| Same selected profile with legacy 5% position cap | +$51.52 | 38 | 97.37% | 94.74% | $1.41 | Conservative legacy cap |
| Same profile with legacy 8% NO-tail | +$377.88 | 33 | 96.97% | 93.94% | $13.81 | Slightly stricter legacy gate |
| Same profile with legacy 9% NO-entry tail | +$376.37 | 35 | 97.14% | 94.29% | $13.70 | Narrowly under-selected clean tail cases |
| Same profile with legacy 9% hold NO-tail | +$363.11 | 38 | 89.47% | 94.74% | $13.40 | Over-churned high-FV holds |
| Same profile with prior 93c NO cap | +$376.81 | 31 | 100.00% | 93.55% | $13.83 | Cleaner realized PnL, lower event accuracy |
| Same profile with legacy 90c NO cap | +$369.54 | 26 | 100.00% | 92.31% | $13.54 | Slightly underfit legacy gate |
| Same profile with loose 20% entry NO-tail | +$1,226.60 | 51 | 88.24% | 90.20% | $40.41 | Higher PnL, not promoted |
| Same profile with disabled entry NO-tail gate | +$1,693.30 | 58 | 89.66% | 89.66% | $65.46 | Stress test only |

The 25c full-size edge trigger remains the selected sizing shape because it improved the aggressive stress replay while still damping marginal low-edge entries. The effective-cap calculation applies the absolute/percentage cap first, then edge-scales that actual cap; this lets the current-equity cap compound after wins while avoiding full-size bets on marginal edges.

The 95c NO cap replaced the 93c cap because it improved PnL and selected-candidate Brier score, not because it made the path cleaner. The separate 15% hold-tail cap then fixed one failure mode in the 95c profile: already-open, high-FV NO positions were being sold down when one model briefly lifted the opposite YES tail above the entry limit.

The 10% NO-entry tail cap remains the promoted paper-test gate. It added clean trades versus the stricter 9% gate without admitting the weaker `12%+` tail cohort. The threshold sweep is not monotonic in quality: 12% and looser variants start admitting lower event hit rates and weather ambiguity, so those remain diagnostics, not defaults.

The generated artifact includes `strategy_recommendation_diagnostics`. The latest artifact keeps the `25%` current-equity cap as current after the expanded cap grid showed that 20%, 22.5%, and 25% preserve the same 38 trades, event hit rate, and weather-validation cleanliness. It still explicitly rejects looser NO-entry tails despite higher PnL because they weaken accuracy/validation quality.

The scored-outcome detail distinguishes `timing_entry_eligible` from `passes_signal_filter`, `signal_eligible`, and `trade_eligible`. Ultra-low-price rows can show large apparent fair-value gaps and still be correctly rejected; tradability analysis should use those signal/trade fields and `signal_filter_reason`.

Cap-fraction robustness slices:

| Cap | Slice | Trades | PnL | Trade hit | Event hit | Buy notional | Max drawdown |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5% | First 50% | 21 | +$25.19 | 100.00% | 95.24% | $65.95 | $0.36 |
| 5% | Second 50% | 18 | +$20.96 | 94.44% | 94.44% | $66.77 | $1.13 |
| 10% | First 50% | 21 | +$54.92 | 100.00% | 95.24% | $150.29 | $0.77 |
| 10% | Second 50% | 18 | +$47.44 | 94.44% | 94.44% | $154.35 | $2.51 |
| 15% | First 50% | 21 | +$89.79 | 100.00% | 95.24% | $256.19 | $1.25 |
| 15% | Second 50% | 18 | +$77.68 | 94.44% | 94.44% | $258.76 | $4.16 |
| 20% | First 50% | 21 | +$130.44 | 100.00% | 95.24% | $387.19 | $1.78 |
| 20% | Second 50% | 18 | +$112.62 | 94.44% | 94.44% | $383.55 | $6.11 |
| 25% | First 50% | 21 | +$177.59 | 100.00% | 95.24% | $542.49 | $2.37 |
| 25% | Second 50% | 18 | +$153.10 | 94.44% | 94.44% | $541.58 | $8.41 |

## Robustness

Selected traded-token calibration:

| Scope | Rows | Model Brier | Market Brier | Actual rate | Avg model FV | Avg market price |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Traded tokens | 38 | 0.051043 | 0.082777 | 94.74% | 97.33% | 81.84% |
| Traded NO tokens | 30 | 0.032519 | 0.048824 | 96.67% | 99.10% | 85.31% |
| Traded YES tokens | 8 | 0.120511 | 0.210100 | 87.50% | 90.69% | 68.82% |

The selected subset is better than market price, but the model is still overconfident: selected fair value averages `97.33%` against a `94.74%` realized rate. This supports continuing with strict gates and fractional Kelly rather than treating fair value as exact truth.

Event accuracy is now reported separately from realized trade PnL:

| Event outcome | Trades | Buy notional | Realized PnL | Return on buy notional |
| --- | ---: | ---: | ---: | ---: |
| Event win | 36 | $1,150.23 | +$363.90 | +31.64% |
| Event loss | 2 | $92.99 | +$20.95 | +22.53% |

The two event-losing trades were still profitable because the replay exited before final settlement. That is useful but fragile. Future promotion decisions should prioritize event hit rate and calibration, not only realized PnL.

Fresh-bankroll chronological slices:

| Slice | Sessions | Trades | PnL | Trade hit | Event hit | Buy notional | Return on buy notional | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| First 50% of sessions | 95 | 21 | +$130.44 | 100.00% | 95.24% | $387.19 | +33.69% | $1.78 |
| Second 50% of sessions | 95 | 18 | +$112.62 | 94.44% | 94.44% | $383.55 | +29.36% | $6.11 |
| First 70% of sessions | 133 | 24 | +$191.49 | 100.00% | 95.83% | $501.07 | +38.22% | $1.78 |
| Last 30% of sessions | 57 | 15 | +$88.05 | 93.33% | 93.33% | $307.85 | +28.60% | $5.41 |

Fresh-bankroll monthly slices:

| Entry month | Trades | PnL | Trade hit | Event hit | Buy notional | Return on buy notional | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-02 | 2 | +$27.67 | 100.00% | 100.00% | $28.88 | +95.83% | $0.00 |
| 2026-03 | 8 | +$53.12 | 100.00% | 87.50% | $98.69 | +53.83% | $1.39 |
| 2026-04 | 11 | +$17.87 | 100.00% | 100.00% | $118.82 | +15.04% | $0.10 |
| 2026-05 | 11 | +$98.76 | 100.00% | 90.91% | $265.47 | +37.20% | $6.05 |
| 2026-06 | 6 | +$5.85 | 83.33% | 100.00% | $53.33 | +10.98% | $0.20 |

The profile is positive across time slices, which is better than a single lucky cluster. The weak pattern is clear anyway: April and June trades are positive but much lower-return, and high-price/low-edge buckets remain the least attractive survivors.

## Patterns

By city:

| City | Trades | Buy notional | PnL | Return on buy notional |
| --- | ---: | ---: | ---: | ---: |
| Seoul | 17 | $758.72 | +$245.98 | +32.42% |
| New York | 6 | $141.99 | +$67.48 | +47.52% |
| Dallas | 2 | $28.94 | +$28.45 | +98.30% |
| Shanghai | 2 | $83.83 | +$15.38 | +18.35% |
| Paris | 2 | $55.32 | +$13.52 | +24.44% |
| Taipei | 3 | $74.42 | +$8.61 | +11.57% |
| Chicago | 3 | $43.17 | +$4.06 | +9.41% |
| London | 2 | $47.07 | +$1.07 | +2.26% |
| Denver | 1 | $9.77 | +$0.30 | +3.04% |

By entry hour:

| UTC hour | Trades | Buy notional | PnL | Return on buy notional |
| --- | ---: | ---: | ---: | ---: |
| 00:00 | 14 | $403.76 | +$177.74 | +44.02% |
| 12:00 | 24 | $839.46 | +$207.10 | +24.67% |

By lead time:

| Lead days | Trades | Buy notional | PnL | Return on buy notional |
| --- | ---: | ---: | ---: | ---: |
| 1 | 12 | $311.27 | +$128.95 | +41.43% |
| 2 | 26 | $931.95 | +$255.89 | +27.46% |

Both simulated run times and both next-day/two-day windows were profitable. The best use of automation remains a small number of broad next-day runs, not hourly day-of churn.

## Data Quality

- Tests passed after the latest code change: 115.
- Price-history errors: 2,686.
- NO price-history errors: 0.
- Forecast payload misses: 622.
- Broad market-resolution errors: 177.
- Unresolved traded tokens: 0.
- Weather cross-check mismatches on trades: 0.
- Signal-eligible weather-checked rows: 64 of 64.
- Signal-eligible weather-matched rows: 64 of 64.
- Traded weather-checked tokens: 38 of 38.
- Traded weather-matched tokens: 38 of 38.
- Ambiguous exact/range station checks: 56 market-level cases, 0 traded cases.
- Future price violations: 0.
- Stale price violations: 0.
- Unavailable forecast violations: 0.
- Minimum forecast availability lag: 6 hours.
- Maximum scored quote age: 3,600 seconds under the 90-minute quote-staleness cap.

## Decision

This backtest is strong enough to keep paper testing and to keep the explicit NO-token path enabled in paper mode. It is not strong enough to go live yet.

The main reasons to continue are the positive PnL across time slices, clean weather cross-checks on all traded tokens, no timestamp-quality violations, and selected-candidate Brier improvement versus market price. The main reasons not to go live yet are the small sample size, city concentration, overconfident fair values, forecast payload misses, and the fact that some realized PnL came from exiting event losers before final settlement.
