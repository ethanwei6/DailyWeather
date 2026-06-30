# Strategy Notes

DailyWeather prices daily-high temperature markets with a conservative fair-value pipeline:

1. Discover live Polymarket temperature contracts.
2. Parse contract text into city, target date, and a temperature interval.
3. Fetch multiple live weather-source distributions for the city/date.
4. Convert those distributions into probability views.
5. De-duplicate identical upstream daily-high payloads, then aggregate probability views by independent weather source.
6. Apply observation-aware adjustment for same-day markets when actual observations are available.
7. Compare fair value to live market price.
8. Paper-trade only when edge, agreement, liquidity, and timing filters all pass.

## Model Views

The forecast engine currently uses these probability perspectives:

- Empirical ensemble frequency.
- Tight kernel smoothing.
- Wide kernel smoothing.
- Parametric normal model.
- Conservative normal model.
- Feature-aware normal model using precipitation, cloud, wind, humidity, pressure, and solar-radiation features.
- Hourly-curve max model using the source's predicted intraday hourly temperature path.

The important implementation detail is that these are not treated as fully independent signals. They are different views of underlying weather-source data. Consensus fair value and agreement are computed at the source level to reduce false confidence, and exact duplicate source payloads are collapsed before scoring.

## Why Source-Level Consensus Matters

Earlier versions counted every transformation of a source as an independent model. That made agreement look stronger than it was. For example, one deterministic forecast could produce several high-probability transformed views and create the appearance of broad consensus.

The current pipeline keeps the detailed model probabilities for diagnostics, but source-level averages drive:

- consensus fair value,
- model count,
- probability dispersion,
- agreement gating.

## Live Observations

The observation layer tries real station data before forecast proxies:

- US NWS station observations when a supported station is configured.
- Global METAR station observations for major international and US cities.
- Open-Meteo hourly data only as a fallback proxy.

Fallback proxy observations are retained for diagnostics, but non-final proxy values do not hard-zero or hard-certify same-day outcomes. That avoids treating forecast-hourly data as if it were an already observed high.

## Same-Day Markets

Daily-high markets become structurally different once the local afternoon begins. If the current observed high already exceeds a bucket upper bound, that bucket is impossible. If an open-ended threshold has already been reached, that outcome is effectively certain.

For same-day markets that are not final, the model applies a remaining-upside path adjustment based on local time. This prevents the strategy from treating a late-afternoon forecast distribution as if the whole thermal day is still ahead.

Entry timing and exit timing are deliberately separate. The strategy can block new same-day entries before the observation-aware window without forcing an existing position to liquidate. Existing positions are rebalanced only when fair value, source agreement, price, and price-aware Kelly edge justify a change.

## Entry Filters

The paper strategy requires:

- positive edge after uncertainty buffer,
- minimum independent-source agreement,
- minimum independent source count,
- price-aware Kelly edge, so high-probability markets can trade on smaller absolute gaps,
- stricter effective edge requirements for low-probability longshots,
- acceptable CLOB spread,
- tradeable price band,
- a `12.5c` general minimum price floor and a stricter `20c` YES-side floor, with a separate `70%` fair-value gate so cheap entries require high model conviction rather than just a large-looking absolute spread,
- a `10c` normal extra edge floor, `2c` high-confidence edge floor, `95c` max-entry price, `10%` opposite-event model-tail cap for experimental NO-token research entries, and a wider `15%` opposite-event tail cap for already-open NO holds,
- entry-eligible lead time,
- one active entry per city/date group,
- current-equity Kelly compounding after gates pass,
- optional edge-scaled position caps, applied after the absolute/percentage cap is chosen, so marginal edges do not receive the same maximum size as large, source-confirmed edges.

This intentionally reduces trade frequency. Strategy changes should be judged with live-like historical replay before they are promoted to the live automation.

## NO-Side Research

Historical replay can optionally score real Polymarket NO tokens. This does not approximate NO prices by taking `1 - YES`; it extracts explicit NO `clobTokenIds` and uses Telonex/CLOB prices for that token.

NO-side gates exist because high-probability NO trades can look safe while still hiding a small but meaningful opposite-event tail. New NO entries therefore use a counter-event cap, side-specific max-entry price, and extra edge floors. Existing NO positions may use a wider hold cap so the strategy does not churn out of a strong position just because one model briefly raises the opposite-event tail.

Current promoted profiles use live-like replay artifacts rather than static report numbers. Generated comparison artifacts should show whether looser NO-tail caps, higher position caps, or broader bounded-bucket entries actually improve PnL after drawdown, concentration, and win-rate costs.

For artifact analysis, `entry_eligible` remains the legacy timing-only field. Use `passes_signal_filter`, `signal_eligible`, `trade_eligible`, and `signal_filter_reason` to determine whether a scored outcome was actually tradable under the current gates.

The live forward paper automation writes the same distinction into `work/logs/live_forward_paper/*.json`, along with signal-filter counts, skipped-market reason counts, per-city coverage, local lead-day timing, and post-run positions. Those artifacts should be treated as forward-test research data: once markets resolve, they can be joined back into calibration and replay studies without relying on chat summaries.

The broad fair-value model can be worse than market price overall while the filtered subset still beats market. Treat this as a selective trading rule, not a general weather-pricing oracle. The backtester reports concentration, side mix, region mix, and calibration precisely because a high-PnL replay can still be fragile if it comes from one city, one side, or one cluster of high-confidence NO trades.

## Accuracy Weighting

The backtest workflow resolves completed forecast rows, measures source/model Brier score and log loss, then writes shrunk weights to `work/data/model_weights.json`. Better-calibrated sources and model transforms receive modestly higher consensus weight; weaker views receive lower weight. The shrinkage prior keeps small samples from dominating live sizing.

These weights feed into the normal forecast engine, so they affect:

- consensus fair value,
- source agreement,
- Kelly target sizing,
- future backtest comparisons between default and calibrated weights.

The backtest is intentionally chronological: earlier resolved rows train weights, later rows are held out for accuracy and paper-trading replay.

## Current Limitations

- Resolution station parsing is still approximate for non-US cities.
- Live market discovery depends on external API availability.
- Historical resolution uses Polymarket settlement as primary in long replay; station METAR/ASOS and archive weather data are cross-checks or fallbacks.
- NO-side trading is implemented for historical replay, paper simulation, and guarded live execution.
- Historical PnL is concentrated by city, month, and a few large winners; the artifact now reports these cohorts explicitly before any promotion to live risk.
- Mutually exclusive bucket portfolios are reduced to one selected entry per city/date instead of solved as a full constrained portfolio.
- Backtest replay uses recorded forecast snapshots and recorded prices, not a complete historical order book.
- Live execution is guarded separately from research replay and should only run with the same profile validated by the live-like backtester.
