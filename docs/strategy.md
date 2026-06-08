# Strategy Notes

DailyWeather prices daily-high temperature markets with a conservative fair-value pipeline:

1. Discover live Polymarket temperature contracts.
2. Parse contract text into city, target date, and a temperature interval.
3. Fetch multiple weather-source distributions for the city/date.
4. Convert those distributions into probability views.
5. Aggregate probability views by independent weather source.
6. Apply observation-aware adjustment for same-day markets.
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

The important implementation detail is that these are not treated as fully independent signals. They are different views of underlying weather-source data. Consensus fair value and agreement are computed at the source level to reduce false confidence.

## Why Source-Level Consensus Matters

Earlier versions counted every transformation of a source as an independent model. That made agreement look stronger than it was. For example, one deterministic forecast could produce several high-probability transformed views and create the appearance of broad consensus.

The current pipeline keeps the detailed model probabilities for diagnostics, but source-level averages drive:

- consensus fair value,
- model count,
- probability dispersion,
- agreement gating.

## Same-Day Markets

Daily-high markets become structurally different once the local afternoon begins. If the current observed high already exceeds a bucket upper bound, that bucket is impossible. If an open-ended threshold has already been reached, that outcome is effectively certain.

For same-day markets that are not final, the model applies a remaining-upside path adjustment based on local time. This prevents the strategy from treating a late-afternoon forecast distribution as if the whole thermal day is still ahead.

## Entry Filters

The paper strategy requires:

- positive edge after uncertainty buffer,
- minimum independent-source agreement,
- minimum independent source count,
- acceptable CLOB spread,
- tradeable price band,
- entry-eligible lead time,
- one active entry per city/date group.

This intentionally reduces trade frequency. The goal is to collect high-quality paper-trading evidence before any real execution work.

## Current Limitations

- Resolution station parsing is still approximate for non-US cities.
- Live market discovery depends on external API availability.
- Calibration needs more resolved observations or a historical backfill.
- Mutually exclusive bucket portfolios are reduced to one selected entry per city/date instead of solved as a full constrained portfolio.
- No live execution adapter is enabled.
