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
- entry-eligible lead time,
- one active entry per city/date group.

This intentionally reduces trade frequency. The goal is to collect high-quality paper-trading evidence before any real execution work.

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
- Historical resolution currently uses final daily highs from archive weather data unless official station observations are available.
- Mutually exclusive bucket portfolios are reduced to one selected entry per city/date instead of solved as a full constrained portfolio.
- Backtest replay uses recorded forecast snapshots and recorded prices, not a complete historical order book.
- No live execution adapter is enabled.
