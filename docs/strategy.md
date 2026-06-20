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

This intentionally reduces trade frequency. The goal is to collect high-quality paper-trading evidence before any real execution work.

## NO-Side Research

The historical replay can optionally score real Polymarket NO tokens with `long-backtest --allow-no-side-entries`. This does not approximate NO prices by taking `1 - YES`; it extracts the explicit NO `clobTokenIds` from Gamma payloads and fetches their own CLOB price histories.

The June 20 promoted paper replay used a true `$100` bankroll with a `25%` current-equity max-position cap and ended at `$640.93`, or `+$540.93`, with explicit YES+NO trading enabled. The latest candidate profile applies a split NO edge floor: `10c` for normal NO entries and `2c` for high-confidence NO entries where the explicit NO token already trades at or above `75c`. It also uses a `12.5c` general price floor, a stricter `20c` YES-side floor, a `95c` max-entry price, a `10%` NO-side entry counter-event cap, a wider `15%` NO-side hold counter-event cap, `0.75` fractional Kelly sizing, current-equity compounding, raw model fair value for sizing, and a corrected edge-scaled cap that first applies the absolute/percentage cap and then floors marginal edges at `35%` until buffered edge reaches `25c`. The entry counter-event cap rejects new NO entries when any underlying model view still gives the original YES event more than a 10% chance; the hold cap is wider so strong existing NO positions are not churned out by a moderate tail-risk wobble.

The result is promising but not yet production-ready. The looser NO-tail counterfactuals on the broad replay had much larger drawdown and ambiguous weather validation. With the corrected live-scaled `25%` current-equity cap enabled, the completed broad replay made `+$540.93`, max drawdown was `$20.40`, realized trade hit rate was `97.37%`, and event hit rate was `94.74%` across 38 trades. The artifact recommendation keeps the `25%` cap as the current aggressive paper-only sizing profile because it preserved the same 38 trades, same event hit rate, and zero ambiguous/mismatched traded tokens versus the prior `20%` profile while raising in-sample PnL; it is still a paper profile, not a live-production default. The prior `9%` entry-tail profile was weaker under the same signal family, so the `10%` entry tail remains a narrow entry-policy promotion: it added three weather-matched event winners without admitting the weaker `12%+` tail cohort. The prior `9%` hold-tail profile over-churned high-FV NO positions, so the `15%` hold tail remains a separate exit-policy promotion that reduces churn without weakening new-entry standards.

The latest sensitivity diagnostics now track max-position fractions directly, including chronological cap-fraction robustness slices. Moving from `5%` to `10%` to `15%` to `20%` to `25%` kept the same 38 weather-matched trades and raised PnL from `+$51.52` to `+$127.51` to `+$234.56` to `+$384.84` to `+$540.93`, while max drawdown rose from `$1.41` to `$3.87` to `$7.83` to `$13.94` to `$20.40`. The `15%` cap stays the safer baseline replay. The `25%` position cap made more in sample, preserved trade count and event hit rate, and stayed profitable in all cap-fraction robustness slices, so it is the current aggressive paper-only profile. A loose `20%` NO-entry tail cap and disabled NO-entry tail gate generated much higher in-sample PnL, but they added larger drawdowns, two ambiguous weather-validated trades, and lower event hit rates, so they are not promoted.

For artifact analysis, `entry_eligible` remains the legacy timing-only field. Use `passes_signal_filter`, `signal_eligible`, `trade_eligible`, and `signal_filter_reason` to determine whether a scored outcome was actually tradable under the current gates.

The live forward paper automation writes the same distinction into `work/logs/live_forward_paper/*.json`, along with signal-filter counts, skipped-market reason counts, per-city coverage, local lead-day timing, and post-run positions. Those artifacts should be treated as forward-test research data: once markets resolve, they can be joined back into calibration and replay studies without relying on chat summaries.

The completed 20-page replay with live/cache-backed data parsed `6,804` resolved markets and scored `20,373` outcomes with `runtime_limited=false`. YES trades made `+$161.46` and NO trades made `+$379.47`, with zero weather cross-check mismatches on traded tokens. Settlement-quality diagnostics show that all 38 traded tokens were independently weather-checked and matched Polymarket settlement, with zero ambiguous, Polymarket-only, or unresolved traded tokens. The new `real_data_audit` block makes those replay assumptions explicit: it checks historical CLOB price timestamps, historical forecast run times, Open-Meteo Single Runs sources, forecast availability lag, explicit NO-token attribution, and weather-matched settlement. The broad discovery pass still logged `177` market-resolution errors, `2,686` unusable YES price histories, and `622` forecast misses; those are explicit data gaps, not synthetic fills. The broad fair-value model was still worse than market price overall, but the selected filtered rows were better than market price, so the strategy remains a selective trading rule rather than a general weather-pricing model. NO-side replay is now mirrored in paper trading with explicit NO-token accounting and inverted settlement; real execution should remain disabled until this survives out-of-sample paper validation.

The robustness diagnostics now replay the selected profile on fresh-bankroll time slices. First-half, second-half, first-70%, last-30%, and every monthly slice remained profitable in the latest artifact. The weakest periods were April and June, and the selected model probabilities were still overconfident: `97.33%` average fair value versus a `94.74%` realized rate on traded tokens. That means the model can beat market price on the filtered subset while still needing fractional Kelly, strict gates, and more paper validation before live trading.

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
- NO-side trading is implemented for long historical replay and paper simulation; the real execution boundary remains disabled.
- Historical PnL is concentrated by city, month, and a few large winners; the artifact now reports these cohorts explicitly before any promotion to live risk.
- Mutually exclusive bucket portfolios are reduced to one selected entry per city/date instead of solved as a full constrained portfolio.
- Backtest replay uses recorded forecast snapshots and recorded prices, not a complete historical order book.
- No live execution adapter is enabled.
