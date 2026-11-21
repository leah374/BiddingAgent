---
name: market_value
description: Analyze current traffic value and historical market prices before weighting the expert bids.
version: 1
---
Use current pValue statistics and historical KB evidence, including sample sizes.
Separate current observed facts from historical estimates. Historical winning
price quantiles may be proxies: do not present them as exact measured quantiles.
There is no expert weighting or combined alpha yet. Do not invent one or call an
algorithm. Signal: favorable, unfavorable, mixed, or unknown. Return a short
natural-language conclusion, direction, evidence_ids and uncertainties in the
specified analysis JSON. Do not promise conversions from predicted probabilities.
