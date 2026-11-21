---
name: budget_control
description: Analyze budget pacing, observed and expected CPA risk, and the remaining horizon.
version: 1
---
Optimize final official beta=2 score, not budget utilization by itself. Compare
spent_ratio with elapsed_ratio; uniform pacing is a diagnostic, not an optimal
spending schedule. A CPA with zero conversions is undefined, not zero. Expected
CPA is an auxiliary estimate. Explain the tradeoff when spending is behind but
CPA risk is high. Signal: favorable, unfavorable, mixed, or unknown. Return the
specified analysis JSON. Recommend a direction, but do not determine the expert weights or final bid.
