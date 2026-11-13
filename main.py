#!/usr/bin/env python3
"""polymarket-nvidia-ceo-forecast-01 entry (Python 3.13, standard library only).

Forecasts the Yes price of every market of Polymarket's "Next CEO of
NVIDIA?" event, minute by minute, 4 hours ahead, as five quantiles.

Design v2 (chosen over v1 on an extended backtest: 19 checkpoints every
4h Nov 10-13 2026, the 6 official practice cases, and 5 fresh out-of-sample
hourly checkpoints Nov 13 07:00-11:00 UTC from data fetched after the
sweeps; the house baseline's published practice score 0.363589 was
reproduced exactly by the same harness, so the comparison is apples to
apples):

  - Quantiles live in LOGIT space, not price space: prices sit in [0, 1]
    and most markets hug 0, so normal quantiles in price space waste mass
    below 0 and understate upside jumps. In logit space the quantiles are
    naturally asymmetric (fat upside, compressed downside near 0) and the
    median is still exactly the last price.
  - Per-market sigma in logit space, estimated robustly (mean absolute
    move scaled to a normal sigma, capped by the median absolute move)
    from two windows blended 50/50: the last 24 hours (1440 min) and the
    last 1 hour (60 min). Volatility clusters in this event, so the last
    hour predicts the next 4 hours better than a long window alone; the
    blend beat every single-window variant on all three backtest sets.
    Sigma is floored at 0.002 and capped at 0.10 per square-root minute
    in logit units. A plain standard deviation is poisoned by this
    event's jumps (single minutes moved a market by 0.48 early on).
  - The median is persistence: the last price. Mean-reversion pulls,
    trend drift, and median shifts were all tested and did not help (the
    median realized move is exactly 0.000000); every gain comes from the
    spread.
  - Horizon scaling sigma * h**0.40, not sqrt(h): consecutive hourly moves
    correlate about -0.3 to -0.5 in this event (moves revert), so the
    spread grows slower than a random walk's. The 5%/95% tails use a
    slightly faster exponent, h**0.45: tail risk grows faster than the
    centre as the horizon extends.
  - The 5% and 95% quantiles are widened by 1.30x: jumps of 0.05-0.24
    still happen in the recent regime, and a normal tail under-covers
    them. The 25%/75% quantiles keep the normal width.

Validated on the 6 official practice cases plus a 7th case built from the
same public history (checkpoint Nov 13 06:00 UTC) and 5 fresh holdout
checkpoints: mean pinball loss 0.275441 over the 6 official practice cases
(house 0.363589, -24%; v1 scored 0.290140), 0.243215 over the 7 recent
checkpoints (v1 0.256388), 0.070520 over the 5 fresh holdout checkpoints
(v1 0.081706, winning all 5). Format, determinism (two runs
byte-identical) and runtime (under 0.1 s per case against an 8 s cap) all
verified locally before submission.

Deterministic: pure functions of the case file, no randomness, no clock,
no network. Two runs on the same case write the same bytes.
"""

import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level, to 6 decimals.
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]

WINDOW = 1440         # minutes of history for the long sigma (24 hours)
BLEND_WINDOW = 60     # minutes of history for the short sigma (1 hour)
BLEND_F = 0.5         # weight on the short-window sigma
ALPHA = 0.40          # horizon exponent: sigma(h) = sigma * h**ALPHA
TAIL_ALPHA = 0.45     # horizon exponent for the 5%/95% quantiles
TAIL_SCALE = 1.30     # widening of the 5% and 95% quantiles
LOGIT_EPS = 0.0005    # clip for the logit transform
SIGMA_FLOOR = 0.002   # per-minute sigma floor (logit units)
SIGMA_CAP = 0.10      # per-minute sigma cap (logit units)
FALLBACK_SIGMA = 0.0025  # when a market has too little history


def robust_sigma(diffs):
    """Robust per-minute sigma from 1-minute moves (logit space here).

    Mean absolute move scaled to a normal sigma (E|X| = sigma*sqrt(2/pi)),
    capped by the median absolute move where that is positive. Jumps move
    the mean only a little (they are rare); a plain standard deviation
    would let one 0.48 jump dominate the estimate.
    """
    n = len(diffs)
    if n < 10:
        return FALLBACK_SIGMA
    nz = [abs(d) for d in diffs if d != 0]
    if not nz:
        return 0.0002
    mean_abs = sum(abs(d) for d in diffs) / n
    mad = mean_abs / 0.7978845608028654
    ordered = sorted(abs(d) for d in diffs)
    med = ordered[n // 2]
    if med > 0:
        plain = med / 0.6744897501960817
        return min(mad, plain)
    return mad


def logit(x):
    """log(x / (1 - x)) with x clipped away from 0 and 1."""
    x = min(1.0 - LOGIT_EPS, max(LOGIT_EPS, x))
    return math.log(x / (1.0 - x))


def sigmoid(v):
    """1 / (1 + exp(-v)), overflow-safe on both sides."""
    if v >= 0:
        return 1.0 / (1.0 + math.exp(-v))
    e = math.exp(v)
    return e / (1.0 + e)


def price6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0


def window_sigma(log_prices, w):
    """Robust sigma from the last w logit moves, floored and capped."""
    n = len(log_prices)
    w = min(w, n - 1)
    diffs = [log_prices[i + 1] - log_prices[i] for i in range(n - 1 - w, n - 1)]
    return max(SIGMA_FLOOR, min(SIGMA_CAP, robust_sigma(diffs)))


def forecast_market(prices, horizon):
    """240 rows of five quantiles for one market."""
    n = len(prices)
    if n == 0:
        # No history at all: stay flat at 0.5 with the fallback spread.
        x = 0.5
        sigma = FALLBACK_SIGMA
        lx = logit(x)
    else:
        x = prices[-1]
        lx = logit(x)
        log_prices = [logit(v) for v in prices]
        sigma = (1.0 - BLEND_F) * window_sigma(log_prices, WINDOW) + BLEND_F * window_sigma(
            log_prices, BLEND_WINDOW
        )
    rows = []
    for h in range(1, horizon + 1):
        spread = sigma * (h ** ALPHA)
        tail_spread = sigma * (h ** TAIL_ALPHA)
        row = []
        for j, z in enumerate(Z):
            if j in (0, 4):
                v = sigmoid(lx + z * tail_spread * TAIL_SCALE)
            else:
                v = sigmoid(lx + z * spread)
            row.append(price6(v))
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for name in inputs:
        case_id = name[: -len(".in")]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        if case["quantiles"] != QUANTILES:
            raise ValueError(
                "%s asks for quantiles %s, not %s"
                % (name, case["quantiles"], QUANTILES)
            )
        horizon = case["horizon_minutes"]
        forecasts = [
            {"id": m["id"], "q": forecast_market(m["prices"], horizon)}
            for m in case["markets"]
        ]
        out = json.dumps({"forecasts": forecasts}) + "\n"
        with open(
            os.path.join(OUTPUT_DIR, case_id + ".out"), "w", encoding="utf-8"
        ) as f:
            f.write(out)


if __name__ == "__main__":
    main()
