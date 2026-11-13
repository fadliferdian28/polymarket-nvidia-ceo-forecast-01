#!/usr/bin/env python3
"""polymarket-nvidia-ceo-forecast-01 entry (Python 3.13, standard library only).

Forecasts the Yes price of every market of Polymarket's "Next CEO of
NVIDIA?" event, minute by minute, 4 hours ahead, as five quantiles.

Design v5: a 70/30 ENSEMBLE of two forecasters whose errors partly cancel.
Leg A (70%) is the v3 forecaster below, with one v5 change: big markets
(last price >= 0.10) use a STRONGER left-skew quantile retune (5% at -2.0,
25% at -1.0, 75% at 0.4, 95% at 1.30) while small markets keep the v3
levels — the big markets are the ones that absorb sharp drops, and the
split won on all four evaluations (backtest ALL/RECENT/PRACTICE plus a
fresh 11-checkpoint holdout). Leg B (30%) is a gated replica of
a friend's submitted v2 (price space, winsorized mean-absolute-move vol,
median sum-reversion toward x/sum). The ensemble was chosen after rounds
14-17, which tested the friend's full write-up (sum-reversion, asymmetric
vol, regime switching, split-normal quantiles, vol decay) plus ensemble
weights. Selection used 19 checkpoints every 4h Nov 10-13 2026, the 6
official practice cases, and 10 fresh out-of-sample half-hourly
checkpoints Nov 13 07:00-11:30 UTC from data fetched after the sweeps; the
house baseline's published practice score 0.363589 was reproduced exactly
by the same harness, so the comparison is apples to apples.

Leg A (v3, 70%):
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
  - The quantile levels are retuned: the lower quantiles sit further out
    (5% at -1.80, 25% at -0.85) and the upper ones closer in (75% at 0.5,
    95% at 1.50), because this event's moves are left-skewed — sudden
    drops are more common than sudden rises. Deeper retunes (25% at -1.50)
    looked better on the backtest but FAILED the fresh holdout, so the
    mild version was kept.

Leg B (gated replica, 30%):
  - Price space, per-market sigma = mean |1-minute move| over the last
    12 hours, winsorized at 0.05 and scaled by 0.75, clipped to
    [0.0002, 0.01]. Horizon scaling sqrt(h). Quantile levels +-2.0 for
    the 5%/95% tails and +-0.5 for the 25%/75% mids (tuned, not normal).
  - Median = last price pulled toward x / S, where S is the cross-market
    sum of last prices: the 24 outcomes are mutually exclusive, so the
    Yes prices should sum to 1. The pull ramps in exponentially over
    ~120 minutes. GATED to sane sums only (0.8 <= S <= 1.2): ungated,
    the pull explodes in wild regimes where the sum is meaningless.
  - On its own the replica scores 0.262723 on the practice cases but is
    fragile on the full 19-checkpoint backtest; as a 30% ensemble leg it
    contributes error diversity without the fragility.

Why the ensemble: leg A alone scores 0.269565 on the practice cases;
leg B alone 0.262723. Their 70/30 average scores 0.260489 — the two legs
make different errors (different space, different vol, different median),
and averaging cancels part of each. The weight was chosen on the backtest
plus the fresh holdout: 50/50 gains slightly more on the backtest
(0.258084) but costs 3x as much on the calm holdout; 70/30 keeps most of
the gain at a third of the cost. Mechanisms that looked better on the
backtest but failed the holdout were rejected outright (ungated and gated
sum-reversion alone, split-normal quantiles, asymmetric vol, vol decay).

Validated on the 6 official practice cases plus a 7th case built from the
same public history (checkpoint Nov 13 06:00 UTC) and 11 fresh holdout
checkpoints: mean pinball loss 0.259288 over the 6 official practice cases
(house 0.363589, -29%; v4 scored 0.260489), 0.229642 over the 7
recent checkpoints (v4 0.230648), 0.058675 over the 11 fresh holdout
checkpoints (v4 0.058926 — the v5 change wins on the holdout too, unlike
the round-19 combos which won the backtest and lost the holdout and were
rejected), and 0.932638 over the full 19-checkpoint backtest (v4 0.938438).
Format, determinism (two runs byte-identical) and runtime
(under 0.3 s per case against an 8 s cap) all verified locally before
submission.

Deterministic: pure functions of the case file, no randomness, no clock,
no network. Two runs on the same case write the same bytes.
"""

import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# Retuned quantile levels (leg A / v3): this event's moves are left-skewed,
# so the lower quantiles sit further out and the upper ones closer in.
Z = [-1.80, -0.85, 0.0, 0.5, 1.50]
# v5: big markets (last price >= 0.10) want a STRONGER left-skew retune —
# they are the ones that absorb sharp drops. Small markets keep Z.
Z_HI = [-2.0, -1.0, 0.0, 0.4, 1.30]
Z_HI_THRESH = 0.10

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

# Leg B (gated replica): price space, winsorized mean-|move| vol.
REP_WINDOW = 720      # minutes of history for the replica sigma (12 hours)
REP_WINSOR = 0.05     # winsorize 1-minute moves at +-0.05 before averaging
REP_VOL_F = 0.75      # scale on the winsorized mean |move|
REP_ALPHA = 0.5       # horizon exponent: sigma(h) = sigma * sqrt(h)
REP_Z = [-2.0, -0.5, 0.0, 0.5, 2.0]  # tuned quantile levels (tails +-2.0)
REP_SIGMA_FLOOR = 0.0002  # per-minute sigma floor (price units)
REP_SIGMA_CAP = 0.01      # per-minute sigma cap (price units)
REP_MSR_TAU = 120.0   # timescale (min) for the median sum-reversion ramp
REP_MSR_LO = 0.8      # only revert when the cross-market sum is >= 0.8
REP_MSR_HI = 1.2      # only revert when the cross-market sum is <= 1.2

ENS_W = 0.7           # weight on leg A (v3); leg B takes the remaining 0.3


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


def replica_sigma(diffs):
    """Leg B sigma: mean |1-minute move|, winsorized, times 0.75."""
    n = len(diffs)
    if n < 10:
        return FALLBACK_SIGMA
    w = [max(-REP_WINSOR, min(REP_WINSOR, d)) for d in diffs]
    s = sum(abs(d) for d in w) / n * REP_VOL_F
    return max(REP_SIGMA_FLOOR, min(REP_SIGMA_CAP, s))


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


def forecast_market_v3(prices, horizon):
    """Leg A: the v3 forecaster — 240 rows of five quantiles, logit space."""
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
    zz = Z_HI if x >= Z_HI_THRESH else Z
    for h in range(1, horizon + 1):
        spread = sigma * (h ** ALPHA)
        tail_spread = sigma * (h ** TAIL_ALPHA)
        row = []
        for j, z in enumerate(zz):
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


def forecast_market_replica(prices, horizon, sum0):
    """Leg B: the gated replica — price space, sum-reverting median."""
    n = len(prices)
    if n == 0:
        return [[0.5] * 5 for _ in range(horizon)]
    x = prices[-1]
    w = min(REP_WINDOW, n - 1)
    diffs = [prices[i + 1] - prices[i] for i in range(n - 1 - w, n - 1)]
    sigma = replica_sigma(diffs)
    gated = bool(sum0) and REP_MSR_LO <= sum0 <= REP_MSR_HI
    rows = []
    for h in range(1, horizon + 1):
        med = x
        if gated:
            med_t = x / sum0
            med += (1.0 - math.exp(-h / REP_MSR_TAU)) * (med_t - x)
        s = sigma * (h ** REP_ALPHA)
        row = []
        for j, z in enumerate(REP_Z):
            row.append(price6(med + z * s))
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def forecast_market(prices, horizon, sum0):
    """70/30 ensemble of leg A (v3) and leg B (gated replica)."""
    leg_a = forecast_market_v3(prices, horizon)
    leg_b = forecast_market_replica(prices, horizon, sum0)
    rows = []
    for ra, rb in zip(leg_a, leg_b):
        row = [price6(ENS_W * a + (1.0 - ENS_W) * b) for a, b in zip(ra, rb)]
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
        sum0 = 0.0
        for m in case["markets"]:
            if m["prices"]:
                sum0 += m["prices"][-1]
        forecasts = [
            {"id": m["id"], "q": forecast_market(m["prices"], horizon, sum0)}
            for m in case["markets"]
        ]
        out = json.dumps({"forecasts": forecasts}) + "\n"
        with open(
            os.path.join(OUTPUT_DIR, case_id + ".out"), "w", encoding="utf-8"
        ) as f:
            f.write(out)


if __name__ == "__main__":
    main()
