#!/usr/bin/env python3
"""polymarket-nvidia-ceo-forecast-01 entry (Python 3.13, standard library only).

Forecasts the Yes price of every market of Polymarket's "Next CEO of
NVIDIA?" event, minute by minute, 4 hours ahead, as five quantiles.

Design v6: a 70/30 ENSEMBLE of two forecasters whose errors partly cancel.
Leg A (70%) is the v3 forecaster below, with one v5 change: big markets
(last price >= 0.10) use a STRONGER left-skew quantile retune (5% at -2.0,
25% at -1.0, 75% at 0.4, 95% at 1.30) while small markets keep the v3
levels — the big markets are the ones that absorb sharp drops, and the
split won on all four evaluations (backtest ALL/RECENT/PRACTICE plus a
fresh 11-checkpoint holdout). Leg B (30%) was a gated replica of
a friend's submitted v2 (price space, winsorized mean-absolute-move vol,
median sum-reversion toward x/sum).

Design v7 (this file): the second leg is REPLACED. The friend shared a
newer model of theirs (practice 0.2461, verified on our own harness —
their number reproduces exactly), and it is structurally different from
leg A: price space, short-memory EWMA vol, a cross-market level-fit
shrink, and a sigma-weighted sum correction. On its own it is fragile
(full backtest 1.58, both fresh holdouts worse than v6), so it is NOT
adopted wholesale — but as a 20% ensemble leg it wins BOTH fresh
holdouts (13 checkpoints 0.086956 vs v6 0.088309; 6 newer checkpoints
0.054528 vs 0.057266) at every weight from 10% to 40%. The 80/20 weight
is the holdout-best (and halves the wild-regime cost of 70/30). The
leg is reimplemented here from the mechanism description, not copied:
the math is the friend's, the code is ours. The old replica leg is gone
(a three-way ensemble with both legs was tested and rejected).

Design v6: leg A gains an ASYMMETRIC MEDIAN (round 25, inspired by the
board leader's "slow asymmetric bounce + down-continuation"): after a
recent rise the median gives the rise back in full, ramping in over
~120 minutes; after a recent fall the median does NOT continue down.
The asymmetry is the whole point — a symmetric give-back and a
down-continuation were both tested and both HURT (the down leg alone
cost +0.004 on the practice cases), while the up-bounce alone won on
all four evaluations (practice, recent, full backtest, fresh holdout).
Rises in this event are noise that reverts; falls are signal that
persists. The give-back uses the 60-minute return and applies in price
space before the logit transform, so the median at long horizons after
a rise returns toward the price an hour ago.

Leg A (v3, 80%):
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

Leg B (short-memory leg, 20% — reimplemented from the friend's mechanism):
  - Price space. Per-market sigma = EWMA-weighted RMS of the last 120
    one-minute returns, 26-minute half-life: recent volatility is nearly
    all that matters in these markets. The raw sigma is then shrunk
    toward a cross-market target: a log-log OLS fit of sigma on
    x*(1-x) (the Bernoulli variance — the natural volatility scale for a
    probability), blended with the median sigma across markets. Final
    sigma = 0.4*raw + 0.6*(0.6*level-fit + 0.4*median), clipped to
    [0.0002, 0.005] and scaled by 0.87.
  - Centre = last price plus a cross-market sum correction: the gap
    (1 - S) * 1.15 is shared across markets in proportion to their
    (pre-level-fit) sigma, phased in as (h/240)**0.7. Markets with more
    vol absorb more of the correction.
  - Quantiles = centre + MULT[tau] * sigma * h**0.47, with horizon-band
    width factors (1.08 for h<=30, 1.05 for h<=120, 0.95 beyond). The
    multipliers are tuned and asymmetric: (-1.813, -0.952, -0.270,
    +0.391, +1.862) — note the median sits at -0.27, not 0: a
    sigma-proportional downward median shift.
  - On its own this leg scores 0.246137 on the practice cases (better
    than v6's 0.248906) but is fragile on the full backtest (1.58) and
    loses both fresh holdouts; as a 20% leg it contributes error
    diversity plus a faster-adapting vol estimate.

Why the ensemble: leg A alone scores 0.269565 on the practice cases;
the short-memory leg alone 0.246137. Their 80/20 average scores
0.251313 on practice — worse than v6's 0.248906 — but wins both FRESH
holdouts (0.086956 vs 0.088309; 0.054528 vs 0.057266), which are the
newest data and the best predictor of the final cases. The backtest
loss concentrates in the wild Nov 10-11 regime (a one-off launch event);
the final cases run in the recent regime, where the short-memory leg
adapts fastest. Every weight from 10% to 40% won both holdouts; 80/20
is the holdout-best. Mechanisms that looked better on the backtest but
failed the holdout were rejected outright (ungated and gated
sum-reversion alone, split-normal quantiles, asymmetric vol, vol decay,
quantile-regression multipliers, gap-momentum median).

Validated on the 6 official practice cases plus a 7th case built from the
same public history (checkpoint Nov 13 06:00 UTC) and 19 fresh holdout
checkpoints: mean pinball loss 0.251313 over the 6 official practice cases
(house 0.363589, -31%; v6 scored 0.248906), 0.225292 over the 7
recent checkpoints (v6 0.223018), 0.086956 over the 13 fresh holdout
checkpoints (v6 0.088309 — the v7 change wins 9 of 13 checkpoints),
0.054528 over 6 newer holdout checkpoints (v6 0.057266), and 0.970502
over the full 19-checkpoint backtest (v6 0.921418 — the wild-regime
cost of the short-memory leg).
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
# v6: asymmetric median — give back the recent rise in full (k_up 1.0),
# never continue a fall (k_down 0). 60-minute return, ~120-min ramp.
ASYM_K_UP = 1.0
ASYM_WINDOW = 60
ASYM_TAU = 120.0

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

# Leg B (short-memory leg): price space, EWMA vol, level-fit shrink.
LB_WINDOW = 120       # returns window for the EWMA sigma (minutes)
LB_HALFLIFE = 26.0    # half-life (minutes) of the sigma EWMA
LB_SHRINK = 0.4       # weight of the median sigma in the shrink target
LB_LVL_SHRINK = 0.6   # weight of the shrink target in the final sigma
LB_FLOOR = 0.0002     # per-minute sigma floor (price units)
LB_CAP = 0.005        # per-minute sigma cap (price units)
LB_SCALE = 0.87       # global sigma scale
LB_BETA = 0.47        # horizon exponent: width = sigma * h**LB_BETA
LB_SUM_K = 1.15       # sum-correction strength (toward a total of 1.0)
LB_MULT = [-1.8130133458944, -0.9521483940509615, -0.27,
           0.3914563862643396, 1.8623702217199494]  # tuned multipliers

ENS_W = 0.8           # weight on leg A (v3); leg B takes the remaining 0.2


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


def ewma_rms(diffs):
    """Leg B sigma: EWMA-weighted RMS of the last LB_WINDOW 1-minute moves.

    Recent volatility is nearly all that matters in these markets, so the
    weights decay with a 26-minute half-life. Unlike a plain RMS this
    adapts within the hour after a regime change.
    """
    r = diffs[-LB_WINDOW:]
    if not r:
        return FALLBACK_SIGMA
    decay = 0.5 ** (1.0 / LB_HALFLIFE)
    weights = [decay ** (len(r) - 1 - i) for i in range(len(r))]
    num = sum(w * v * v for w, v in zip(weights, r))
    return math.sqrt(num / sum(weights))


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
        aret = 0.0
    else:
        x = prices[-1]
        lx = logit(x)
        log_prices = [logit(v) for v in prices]
        sigma = (1.0 - BLEND_F) * window_sigma(log_prices, WINDOW) + BLEND_F * window_sigma(
            log_prices, BLEND_WINDOW
        )
        # v6: the 60-minute return, for the asymmetric median.
        aw = min(ASYM_WINDOW, n - 1)
        aret = x - prices[n - 1 - aw]
    rows = []
    zz = Z_HI if x >= Z_HI_THRESH else Z
    for h in range(1, horizon + 1):
        # v6: after a recent rise, the median gives the rise back in full
        # (ramping in over ~120 min); after a fall, no continuation.
        if aret > 0:
            lm = logit(x - (1.0 - math.exp(-h / ASYM_TAU)) * ASYM_K_UP * aret)
        else:
            lm = lx
        spread = sigma * (h ** ALPHA)
        tail_spread = sigma * (h ** TAIL_ALPHA)
        row = []
        for j, z in enumerate(zz):
            if j in (0, 4):
                v = sigmoid(lm + z * tail_spread * TAIL_SCALE)
            else:
                v = sigmoid(lm + z * spread)
            row.append(price6(v))
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def forecast_case_leg_b(markets, horizon):
    """Leg B: the short-memory forecaster, for a whole case at once.

    The level-fit shrink and the sum correction are cross-market, so this
    leg is computed case-wide. Returns {market id: 240 rows of quantiles}.
    """
    n = len(markets)
    last = [m["prices"][-1] if m["prices"] else 0.5 for m in markets]
    raw = []
    for m in markets:
        prices = m["prices"]
        diffs = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        raw.append(ewma_rms(diffs))
    ordered = sorted(raw)
    med = ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])

    # Level-fit shrink target: log sigma ~ a + b * log(x * (1 - x)).
    # Volatility scales with the Bernoulli variance of the price level.
    lx = [math.log(max(1e-6, x * (1.0 - x))) for x in last]
    ly = [math.log(max(1e-9, r)) for r in raw]
    mx = sum(lx) / n
    my = sum(ly) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    sxx = sum((a - mx) ** 2 for a in lx)
    slope = sxy / sxx if sxx > 0 else 0.0
    icept = my - slope * mx

    sigmas = []
    plain = []
    for i in range(n):
        level = math.exp(icept + slope * lx[i])
        target = level * (1.0 - LB_SHRINK) + med * LB_SHRINK
        s = raw[i] * (1.0 - LB_LVL_SHRINK) + target * LB_LVL_SHRINK
        sigmas.append(min(LB_CAP, max(LB_FLOOR, s)) * LB_SCALE)
        p = raw[i] * (1.0 - LB_SHRINK) + med * LB_SHRINK
        plain.append(min(LB_CAP, max(LB_FLOOR, p)) * LB_SCALE)

    # Sum correction toward a total of exactly 1.0, shared in proportion
    # to the plain (pre-level-fit) sigma: volatile markets absorb more.
    gap = (1.0 - sum(last)) * LB_SUM_K
    tot = sum(plain) or 1.0
    shifts = [gap * s / tot for s in plain]

    widths = [(h ** LB_BETA) * (1.08 if h <= 30 else 1.05 if h <= 120 else 0.95)
              for h in range(1, horizon + 1)]
    out = {}
    for idx, m in enumerate(markets):
        x = last[idx]
        sig = sigmas[idx]
        sh = shifts[idx]
        rows = []
        for h in range(1, horizon + 1):
            centre = x + sh * min(1.0, h / 240.0) ** 0.7
            w = sig * widths[h - 1]
            row = [price6(centre + LB_MULT[j] * w) for j in range(5)]
            for j in range(1, 5):
                if row[j] < row[j - 1]:
                    row[j] = row[j - 1]
            rows.append(row)
        out[m["id"]] = rows
    return out


def forecast_case(case):
    """80/20 ensemble of leg A (v3, per market) and leg B (short-memory)."""
    horizon = case["horizon_minutes"]
    leg_b = forecast_case_leg_b(case["markets"], horizon)
    forecasts = []
    for m in case["markets"]:
        leg_a = forecast_market_v3(m["prices"], horizon)
        rows = []
        for ra, rb in zip(leg_a, leg_b[m["id"]]):
            row = [price6(ENS_W * a + (1.0 - ENS_W) * b) for a, b in zip(ra, rb)]
            for j in range(1, 5):
                if row[j] < row[j - 1]:
                    row[j] = row[j - 1]
            rows.append(row)
        forecasts.append({"id": m["id"], "q": rows})
    return forecasts


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
        forecasts = forecast_case(case)
        out = json.dumps({"forecasts": forecasts}) + "\n"
        with open(
            os.path.join(OUTPUT_DIR, case_id + ".out"), "w", encoding="utf-8"
        ) as f:
            f.write(out)


if __name__ == "__main__":
    main()
