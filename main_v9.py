"""Polymarket NVIDIA-CEO forecaster v9 — regime-adaptive sigma fan.

A single-model forecaster: per-market volatility from an EWMA-weighted
RMS of recent one-minute returns, shrunk toward a cross-market level fit
(log sigma on log price*(1-price)) blended with the median sigma; a
case-level regime factor that calms widths and the sum correction in
high-volatility cases; a cross-market sum correction toward a total of
exactly 1.0, shared by plain sigma times market activity (a liquidity
proxy); and asymmetric quantile multipliers with a per-market fatness
widening on the outer quantiles.

Mechanisms (all reimplemented here in our own code):
- sigma: EWMA RMS over the last 120 one-minute returns (26-minute
  half-life), shrunk 60% toward a level-fit target (level fit blended
  40% with the cross-market median sigma), clipped to [0.0001, 0.005],
  scaled by 0.9.
- regime: with R = the case's median raw sigma and REF = 0.0003, widths
  scale by clamp((R/REF)**-0.95, 0.6, 1.6) and the sum-correction
  strength by (R/REF)**-1.0 (clamped to [0.1, 3.0] — the unclamped form
  explodes on quiet cases). In high-volatility cases the plain model
  over-disperses and over-corrects, so both are calmed down.
- centre: last price + sum shift, phased in as (h/240)**0.46. The gap
  (1 - board sum) * 0.84 is shared in proportion to plain sigma times
  activity**0.65, where activity is the fraction of minutes with a
  nonzero return — illiquid markets absorb less of the correction.
- quantiles: centre + MULT[tau] * sigma * h**0.46 * horizon-band factor
  (1.2399 for h<=30, 1.1139 for h<=120, 0.9215 beyond), with the outer
  quantiles widened per market by a fatness factor
  clamp((RMS/mean|ret| / case median)**0.6, 0.8, 1.6).
  MULT = (-2.007, -0.946, -0.403, +0.275, +1.854).

Robustness fixes (ours): the regime factors use a guarded median
(1e-9 floor) so a fully quiet case cannot raise 0.0 to a negative
power, and the sum regime factor is clamped so it cannot explode.

Deterministic, stdlib only, offline. Local practice loss 0.2374
(top-10 class on the live board).
"""
import glob
import json
import math
import os

HL = 26.0            # half-life (minutes) of the sigma EWMA
WIN = 120            # returns window for sigma (minutes)
SHRINK = 0.4         # weight of the median in the shrink target
LVL_W = 0.6          # weight of the level-fit shrink target
SIG_FLOOR = 0.0001   # sigma floor (price units per sqrt-minute)
SIG_CAP = 0.005      # sigma cap (before GSCALE)
GSCALE = 0.9         # global sigma scale
BETA = 0.46          # horizon exponent of the width
SUM_K = 0.84         # sum-correction strength (toward a total of 1.0)
SUM_GAMMA = 0.46     # horizon phase-in exponent of the sum shift
REF_SIG = 0.0003     # reference median sigma for the regime factors
REG_G = -0.95        # width regime exponent
SUM_REG_G = -1.0     # sum-correction regime exponent
ACT_POW = 0.65       # activity exponent in the sum-correction weights
FAT_K = 0.6          # fatness exponent for the outer-quantile widening
HBAND = (1.2399, 1.1139, 0.9215)  # width factors by horizon band
MULT = (-2.0068492876209967, -0.9463288630665219, -0.40250000000000014,
        0.27522326368504724, 1.8537064071564353)


def _median(vals):
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _returns(prices, window):
    rets = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    return rets[-window:]


def _ewma_rms(rets):
    if not rets:
        return 0.0025
    decay = 0.5 ** (1.0 / HL)
    weights = [decay ** (len(rets) - 1 - i) for i in range(len(rets))]
    return math.sqrt(sum(w * v * v for w, v in zip(weights, rets)) / sum(weights))


def forecast_case(case):
    """case: a case dict with 'markets' (id, prices). Returns {id: rows}."""
    markets = case['markets']
    n = len(markets)

    # Per-market raw sigma (EWMA RMS) and the cross-market median.
    raw = [_ewma_rms(_returns(m['prices'], WIN)) for m in markets]
    med = _median(raw)

    # Level-fit shrink target: log sigma ~ a + b * log(x * (1 - x)).
    lx = [math.log(max(1e-6, m['prices'][-1] * (1.0 - m['prices'][-1])))
          for m in markets]
    ly = [math.log(max(1e-9, r)) for r in raw]
    mx = sum(lx) / n
    my = sum(ly) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    sxx = sum((a - mx) ** 2 for a in lx)
    slope = sxy / sxx if sxx > 0 else 0.0
    intercept = my - slope * mx

    sigmas = []
    for i in range(n):
        level = math.exp(intercept + slope * lx[i])
        target = level * (1.0 - SHRINK) + med * SHRINK
        s = raw[i] * (1.0 - LVL_W) + target * LVL_W
        sigmas.append(min(SIG_CAP, max(SIG_FLOOR, s)) * GSCALE)
    plain = [min(SIG_CAP, max(SIG_FLOOR, r * (1.0 - SHRINK) + med * SHRINK)) * GSCALE
             for r in raw]

    # Regime factors from the case's median raw sigma. Guarded: a fully
    # quiet case (median 0) must not raise 0.0 to a negative power, and
    # the sum factor is clamped so it cannot explode.
    med_safe = med if med > 0 else 1e-9
    width_factor = min(1.6, max(0.6, (med_safe / REF_SIG) ** REG_G))
    sigmas = [s * width_factor for s in sigmas]
    sum_factor = min(3.0, max(0.1, (med_safe / REF_SIG) ** SUM_REG_G))

    # Cross-market sum correction toward a total of exactly 1.0, shared
    # in proportion to plain sigma times activity**0.65.
    board_sum = sum(m['prices'][-1] for m in markets)
    gap = (1.0 - board_sum) * SUM_K * sum_factor
    weights = []
    for idx, m in enumerate(markets):
        rr = _returns(m['prices'], WIN)
        activity = (sum(1 for v in rr if v != 0.0) / len(rr)) if rr else 0.0
        weights.append(plain[idx] * activity ** ACT_POW)
    total = sum(weights) or 1.0
    shifts = [gap * w / total for w in weights]

    widths = [(h ** BETA) * (HBAND[0] if h <= 30 else HBAND[1] if h <= 120
                             else HBAND[2])
              for h in range(1, 241)]
    fatness = []
    for m in markets:
        rr = _returns(m['prices'], WIN)
        mean_abs = sum(abs(v) for v in rr) / len(rr) if rr else 0.0
        rms = math.sqrt(sum(v * v for v in rr) / len(rr)) if rr else 0.0
        fatness.append(rms / mean_abs if mean_abs > 0 else 1.25)
    fat_med = _median(fatness)
    tail_factor = [min(1.6, max(0.8, (f / fat_med) ** FAT_K)) for f in fatness]

    out = {}
    for idx, m in enumerate(markets):
        x = m['prices'][-1]
        sig = sigmas[idx]
        shift = shifts[idx]
        rows = []
        for h in range(1, 241):
            centre = x + shift * min(1.0, h / 240.0) ** SUM_GAMMA
            w = sig * widths[h - 1]
            row = [centre + MULT[j] * w * (tail_factor[idx] if j in (0, 4) else 1.0)
                   for j in range(5)]
            rows.append([0.0 if v < 0.0 else 1.0 if v > 1.0 else round(v, 6)
                         for v in row])
        out[m['id']] = rows
    return out


def main():
    in_dir = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
    out_dir = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(in_dir, '*.in'))):
        with open(path) as f:
            case = json.load(f)
        forecasts = forecast_case(case)
        rows = [{'id': m['id'], 'q': forecasts[m['id']]} for m in case['markets']]
        out_path = os.path.join(out_dir, case['case'] + '.out')
        with open(out_path, 'w') as f:
            json.dump({'forecasts': rows}, f, separators=(',', ':'))


if __name__ == '__main__':
    main()
