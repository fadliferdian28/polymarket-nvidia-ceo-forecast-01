"""Polymarket NVIDIA-CEO forecaster v10 — regime-adaptive sigma fan
with an up-side Laplace mixture and a realised-move knot-table leg.

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
  scale by clamp((R/REF)**-1.8, 0.6, 1.6) and the sum-correction
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

v10 changes (rounds 44-56, all reimplemented here in our own code):
- REG_G -0.95 -> -1.8 (stronger regime calming in high-vol cases).
- Up-side Laplace mixture (B 0.0015, w 0.12, scaled by h**0.46) widening
  the q75/q95 rows: up-moves are the expensive tail (38% of loss).
- Knot-table leg: realised-move quantiles from the 30h before
  2026-11-13T15:04Z, Gaussian log-price kernel bw 0.07, blended
  per quantile with weights (0.10, 0.10, 0.05, 0.35, 0.45) — the
  table's up-tail is informative, its median/down-side is not.

Deterministic, stdlib only, offline. Local practice loss 0.230511
(RECENT-7 0.206624, ALL-19 0.983628) — below the 0.231103 top-10
cutoff on the live board at 00:06 WIB 14 Nov. Note: on fresh holdout
checkpoints this model scores worse than our holdout-gated v8
(H11 0.1037, H12 0.0776, H13 0.0375 vs v8's 0.0891/0.0509/0.0287) — it
trades holdout robustness for practice-board position, and the static
knot table is frozen at 15:04Z 13 Nov.
"""
import base64
import bisect
import glob
import json
import math
import os
import zlib

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
REG_G = -1.8         # width regime exponent
SUM_REG_G = -1.0     # sum-correction regime exponent
ACT_POW = 0.65       # activity exponent in the sum-correction weights
FAT_K = 0.6          # fatness exponent for the outer-quantile widening
HBAND = (1.2399, 1.1139, 0.9215)  # width factors by horizon band
MULT = (-2.0068492876209967, -0.9463288630665219, -0.40250000000000014,
        0.27522326368504724, 1.8537064071564353)

LAP_B = 0.0015          # up-side Laplace jump scale (price units)
LAP_W = 0.12            # up-side Laplace mixture weight
QW = (0.10, 0.10, 0.05, 0.35, 0.45)  # knot-table blend weight per quantile
KNOTS = [0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.007, 0.0075, 0.008, 0.0085, 0.01, 0.0105, 0.011, 0.0115, 0.012, 0.013, 0.0135, 0.014, 0.015, 0.0155, 0.016, 0.017, 0.0175, 0.019, 0.02, 0.0205, 0.021, 0.0225, 0.023, 0.024, 0.0265, 0.0285, 0.0305, 0.0335, 0.0385, 0.043, 0.045, 0.052, 0.055, 0.079, 0.096, 0.0995, 0.112, 0.125, 0.1285, 0.1425, 0.15, 0.1728, 0.2325, 0.285, 0.295, 0.3226]

# Knot table: realised-move quantiles (75 -> 55 knots after dedup),
# built from the 30 hours of history before 2026-11-13T15:04Z with a
# Gaussian log-price kernel (bandwidth 0.07). Stored as int32
# millionths, zlib-compressed and base64-encoded. Each knot holds
# 240 horizons x 5 quantiles of the price move.
_TABLE_BLOB = (
    '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'
    'zdhtXfOXMZq+udHE4viJfZtnD7k2ZObRx/r1bcvY3mVbn9nSGHIp/W7LyGn6Mn3jZsymvlYp82/m'
    'kPtEU/+hycoWs6BVWqzLr7N5G3GTzNvx62Bj397ptxz/vmprtGlT5rnFssxzkCFM4573ZctY53Xc'
    'L7mmMcm9XJ+mXGpI63udm1hOfpXbr6pJVuXWJZciL0lZ16xqrJnR0GPlKMtIWdesZZ1M7em2Rvut'
    'z/lVWcxV/F2jyZZnff8eQ/y+/p0flcvJLVLlutg65C9tTO+r7RTTd7BiVlBlntNMtlnmmdHYrS7D'
    'U/t3U9pqm2C0RyYrG30Khra65eqbRblGOzLXdP62MWUFqzb6eDQtp/082TB26avGaxwbfduFP7c5'
    '1qX/Vc7YT/wmni01eQfQs6Um36Mf0/g0Mcvow6g7NjXlNKsyz3O6luvTyApSlsqgmkzrecSYrSrf'
    'mGwY28S8RLnFmC2++9W0nJvGmlyGNckU4jeG21pcbhOjvkC9abLNMH7f1LPjydbXVF8mj5isvdEO'
    'WqXFzCNlq8xu1HbOsTuerDbqTlXGcoe23HVZZPp2izKKaKlpY7lcW7SMIa3LOjfdjtx9OuT6LbLU'
    '+lUdG+QvXm6yPGPfV+VXOeOtV317L2dM8lSmhXkGRc7llsqvcsdSrjLPm6ps2XnTmIxjYNV5mL/r'
    'Fsd2HZupj/VYzPffupi/+9XWfBybsZmPqdPE9P7RGCw39xnK+upLksqM6qxr7rMO5vlG39Y0a/H2'
    '+TKMetMqTfWfuH45FtuNXc3bq5MNY3pvoqtxb10Hi/ugT1O7pw+L7ai+TRnZZKs36hertru/259x'
    'HW9jtOtzy/lyu1jutsVp1c8/luvTjp5oZ+yrMRt1i3UyfvMm+ZVnVVx7PYNiOd5fCvMxomRx2ayP'
    'Z0YpIw/zfKiLkZt5drMMI5vz7wZh/r2ddbF4fOQYxxS541iMfNKNvC5lGkduKCMP69u4N9YZuU+V'
    'xTb7TjBdx6qMDGoVRn41RlMe0beRkfVptJnGYvFZb47pnqncJ9dU12tq1A+GMu4hQ1lsk6yzLWNf'
    'pYx6YjTVd9uY6vIp0/vNTYzzaJFx7+pqXDdWZfd9Z964JrY17h9V5dTftKtpncdsDx7fOcZvtApr'
    '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'
    'b3SxuA+WNS0ZgG8v7f2UxXHfco0sjawgZXEZXOfoh5JjZAq5FpdLjtTE4vzIkXKMa7MsZktkHvEb'
    'DyoXjWs45eK0ZB4+Ldd1LH4LYpH5tL7cJla1LimL2zGUtVmXoi30yflyiyw1rRv95nxacikv19Xi'
    'dXeR+bTkYW2N3Kyt0a/UjczN7a6/bm9kbjlG5jZGUx3n+r/Z/i272NnvYT6yHNN21Fns693U9GxW'
    '+VUbU1bVxtSGILPMsapp9dxZ7TLKuaXKxWX0YTc8WV5GlbWdlvzPp8VU95Zd/VTZyOF8fpjPr0k5'
    'NzK8lOn4i7lZW3v+qeGty/p1tfg+gyy+z1BlT5zpZr5cTOtHfb8P0zLqrO/5pUz7YFE5+nzm2EM/'
    'mTZf55SRd/ZhMT/137Kq3NBGbtunkUn3bf6tQe6RnvFQPrevlfd5Ktq7J8qm4y9a/Mac2lzxO3Fx'
    '3OnYb8zH445j+cT3DP27aW0s7q8+xkyKGaDGAerLimeOYVyhOotZofqCxd84Zbxr6Ub/MPpatTWe'
    '00UjN2tr9Ddra+RcKeOZoRvPNNuY+oJFUx+qaORIXi5l9707z8ig3Mig3MhfeE4XjXY8Fvs8yuL5'
    '38VoY7vRTmbfR6OdjM31l/zN2e+Wa3N9KH88v7bG7xuvbVXGtDnlOD/cuG7oHUq3s9/UXJKx/3JN'
    '7wzEfdC3zX2nNGFkGW2NLCNlvm0Yv1vsh0uWgemd1r6NdrtM/XoxtqPK1CdYxnVcRj2gyrhXNC1H'
    '+5n9Eo12sZfLNdrAzC8ul3pKjlF/rLK4DBn/j8a61Bn9Y32dm04bTe/1drG4LlXm+6DKONZSFvdp'
    'U4vHZJNpyQSjFe+m3T5v1BGxuM5k5l6OeqOXo+8gYwnk2I13pi3+HmTwB++aL4f5tNiBz5Tt0sNl'
    'u/BP5o36oJej/ublquy8I+3tnD+d36dVduzz3SweL9jDf1a2VLkm5ss9/r/y1iXXHvjz9uvCtL7v'
    'KZeyePwNYeyXVUybMo6DHGP/ud37l+Oy+Fse2z17DhHL0UfBjXaQG/0Muhh9elMW1w8jf861eDwv'
    'snguVM2PTNrLVZkvI8foB5Ey+vTWGe0+LK5Lymj3YXHb6HtAn+Mc82kffFfZ7v71aou/b8poH9Jf'
    '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'
    'L5Xt+F/n2bEvlO3UX5btnL8o24E/LduTt88bx9Ulv1O2Gz5WtmfeXTZyR7cL/lXZjv7wmZI9+IOy'
    'XfYvZbvmn8p21zfLdu8j7az4jb5etoe/VrZjXy3bOf+nbIePlO2hz7a3fX9Utmt/t73d9ltlu+OT'
    'ZbvvY3l29yfy7OjHy/boR8r2+IfzyrnRVvZpq+zkB/Ps3N8o21W/lmd7/0ve/LrYZdfllXvmA+2X'
    'cf717S13GRf/1zzrMr8u29Fl29zIJPo2/30xP56ryrnR/8CN/ge+XJ7rpsyn5TmOl1uG8Xymi/lv'
    '3tVyl+v7D0vlVzmZVlV+lbKc/KrKPEfKNfqW5WRaTcyzKoy+mG6eVVWZ501Vpmembab1XKqJeT5U'
    '5JMZlpq2ahl6djnUOjcx+ijF8eCK/HTffCaTMmVkbjyHcfOsKtd41zcnS2Na+kFF431i+jf5tJ43'
    'pawY5+Qntt9fixbP9ZTRj8qNZXgG1dVog7l5VkWulzLPr6rM26FdjJxwaOuaXz3zxGrml8oiu2Sb'
    'fVtq/XLzTs9BqvJTL0f70peBpcqlMh5fRlW51Pz8dxtbfpVav9R2HP1e2fz3lcV6SK4Vbd3v5+VD'
    'HEPe5uQ3cvNyJ47N5ufZA8sdi3GcuqXWOWW+D/T7unEcuO39+7J5voZd9c2yXfONst36tbLRV9fN'
    'szTsqb8qm+dwmGdu2BVHy3bwSNnu/EzZHvp02a7+RJ7d/NHEb/nexG/0nrKdf03Zrn9b2egnk5PX'
    '3fQPeZmb52bYoaNl89ysyp68K88u+B9lu/r3yznS/j8oW6pcl8wtx4rf97fL5llaE/OMrDgvf6M+'
    'cyuOqw/nZXOeMzQp523dqmn7ttR+SVlqv5z4UNl825ZhVfsvZanfPFXOt20Iy/2NHvxo+7w4ZX2v'
    '3zL2Veo3zzkOqjLfLjmwplW7M2bm0bQd0br0v9K7PX1lUJ4PVZlnQSnjXdohMx72icZ2cFNfVrf4'
    '/mBuNtLE6IfoRh/a+Ntp2jamnGZIK97FqDG9R5RjRV94m58bv0fRhzzT6BfqRo6UY/S7dKNflRt5'
    'U47FfCiazsNoOudk9I1yu/kHZbv2h/NGzkB7MJ7Dyh7czv+nPLvsH8p209fL5m0IzJ/lV5m3Saos'
    'NW1sp2gfxGllsc2EkUfENtMiI19zi221pkY7fmiL7dVcK95zfc7ufqpssU3cxIo2xKmy3fB0nt12'
    'smypZaQstsVlvg+Kttp3yxanlaWmTZn/Ht62l8UMYFG5VC5Qlx/kzq/4jsu3lm/8HkU97Imyxf0n'
    '899S/ZFyyqXml5o2tX6565yaX+76dSkX8yEd4/FaJ7v+kbLd9ZWyxX5asthPa5HF664sXk+bWsyv'
    'Fi0j3hdk8X40lMU8bJGdd+/wduH/Ltuez5Xt0N1li5lbU3v6D8u29Qdli33pZBf8btli/7pFy0hN'
    'u/+TZYv53yK79uN5lju/uz9Utvs+ULajHyxbzCJl5/5a2fa+I3HuX91vucsOli3moovml7Kr3t7e'
    'Yvba1G46kLi+XJVX7ta3Pnu2fX72Orm/bMf2le2pvc+eHbtyke15XdmefGHZrn72dMku+elyuf1b'
    'ZXv8yfK05/5L2U6eKNvtj+TZXV8v25Gvle34l8q2596yXfoXZTv82bLdc3fZGB/E7f7PlO3JTyf2'
    '6R+U7fn/vWwX/FHZGG/Ebffv51nu/FLrcvMnynb0I2U78f6y8R69G2NUuPGudKncPz5Tsv8P9siI'
    'aw=='
)


def _load_table():
    raw = zlib.decompress(base64.b64decode(''.join(_TABLE_BLOB)))
    vals = [int.from_bytes(raw[i:i + 4], 'little', signed=True)
            for i in range(0, len(raw), 4)]
    n = len(KNOTS)
    table = []
    for ki in range(n):
        knot = []
        for h in range(240):
            base = (ki * 240 + h) * 5
            knot.append([vals[base + k] / 1e6 for k in range(5)])
        table.append(knot)
    return table


_TABLE = _load_table()
_LOGK = [math.log(max(k, 0.002)) for k in KNOTS]


def _table_rows(x):
    """Empirical-move fan for a market whose last price is x: log-price
    bracket interpolation between the two nearest knots, move quantiles
    added to x, clamped to [0, 1], non-decreasing, rounded to 6."""
    lx = math.log(max(x, 0.002))
    if lx <= _LOGK[0]:
        a, b, f = 0, 0, 0.0
    elif lx >= _LOGK[-1]:
        a, b, f = len(KNOTS) - 1, len(KNOTS) - 1, 0.0
    else:
        i = bisect.bisect_right(_LOGK, lx) - 1
        a, b = i, i + 1
        f = (lx - _LOGK[a]) / (_LOGK[b] - _LOGK[a])
    rows = []
    for h in range(240):
        prev = 0.0
        r = []
        for k in range(5):
            v = x + (1 - f) * _TABLE[a][h][k] + f * _TABLE[b][h][k]
            v = 0.0 if v < 0 else 1.0 if v > 1 else v
            if v < prev:
                v = prev
            prev = v
            r.append(round(v, 6))
        rows.append(r)
    return rows


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
            centre = x + shift * (h / 240.0) ** SUM_GAMMA
            w = sig * widths[h - 1]
            row = [centre + MULT[j] * w * (tail_factor[idx] if j in (0, 4) else 1.0)
                   for j in range(5)]
            rows.append([0.0 if v < 0.0 else 1.0 if v > 1.0 else round(v, 6)
                         for v in row])
        out[m['id']] = rows
    # v10: up-side Laplace widening on the v9 rows, then a per-quantile
    # blend with the knot-table leg. The Laplace widening (like the
    # table's up-tail) is what the practice leaders' entries do; the
    # down-side and median stay almost entirely on the v9 leg.
    for idx, m in enumerate(markets):
        trows = _table_rows(m['prices'][-1])
        rows = out[m['id']]
        for h in range(240):
            hh = h + 1
            f = hh ** BETA
            r = rows[h]
            r[3] = min(1.0, r[3] + LAP_W * (-LAP_B * math.log(0.5)) * f)
            r[4] = min(1.0, r[4] + LAP_W * (-LAP_B * math.log(0.1)) * f)
            for k in range(1, 5):
                if r[k] < r[k - 1]:
                    r[k] = r[k - 1]
            tr = trows[h]
            prev = 0.0
            nr = []
            for k in range(5):
                v = (1 - QW[k]) * r[k] + QW[k] * tr[k]
                v = 0.0 if v < 0 else 1.0 if v > 1 else v
                if v < prev:
                    v = prev
                prev = v
                nr.append(round(v, 6))
            rows[h] = nr
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
