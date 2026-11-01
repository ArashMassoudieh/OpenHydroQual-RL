"""
Functional-flow targets and scores, a simplified version of the California
Environmental Flows Framework (CEFF; Yarnell et al. 2020, River Research and Applications 36:318).

CEFF describes a healthy river hydrograph by five functional-flow components.
Here each component gets a fixed calendar window (CEFF detects the timing from
the hydrograph; fixed windows keep the rules easy to read) and a target that
is taken from the unimpaired (natural) flow of the same river, separately for
dry, moderate and wet water years:

  component             window (water-year days)       requirement
  FA fall pulse         Oct 15 - Dec 15                >= D_FA days with Q >= Q_FA
  WB wet baseflow       Dec 16 - Mar 31                Q >= Q_WB every day
  PK peak flow          Dec  1 - Apr 30                >= D_PK days with Q >= Q_PK
  SP spring pulse       Apr  1 - May 31                >= D_SP days with Q >= Q_SP
     and recession      Apr 15 - Jun 30                decline of 7-day mean <= ROC_max
  DS dry baseflow       Jul  1 - Oct 14                Q >= Q_DS every day

Year score in [0, 1] per component, with graded day credit (1 at the
target, falling linearly to 0 at 90% of it; see credit()):
  baseflows   mean day credit over the window
  pulses      min(1, sum of day credits / days required)
  spring      0.5 * pulse score + 0.5 * mean recession credit (1 within
              ROC_max, falling linearly to 0 at twice ROC_max)
The functional-flow score of a year is the mean over the components that apply
(a pulse whose required duration is zero, e.g. peak flows in dry years, does
not apply).

Water years start on Oct 1; day index 0 is Oct 1, Feb 29 is ignored (day 151
of a leap year is treated as Mar 1), and day 365 of a leap year is folded onto
day 364.
"""
import numpy as np

YEAR_TYPES = ("dry", "moderate", "wet")
TOL = 0.01      # a flow within 1% of a target counts as meeting it

# Windows as half-open water-year day ranges [start, end).
WINDOWS = {
    "FA": [(14, 76)],
    "WB": [(76, 182)],
    "PK": [(61, 212)],
    "SP": [(182, 243)],
    "REC": [(196, 273)],
    "DS": [(0, 14), (273, 365)],
}


def in_window(name, day):
    return any(a <= day < b for a, b in WINDOWS[name])


def window_mask(name):
    m = np.zeros(365, bool)
    for a, b in WINDOWS[name]:
        m[a:b] = True
    return m


def water_year(date):
    y, m = int(str(date)[:4]), int(str(date)[5:7])
    return y + (m >= 10)


def split_water_years(dates, q):
    """{water year: 365 daily values}, leap days dropped."""
    out = {}
    wy = np.array([water_year(d) for d in dates])
    feb29 = np.array([str(d)[5:10] == "02-29" for d in dates])
    for y in np.unique(wy):
        x = q[(wy == y) & ~feb29]
        if len(x) == 365:
            out[int(y)] = x
    return out


def classify_years(annual_volume):
    """Tercile year types from {year: volume}."""
    v = np.array(list(annual_volume.values()))
    lo, hi = np.percentile(v, [100 / 3, 200 / 3])
    return {y: ("dry" if x < lo else "wet" if x > hi else "moderate")
            for y, x in annual_volume.items()}


def _days_above(x, thr):
    return int(np.sum(x >= (1.0 - TOL) * thr))


def natural_targets(years_q, year_type, Q_safe, max_pulse_days=7):
    """Targets per year type from unimpaired flow {year: 365 values} [m3/s].

    Following the CEFF metrics, the wet-season floor Q_WB is the median over
    the natural years of that type of the window's 10th-percentile flow
    (Wet_BFL_Mag_10) and the dry-season floor Q_DS the 25th percentile over
    those years of the window's median flow (DS_Mag_50); the fall pulse is the 25th percentile of
    the natural fall maximum and the spring pulse half the median natural
    spring maximum, so unregulated flow meets most targets. Pulse targets
    are capped for regulated delivery: the peak-flow magnitude at 0.8*Q_safe
    (bigger releases would flood the channel) and pulse durations at
    max_pulse_days.
    """
    m = {k: window_mask(k) for k in WINDOWS}
    annual_peaks = [x.max() for x in years_q.values()]
    Q_PK = min(float(np.median(annual_peaks)), 0.8 * Q_safe)
    # Recession: 90th percentile of the natural daily relative declines of the
    # 7-day mean flow (raw daily FNF is spiky; regulated releases are not).
    declines = []
    for x in years_q.values():
        xs = np.convolve(x, np.ones(7) / 7, "same")
        r = -np.diff(xs)[m["REC"][1:]] / np.maximum(xs[:-1][m["REC"][1:]], 1e-6)
        declines.extend(r[r > 0])
    ROC_max = float(np.percentile(declines, 90))
    T = {}
    for t in YEAR_TYPES:
        ys = [y for y in years_q if year_type[y] == t]
        X = [years_q[y] for y in ys]
        Q_FA = float(np.percentile([x[m["FA"]].max() for x in X], 25))
        Q_SP = 0.5 * float(np.median([x[m["SP"]].max() for x in X]))
        T[t] = dict(
            Q_FA=Q_FA, D_FA=2,
            Q_WB=float(np.median([np.percentile(x[m["WB"]], 10) for x in X])),
            Q_PK=Q_PK,
            D_PK=int(min(max_pulse_days, np.median([_days_above(x[m["PK"]], Q_PK) for x in X]))),
            Q_SP=Q_SP,
            D_SP=int(min(max_pulse_days, np.median([_days_above(x[m["SP"]], Q_SP) for x in X]))),
            ROC_max=ROC_max,
            Q_DS=float(np.percentile([np.median(x[m["DS"]]) for x in X], 25)),
        )
    return T


def credit(q, thr):
    """Day credit in [0, 1] for a flow q against a target thr: 1 at or above
    the target (within TOL), falling linearly to 0 at 90% of it. The same
    graded credit is used by the reward, so small release errors cost little
    instead of failing a whole day."""
    return np.clip((np.asarray(q) / thr - 0.9) / (0.1 - TOL), 0.0, 1.0)


def component_scores(q, T):
    """Year scores of a 365-day river flow series q under targets T (one type)."""
    m = {k: window_mask(k) for k in WINDOWS}
    s = {}
    s["FA"] = min(1.0, float(credit(q[m["FA"]], T["Q_FA"]).sum()) / T["D_FA"])
    s["WB"] = float(credit(q[m["WB"]], T["Q_WB"]).mean())
    if T["D_PK"] > 0:
        s["PK"] = min(1.0, float(credit(q[m["PK"]], T["Q_PK"]).sum()) / T["D_PK"])
    pulse = min(1.0, float(credit(q[m["SP"]], T["Q_SP"]).sum()) / T["D_SP"]) if T["D_SP"] > 0 else 1.0
    rec_days = np.flatnonzero(m["REC"])
    rate = np.array([recession_rate(q[d - 7:d + 1]) for d in rec_days])
    r = T["ROC_max"]
    rec = 1.0 - np.minimum(1.0, np.maximum(0.0, rate - (1.0 + TOL) * r) / r)
    s["SP"] = 0.5 * pulse + 0.5 * float(rec.mean())
    s["DS"] = float(credit(q[m["DS"]], T["Q_DS"]).mean()) if T["Q_DS"] > 0 else 1.0
    return s


def recession_rate(q8):
    """Relative decline of the trailing 7-day mean flow from yesterday to
    today, from the last 8 daily flows (the same smoothing used to derive
    ROC_max from the natural flow)."""
    m_prev, m_now = np.mean(q8[:7]), np.mean(q8[1:])
    return (m_prev - m_now) / max(m_prev, 1e-6)


def ff_score(q, T):
    s = component_scores(q, T)
    return float(np.mean(list(s.values()))), s


def prescribed_hydrograph(T, fa_day=20, pk_day=137, sp_day=201):
    """A fixed functional-flow release schedule (minimum river flow) for one
    water year that meets targets T by construction: baseflow floors, a fall
    pulse starting on day fa_day (Oct 21), a peak-flow block on pk_day
    (Feb 15), a spring pulse on sp_day (Apr 20) followed by a recession at
    0.8 ROC_max down to the dry-season floor. Used by the 'prescriptive' scenario."""
    q = np.zeros(365)
    for k, key in (("DS", "Q_DS"), ("WB", "Q_WB")):
        q[window_mask(k)] = T[key]
    q[window_mask("FA")] = np.maximum(q[window_mask("FA")], min(T["Q_WB"], T["Q_DS"]))
    q[q == 0] = min(T["Q_WB"], T["Q_DS"])  # gaps between windows
    q[182:sp_day] = T["Q_WB"]             # hold the wet baseflow until the spring pulse
    q[fa_day:fa_day + T["D_FA"]] = np.maximum(q[fa_day:fa_day + T["D_FA"]], T["Q_FA"])
    if T["D_PK"] > 0:
        q[pk_day:pk_day + T["D_PK"]] = np.maximum(q[pk_day:pk_day + T["D_PK"]], T["Q_PK"])
    if T["D_SP"] > 0:
        end = sp_day + T["D_SP"]
        q[sp_day:end] = np.maximum(q[sp_day:end], T["Q_SP"])
        level = T["Q_SP"]
        for d in range(end, 365):
            level *= 1.0 - 0.8 * T["ROC_max"]
            if level <= q[d]:
                break
            q[d] = level
    return q
