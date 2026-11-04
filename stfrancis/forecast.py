"""
Perfect-forecast indicator of the St. Francis valve rule, computed outside the
model so a policy can use it.

The valve's kernel_smoothed_forcast with a perfect forecast reduces to
_ekr(timeseries; 1/tau_k), OHQ's forward exponential kernel
(TimeSeries::Exponential_Kernel), a trapezoidal rule for

  Ihat(t) = int_0^{2 tau_k} lam exp(-lam s) c(t + s) ds,   lam = 1/tau_k:

  Ihat(t) = sum_{i0 <= i < i1} c_i * lam * (exp(-lam (t_i - t)) + exp(-lam (t_{i+1} - t))) / 2 * (t_{i+1} - t_i)

with i0 the sample at or before t and i1 the sample at t + 2/lam (truncation),
exactly as OHQ evaluates it. (OHQ before the 2026-11-02 fix omitted the 1/2;
parameters tuned then convert as S -> S/2, I0 -> I0/2.)
"""
import numpy as np

RAIN = ("/home/arash/Dropbox/St. Francis RTC paper/Residence Time Runs 2026/data/"
        "Precipitation/Precipitation 2020_2023.txt")


def load_series(path=RAIN):
    """The valve's 'timeseries' as OHQ reads it: (t, c) from the first two columns."""
    d = np.loadtxt(path, delimiter=",", ndmin=2)
    return d[:, 0], d[:, 1]


def load_rain(path=RAIN):
    """Precipitation bins (start, end, depth)."""
    d = np.loadtxt(path, delimiter=",", ndmin=2)
    return d[:, 0], d[:, 1], d[:, 2]


class Indicator:
    def __init__(self, t, c):
        self.t, self.c = np.asarray(t, float), np.asarray(c, float)

    def ekr(self, t, lam):
        T, Cv = self.t, self.c
        i0 = max(int(np.searchsorted(T, t, side="right")) - 1, 0)
        i1 = min(max(int(np.searchsorted(T, t + 2.0 / lam, side="right")) - 1, 0), len(T) - 1)
        if i1 <= i0:
            return 0.0
        ti, tj, ci = T[i0:i1], T[i0 + 1:i1 + 1], Cv[i0:i1]
        return float(np.sum(0.5 * ci * lam * (np.exp(-lam * (ti - t)) + np.exp(-lam * (tj - t))) * (tj - ti)))


def series_from_kernel_header(header, member="timeseries_"):
    """The valve's forecast series exactly as the compiled kernel holds it
    (ohq_generate embeds it resampled to the model's dt0 grid)."""
    import re
    tag = f"Catch_basin__2___{member}.push(D[2*k], D[2*k+1]);"
    lines = open(header).read().split("\n")
    end = next(i for i, ln in enumerate(lines) if tag in ln)
    start = max(i for i in range(end) if "static const double D[] = {" in lines[i])
    body = " ".join(lines[start:end]).split("{", 2)[-1].split("}")[0]
    v = np.array([float(x) for x in re.split(r"[,\s]+", body) if x])
    return v[0::2], v[1::2]
