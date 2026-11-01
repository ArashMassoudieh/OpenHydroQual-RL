"""
Convert a NOAA NCEI Hourly Precipitation Data (HPD v2) station file into a
clean hourly rainfall series for the stormwater-pond environment.

    python -I prepare_rain.py data/noaa_hpd/USC00180700.csv data/beltsville_hourly.npz

Source: NOAA NCEI, Hourly Precipitation Data (HPD) Network, Version 2.
Station USC00180700, BELTSVILLE, MD (39.03 N, 76.93 W), between Baltimore
and Washington DC. Values are hourly totals in hundredths of an inch; -9999
is missing. Any quality flag marks the hour as suspect; such hours are kept
as gaps (NaN) rather than zeros, so the environment can avoid windows with
missing data.
"""
import csv
import sys
from datetime import datetime

import numpy as np

INCH = 0.0254  # m


def main(src, dst):
    hours, rain = [], []
    with open(src, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}
        for row in reader:
            day = datetime.strptime(row[col["DATE"]], "%Y-%m-%d")
            base = (day - datetime(1900, 1, 1)).days * 24
            for h in range(24):
                v = float(row[col[f"HR{h:02d}Val"]])
                qf = row[col[f"HR{h:02d}QF"]].strip()
                bad = v <= -9999 or qf != ""
                hours.append(base + h)
                rain.append(np.nan if bad else v * 0.01 * INCH)
    hours = np.array(hours, dtype=np.int64)
    rain = np.array(rain)
    order = np.argsort(hours)
    hours, rain = hours[order], rain[order]
    # Dense hourly grid; hours absent from the file are gaps too.
    grid = np.arange(hours[0], hours[-1] + 1)
    series = np.full(grid.size, np.nan)
    series[hours - hours[0]] = rain
    years = 1900 + (grid / 24 / 365.2425)
    np.savez_compressed(dst, hour_since_1900=grid, rain_m=series, station="USC00180700 BELTSVILLE, MD")
    print(f"{dst}: {grid.size} hours, {np.isnan(series).mean():.1%} missing overall")
    for y in range(int(years.min()), int(years.max()) + 1):
        m = (years >= y) & (years < y + 1)
        ok = ~np.isnan(series[m])
        if m.sum():
            print(f"  {y}: coverage {ok.mean():6.1%}  total {np.nansum(series[m]) * 1000:7.0f} mm")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
