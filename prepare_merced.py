"""
Download the daily hydrology of the Merced River / Lake McClure (California)
and save it as data/merced_daily.npz for the environmental-flow problem.

Sources (all public, no key needed):
    CDEC MRC sensor 8   full natural (unimpaired) flow, Merced R. nr Merced Falls [cfs]
    CDEC EXC sensor 15  Lake McClure (New Exchequer) storage [AF]
    CDEC EXC sensor 23  Lake McClure outflow [cfs]
    USGS 11272500       Merced R. near Snelling, the river below the
                        Crocker-Huffman irrigation diversions [cfs]

Saved arrays (one value per day, water years 1989-2025, metric units):
    date        numpy datetime64[D]
    fnf         unimpaired inflow to the reservoir [m3/s]
    storage     observed storage [hm3]                   (NaN where missing)
    outflow     observed reservoir outflow [m3/s]        (NaN where missing)
    river       observed river flow below the diversions [m3/s] (NaN where missing)
The raw downloads are cached in data/raw_merced/ (not committed).
"""
import os
import urllib.request

import numpy as np

CFS = 0.0283168466      # m3/s per cfs
AF = 1233.48184e-6      # hm3 per acre-foot
START, END = "1988-10-01", "2025-09-30"
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw_merced")


def fetch(url, name):
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        os.makedirs(RAW, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def cdec(station, sensor):
    url = ("https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?"
           f"Stations={station}&SensorNums={sensor}&dur_code=D&Start={START}&End={END}")
    out = {}
    for line in open(fetch(url, f"cdec_{station}_{sensor}.csv")).readlines()[1:]:
        f = line.split(",")
        try:
            v = float(f[6])
        except ValueError:
            continue
        d = f[4][:8]
        out[np.datetime64(f"{d[:4]}-{d[4:6]}-{d[6:8]}")] = v
    return out


def usgs(site):
    url = ("https://waterservices.usgs.gov/nwis/dv/?format=rdb"
           f"&sites={site}&parameterCd=00060&startDT={START}&endDT={END}")
    out = {}
    for line in open(fetch(url, f"usgs_{site}.rdb")):
        f = line.rstrip("\n").split("\t")
        if f[0] == "USGS":
            try:
                out[np.datetime64(f[2])] = float(f[3])
            except ValueError:
                pass
    return out


def on_grid(series, dates, scale):
    return np.array([series.get(d, np.nan) for d in dates]) * scale


def main():
    dates = np.arange(np.datetime64(START), np.datetime64(END) + 1)
    fnf = on_grid(cdec("MRC", 8), dates, CFS)
    # CDEC full natural flow is a computed series and occasionally negative or
    # missing on single days; clip and fill by linear interpolation.
    fnf = np.where(fnf < 0, np.nan, fnf)
    good = ~np.isnan(fnf)
    print(f"FNF: {good.sum()} of {len(fnf)} days present, {np.sum(~good)} filled")
    fnf = np.interp(np.arange(len(fnf)), np.flatnonzero(good), fnf[good])
    storage = on_grid(cdec("EXC", 15), dates, AF)
    outflow = on_grid(cdec("EXC", 23), dates, CFS)
    river = on_grid(usgs("11272500"), dates, CFS)
    for name, x in (("storage", storage), ("outflow", outflow), ("river", river)):
        print(f"{name}: {np.sum(~np.isnan(x))} days present")
    out = os.path.join(HERE, "data", "merced_daily.npz")
    np.savez_compressed(out, date=dates, fnf=fnf, storage=storage, outflow=outflow, river=river)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
