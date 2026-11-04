"""Probe run: step a model hourly over the whole record, record flows and timing.
python stfrancis/probe.py rule|passive|rl [S I0 tau_k] """
import os
import sys
import time
import numpy as np

sys.path.append("/home/arash/Projects/OpenHydroQual")
from ohqrl.env import _import_ohq, OHQ_RESOURCES

HERE = os.path.dirname(os.path.abspath(__file__))
VALVE = "Aggregate_storage_layer (1) - Catch basin (2)"
OUTLET = "Catch basin (2) - fixed_head (3)"
WEIR = "Pond (1) - Catch basin (2)"

kind = sys.argv[1]
ohq = _import_ohq()
s = ohq.System()
s.set_default_template_path(OHQ_RESOURCES)
wd = os.path.join(HERE, "work", f"probe_{kind}")
os.makedirs(wd, exist_ok=True)
s.set_working_folder(wd + "/")
s.set_silent(True)
assert s.create_from_script(os.path.join(HERE, "models", f"{kind}.ohq"), OHQ_RESOURCES + "settings.json")
s.set_system_value("maximum_number_of_matrix_inverstions", "2000000000")
s.set_system_value("maximum_time_allowed", "1e9")
t0, t1 = s.start_time(), s.end_time()
s.begin_stepping(True)
if kind == "rule":
    S, I0, tk = map(float, sys.argv[2:5])
    s.set_value(VALVE, "flow_scale", S)
    s.set_value(VALVE, "threshold", I0)
    s.set_value(VALVE, "kernel_delay_parameter", tk)
dt = 1 / 24
rec = []
tic = time.time()
t = t0
q = [(OUTLET, "flow"), (VALVE, "flow"), (WEIR, "flow"), ("Pond (1)", "Storage"),
     ("Aggregate_storage_layer (1)", "Storage")]
if kind == "rule":
    q.append((VALVE, "kernel_smoothed_forcast"))
while t < t1 - 1e-9:
    t = min(t + dt, t1)
    if not s.advance_to(t):
        print("solver failed at", t)
        break
    rec.append([t] + list(s.get_values(q)))
toc = time.time() - tic
names = [o["name"] if isinstance(o, dict) else o for o in s.objective_names()]
tot = {n: s.objective_integral(n) for n in names}
s.end_stepping(False)
rec = np.array(rec)
np.save(os.path.join(wd, "rec.npy"), rec)
print(f"{kind}: {toc:.1f} s for {len(rec)} hourly steps; objectives {tot}")
Q = rec[:, 1]
print("outlet flow quantiles (m3/d): 50% %.2f 90% %.2f 99% %.2f 99.9%% %.2f max %.1f" %
      tuple(np.quantile(Q, [0.5, 0.9, 0.99, 0.999]).tolist() + [Q.max()]))
for lim in [10, 20, 30, 50, 75, 100]:
    print(f"  Q>{lim}: {np.mean(Q > lim) * 100:.2f}% of hours, excess {np.sum(np.maximum(Q - lim, 0)) * dt:.1f} m3")
