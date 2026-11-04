"""Compiled kernel vs interpreted OHQ (Python bindings), orifice model, 30 days
with the opening switched every 12 h."""
import os, sys, time
import numpy as np
sys.path.append("/home/arash/Projects/OpenHydroQual")
from ohqrl.env import _import_ohq, OHQ_RESOURCES
from common import HERE, KERNELS, Kernel, out_index
VALVE = "Aggregate_storage_layer (1) - Catch basin (2)"
OUT = "Catch basin (2) - fixed_head (3)"
t0, t1, dt = 43831.0, 43861.0, 1 / 24
sched = lambda t: 0.002 if int((t - t0) / 0.5) % 2 == 0 else 0.008
ohq = _import_ohq(); s = ohq.System(); s.set_default_template_path(OHQ_RESOURCES)
wd = os.path.join(HERE, "work", "parity"); os.makedirs(wd, exist_ok=True)
s.set_working_folder(wd + "/"); s.set_silent(True)
assert s.create_from_script(os.path.join(HERE, "models", "rl.ohq"), OHQ_RESOURCES + "settings.json")
for kk, v in {"simulation_end_time": "43861", "maximum_number_of_matrix_inverstions": "2000000000", "maximum_time_allowed": "1e9"}.items():
    s.set_system_value(kk, v)
s.begin_stepping(True)
k = Kernel(KERNELS["rl"]); k.initialize(); ix = out_index(k)
qa, qb = [], []; t = t0; tic = time.time()
while t < t1 - 1e-9:
    u = sched(t); s.set_value(VALVE, "opening", u); k.set_parameters({"Opening": u})
    t = t + dt; s.advance_to(t); k.run_to(t)
    qa.append(s.get_value("Pond (1)", "Storage")); qb.append(k.outputs()[ix["pond"]])
qa, qb = np.array(qa), np.array(qb)
ia = {n: s.objective_integral(n) for n in ("Excess", "Age_Load")}
print("interpreted", ia); print("kernel     ", k.objective_integrals())
print("outflow: max |diff| %.4f, max %.3f, total vol interp %.2f kernel %.2f" % (np.abs(qa - qb).max(), qa.max(), qa.sum() * dt, qb.sum() * dt))
print("%.1f s" % (time.time() - tic)); np.save(os.path.join(wd,"q.npy"), np.c_[qa,qb])
