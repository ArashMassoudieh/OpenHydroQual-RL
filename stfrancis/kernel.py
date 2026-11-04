"""
ctypes wrapper of a compiled OpenHydroQual kernel (ohq_generate --project
shared; ABI v3, see OpenHydroQual/codegen/tools/ohq_kernel.h).

Closed-loop use: set_parameter(...) + apply_parameters() before each control
interval, run_to(t), then read outputs() and the objective integrals. The
objectives are integrated here from the kernel's raw per-step series the way
OHQ's Objective_Function accumulates them (value * (t - t_prev)), and the
series are cleared after every read so memory stays flat over long runs.
"""
import ctypes as C
import numpy as np

ABI = 3


class Kernel:
    def __init__(self, path):
        L = self.L = C.CDLL(path)
        vp, d, i, cp = C.c_void_p, C.c_double, C.c_int, C.c_char_p
        dp, ip = C.POINTER(C.c_double), C.POINTER(C.c_int)
        sig = {
            "abi_version": (i, []), "create": (vp, []), "destroy": (None, [vp]),
            "initialize": (None, [vp]), "initialize_at": (None, [vp, d, d]),
            "run_to": (i, [vp, d]), "time": (d, [vp]),
            "simulation_start": (d, []), "simulation_end": (d, []),
            "n_parameters": (i, []), "parameter_name": (cp, [i]),
            "set_parameter": (None, [vp, i, d]), "apply_parameters": (None, [vp]),
            "n_objectives": (i, []), "objective_name": (cp, [i]),
            "objective_count": (i, [vp, i]), "objective_at": (i, [vp, i, i, dp, dp]),
            "clear_objectives": (None, [vp]),
            "n_outputs": (i, []), "output_name": (cp, [i]), "outputs": (None, [vp, dp]),
            "n_states": (i, []), "n_mass": (i, []),
            "export_state": (None, [vp, dp, dp, ip, dp]),
            "import_state": (None, [vp, d, dp, dp, ip, dp]),
            "solution_failed": (i, [vp]), "reset_status": (None, [vp]),
        }
        for name, (res, args) in sig.items():
            f = getattr(L, "ohq_kernel_" + name)
            f.restype, f.argtypes = res, args
            setattr(self, "_" + name, f)
        if self._abi_version() != ABI:
            raise RuntimeError(f"{path}: kernel ABI {self._abi_version()}, expected {ABI}")
        self.params = [self._parameter_name(k).decode() for k in range(self._n_parameters())]
        self.objectives = [self._objective_name(k).decode() for k in range(self._n_objectives())]
        self.output_names = [self._output_name(k).decode() for k in range(self._n_outputs())]
        self.ns, self.nm = self._n_states(), self._n_mass()
        self.t_start, self.t_end = self._simulation_start(), self._simulation_end()
        self.h = self._create()
        self._out = (C.c_double * len(self.output_names))()

    def __del__(self):
        if getattr(self, "h", None):
            self._destroy(self.h)
            self.h = None

    # ------------------------------------------------------------- run
    def initialize(self, t0=None, dt0=0.01):
        if t0 is None:
            self._initialize(self.h)
            t0 = self.t_start
        else:
            self._initialize_at(self.h, t0, dt0)
        self._clear_objectives(self.h)
        self.t_prev = np.full(len(self.objectives), float(t0))
        self.integral = np.zeros(len(self.objectives))

    def set_parameters(self, values):
        """values: {name: value}; applied at once (re-derives dependent constants)."""
        for k, v in values.items():
            self._set_parameter(self.h, self.params.index(k), float(v))
        self._apply_parameters(self.h)

    def run_to(self, t):
        ok = bool(self._run_to(self.h, float(t))) and not self._solution_failed(self.h)
        self._accumulate()
        return ok

    def time(self):
        return self._time(self.h)

    def _accumulate(self):
        t, v = C.c_double(), C.c_double()
        for j in range(len(self.objectives)):
            for k in range(self._objective_count(self.h, j)):
                self._objective_at(self.h, j, k, C.byref(t), C.byref(v))
                if t.value > self.t_prev[j]:
                    self.integral[j] += v.value * (t.value - self.t_prev[j])
                    self.t_prev[j] = t.value
        self._clear_objectives(self.h)

    def objective_integrals(self):
        return dict(zip(self.objectives, self.integral.copy()))

    def outputs(self):
        self._outputs(self.h, self._out)
        return np.frombuffer(self._out, dtype=np.float64).copy()

    # ----------------------------------------------------------- state
    def export_state(self):
        st = np.zeros(self.ns)
        ms = np.zeros(max(self.nm, 1))
        lim = np.zeros(self.ns, dtype=np.int32)
        lf = np.ones(self.ns)
        self._export_state(self.h, st.ctypes.data_as(C.POINTER(C.c_double)),
                           ms.ctypes.data_as(C.POINTER(C.c_double)),
                           lim.ctypes.data_as(C.POINTER(C.c_int)),
                           lf.ctypes.data_as(C.POINTER(C.c_double)))
        return {"storage": st, "mass": ms, "limited": lim, "limit_factor": lf}

    def import_state(self, t, s):
        self._import_state(self.h, float(t), s["storage"].ctypes.data_as(C.POINTER(C.c_double)),
                           s["mass"].ctypes.data_as(C.POINTER(C.c_double)),
                           s["limited"].ctypes.data_as(C.POINTER(C.c_int)),
                           s["limit_factor"].ctypes.data_as(C.POINTER(C.c_double)))
        self._clear_objectives(self.h)
        self.t_prev[:] = t
