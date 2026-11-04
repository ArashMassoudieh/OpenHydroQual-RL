"""
Build the St. Francis bioretention models (half footprint, 30.2 mm, perfect
forecast) used for the RL vs rule-based control comparison.

Sources (read only) are the residence-time campaign models of the forecast-
uncertainty study. Three variants are written, all with the same two additive
objective functions on the catch-basin outflow:

  Excess    integral of max(Q - Q_lim, 0) dt   [m^3]  flow above a preset limit
  Age_Load  integral of the young-water load (Laplace age tracker, decay 1/d)

  rule.ohq     forecast-informed valve (S, I_0, tau_k set by the caller via set_value)
  rl.ohq       the valve replaced by a controlled orifice whose opening is set
               each control interval; with head_loss_coefficient = 1,
               max_area = valve area and a 1 mm regularization head its flow is
               identical to the rule valve's at opening = sqrt(Ihat/S)
  passive.ohq  the valve replaced by a free-draining 0.2 m underdrain

Usage: python stfrancis/build_models.py --qlim 30
"""
import argparse
import json
import os
import re

STUDY = "/home/arash/Dropbox/St. Francis RTC paper/Exceedance 1m Weir Runs 2026/rt_campaign"
SRC_RULE = os.path.join(STUDY, "models/R1_f0.50_half_30.2mm/sigma_0_ts_2/model.ohq")
SRC_PASSIVE = os.path.join(STUDY, "step_check/passive_s0025_half/model.ohq")
OLD_RES = "/home/arash/Projects/OHQ-ut-6ac7912/resources/"
NEW_RES = "/home/arash/Projects/OpenHydroQual/resources/"

OUTLET = "Catch basin (2) - fixed_head (3)"
VALVE = "Aggregate_storage_layer (1) - Catch basin (2)"
VALVE_AREA = 3.1415 * 0.2 ** 2 / 4          # as in the valve type's 'area' expression
HERE = os.path.dirname(os.path.abspath(__file__))

SETTINGS = {"initial_time_step": "0.01", "max_timestep_increase_factor": "10",
            "n_threads": "1", "simulation_start_time": "43831", "simulation_end_time": "45291"}


def objectives(qlim):
    return [f"create objectivefunction;type=Objective_Function,exceedance_probability=0.05,"
            f"expression=_pos(flow-{qlim:g}),method=integrate,name=Excess,object={OUTLET},weight=1",
            f"create objectivefunction;type=Objective_Function,exceedance_probability=0.05,"
            f"expression=AgeTracker_laplace_1:advective_masstransfer,method=integrate,"
            f"name=Age_Load,object={OUTLET},weight=1"]


def clean(lines, qlim):
    out = []
    for ln in lines:
        ln = ln.rstrip("\n").replace(OLD_RES, NEW_RES)
        if (ln.startswith("create objectivefunction") or ln.startswith("create observation")
                or ln.startswith("create parameter") or ln.startswith("setasparameter")):
            continue
        m = re.match(r"setvalue; object=system, quantity=(\w+), value=", ln)
        if m and m.group(1) in SETTINGS:
            ln = f"setvalue; object=system, quantity={m.group(1)}, value={SETTINGS[m.group(1)]}"
        out.append(ln)
    return out + objectives(qlim)


def build(qlim, outdir):
    os.makedirs(outdir, exist_ok=True)
    rule = clean(open(SRC_RULE).readlines(), qlim)
    passive = clean(open(SRC_PASSIVE).readlines(), qlim)
    valve = [i for i, ln in enumerate(rule) if ln.startswith("create link") and f"name={VALVE}" in ln]
    assert len(valve) == 1, "valve link not found"
    rl = list(rule)
    rl[valve[0]] = (f"create link;from=Aggregate_storage_layer (1),to=Catch basin (2),type=controlled_orifice,"
                    f"AgeTracker_laplace_1:decay_parameter=0,head_loss_coefficient=1,max_area={VALVE_AREA:.10g},"
                    f"name={VALVE},opening=0,regularization_head=0.001")
    # Parameters: the kernel's runtime handles (set_parameter + apply_parameters)
    rule += ["create parameter;type=Parameter,high=4000,low=10,name=S,prior_distribution=log-normal,value=500",
             "create parameter;type=Parameter,high=0.03,low=1e-06,name=I0,prior_distribution=log-normal,value=0.01",
             "create parameter;type=Parameter,high=100,low=0.001,name=tau_k,prior_distribution=log-normal,value=2",
             f"setasparameter; object= {VALVE}, parametername= S, quantity= flow_scale",
             f"setasparameter; object= {VALVE}, parametername= I0, quantity= threshold",
             f"setasparameter; object= {VALVE}, parametername= tau_k, quantity= kernel_delay_parameter"]
    rl += ["create parameter;type=Parameter,high=1,low=0,name=Opening,prior_distribution=normal,value=0",
           f"setasparameter; object= {VALVE}, parametername= Opening, quantity= opening"]
    for name, lines in [("rule", rule), ("rl", rl), ("passive", passive)]:
        with open(os.path.join(outdir, f"{name}.ohq"), "w") as f:
            f.write("\n".join(lines) + "\n")
    meta = {"qlim": qlim, "valve_area": VALVE_AREA, "outlet": OUTLET, "valve": VALVE,
            "sources": {"rule": SRC_RULE, "passive": SRC_PASSIVE}}
    json.dump(meta, open(os.path.join(outdir, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qlim", type=float, required=True, help="flow limit for Excess [m^3/d]")
    ap.add_argument("--out", default=os.path.join(HERE, "models"))
    a = ap.parse_args()
    build(a.qlim, a.out)
