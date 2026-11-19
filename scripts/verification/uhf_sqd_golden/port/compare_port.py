"""PR3: compare a port run (run_port.py layout) against the PR0 golden set.

Usage: python -I compare_port.py PR0_HARNESS GOLDEN_DIR RUN_DIR SYSTEM CONFIG
Exit 0 only if both parts pass:

1. Computed outputs, with PR0's own reduction (make_golden.reduce_run) and diff rules
   (compare_golden.diff, exact equality), in --fcidump-records mode: flow energy, reference
   scalars and array hashes, telemetry rows, solver job inputs/outputs, sqd_data content hashes,
   hashed arrays, per-walker arrays and the reference arrays.
2. Settings: the golden "flow_parameters" and "solver_block" are the source's own models, whose
   fields differ from the port's. Each source field is matched by the table below to the port
   setting it stands for, or must be at the value with which its feature is off (the feature is
   not in the port). "source" (the source revision) is not compared.
"""

import json
import sys
from pathlib import Path

import numpy as np

harness, gdir, run, system, cfg = (
    sys.argv[1],
    Path(sys.argv[2]),
    Path(sys.argv[3]),
    sys.argv[4],
    sys.argv[5],
)
sys.path.insert(0, harness)
from compare_golden import diff  # noqa: E402
from make_golden import reduce_run  # noqa: E402

golden = json.loads((gdir / f"{system}__{cfg}.json").read_text())
garr = np.load(gdir / f"{system}__{cfg}.npz")
gref = np.load(gdir / f"{system}__reference.npz")
new, arrays, ref = reduce_run(run)
out = []

# --- 1. computed outputs ---------------------------------------------------------------------
SETTINGS = ("source", "flow_parameters", "solver_block")


def drop_fcidump(jobs):
    return sorted(
        json.dumps({k: v for k, v in json.loads(j).items() if k != "fcidump.txt"}, sort_keys=True)
        for j in jobs
    )


g = {k: v for k, v in golden.items() if k not in SETTINGS}
n = {k: v for k, v in new.items() if k not in SETTINGS}
g["solver_jobs"], n["solver_jobs"] = drop_fcidump(g["solver_jobs"]), drop_fcidump(n["solver_jobs"])
diff("$", g, n, out)
for k in sorted(set(garr.files) | set(arrays)):
    if k not in garr.files or k not in arrays:
        out.append(f"array {k}: missing on {'golden' if k not in garr.files else 'run'} side")
        continue
    x, y = garr[k], arrays[k]
    if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
        out.append(f"array {k}: differs")
for k in gref.files:
    if k not in ref or gref[k].tobytes() != np.ascontiguousarray(ref[k]).tobytes():
        out.append(f"reference array {k}: differs")
computed = list(out)

# --- 2. settings -----------------------------------------------------------------------------
src = golden["flow_parameters"]
port = json.loads((run / "flow_parameters.json").read_text())
summary = json.loads((run / "summary.json").read_text())
SAME = ["sqd_dim", "n_recovery_steps", "n_batches", "random_seed"]
SAME_CIRC = [
    "n_lucj_layers",
    "use_reset_mitigation",
    "optimization_level",
    "sabre_max_iterations",
    "sabre_swap_trials",
    "sabre_layout_trials",
    "ab_stride",
    "ucj_optimize",
]
OFF = {  # source fields of features not in the port: the value with which they are off
    "active_space_manifest": None,
    "seed_cisd": 0,
    "seed_budget_frac": 1.0,
    "hci_boost": 0,
}
OFF_CIRC = {"use_error_aware_layout": False}
OFF_SAMPLER = {
    "n_shot_batches": 1,
    "dynamical_decoupling": False,
    "measure_twirling": False,
    "gate_twirling": False,
    "saved_samples": [],
}
# Fields used only when another (off) field turns their feature on.
DEPENDENT = {"hci_boost_max": "hci_boost", "hci_boost_ncore": "hci_boost"}
DEPENDENT_SAMPLER = {"dd_sequence": "dynamical_decoupling"}
checks = []


def check(name, ok, detail):
    checks.append((name, ok, detail))


check("fcidump", Path(port["fcidump"]).name == src["fcidump"], (src["fcidump"], port["fcidump"]))
for k in SAME:
    check(k, src[k] == port[k], (src[k], port[k]))
for k in SAME_CIRC:
    check(
        f"circ_params.{k}",
        src["circ_params"][k] == port["circ_params"][k],
        (src["circ_params"][k], port["circ_params"][k]),
    )
check(
    "sampler.shots",
    src["sampler"]["shots"] == port["sampler"]["shots"],
    (src["sampler"]["shots"], port["sampler"]["shots"]),
)
check("de_params", src["de_params"] == port["de_params"], (src["de_params"], port["de_params"]))
for k, v in OFF.items():
    check(f"{k} (feature not ported, must be off)", src[k] == v, src[k])
for k, v in OFF_CIRC.items():
    check(
        f"circ_params.{k} (not ported, must be off)",
        src["circ_params"][k] == v,
        src["circ_params"][k],
    )
for k, v in OFF_SAMPLER.items():
    check(f"sampler.{k} (not ported, must be off)", src["sampler"][k] == v, src["sampler"][k])
for k, switch in DEPENDENT.items():
    check(f"{k} (used only when {switch} is on)", src[switch] == OFF[switch], (src[k], src[switch]))
for k, switch in DEPENDENT_SAMPLER.items():
    check(
        f"sampler.{k} (used only when {switch} is on)",
        src["sampler"][switch] == OFF_SAMPLER[switch],
        (src["sampler"][k], src["sampler"][switch]),
    )
check(
    "quantum_source: 'random' <-> a QuantumSourceBlock of kind random",
    src["quantum_source"] == "random"
    and summary["block_names"]["quantum_sources"].get(port["quantum_source"]) == "random",
    (src["quantum_source"], port["quantum_source"], summary["block_names"]["quantum_sources"]),
)
check(
    "reference: source environment defaults (no AF groups, cache read, UCCSD 50 x 10)",
    port["reference"]["af_groups"] is None
    and port["reference"]["cache_read"] is True
    and port["reference"]["ccsd_max_cycle"] == 50
    and port["reference"]["ccsd_restarts"] == 10,
    port["reference"],
)
check(
    "oo_* (orbital optimization runs only with do_rdm != 0; source do_rdm = 0)",
    golden["solver_block"]["do_rdm"] == 0,
    golden["solver_block"]["do_rdm"],
)
sb_src, sb_port = golden["solver_block"], summary["solver_block"]
for k in ("block", "iteration", "tolerance", "carryover_ratio", "carryover_type", "solver_mode"):
    check(f"solver_block.{k}", sb_src[k] == sb_port[k], (sb_src[k], sb_port[k]))
check(
    "solver_block.method 'uhf' <-> integrals 'unrestricted'",
    sb_src["method"] == "uhf" and sb_port["integrals"] == "unrestricted",
    (sb_src["method"], sb_port["integrals"]),
)
check(
    "solver_block.do_rdm 0 <-> the port always passes --rdm 0",
    sb_src["do_rdm"] == 0,
    sb_src["do_rdm"],
)
unmapped = (
    set(src)
    - set(SAME)
    - set(OFF)
    - set(DEPENDENT)
    - {"fcidump", "circ_params", "sampler", "de_params", "quantum_source", "solver_block_ref"}
    - {k for k in src if k.startswith("oo_")}
)
check("every source flow parameter is mapped", not unmapped, sorted(unmapped))
unmapped_c = (
    set(src["circ_params"])
    - set(SAME_CIRC)
    - set(OFF_CIRC)
    - {"two_qubit_error_threshold", "readout_error_threshold", "layout_connectivity"}
)
check(
    "every source circuit parameter is mapped (thresholds/connectivity: error-aware layout only)",
    not unmapped_c,
    sorted(unmapped_c),
)
check(
    "every source sampler parameter is mapped",
    not (set(src["sampler"]) - {"shots"} - set(OFF_SAMPLER) - set(DEPENDENT_SAMPLER)),
    sorted(set(src["sampler"]) - {"shots"} - set(OFF_SAMPLER) - set(DEPENDENT_SAMPLER)),
)

bad = [c for c in checks if not c[1]]
ok = not computed and not bad
outputs = "IDENTICAL" if not computed else f"{len(computed)} DIFFERENCES"
print(
    f"{system}/{cfg}: computed outputs {outputs}; "
    f"settings {len(checks) - len(bad)}/{len(checks)} mapped"
)
for line in computed[:50]:
    print("   ", line)
for name, _, detail in bad:
    print("    settings:", name, detail)
sys.exit(0 if ok else 1)
