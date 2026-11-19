"""Compare the ported uhf_sqd functions with the source (3968e58) at the AST level.

Docstrings, comments and formatting are ignored; ElectronicProperties -> UHFElectronicProperties
is normalized. Prints IDENTICAL or a unified diff (ast.unparse) per function.
Usage: python -I ast_compare_workflow.py SOURCE_SBD_PACKAGE_DIR PORT_UHF_SQD_PACKAGE_DIR
"""

import ast
import difflib
import sys
from pathlib import Path

src_dir, port_dir = Path(sys.argv[1]), Path(sys.argv[2])
RENAME = {"ElectronicProperties": "UHFElectronicProperties"}


class Norm(ast.NodeTransformer):
    def visit_Name(self, node):
        node.id = RENAME.get(node.id, node.id)
        return node

    def _strip(self, node):
        self.generic_visit(node)
        b = node.body
        if (
            b
            and isinstance(b[0], ast.Expr)
            and isinstance(getattr(b[0], "value", None), ast.Constant)
            and isinstance(b[0].value.value, str)
        ):
            node.body = b[1:] or [ast.Pass()]
        return node

    visit_FunctionDef = visit_ClassDef = _strip


def defs(path):
    tree = Norm().visit(ast.parse(path.read_text()))
    return {
        n.name: ast.unparse(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }


PAIRS = {
    ("sqd", "sqd"): [
        "_spin_halves_as_ints",
        "_compute_excitation_counts",
        "_excitation_summary",
        "_reference_telemetry",
        "_pool_ref_meta",
        "_pool_ref_meta_array",
        "walker_sqd",
        "_split_spin_carryover",
        "_drop_padding_rows",
        "postselect_bitstrings",
        "_safe_run_logger",
        "_cisd_strings",
        "_merge_with_seed",
        "subsample_open_shell",
        "_subsample_one_spin",
        "_deduplicate_and_accumurate_probs",
    ],
    ("main", "main"): [
        "_build_ab_indices",
        "OptimizerState",
        "differential_evolution_trial",
        "mutation_and_crossover",
        "selection",
        "deploy",
    ],
    ("main", "main:flow"): ["riken_sqd_de"],
    ("lucj", "lucj"): [
        "_default_bb_indices",
        "_antisymmetric_noise",
        "initialize_ucj_parameters",
        "_initialize_ucj_parameters_uhf",
        "create_lucj_circuit",
    ],
    ("data_io", "data_io"): [
        "_flow_scoped_subdir",
        "_local_storage_dir",
        "save_ndarray",
        "extend_table_artifact",
    ],
    ("flow_params", "flow_params"): [
        "CircuitParameters",
        "DEParameters",
        "SamplerParameters",
        "FlowParameters",
    ],
    ("transpile_custom", "transpilation"): ["transpile_circuit", "find_optimal_layout"],
}
PORT_NAME = {
    "riken_sqd_de": "uhf_sqd_de",
    "transpile_circuit": "_transpile_circuit",
    "find_optimal_layout": "_find_optimal_layout",
}
counts = {"IDENTICAL": 0, "DIFFERS": 0}
for (smod, pmod), names in PAIRS.items():
    s, p = defs(src_dir / f"{smod}.py"), defs(port_dir / f"{pmod.split(':')[0]}.py")
    for name in names:
        pname = PORT_NAME.get(name, name)
        a = s[name].replace(f"def {name}(", f"def {pname}(", 1)
        b = p[pname]
        status = "IDENTICAL" if a == b else "DIFFERS"
        counts[status] += 1
        print(f"== {smod}.{name} -> {pmod.split(':')[0]}.{pname}: {status}")
        if status == "DIFFERS":
            sys.stdout.writelines(
                difflib.unified_diff(a.splitlines(True), b.splitlines(True), "source", "port", n=0)
            )
            print()
print(counts)
