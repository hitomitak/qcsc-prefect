"""Compare ported functions with the source at the AST level.

Docstrings, comments and formatting are ignored; renamed helpers are normalized. Prints, for
each pair, IDENTICAL or a unified diff of ast.unparse output.
Usage: python -I ast_compare_reference.py SOURCE_CHEM SOURCE_ACTIVE_SPACE PORT_UHF
"""

import ast
import difflib
import sys

src_chem, src_as, port = (open(p).read() for p in sys.argv[1:4])
RENAME = {
    "validate_orbital_indices": "_validate_orbital_indices",
    "density_from_af_groups": "_density_from_af_groups",
    "_AF_OPTION_KEYS": "AF_OPTION_KEYS",
}


class Norm(ast.NodeTransformer):
    def visit_Name(self, node):
        node.id = RENAME.get(node.id, node.id)
        return node

    def _strip(self, node):
        self.generic_visit(node)
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
        return node

    visit_FunctionDef = _strip


def funcs(text):
    tree = Norm().visit(ast.parse(text))
    return {n.name: ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef)}


s = {**funcs(src_chem), **funcs(src_as)}
p = funcs(port)
PAIRS = [
    ("file_sha256", "file_sha256"),
    ("validate_orbital_indices", "_validate_orbital_indices"),
    ("density_from_af_groups", "_density_from_af_groups"),
    ("_check_af_groups", "_check_af_groups"),
    ("_converge_broken_symmetry_uhf", "_converge_broken_symmetry_uhf"),
    ("_fcidump_uhf", "_fcidump_uhf"),
    ("_uhf_cache_path", "_uhf_cache_path"),
    ("_uhf_cache_load", "_uhf_cache_load"),
    ("_uhf_cache_save", "_uhf_cache_save"),
    ("_build_property_uhf", "_build_property_uhf"),
    ("_get_logger", "_get_logger"),
]
for a, b in PAIRS:
    sa = s[a].replace(f"def {a}(", f"def {b}(", 1)
    if sa == p[b]:
        print(f"== {a} -> {b}: IDENTICAL")
    else:
        print(f"== {a} -> {b}: DIFFERS")
        sys.stdout.writelines(
            difflib.unified_diff(sa.splitlines(True), p[b].splitlines(True), "source", "port", n=1)
        )
print("port-only functions:", sorted(set(p) - {b for _, b in PAIRS}))
