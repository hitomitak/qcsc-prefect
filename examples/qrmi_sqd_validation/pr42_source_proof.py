"""Verify recovery aliases and preserve all code outside the sampling boundary."""

from __future__ import annotations

import ast
import hashlib
from difflib import unified_diff
from pathlib import Path

from .pinned_recovery import verify_sources
from .pr42 import FIXTURE_DIR, source_metadata


def verify_workflow_copy() -> dict:
    verify_sources()
    original = (FIXTURE_DIR / "sqd_fulqrum_main.py.txt").read_text(encoding="utf-8")
    directory = Path(__file__).parent
    adapted = (directory / "pr42_workflow.py").read_text(encoding="utf-8")
    provenance = source_metadata()
    if hashlib.sha256(original.encode()).hexdigest() != provenance["sha256"]:
        raise ValueError("Original PR #42 workflow hash mismatch")
    before, after = ast.parse(original), ast.parse(adapted)

    def declarations(tree):
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }

    old, new = declarations(before), declarations(after)
    if old.keys() != new.keys():
        raise ValueError("The adapted copy added or removed workflow definitions")
    original_selection = ast.get_source_segment(original, old["sample_bitstrings"]).split(
        "    # Post-process bitstrings\n", 1
    )[1]
    adapted_selection = ast.get_source_segment(adapted, new["sample_bitstrings"]).split(
        "    # Post-process bitstrings\n", 1
    )[1]
    if original_selection != adapted_selection:
        raise ValueError("Original meas/test postprocessing changed")
    unchanged = []
    for name, node in old.items():
        if name == "sample_bitstrings":
            continue
        if ast.get_source_segment(original, node) != ast.get_source_segment(adapted, new[name]):
            raise ValueError(f"Calculation definition changed: {name}")
        if ast.dump(node, include_attributes=False) != ast.dump(
            new[name], include_attributes=False
        ):
            raise ValueError(f"Calculation decorators or annotations changed: {name}")
        unchanged.append(name)

    # Only these explicit bindings may be added outside the sampling function.
    bindings = {
        ast.dump(node, include_attributes=False)
        for node in ast.parse(
            "SolveFn = load_recovery().SolveFn\n"
            "run_configuration_recovery = load_recovery().run_configuration_recovery\n"
        ).body
    }
    module_nodes = [ast.dump(node, include_attributes=False) for node in after.body]
    if any(module_nodes.count(binding) != 1 for binding in bindings):
        raise ValueError("Verified original recovery bindings changed")

    # Constants and the module docstring stay unchanged; imports are visible in the patch.
    def other_nodes(tree):
        return [
            ast.dump(node, include_attributes=False)
            for node in tree.body
            if not isinstance(
                node,
                (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            )
            and ast.dump(node, include_attributes=False) not in bindings
        ]

    if other_nodes(before) != other_nodes(after):
        raise ValueError("Workflow constants changed outside the sampling boundary")
    patch = "".join(
        unified_diff(
            original.splitlines(keepends=True),
            adapted.splitlines(keepends=True),
            fromfile="a/algorithms/sqd/sqd_fulqrum/main.py",
            tofile="b/algorithms/sqd/sqd_fulqrum/main.py",
        )
    )
    # Match diff --suppress-blank-empty; keep patch context valid without trailing spaces.
    patch = "".join("\n" if line == " \n" else line for line in patch.splitlines(keepends=True))
    if patch != (directory / "pr42_qrmi.patch").read_text(encoding="utf-8"):
        raise ValueError("The displayed patch does not match the executable workflow copy")
    return {
        "source_commit": provenance["commit"],
        "original_sha256": provenance["sha256"],
        "adapted_sha256": hashlib.sha256(adapted.encode()).hexdigest(),
        "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
        "unchanged_definitions": unchanged,
        "sampling_postprocessing_unchanged": True,
        "changed_definition": "sample_bitstrings",
    }
