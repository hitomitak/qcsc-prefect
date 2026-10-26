"""Verify and load the unmodified recovery functions from the fixed PR snapshot."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from functools import lru_cache
from types import ModuleType

from .pr42 import FIXTURE_DIR, ROOT, source_metadata

NAMESPACE = "_qrmi_pr42_postprocessing"


def verify_sources() -> dict:
    manifest = json.loads((FIXTURE_DIR / "postprocessing.json").read_text())
    if manifest["commit"] != source_metadata()["commit"]:
        raise ValueError("Sampling and recovery source commits differ")
    for record in manifest["files"]:
        path = ROOT / record["local_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Pinned postprocessing source hash mismatch: {record['path']}")
    return manifest


@lru_cache(maxsize=1)
def load_recovery():
    manifest = verify_sources()
    package = ModuleType(NAMESPACE)
    package.__path__ = []
    sys.modules[NAMESPACE] = package
    for name in ["subsample", "recovery"]:
        original = f"algorithms/sqd/sqd_fulqrum/{name}.py"
        record = next(item for item in manifest["files"] if item["path"] == original)
        path = ROOT / record["local_path"]
        tree = ast.parse(path.read_text(), filename=str(path))
        # Isolate this snapshot from any separately installed sqd_fulqrum.
        # Only the module import is redirected; function bodies are unchanged.
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "sqd_fulqrum.subsample":
                node.module = NAMESPACE + ".subsample"
        module = ModuleType(NAMESPACE + "." + name)
        module.__file__ = str(path)
        module.__package__ = NAMESPACE
        sys.modules[module.__name__] = module
        try:
            exec(compile(tree, str(path), "exec"), module.__dict__)
        except Exception:
            for key in [NAMESPACE, NAMESPACE + ".subsample", NAMESPACE + ".recovery"]:
                sys.modules.pop(key, None)
            raise
    return sys.modules[NAMESPACE + ".recovery"]
