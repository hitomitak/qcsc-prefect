#!/usr/bin/env python3
"""Bump qcsc-prefect package versions and internal dependency pins."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_GLOB = "packages/*/pyproject.toml"
INTERNAL_PIN_NAME = r"qcsc-prefect(?:-[A-Za-z0-9]+)*"
VERSION_TAIL = r"(?![A-Za-z0-9.!+_-])"


def package_pyprojects() -> list[Path]:
    paths = sorted(ROOT.glob(PACKAGE_GLOB))
    if not paths:
        raise SystemExit(f"No package pyproject.toml files found with {PACKAGE_GLOB}.")
    return paths


def replace_versions(path: Path, old_version: str, new_version: str) -> tuple[str, bool]:
    text = path.read_text(encoding="utf-8")
    old = re.escape(old_version)

    version_pattern = re.compile(rf'(?m)^(version\s*=\s*)"{old}"$')
    if not version_pattern.search(text):
        rel = path.relative_to(ROOT)
        raise ValueError(f"{rel}: expected version = \"{old_version}\"")

    text, version_count = version_pattern.subn(rf'\1"{new_version}"', text)

    pin_pattern = re.compile(rf"({INTERNAL_PIN_NAME}==){old}{VERSION_TAIL}")
    text, pin_count = pin_pattern.subn(rf"\g<1>{new_version}", text)

    return text, version_count > 0 or pin_count > 0


def validate_no_stale_versions(paths: list[Path], old_version: str, new_version: str) -> list[str]:
    old = re.escape(old_version)
    new = re.escape(new_version)
    stale_version_pattern = re.compile(rf'(?m)^version\s*=\s*"{old}"$')
    stale_pin_pattern = re.compile(rf"{INTERNAL_PIN_NAME}=={old}{VERSION_TAIL}")
    new_version_pattern = re.compile(rf'(?m)^version\s*=\s*"{new}"$')
    errors: list[str] = []

    for path in paths:
        rel = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        if stale_version_pattern.search(text):
            errors.append(f"{rel}: stale project version {old_version} remains")
        if stale_pin_pattern.search(text):
            errors.append(f"{rel}: stale internal dependency pin {old_version} remains")
        if not new_version_pattern.search(text):
            errors.append(f"{rel}: project version was not updated to {new_version}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update package versions and internal qcsc-prefect pins."
    )
    parser.add_argument("old_version")
    parser.add_argument("new_version")
    args = parser.parse_args()

    paths = package_pyprojects()
    updates: dict[Path, str] = {}
    errors: list[str] = []

    for path in paths:
        try:
            updated_text, changed = replace_versions(path, args.old_version, args.new_version)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if changed:
            updates[path] = updated_text

    if errors:
        print("Version bump failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    for path, text in updates.items():
        path.write_text(text, encoding="utf-8")

    stale_errors = validate_no_stale_versions(paths, args.old_version, args.new_version)
    if stale_errors:
        print("Version bump left stale versions:", file=sys.stderr)
        for error in stale_errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Updated files:")
    for path in sorted(updates):
        print(f"- {path.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
