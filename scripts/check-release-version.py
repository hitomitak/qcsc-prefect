#!/usr/bin/env python3
"""Validate release tag, package versions, and internal dependency pins."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_GLOB = "packages/*/pyproject.toml"
INTERNAL_PIN_PATTERN = re.compile(
    r"(?P<name>qcsc-prefect(?:-[A-Za-z0-9]+)*)==(?P<version>[A-Za-z0-9.!+_-]+)"
)


def package_pyprojects() -> list[Path]:
    paths = sorted(ROOT.glob(PACKAGE_GLOB))
    if not paths:
        raise SystemExit(f"No package pyproject.toml files found with {PACKAGE_GLOB}.")
    return paths


def version_from_tag(tag: str) -> str:
    if not tag:
        raise ValueError("tag must not be empty")
    return tag[1:] if tag.startswith("v") else tag


def main() -> int:
    parser = argparse.ArgumentParser(description="Check release package versions.")
    parser.add_argument("--tag", required=True, help="Release tag, for example v0.1.1")
    args = parser.parse_args()

    try:
        expected_version = version_from_tag(args.tag)
    except ValueError as exc:
        print(f"Invalid tag: {exc}", file=sys.stderr)
        return 1

    errors: list[str] = []
    for path in package_pyprojects():
        rel = path.relative_to(ROOT)
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        project = data.get("project", {})
        actual_version = project.get("version")
        if actual_version != expected_version:
            errors.append(
                f"{rel}: project.version is {actual_version!r}, expected {expected_version!r}"
            )

        text = path.read_text(encoding="utf-8")
        for match in INTERNAL_PIN_PATTERN.finditer(text):
            pin_name = match.group("name")
            pin_version = match.group("version")
            if pin_version != expected_version:
                errors.append(
                    f"{rel}: {pin_name} pin is {pin_version!r}, expected {expected_version!r}"
                )

    if errors:
        print(f"Release version check failed for tag {args.tag}:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"All package versions and internal pins match {expected_version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
