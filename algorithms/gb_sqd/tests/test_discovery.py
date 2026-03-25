from __future__ import annotations

from pathlib import Path

from gb_sqd.discovery import discover_target_directories


def _write_case(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "count_dict.txt").write_text("count")
    (path / "fci_dump.txt").write_text("fcidump")


def test_discover_target_directories_returns_leaf_matches_only(tmp_path: Path):
    parent_case = tmp_path / "ligand" / "case_a"
    child_case = parent_case / "atom_1"
    _write_case(parent_case)
    _write_case(child_case)
    _write_case(tmp_path / "ligand" / "case_b" / "atom_2")

    discovered = discover_target_directories(tmp_path / "ligand")

    assert [path.relative_to(tmp_path / "ligand").as_posix() for path in discovered] == [
        "case_a/atom_1",
        "case_b/atom_2",
    ]


def test_discover_target_directories_can_return_non_leaf_matches(tmp_path: Path):
    parent_case = tmp_path / "ligand" / "case_a"
    child_case = parent_case / "atom_1"
    _write_case(parent_case)
    _write_case(child_case)

    discovered = discover_target_directories(tmp_path / "ligand", leaf_only=False)

    assert [path.relative_to(tmp_path / "ligand").as_posix() for path in discovered] == [
        "case_a",
        "case_a/atom_1",
    ]
