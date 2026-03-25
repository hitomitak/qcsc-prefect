"""Directory discovery helpers for bulk GB-SQD runs."""

from __future__ import annotations

from pathlib import Path


def discover_target_directories(
    input_root_dir: str | Path,
    *,
    count_dict_filename: str = "count_dict.txt",
    fcidump_filename: str = "fci_dump.txt",
    leaf_only: bool = True,
) -> list[Path]:
    """Discover directories that contain both required GB-SQD input files."""

    root = Path(input_root_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input root directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {root}")

    matches = [
        path
        for path in root.rglob("*")
        if path.is_dir()
        and (path / count_dict_filename).is_file()
        and (path / fcidump_filename).is_file()
    ]
    matches.sort(key=lambda path: path.relative_to(root).as_posix())

    if not leaf_only:
        return matches

    leaf_matches: list[Path] = []
    for candidate in matches:
        if any(candidate != other and candidate in other.parents for other in matches):
            continue
        leaf_matches.append(candidate)
    return leaf_matches
