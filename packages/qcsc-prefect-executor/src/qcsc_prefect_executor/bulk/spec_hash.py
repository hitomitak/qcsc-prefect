from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import PurePath
from typing import Any, Mapping

BULK_SPEC_HASH_SCHEMA_VERSION = "qcsc-prefect-bulk-spec-v1"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalize(value: Any) -> Any:
    if value is None:
        return {"type": "none"}
    if isinstance(value, Enum):
        return _normalize(value.value)
    if isinstance(value, PurePath):
        return _normalize(value.as_posix())
    if isinstance(value, str):
        return {"type": "string", "sha256": _sha256_text(value)}
    if isinstance(value, bytes):
        return {"type": "bytes", "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, bool):
        return {"type": "bool", "sha256": _sha256_text("true" if value else "false")}
    if isinstance(value, int):
        return {"type": "int", "sha256": _sha256_text(str(value))}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Bulk spec values must not contain NaN or infinity.")
        return {"type": "float", "sha256": _sha256_text(repr(value))}
    if is_dataclass(value) and not isinstance(value, type):
        return _normalize(asdict(value))
    if isinstance(value, Mapping):
        items = [[_normalize(key), _normalize(item)] for key, item in value.items()]
        items.sort(key=lambda pair: _canonical_json(pair[0]))
        return {"type": "mapping", "items": items}
    if isinstance(value, (list, tuple)):
        return {"type": "sequence", "items": [_normalize(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        items = [_normalize(item) for item in value]
        items.sort(key=_canonical_json)
        return {"type": "set", "items": items}
    raise TypeError(
        f"Unsupported bulk spec value type: {type(value).__module__}.{type(value).__qualname__}"
    )


def canonical_bulk_spec_json(spec: Mapping[str, Any]) -> str:
    """Return the versioned canonical JSON used as the bulk spec hash input.

    All dynamic string and scalar values are represented by typed SHA-256
    fingerprints. The returned payload is therefore safe to use in tests and
    diagnostics without exposing plaintext command arguments or environment
    values.
    """

    envelope = {
        "schema": BULK_SPEC_HASH_SCHEMA_VERSION,
        "spec": _normalize(spec),
    }
    return _canonical_json(envelope)


def build_bulk_spec_hash(spec: Mapping[str, Any]) -> str:
    """Return a deterministic, version-prefixed SHA-256 hash for ``spec``."""

    canonical = canonical_bulk_spec_json(spec)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{BULK_SPEC_HASH_SCHEMA_VERSION}:sha256:{digest}"
