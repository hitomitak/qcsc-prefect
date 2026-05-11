"""Input digest helpers for native Qiskit Prefect submit tasks.

These helpers build stable digests for Qiskit primitive inputs. The digest is
intended to be passed as ``input_digest`` to submit/run tasks and used by the
submit cache helpers. By default, digests are scoped to the current Prefect Flow
when available. Set ``cache_scope="global"`` to allow different Flows to reuse
the same cached submit reference when Qiskit inputs and execution conditions are
the same.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
import os
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any


def build_qiskit_input_digest_payload(
    *,
    program_type: str,
    pubs: Iterable[Any],
    backend_name: str | None = None,
    runtime_block_name: str | None = None,
    shots: int | None = None,
    precision: float | None = None,
    options: Mapping[str, Any] | None = None,
    cache_scope: str = "flow",
    cache_namespace: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical payload for a Qiskit input digest.

    Include ``backend_name`` when submit caching should be invalidated by a
    backend change. By default, the digest is scoped to the current Prefect Flow
    when called inside a Flow. Set ``cache_scope="global"`` to allow cache reuse
    across different Flows with identical Qiskit inputs and execution settings.
    Include ``runtime_block_name`` only when separate Prefect blocks should
    intentionally produce separate digests.
    """

    normalized_program_type = str(program_type).strip().lower()
    payload: dict[str, Any] = {
        "program_type": normalized_program_type,
        "pubs": [
            _canonical_pub(pub, normalized_program_type)
            for pub in _pub_list(pubs, normalized_program_type)
        ],
    }
    _add_if_present(payload, "backend_name", _stable_name(backend_name))
    _add_if_present(payload, "runtime_block_name", runtime_block_name)
    _add_if_present(payload, "shots", shots)
    _add_if_present(payload, "precision", precision)
    if options is not None:
        payload["options"] = _canonical_value(options)
    scope = _cache_scope_payload(cache_scope=cache_scope, cache_namespace=cache_namespace)
    if scope is not None:
        payload["cache_scope"] = scope
    if extra is not None:
        payload["extra"] = _canonical_value(extra)
    return payload


def qiskit_input_digest_from_payload(payload: Mapping[str, Any]) -> str:
    """Return a readable SHA-256 digest for a canonical Qiskit input payload."""

    canonical_payload = _canonical_value(payload)
    serialized = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    program_type = str(canonical_payload.get("program_type", "runtime")).strip().lower()
    if program_type not in {"sampler", "estimator"}:
        program_type = "runtime"
    return f"qiskit-{program_type}-input-{digest}"


def build_qiskit_sampler_input_digest(
    pubs: Iterable[Any],
    *,
    backend_name: str | None = None,
    runtime_block_name: str | None = None,
    shots: int | None = None,
    options: Mapping[str, Any] | None = None,
    cache_scope: str = "flow",
    cache_namespace: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Build an ``input_digest`` for native Qiskit Sampler submit caching.

    Example:

        input_digest = build_qiskit_sampler_input_digest(
            pubs,
            backend_name="ibm_kawasaki",
            shots=1024,
            options=options,
        )

    The digest is summary-only input identity. It is not a result serializer and
    does not include Prefect task/run identity. By default it includes the
    current Prefect Flow name when available. Set ``cache_scope="global"`` to
    allow different Flows to share cached submit references.
    """

    payload = build_qiskit_input_digest_payload(
        program_type="sampler",
        pubs=pubs,
        backend_name=backend_name,
        runtime_block_name=runtime_block_name,
        shots=shots,
        options=options,
        cache_scope=cache_scope,
        cache_namespace=cache_namespace,
        extra=extra,
    )
    return qiskit_input_digest_from_payload(payload)


def build_qiskit_estimator_input_digest(
    pubs: Iterable[Any],
    *,
    backend_name: str | None = None,
    runtime_block_name: str | None = None,
    precision: float | None = None,
    options: Mapping[str, Any] | None = None,
    cache_scope: str = "flow",
    cache_namespace: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Build an ``input_digest`` for native Qiskit Estimator submit caching.

    The digest includes estimator pubs, observables, backend name, precision,
    options, and optional extra stable configuration. It intentionally excludes
    Prefect task/run identity. By default it includes the current Prefect Flow
    name when available. Set ``cache_scope="global"`` to allow different Flows
    to share cache entries.
    """

    payload = build_qiskit_input_digest_payload(
        program_type="estimator",
        pubs=pubs,
        backend_name=backend_name,
        runtime_block_name=runtime_block_name,
        precision=precision,
        options=options,
        cache_scope=cache_scope,
        cache_namespace=cache_namespace,
        extra=extra,
    )
    return qiskit_input_digest_from_payload(payload)


def _add_if_present(payload: dict[str, Any], key: str, value: Any | None) -> None:
    if value is not None:
        payload[key] = _canonical_value(value)


def _cache_scope_payload(*, cache_scope: str, cache_namespace: str | None) -> dict[str, Any] | None:
    normalized_scope = str(cache_scope).strip().lower()
    if normalized_scope in {"global", "cross-flow", "cross_flow"}:
        payload: dict[str, Any] = {"mode": "global"}
    elif normalized_scope == "flow":
        payload = {"mode": "flow"}
        flow_key = _current_prefect_flow_key()
        if flow_key is not None:
            payload["flow"] = flow_key
    else:
        payload = {"mode": "custom", "value": normalized_scope}

    if cache_namespace is not None:
        payload["namespace"] = str(cache_namespace)
    return payload


def _current_prefect_flow_key() -> str | None:
    try:
        from prefect.context import get_run_context
    except Exception:
        return None

    try:
        context = get_run_context()
    except Exception:
        return None

    flow = getattr(context, "flow", None)
    flow_name = getattr(flow, "name", None)
    if flow_name is not None:
        return f"name:{flow_name}"

    flow_run = getattr(context, "flow_run", None)
    flow_id = getattr(flow_run, "flow_id", None)
    if flow_id is not None:
        return f"id:{flow_id}"
    return None


def _pub_list(pubs: Any, program_type: str) -> list[Any]:
    if _is_single_pub(pubs, program_type):
        return [pubs]
    if isinstance(pubs, str | bytes | bytearray | Mapping):
        return [pubs]
    if isinstance(pubs, Iterable):
        return list(pubs)
    return [pubs]


def _is_single_pub(value: Any, program_type: str) -> bool:
    if _is_circuit_like(value):
        return True
    if _has_any_attr(value, ("circuit", "observables", "observable")):
        return True
    if isinstance(value, tuple):
        return _looks_like_pub_tuple(value, program_type)
    return False


def _looks_like_pub_tuple(value: tuple[Any, ...], program_type: str) -> bool:
    if not value:
        return False
    if program_type == "estimator":
        return len(value) >= 2 and _is_circuit_like(value[0])
    return _is_circuit_like(value[0])


def _canonical_pub(pub: Any, program_type: str) -> Any:
    if _is_circuit_like(pub):
        return {"circuit": _canonical_circuit(pub)}
    if isinstance(pub, Mapping):
        return _canonical_pub_mapping(pub)
    if isinstance(pub, tuple | list) and pub and _is_circuit_like(pub[0]):
        return _canonical_pub_sequence(pub, program_type)
    if _has_any_attr(pub, ("circuit", "observables", "observable")):
        return _canonical_pub_object(pub, program_type)
    return _canonical_value(pub)


def _canonical_pub_mapping(pub: Mapping[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    for key, value in sorted(pub.items(), key=lambda item: str(item[0])):
        key_text = str(key)
        canonical[key_text] = _canonical_pub_field(key_text, value)
    return canonical


def _canonical_pub_sequence(pub: tuple[Any, ...] | list[Any], program_type: str) -> dict[str, Any]:
    if program_type == "estimator":
        field_names = ("circuit", "observables", "parameter_values", "precision")
    else:
        field_names = ("circuit", "parameter_values", "shots")

    canonical: dict[str, Any] = {}
    for index, value in enumerate(pub):
        key = field_names[index] if index < len(field_names) else f"item_{index}"
        canonical[key] = _canonical_pub_field(key, value)
    return canonical


def _canonical_pub_object(pub: Any, program_type: str) -> dict[str, Any]:
    field_names = ["circuit"]
    if program_type == "estimator":
        field_names.extend(["observables", "observable", "parameter_values", "precision"])
    else:
        field_names.extend(["parameter_values", "shots"])

    canonical: dict[str, Any] = {}
    for field_name in field_names:
        if not hasattr(pub, field_name):
            continue
        value = getattr(pub, field_name)
        if callable(value):
            continue
        canonical[field_name] = _canonical_pub_field(field_name, value)
    return canonical


def _canonical_pub_field(field_name: str, value: Any) -> Any:
    if field_name == "circuit":
        return _canonical_circuit(value) if _is_circuit_like(value) else _canonical_value(value)
    if field_name in {"observable", "observables"}:
        return _canonical_observable(value)
    return _canonical_value(value)


def _canonical_circuit(circuit: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "type": _type_name(circuit),
        "num_qubits": _safe_attr(circuit, "num_qubits"),
        "num_clbits": _safe_attr(circuit, "num_clbits"),
    }

    try:
        from qiskit import qasm3

        qasm3_text = qasm3.dumps(circuit)
        base["format"] = "qasm3"
        base["qasm3_sha256"] = hashlib.sha256(qasm3_text.encode("utf-8")).hexdigest()
        base["qasm3_length"] = len(qasm3_text)
        return base
    except Exception:
        pass

    try:
        from qiskit import qpy

        buffer = io.BytesIO()
        qpy.dump([circuit], buffer)
        base["format"] = "qpy-sha256"
        base["qpy_sha256"] = hashlib.sha256(buffer.getvalue()).hexdigest()
        return base
    except Exception:
        pass

    base["format"] = "str"
    base["value"] = str(circuit)
    return base


def _canonical_observable(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_observable(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, tuple | list):
        return [_canonical_observable(item) for item in value]

    to_list = getattr(value, "to_list", None)
    if callable(to_list):
        try:
            return {"type": _type_name(value), "items": _canonical_value(to_list())}
        except Exception:
            pass

    to_label = getattr(value, "to_label", None)
    if callable(to_label):
        try:
            return {"type": _type_name(value), "label": str(to_label())}
        except Exception:
            pass

    return _canonical_value(value)


def _canonical_value(value: Any, _seen: set[int] | None = None) -> Any:
    if _seen is None:
        _seen = set()

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"type": "float", "value": str(value)}
    if isinstance(value, complex):
        return {"imag": _canonical_value(value.imag), "real": _canonical_value(value.real)}
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, bytes | bytearray):
        return {
            "type": "bytes",
            "sha256": hashlib.sha256(bytes(value)).hexdigest(),
        }
    if isinstance(value, Enum):
        return _canonical_value(value.value, _seen)
    if isinstance(value, Path | os.PathLike):
        return os.fspath(value)
    if _is_circuit_like(value):
        return _canonical_circuit(value)

    value_id = id(value)
    if _should_track(value):
        if value_id in _seen:
            return {"type": _type_name(value), "recursive": True}
        _seen.add(value_id)

    try:
        if isinstance(value, Mapping):
            return {
                str(key): _canonical_value(item, _seen)
                for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, tuple | list):
            return [_canonical_value(item, _seen) for item in value]
        if isinstance(value, set | frozenset):
            items = [_canonical_value(item, _seen) for item in value]
            return sorted(items, key=_canonical_sort_key)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return _canonical_value(dataclasses.asdict(value), _seen)

        array_value = _array_value(value)
        if array_value is not None:
            return _canonical_value(array_value, _seen)

        object_mapping = _object_mapping(value)
        if object_mapping is not None:
            return _canonical_value(object_mapping, _seen)

        return {"type": _type_name(value), "value": str(value)}
    finally:
        _seen.discard(value_id)


def _array_value(value: Any) -> Any | None:
    module_name = type(value).__module__
    if not module_name.startswith("numpy"):
        return None
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            return None
    return None


def _object_mapping(value: Any) -> Mapping[str, Any] | None:
    for method_name in ("model_dump", "to_dict", "dict"):
        method = getattr(value, method_name, None)
        if not callable(method):
            continue
        try:
            result = method()
        except Exception:
            continue
        if isinstance(result, Mapping):
            return {"type": _type_name(value), "value": result}

    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, Mapping):
        public_attrs = {
            str(key): item
            for key, item in attrs.items()
            if not str(key).startswith("_") and not callable(item)
        }
        if public_attrs:
            return {"type": _type_name(value), "value": public_attrs}
    return None


def _canonical_sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _is_circuit_like(value: Any) -> bool:
    return (
        type(value).__name__ == "QuantumCircuit"
        and hasattr(value, "num_qubits")
        and hasattr(value, "num_clbits")
    )


def _has_any_attr(value: Any, names: Iterable[str]) -> bool:
    return any(hasattr(value, name) for name in names)


def _safe_attr(value: Any, name: str) -> Any:
    try:
        attr = getattr(value, name, None)
        if callable(attr):
            attr = attr()
        return _canonical_value(attr)
    except Exception:
        return None


def _stable_name(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    for attr_name in ("name", "backend_name"):
        try:
            attr = getattr(value, attr_name, None)
            if callable(attr):
                attr = attr()
            if attr is not None:
                return str(attr)
        except Exception:
            continue
    return str(value)


def _should_track(value: Any) -> bool:
    return not isinstance(
        value,
        str | bytes | bytearray | int | float | bool | complex | Decimal | datetime | date,
    )


def _type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"
