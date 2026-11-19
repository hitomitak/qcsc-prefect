"""Block creation: UHF SBD blocks with the source's Davidson defaults, and source Blocks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import qcsc_prefect_sbd
from qcsc_prefect.integrations.qiskit.sources import QuantumSourceBlock
from uhf_sqd import create_blocks

EXAMPLE = Path(__file__).resolve().parents[1] / "uhf_sqd_blocks.example.toml"


@pytest.fixture
def saved(monkeypatch):
    # Saved QuantumSourceBlocks by name, as Prefect keeps them: sqd_de's device source exists.
    calls = {"sbd": [], "sources": [], "store": {"quantum-source": "qrmi"}}

    def fake_create_sbd_blocks(**kwargs):
        calls["sbd"].append(kwargs)
        return {"solver_block_name": "sbd-uhf-solver"}

    monkeypatch.setattr(qcsc_prefect_sbd, "create_sbd_blocks", fake_create_sbd_blocks)

    def save(self, name, overwrite=False):
        if name in calls["store"] and not overwrite:
            raise ValueError("You are attempting to save values with a name that is already in use")
        calls["store"][name] = self.config.kind
        calls["sources"].append((name, self.config.kind))

    monkeypatch.setattr(QuantumSourceBlock, "save", save)
    return calls


def test_solver_blocks_are_unrestricted_with_the_source_defaults(saved):
    names = create_blocks.create_blocks(
        {"solver": {"hpc_target": "local", "root_dir": "/w", "sbd_executable": "/x/diag_uhf"}}
    )
    (kwargs,) = saved["sbd"]
    assert kwargs["integrals"] == "unrestricted"
    for key, value in create_blocks.SOLVER_DEFAULTS.items():
        assert kwargs[key] == value
    assert create_blocks.SOLVER_DEFAULTS == {
        "block": 4,
        "iteration": 1,
        "tolerance": 1e-2,
        "carryover_ratio": 0.1,
        "carryover_type": 0,
    }
    assert names["solver_block_name"] == "sbd-uhf-solver" and names["quantum_sources"] == {}


def test_explicit_solver_settings_override_the_defaults(saved):
    create_blocks.create_blocks(
        {
            "solver": {
                "hpc_target": "local",
                "root_dir": "/w",
                "sbd_executable": "/x",
                "block": 30,
                "tolerance": 1e-10,
            }
        }
    )
    (kwargs,) = saved["sbd"]
    assert (kwargs["block"], kwargs["tolerance"], kwargs["iteration"]) == (30, 1e-10, 1)


def test_restricted_integrals_and_unknown_sections_are_refused(saved):
    with pytest.raises(ValueError, match="unrestricted"):
        create_blocks.create_blocks({"solver": {"integrals": "restricted"}})
    with pytest.raises(ValueError, match="Unknown sections: blocks"):
        create_blocks.create_blocks({"blocks": {}})
    assert saved["sbd"] == []


def test_quantum_source_blocks_are_saved_by_name(saved, tmp_path, capsys):
    path = tmp_path / "blocks.toml"
    path.write_text(
        '[solver]\nhpc_target = "local"\nroot_dir = "/w"\nsbd_executable = "/x"\n\n'
        '[quantum_sources.uhf-sqd-random]\nconfig = { kind = "random" }\n'
    )
    create_blocks.main([str(path)])
    assert saved["sources"] == [("uhf-sqd-random", "random")]
    assert json.loads(capsys.readouterr().out)["quantum_sources"] == {"uhf-sqd-random": "random"}


def test_the_example_leaves_another_workflows_source_block_alone(saved):
    example = create_blocks.load_config(EXAMPLE)
    create_blocks.create_blocks(example)
    assert saved["store"]["quantum-source"] == "qrmi"  # sqd_de's device source is unchanged
    assert saved["store"]["uhf-sqd-random"] == "random"


def test_an_existing_source_block_is_replaced_only_on_request(saved):
    random_source = {"config": {"kind": "random"}}
    with pytest.raises(ValueError, match="already exists and may be used by another workflow"):
        create_blocks.create_blocks({"quantum_sources": {"quantum-source": random_source}})
    assert saved["store"]["quantum-source"] == "qrmi"
    create_blocks.create_blocks(
        {"quantum_sources": {"quantum-source": {**random_source, "overwrite": True}}}
    )
    assert saved["store"]["quantum-source"] == "random"
    with pytest.raises(ValueError, match="overwrite must be true or false"):
        create_blocks.create_blocks(
            {"quantum_sources": {"x": {**random_source, "overwrite": "yes"}}}
        )
