"""Flow parameters: the source's defaults, explicit reference settings, unknown keys refused.

The strictness tests follow kunalroy332/qcsc-prefect at 3968e58
(``algorithms/sbd/tests/test_flow_params_strict.py``, d3d97b0 by Boseong Kim).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from qcsc_workflow_utility.uhf import DEFAULT_UHF_CACHE_DIR
from uhf_sqd.flow_params import (
    CircuitParameters,
    DEParameters,
    FlowParameters,
    ReferenceParameters,
    SamplerParameters,
)


def test_defaults_are_the_source_defaults():
    fp = FlowParameters(fcidump="x.fcidump")
    assert (fp.sqd_dim, fp.n_recovery_steps, fp.n_batches, fp.random_seed) == (1_000_000, 1, 1, 24)
    assert fp.sampler.shots == 10_000
    c = fp.circ_params
    assert (c.n_lucj_layers, c.use_reset_mitigation, c.optimization_level) == (2, True, 3)
    assert (c.sabre_max_iterations, c.sabre_swap_trials, c.sabre_layout_trials) == (8, 10, 1024)
    assert (c.ab_stride, c.ucj_optimize) == (4, True)
    d = fp.de_params
    assert (d.num_walkers, d.iterations, d.randomization_factor, d.fxc, d.cr_prob) == (
        4,
        1,
        0.2,
        0.6,
        0.9,
    )
    r = fp.reference
    assert (r.af_groups, r.cache_dir, r.cache_read) == (None, DEFAULT_UHF_CACHE_DIR, True)
    assert (r.ccsd_max_cycle, r.ccsd_restarts) == (50, 10)
    assert fp.quantum_source == "uhf-sqd-random"
    assert fp.solver_block_ref == "sbd_solver_job/sbd-uhf-solver"


def test_fcidump_is_required():
    with pytest.raises(ValidationError, match="fcidump"):
        FlowParameters()


@pytest.mark.parametrize(
    "model, field",
    [
        (FlowParameters, "seed_cisd"),
        (FlowParameters, "hci_boost"),
        (FlowParameters, "active_space_manifest"),
        (FlowParameters, "oo_grad_tol"),
        (CircuitParameters, "use_error_aware_layout"),
        (SamplerParameters, "n_shot_batches"),
        (SamplerParameters, "dynamical_decoupling"),
        (SamplerParameters, "saved_samples"),
        (DEParameters, "num_walker"),  # misspelled
        (ReferenceParameters, "af_group"),  # misspelled
    ],
)
def test_unknown_and_deferred_keys_are_rejected(model, field):
    kwargs = {"fcidump": "x.fcidump"} if model is FlowParameters else {}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model(**kwargs, **{field: 1})


def test_reference_settings_validate():
    with pytest.raises(ValidationError):
        ReferenceParameters(ccsd_max_cycle=0)
    groups = {"a": [0, 1], "b": [2], "up": ["a"], "down": ["b"], "pol": 0.6}
    assert ReferenceParameters(af_groups=groups).af_groups == groups
    assert ReferenceParameters(cache_dir=None).cache_dir is None


def test_json_round_trip():
    fp = FlowParameters(fcidump="x.fcidump", reference={"af_groups": {"a": [0], "up": ["a"]}})
    assert FlowParameters.model_validate_json(fp.model_dump_json()) == fp
