"""Per-spin carryover, subsampling, selection, LUCJ noise and the walker_sqd recovery loop.

Most tests are ported from kunalroy332/qcsc-prefect at 3968e58
(``algorithms/sbd/tests/test_uhf_spin_unbalanced.py``, 9f430e1 by Boseong Kim and da4bdf6 by
kunalroy332), adapted to the UHF-only port: the walker samples through a QuantumSource and returns
``(energy, (alpha, beta))``. The RHF, HCI booster, CC refresh and checkpoint tests are not
ported, because those paths are not.
"""

from __future__ import annotations

import asyncio
from unittest import mock

import numpy as np
import pytest
from uhf_sqd import lucj, main, sqd
from uhf_sqd_helpers import (
    NA,
    NB,
    NORB,
    FakeSolver,
    all_strings,
    ints_to_rows,
    prepare_walker,
    random_strings,
    rows_to_ints,
    run_walker,
)


@pytest.fixture(autouse=True)
def _silence_prefect_logger(monkeypatch):
    for mod in (sqd, main, lucj):
        if hasattr(mod, "get_run_logger"):
            monkeypatch.setattr(mod, "get_run_logger", lambda: mock.MagicMock())


# ── per-spin carryover ─────────────────────────────────────────────────────────────────────


def test_split_spin_carryover_per_spin_passthrough():
    norb = 6
    rng = np.random.default_rng(0)
    co_a = ints_to_rows(random_strings(rng, norb, 3, 5), norb)
    co_b = ints_to_rows(random_strings(rng, norb, 2, 2), norb)
    a, b = sqd._split_spin_carryover(co_a, norb, co_b)
    assert a.shape == (5, norb) and b.shape == (2, norb)
    assert np.array_equal(a, co_a) and np.array_equal(b, co_b)


def test_split_spin_carryover_empty_and_none_beta():
    norb = 6
    for empty in (np.full((0, norb), False), np.empty((0, 0), dtype=bool)):
        a, b = sqd._split_spin_carryover(empty, norb)
        assert a.shape == (0, norb) and b.shape == (0, norb)
    co_a = ints_to_rows([0b000111, 0b001011], norb)
    a, b = sqd._split_spin_carryover(co_a, norb, np.empty((0, norb), dtype=bool))
    assert a.shape == (2, norb) and b.shape == (0, norb)


def test_split_spin_carryover_legacy_stacked_drops_padding():
    norb, na, nb = 6, 3, 2
    rng = np.random.default_rng(1)
    co_a = ints_to_rows(random_strings(rng, norb, na, 5), norb)
    co_b = ints_to_rows(random_strings(rng, norb, nb, 3), norb)
    stacked = np.zeros((5, 2 * norb), dtype=bool)
    stacked[:3, :norb] = co_b
    stacked[:5, norb:] = co_a
    a, b = sqd._split_spin_carryover(stacked, norb)
    assert np.array_equal(a, co_a)
    assert np.array_equal(b, co_b)
    assert set(a.sum(axis=1).tolist()) == {na} and set(b.sum(axis=1).tolist()) == {nb}


def test_split_rejects_a_width_that_is_neither_per_spin_nor_stacked():
    with pytest.raises(ValueError, match="per-spin"):
        sqd._split_spin_carryover(np.ones((2, 7), dtype=bool), 4)


def test_split_is_idempotent_and_does_not_mutate_inputs():
    norb = 6
    a = ints_to_rows(all_strings(norb, 3)[:5], norb)
    b = ints_to_rows(all_strings(norb, 2)[:2], norb)
    a0, b0 = a.copy(), b.copy()
    a1, b1 = sqd._split_spin_carryover(a, norb, b)
    a2, b2 = sqd._split_spin_carryover(a1, norb, b1)
    assert np.array_equal(a2, a) and np.array_equal(b2, b)
    assert np.array_equal(a, a0) and np.array_equal(b, b0)


def test_split_uses_beta_left_alpha_right_for_legacy_layout():
    norb = 4
    stacked = np.zeros((2, 2 * norb), dtype=bool)
    stacked[0, :norb] = [1, 1, 0, 0]  # beta
    stacked[0, norb:] = [1, 1, 1, 0]  # alpha
    stacked[1, norb:] = [0, 1, 1, 1]  # alpha only
    a, b = sqd._split_spin_carryover(stacked, norb)
    assert a.tolist() == [[1, 1, 1, 0], [0, 1, 1, 1]]
    assert b.tolist() == [[1, 1, 0, 0]]


def test_split_legacy_keeps_valid_empty_determinant_when_spin_has_zero_electrons():
    norb = 4
    stacked = np.zeros((2, 2 * norb), dtype=bool)
    stacked[:, norb:] = [[1, 0, 0, 0], [0, 1, 0, 0]]
    _, b = sqd._split_spin_carryover(stacked, norb, None, num_electrons=(1, 0))
    assert b.shape == (1, norb) and not b.any()
    _, b_default = sqd._split_spin_carryover(stacked, norb)
    assert b_default.shape == (0, norb)


# ── subsampling ────────────────────────────────────────────────────────────────────────────


def test_subsample_open_shell_unequal_carryover_keeps_sectors():
    norb, na, nb = 8, 4, 3
    rng = np.random.default_rng(2)
    bm = np.array(
        [
            np.r_[ints_to_rows([b], norb)[0], ints_to_rows([a], norb)[0]]
            for a, b in zip(random_strings(rng, norb, na, 40), random_strings(rng, norb, nb, 40))
        ],
        dtype=bool,
    )
    probs = np.ones(len(bm)) / len(bm)
    co_a_ints = random_strings(rng, norb, na, 9)
    co_b_ints = random_strings(rng, norb, nb, 4)
    ci_a, ci_b = sqd.subsample_open_shell.fn(
        bitstring_matrix=bm,
        probabilities=probs,
        carryover_a=ints_to_rows(co_a_ints, norb),
        carryover_b=ints_to_rows(co_b_ints, norb),
        subspace_dim=400,
        norb=norb,
        num_elec_a=na,
        num_elec_b=nb,
        rng=np.random.default_rng(3),
    )
    assert {int(x).bit_count() for x in ci_a} == {na}
    assert {int(x).bit_count() for x in ci_b} == {nb}
    assert len(set(ci_a.tolist())) == len(ci_a) and len(set(ci_b.tolist())) == len(ci_b)
    assert set(co_a_ints) <= set(ci_a.tolist())
    assert set(co_b_ints) <= set(ci_b.tolist())
    assert ci_a[0] == (1 << na) - 1 and ci_b[0] == (1 << nb) - 1


def test_recombination_booster_is_not_available():
    with pytest.raises(NotImplementedError, match="recomb_boost"):
        sqd._subsample_one_spin(
            ci_strs=np.array([0b0111], dtype=np.longlong),
            probabilities=np.ones(1),
            carryover=np.empty((0, 4), dtype=bool),
            subspace_dim=16,
            norb=4,
            num_elec=3,
            rng=np.random.default_rng(0),
            recomb_boost=True,
        )


# ── OptimizerState and selection ───────────────────────────────────────────────────────────


def _state(norb=6, nw=3, with_b=False):
    st = main.OptimizerState(
        energies=np.zeros(nw),
        populations=np.full((nw, 4), np.nan),
        carryover=np.full((0, norb), False),
    )
    if with_b:
        st.carryover_b = np.full((0, norb), False)
    return st


def test_selection_stores_per_spin_carryover():
    norb = 6
    state = main.OptimizerState(
        energies=np.zeros(2),
        populations=np.full((2, 3), np.nan),
        carryover=np.full((0, norb), False),
    )
    co_a = ints_to_rows([0b000111, 0b001011, 0b010011], norb)
    co_b = ints_to_rows([0b000011], norb)
    new = main.selection.fn(
        trial_populations=[np.zeros(3), np.ones(3)],
        trial_energies=np.array([-1.0, -2.0]),
        trial_carryovers=[(None, None), (co_a, co_b)],
        current_state=state,
    )
    assert new.best_index == 1
    assert np.array_equal(new.carryover, co_a) and np.array_equal(new.carryover_b, co_b)
    assert new.copy().carryover_b.shape == (1, norb)


def test_optimizer_state_copy_is_deep_for_carryover_b():
    st = _state(with_b=True)
    st.carryover_b = ints_to_rows([0b000011], 6)
    cp = st.copy()
    cp.carryover_b[0, 0] = not cp.carryover_b[0, 0]
    assert st.carryover_b[0, 0] != cp.carryover_b[0, 0]


def test_selection_does_not_overwrite_carryover_without_improvement():
    st = _state(with_b=True)
    st.energies[:] = -5.0
    st.best_index = 0
    keep_a, keep_b = ints_to_rows([0b000111], 6), ints_to_rows([0b000011], 6)
    st.carryover, st.carryover_b = keep_a, keep_b
    with mock.patch.object(main.OptimizerState, "best_energy", return_value=-5.0):
        new = main.selection.fn(
            trial_populations=[np.zeros(4)] * 3,
            trial_energies=np.array([-1.0, -2.0, -3.0]),
            trial_carryovers=[(ints_to_rows([0b001011], 6), ints_to_rows([0b000101], 6))] * 3,
            current_state=st,
        )
    assert np.array_equal(new.carryover, keep_a) and np.array_equal(new.carryover_b, keep_b)


def test_selection_skips_nan_walkers_when_picking_best_carryover():
    co_good = ints_to_rows([0b000111], 6)
    new = main.selection.fn(
        trial_populations=[np.zeros(4)] * 3,
        trial_energies=np.array([np.nan, -2.0, np.nan]),
        trial_carryovers=[(None, None), (co_good, None), (None, None)],
        current_state=_state(),
    )
    assert new.best_index == 1 and np.array_equal(new.carryover, co_good)


def test_selection_accepts_a_first_positive_energy():
    # Behaviour change from the source, which compared with the initial 0.0 and never accepted
    # a positive total energy: the flow then returned 0.0.
    new = main.selection.fn(
        trial_populations=[np.full(4, 1.0), np.full(4, 2.0), np.full(4, 3.0)],
        trial_energies=np.array([37.5, 37.3, np.nan]),
        trial_carryovers=[(None, None)] * 3,
        current_state=_state(),
    )
    assert new.best_index == 1 and new.best_energy() == 37.3
    assert new.energies.tolist() == [37.5, 37.3, 0.0]
    assert new.populations[:2].tolist() == [[1.0] * 4, [2.0] * 4]
    assert np.all(np.isnan(new.populations[2]))  # a NaN walker is still not accepted
    # After the first acceptance, only a lower energy replaces it (the source's rule).
    later = main.selection.fn(
        trial_populations=[np.full(4, 9.0)] * 3,
        trial_energies=np.array([37.6, 37.2, 40.0]),
        trial_carryovers=[(None, None)] * 3,
        current_state=new,
    )
    assert later.energies.tolist() == [37.5, 37.2, 40.0]
    assert later.populations[0].tolist() == [1.0] * 4 and later.best_energy() == 37.2


def test_selection_resizes_nan_population_width_for_uhf():
    new = main.selection.fn(
        trial_populations=[np.arange(9.0)] * 3,
        trial_energies=np.array([-1.0, -2.0, -3.0]),
        trial_carryovers=[(ints_to_rows([0b000111], 6), None)] * 3,
        current_state=_state(),
    )
    assert new.populations.shape == (3, 9)


# ── LUCJ noise ─────────────────────────────────────────────────────────────────────────────


def _seed_rng(monkeypatch, seed=0):
    monkeypatch.setattr(lucj, "MODULE_RNG", np.random.default_rng(seed))


def test_antisymmetric_noise_has_full_antisymmetry_and_zero_diagonals(monkeypatch):
    _seed_rng(monkeypatch)
    n = lucj._antisymmetric_noise((4, 4, 5, 5), 0.3)
    assert np.allclose(n, -n.transpose(1, 0, 2, 3)) and np.allclose(n, -n.transpose(0, 1, 3, 2))
    assert np.allclose(n[np.arange(4), np.arange(4)], 0.0)
    assert np.allclose(n[:, :, np.arange(5), np.arange(5)], 0.0)


def test_noise_is_reproducible_and_linear_in_the_factor(monkeypatch):
    _seed_rng(monkeypatch, 1)
    a = lucj._antisymmetric_noise((4, 4, 5, 5), 0.1)
    _seed_rng(monkeypatch, 1)
    b = lucj._antisymmetric_noise((4, 4, 5, 5), 0.3)
    assert np.allclose(b, 3 * a)
    _seed_rng(monkeypatch)
    assert not np.any(lucj._antisymmetric_noise((3, 3, 4, 4), 0.0))


def test_initialize_ucj_parameters_needs_an_unrestricted_reference():
    with pytest.raises(ValueError, match="unrestricted"):
        lucj.initialize_ucj_parameters.fn(
            elec_props=mock.Mock(unrestricted=False),
            aa_indices=[],
            ab_indices=[],
            num_walkers=1,
            randomization_factor=0.0,
            n_lucj_layers=1,
        )


def test_initialize_ucj_parameters_keeps_prefect_task_api():
    assert hasattr(lucj.initialize_ucj_parameters, "fn")
    assert not hasattr(lucj._antisymmetric_noise, "fn")


# ── walker_sqd end to end with a fake per-spin solver ──────────────────────────────────────


def _sector_clean(calls):
    for rec in calls:
        assert {x.bit_count() for x in rec.alpha} == {NA}, rec.alpha
        assert {x.bit_count() for x in rec.beta} == {NB}, rec.beta
        assert len(set(rec.alpha)) == len(rec.alpha)
        assert len(set(rec.beta)) == len(rec.beta)


def test_walker_sqd_end_to_end_unbalanced_every_step_is_sector_clean(monkeypatch):
    solver = FakeSolver(n_alpha_co=5, n_beta_co=2)
    prepare_walker(monkeypatch, solver=solver)
    (energy, (co_a, co_b)), telemetry = run_walker(n_steps=4)
    assert len(solver.calls) == 4
    _sector_clean(solver.calls)
    assert co_a.shape == (5, NORB) and co_b.shape == (2, NORB)
    assert energy < -1.0
    assert [row["step"] for row in telemetry["recovery_trace"]] == [0, 1, 2, 3]
    assert "shot_retention_rate" not in telemetry and "raw_samples_path" not in telemetry


def test_walker_sqd_passes_the_unrestricted_integrals(monkeypatch):
    solver = FakeSolver()
    prepare_walker(monkeypatch, solver=solver)
    run_walker()
    (call,) = solver.calls
    assert set(call.kwargs) == {
        "one_body_tensor",
        "two_body_tensor",
        "one_body_tensor_b",
        "two_body_tensor_ab",
        "two_body_tensor_bb",
    }


def test_walker_sqd_carryover_is_fed_forward_per_spin(monkeypatch):
    solver = FakeSolver(n_alpha_co=4, n_beta_co=2)
    prepare_walker(monkeypatch, solver=solver)
    run_walker(n_steps=3)
    for prev, nxt in zip(solver.calls, solver.calls[1:]):
        assert set(prev.alpha[:4]) <= set(nxt.alpha)
        assert set(prev.beta[:2]) <= set(nxt.beta)


def test_walker_sqd_k_batches_solve_each_batch_and_keep_the_lowest(monkeypatch):
    solver = FakeSolver()
    prepare_walker(monkeypatch, solver=solver)
    (energy, _), telemetry = run_walker(n_steps=2, n_batches=3)
    assert len(solver.calls) == 6
    assert energy == pytest.approx(-1.0 - 0.01 * 6)
    trace = telemetry["recovery_trace"]
    assert trace[0]["batch_min"] == pytest.approx(-1.03) and trace[1]["batch_min"] == pytest.approx(
        -1.06
    )


def test_walker_sqd_accepts_legacy_stacked_carryover_input(monkeypatch):
    a = ints_to_rows(all_strings(NORB, NA)[:5], NORB)
    b = ints_to_rows(all_strings(NORB, NB)[:2], NORB)
    stacked = np.zeros((5, 2 * NORB), dtype=bool)
    stacked[:2, :NORB], stacked[:5, NORB:] = b, a
    solver = FakeSolver()
    prepare_walker(monkeypatch, solver=solver)
    run_walker(carryover=stacked)
    _sector_clean(solver.calls)
    assert set(rows_to_ints(a)) <= set(solver.calls[0].alpha)
    assert set(rows_to_ints(b)) <= set(solver.calls[0].beta)


def test_walker_sqd_accepts_explicit_per_spin_carryover_input(monkeypatch):
    a = ints_to_rows(all_strings(NORB, NA)[:5], NORB)
    b = ints_to_rows(all_strings(NORB, NB)[:2], NORB)
    solver = FakeSolver()
    prepare_walker(monkeypatch, solver=solver)
    run_walker(carryover=a, carryover_b=b)
    _sector_clean(solver.calls)
    assert set(rows_to_ints(a)) <= set(solver.calls[0].alpha)
    assert set(rows_to_ints(b)) <= set(solver.calls[0].beta)


def test_carryover_roundtrip_across_recovery_steps_never_grows_padding():
    solver = FakeSolver(n_alpha_co=6, n_beta_co=2)
    co_a, co_b = np.empty((0, NORB), bool), np.empty((0, NORB), bool)
    for step in range(6):
        ci_a = sorted(set(all_strings(NORB, NA)[: 6 + step]) | set(rows_to_ints(co_a)))
        ci_b = sorted(set(all_strings(NORB, NB)[: 4 + step]) | set(rows_to_ints(co_b)))
        res = asyncio.run(
            solver.run(ci_strings=(np.array(ci_a), np.array(ci_b)), nelec=(NA, NB), norb=NORB)
        )
        co_a, co_b = sqd._split_spin_carryover(
            res.carryover_bitstrings, NORB, res.carryover_bitstrings_b
        )
        assert {int(x).bit_count() for x in rows_to_ints(co_a)} == {NA}
        assert {int(x).bit_count() for x in rows_to_ints(co_b)} == {NB}
    _sector_clean(solver.calls)
