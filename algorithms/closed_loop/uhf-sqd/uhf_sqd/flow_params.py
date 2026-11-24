"""Flow parameters of the UHF-SQD workflow.

Ported from kunalroy332/qcsc-prefect at 3968e58 (``algorithms/sbd/sbd/flow_params.py``) with the
first integration's scope: the fields of deferred features are not present (see
``UHF_SQD_DEFERRED_FEATURES.md``). Unknown keys are rejected, so adding them back later is
backward compatible. The defaults of the remaining fields are the source's.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from qcsc_workflow_utility.uhf import DEFAULT_UHF_CACHE_DIR

# Unknown keys are rejected (not silently dropped): a misspelled or unsupported option in a params
# JSON / launcher must fail at validation instead of running with the default.
_STRICT = ConfigDict(extra="forbid")


class CircuitParameters(BaseModel):
    """Configuration for LUCJ circuit construction."""

    model_config = _STRICT

    n_lucj_layers: int = Field(
        default=2,
        description="Number of LUCJ circuit block repetitions.",
        title="LUCJ Circuit Layers",
        ge=1,
    )

    use_reset_mitigation: bool = Field(
        default=True,
        description="Set True to use reset error mitigation scheme.",
        title="Reset Mitigation",
    )

    optimization_level: int = Field(
        default=3,
        description="Optimization level of Qiskit transpiler",
        title="Optimization Level",
        ge=0,
        le=3,
    )

    sabre_max_iterations: int = Field(
        default=8,
        description=(
            "The number of forward-backward routing iterations to refine the layout "
            "and reduce routing costs."
        ),
        title="Sabre Max Iteration",
        ge=1,
    )

    sabre_swap_trials: int = Field(
        default=10,
        description=(
            "The number of routing trials for each layout, refining gate placement "
            "for better routing."
        ),
        title="Sabre SWAP Trials",
        ge=1,
    )

    sabre_layout_trials: int = Field(
        default=1_024,
        description="The number of random seed trials to run layout with.",
        title="Sabre Layout Trials",
        ge=1,
    )

    # ── Alpha-beta coupling density ─────────────────────────────────────────────────────
    # The alpha and beta spin channels of the LUCJ ansatz are coupled by the ab interaction
    # pairs (p, p). On heavy-hex hardware those couplings are ancilla-mediated, so the stock
    # layout only couples every 4th orbital (stride 4) to keep SWAP/2q-gate depth low. That
    # undercouples inter-spin correlation. ab_stride < 4 requests denser coupling (2 -> ~2x,
    # 1 -> full per-orbital) at the cost of more ECR gates / depth.
    ab_stride: int = Field(
        default=4,
        description=(
            "Stride for alpha-beta LUCJ coupling pairs (p, p) for p in range(0, norb, stride). "
            "4 = stock heavy-hex (couple every 4th orbital); 2 = ~2x denser coupling; "
            "1 = couple every orbital. Denser coupling captures more inter-spin correlation "
            "but adds ECR gates / circuit depth."
        ),
        title="Alpha-Beta Coupling Stride",
        ge=1,
    )

    ucj_optimize: bool = Field(
        default=True,
        description=(
            "Build the LUCJ ansatz with ffsim's compressed double factorization (optimize=True): "
            "re-optimize the kept DF terms to best approximate the full UCCSD generator. This "
            "spreads the prepared state over many more configurations -> a far more diverse SQD "
            "sample set (bare optimize=False collapses toward Hartree-Fock). Set False only to "
            "study the ansatz effect."
        ),
        title="UCJ Compressed-DF Optimize",
    )


class DEParameters(BaseModel):
    """Configuration for differential evoluation."""

    model_config = _STRICT

    num_walkers: int = Field(
        default=4,
        description=(
            "Number of populations for differential evolution. "
            "Differential-evolution mutation (iterations > 1) needs at least 4 walkers; "
            "a single evaluation pass (iterations = 1, e.g. a recovery-depth test) accepts 1."
        ),
        title="Walkers",
        ge=1,
    )

    iterations: int = Field(
        default=1,
        description="Number of DE optimization iterations.",
        title="Differential Evolution Iterations",
        ge=1,
    )

    randomization_factor: float = Field(
        default=0.2,
        description="Degree of ansatz parameter perturbation from CCSD amplitude.",
        title="Randomization Factor",
        ge=0,
    )

    fxc: float = Field(
        default=0.6,
        description=(
            "Factor to scale the difference between individuals when generating mutants. "
            "Controls step size and search aggressiveness (typically 0.4 to 1.0)."
        ),
        title="Scaling Factor",
        gt=0.0,
        le=1.0,
    )

    cr_prob: float = Field(
        default=0.9,
        description=(
            "Probability of mixing components from the mutant vector into the trial solution. "
            "Controls exploration vs. exploitation (0 to 1)."
        ),
        title="Crossover Rate",
        gt=0.0,
        le=1.0,
    )


class SamplerParameters(BaseModel):
    """Quantum sampling options."""

    model_config = _STRICT

    shots: int = Field(
        default=10_000,
        description=(
            "Total number of measurement shots per walker. Also the number of bitstrings drawn "
            "from a random source."
        ),
        title="Shots",
        ge=1,
    )


class ReferenceParameters(BaseModel):
    """Broken-symmetry UHF reference settings (the source's environment variables)."""

    model_config = _STRICT

    af_groups: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Atom-localized antiferromagnetic guess: fragment names mapped to 0-based orbital "
            "indices, with the option keys 'up' and 'down' (fragments seeded spin-up and "
            "spin-down), 'free' (seeded half-filled) and 'pol' (polarization, default 1.0). "
            "None tries only the energy-ordered AF guess. The source read it from "
            "FE4S4_AF_GROUPS, FE4S4_AF_POL and FE4S4_AF_FREE_S."
        ),
        title="AF Guess Groups",
    )

    cache_dir: str | None = Field(
        default=DEFAULT_UHF_CACHE_DIR,
        description=(
            "Directory of the content-keyed BS-UHF reference cache (the source's "
            "SBD_UHF_CACHE_DIR). None disables reading and writing the cache."
        ),
        title="Reference Cache Directory",
    )

    cache_read: bool = Field(
        default=True,
        description=(
            "Reuse a cached reference (the source's SBD_UHF_CACHE). A fresh result is "
            "written to the cache either way."
        ),
        title="Reuse Cached Reference",
    )

    ccsd_max_cycle: int = Field(
        default=50,
        description="UCCSD cycles per run (the source's SEED_CCSD_MAX_CYCLE).",
        title="UCCSD Cycles per Run",
        ge=1,
    )

    ccsd_restarts: int = Field(
        default=10,
        description=(
            "Number of UCCSD runs; each restart continues from the last amplitudes with a "
            "fresh DIIS (the source's SEED_CCSD_RESTARTS)."
        ),
        title="UCCSD Runs",
        ge=1,
    )


class FlowParameters(BaseModel):
    """Workflow Parameters."""

    model_config = _STRICT

    fcidump: str = Field(
        description=(
            "A path to a restricted, spatial-orbital FCIDUMP of the target molecule (all "
            "orbitals active; MS2 sets n_alpha - n_beta; ECORE is the constant term). The UHF "
            "reference is built from it."
        ),
        title="FCIDump File",
    )

    sqd_dim: int = Field(
        default=100_000_0,
        description="Dimension of subsampled bitstrings for diagonalization.",
        title="SQD Subspace Dimension",
        ge=1,
    )

    n_recovery_steps: int = Field(
        default=1,
        description=(
            "Number of SQD self-consistency configuration-recovery passes per walker. "
            "Each pass re-runs configuration recovery (from the same quantum samples) using "
            "the orbital occupancies from the previous pass's diagonalization, then "
            "re-subsamples and re-diagonalizes. 1 = single pass (no self-consistency); "
            "3-5 is the canonical SQD recovery loop."
        ),
        title="SQD Recovery Steps",
        ge=1,
    )

    n_batches: int = Field(
        default=1,
        description=(
            "Number of independent subsample batches per recovery pass (the K-batch SQD scheme "
            "of arXiv:2405.05068). Each batch draws its own subspace of dimension sqd_dim from the "
            "recovered distribution and is diagonalized separately; the reported energy is the "
            "minimum over batches and the occupancies fed to the next recovery pass are the "
            "average over batches. 1 = single batch (legacy). 5-10 stabilizes the occupancy and "
            "improves the chance of capturing low-excitation configurations. Cost scales linearly "
            "with this value."
        ),
        title="SQD Batches per Recovery",
        ge=1,
    )

    quantum_source: str = Field(
        default="uhf-sqd-random",
        description=(
            "Name of a saved QuantumSourceBlock. Its settings select the sampling platform "
            "(random, QRMI or QFw); changing this name does not change the algorithm. A Block "
            "shared with another workflow can be named here."
        ),
        title="Quantum Source Block",
        min_length=1,
    )

    random_seed: int = Field(
        default=24,
        description="Base seed for deterministic random samples, if the source uses them.",
        title="Random Seed",
        ge=0,
    )

    circ_params: CircuitParameters = Field(
        default_factory=CircuitParameters,
        title="Circuit Parameters",
    )

    sampler: SamplerParameters = Field(
        default_factory=SamplerParameters,
        title="Sampler Parameters",
    )

    de_params: DEParameters = Field(
        default_factory=DEParameters,
        title="Differential Evoluation Parameters",
    )

    reference: ReferenceParameters = Field(
        default_factory=ReferenceParameters,
        title="Reference Parameters",
    )

    solver_block_ref: str = Field(
        default="sbd_solver_job/sbd-uhf-solver",
        description=(
            "Solver block reference in 'sbd_solver_job/<block_document_name>' format. The block "
            "must have integrals='unrestricted' (a -D_UHF SBD driver)."
        ),
    )
