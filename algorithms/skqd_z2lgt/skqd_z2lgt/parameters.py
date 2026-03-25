"""Workflow parameters."""
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class LGTParameters(BaseModel):
    """Parameters to specify the Z2 LGT."""

    lattice: str | tuple[int, int] = Field(
        default='full_156q',
        description='Two-dimensional lattice configuration.',
        title='Lattice Configuration'
    )
    plaquette_energy: float = Field(
        default=1.,
        description='Plaquette energy in the Hamiltonian (transverse field strength in the dual'
                    ' Ising model).',
        title='Plaquette Energy',
        ge=0.
    )
    charged_vertices: list[int] = Field(
        default_factory=list,
        description='Vertex indices where static charges are to be located.',
        title='Charged vertices'
    )

    @field_validator('charged_vertices')
    @classmethod
    def check_num_vertices(cls, vlist: list[int]) -> list[int]:
        if len(vlist) % 2 != 0:
            raise ValueError('There must be even number of charged vertices')
        return vlist

    def model_post_init(self, _):  # pylint: disable=arguments-differ
        if self.lattice == 'full_156q':
            self.lattice = '''
            *-*-*-*-*╷
             * * * * *
            * * * * *╎
             * * * * *
            * * * * *╎
             * * * * *
            * * * * *╎
             * * * * *
            *-*-*-*-*╵
            '''


class DMRGParameters(BaseModel):
    """Configuration for DMRG."""

    nsweeps: int = Field(
        default=5,
        description='Number of DMRG sweeps.',
        title='Number of sweeps'
    )
    maxdim: list[int] = Field(
        default=[10, 20, 100, 100, 200],
        description='Maximum size allowed for the bond dimension or rank of the MPS.',
        title='Maximum bond dimensions'
    )
    cutoff: float = Field(
        default=1.e-10,
        description='Truncation error cutoff or threshold to use for truncating the bond dimension'
                    ' or rank of the MPS.',
        title='Truncation error cutoff'
    )
    num_samples: int = Field(
        default=100_000,
        description='Number of times to sample the MPS to estimate the probability distribution of'
                    ' the ground state.',
        title='MPS sampling number'
    )
    julia_sysimage: Optional[str] = Field(
        default=None,
        description='Path to the precompiled ITensors sysimage.',
        title='ITensors.jl sysimage path'
    )


class CircuitParameters(BaseModel):
    """Configuration for circuit."""

    layout: Optional[list[int] | dict[tuple[str, int], int]] = Field(
        default=None,
        description='Qubit layout.',
        title='Qubit Layout'
    )
    trotter_steps_per_dt: Optional[list[int]] = Field(
        default=None,
        description='Number of Trotter steps for each evolution step.',
        title='Trotter Steps Per Evolution Step'
    )
    optimization_level: int = Field(
        default=3,
        description="Transpile: Optimization level of transpiler",
        title="Optimization Level",
        ge=0,
        le=3,
    )


class RuntimeParameters(BaseModel):
    """Configuration for runtime sampler."""

    job_id: str | None = Field(
        default=None,
        description='ID of an existing IBM Quantum workload.',
        title='Runtime Job ID'
    )
    instance: str | None = Field(
        default=None,
        description='IBM Quantum instance.',
        title='IBM Quantum Instance'
    )
    backend: str | None = Field(
        default=None,
        description='Backend name.',
        title='Backend Name'
    )
    shots_exp: int = Field(
        default=100_000,
        description='Shots per experiment circuit.',
        title='Shots (Experiment)'
    )
    shots_ref: int = Field(
        default=100_000,
        description='Shots per reference circuit.',
        title='Shots (Reference)'
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description='Runtime options.',
        title='Runtime Options'
    )
    options_name: str | None = Field(
        default=None,
        description='Prefect Variable name for runtime options.',
        title='Options Variable Name'
    )


class CRBMParameters(BaseModel):
    """Configuration for conditional restricted Boltzmann machine used in configuration recovery."""

    num_h: int = Field(
        default=256,
        description='Number of hidden units of the restricted Boltzmann machine.',
        title='Number of Hidden Units',
        ge=1
    )
    l2w_weights: float = Field(
        default=1.,
        description='Weight of the L2 regularization term for weight parameters of the model.',
        title='L2 Weight (W)',
        ge=0.
    )
    l2w_biases: float = Field(
        default=0.2,
        description='Weight of the L2 regularization term for bias parameters of the model.',
        title='L2 Weight (B)',
        ge=0.
    )
    train_batch_size: int = Field(
        default=32,
        description='Batch size for stochastic gradient descent.',
        title='Training Batch Size',
        ge=1
    )
    gen_batch_size: int = Field(
        default=10_000,
        description='Batch size for generation.',
        title='Generation Batch Size',
        ge=1
    )
    learning_rate: float = Field(
        default=0.001,
        description='Learning rate for the Adam optimizer with weight decay',
        title='Learning Rate',
        gt=0.
    )
    init_h_sparsity: float = Field(
        default=0.01,
        description='Sparsity parameter for initializing the bias parameters of the hidden units.',
        title='Initial Hidden Unit Sparsity',
        ge=0.
    )
    num_epochs: int = Field(
        default=100,
        description='Maximum number of epochs to train.',
        title='Maximum Epochs',
        ge=1
    )


class SKQDParameters(BaseModel):
    """Configuration for SKQD algorithm execution."""

    num_krylov: int = Field(
        default=4,
        description="Number of Krylov vectors excluding the initial state.",
        title="Number of Unitary Krylov Vectors",
        ge=1,
    )
    time_steps: list[float] = Field(
        default=[0.4],
        description="Time Steps for Unitary Krylov Vectors.",
        title="Time Steps",
    )
    num_gen: int = Field(
        default=3,
        description="Number of bitstrings to generate in each iteration of configuration recovery.",
        title="Generated Samples",
        ge=1
    )
    extensions: list[str] = Field(
        default_factory=list,
        description="Names of subspace extension functions to apply.",
        title="Subspace extension functions"
    )
    max_iterations: int = Field(
        default=10,
        description="Number of configuration recovery iterations.",
        title="Max Iteration",
        ge=0
    )
    probability_cutoff: float = Field(
        default=1.e-20,
        description="Cutoff for probability of computational basis state to be kept for the next"
                    " round of iterative configuration recovery.",
        title="State probability cutoff",
        ge=0.
    )
    max_subspace_dim: int = Field(
        default=1_000_000,
        description="Maximum subspace dimension in iterative configuration recovery.",
        title="Maximum Subspace Dimension",
        ge=1
    )
    delta_e: float = Field(
        default=0.005,
        description="Convergence condition (change in energy value between configuration recovery"
                    "iterations)",
        title="Energy Convergence",
        ge=0.
    )


class Parameters(BaseModel):
    """Workflow configuration parameters."""

    lgt: LGTParameters = Field(
        default_factory=LGTParameters,
        description='Lattice gauge theory definition.',
        title='Lattice Gauge Theory'
    )

    dmrg: Optional[DMRGParameters] = Field(
        default=None,
        description='DMRG and MPS sampling parameters.',
        title='DMRG'
    )

    circuit: CircuitParameters = Field(
        default_factory=CircuitParameters,
        description='Trotter circuit parameters and transpiler settings.',
        title='Circuit'
    )

    runtime: RuntimeParameters = Field(
        default_factory=RuntimeParameters,
        description='Runtime configuration parameters.',
        title='Runtime'
    )

    crbm: CRBMParameters = Field(
        default_factory=CRBMParameters,
        description='Settings for conditional restricted Boltzmann machine.',
        title='CRBM'
    )

    skqd: SKQDParameters = Field(
        default_factory=SKQDParameters,
        description='Control of SKQD algorithm execution and solver parameters.',
        title='SKQD',
    )

    pkgpath: str | None = Field(
        default=None,
        description='Path of the directory tree where intermediate and final output are stored.',
        title='Output package name'
    )
