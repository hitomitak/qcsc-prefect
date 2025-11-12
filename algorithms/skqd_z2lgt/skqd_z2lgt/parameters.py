"""Workflow parameters."""
from pydantic import BaseModel, Field, field_validator


class LGTParameters(BaseModel):
    """Parameters to specify the Z2 LGT."""

    lattice: str = Field(
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


class CircuitParameters(BaseModel):
    """Configuration for circuit."""

    basis_2q: str = Field(
        default='rzz',
        description='Two-qubit gate used in the Trotter circuit ("cx", "cz", or "rzz"). Note that'
                    ' the selection "rzz" will utilize CZ gates together with Rzz.',
        title='Base two-qubit gate'
    )
    optimization_level: int = Field(
        default=3,
        description="Transpile: Optimization level of transpiler",
        title="Optimization Level",
        ge=0,
        le=3,
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
    batch_size: int = Field(
        default=32,
        description='Batch size for stochastic gradient descent.',
        title='Training Batch Size',
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
        default=10,
        description='Maximum number of epochs to train.',
        title='Maximum Epochs',
        ge=1
    )
    rtol: float = Field(
        default=2.,
        description='Convergence condition (max|loss - mean| of the last 5 epochs < rtol * stddev)',
        title='Convergence Relative Tolerance',
        ge=0.
    )


class SKQDParameters(BaseModel):
    """Configuration for SKQD algorithm execution."""

    n_trotter_steps: int = Field(
        default=8,
        description="Number of Trotter steps (Krylov dimension).",
        title="Trotter Steps",
        ge=1,
    )
    dt: float = Field(
        default=0.02,
        description="Time Interval for Trotterization.",
        title="Time Interval",
        ge=0,
    )
    max_subspace_dim: int = Field(
        default=1_000_000,
        description="Maximum subspace dimension in iterative configuration recovery.",
        title="Maximum Subspace Dimension",
        ge=1
    )
    num_gen: int = Field(
        default=3,
        description="Number of bitstrings to generate in each iteration of configuration recovery.",
        title="Generated Samples",
        ge=1
    )
    max_iterations: int = Field(
        default=10,
        description="Number of configuration recovery iterations.",
        title="Max Iteration",
        ge=0
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

    circuit: CircuitParameters = Field(
        default_factory=CircuitParameters,
        description='Trotter circuit parameters and transpiler settings.',
        title='Circuit'
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

    runtime_job_id: str | None = Field(
        default=None,
        description='ID of an existing IBM Quantum workload.',
        title='Runtime Job ID'
    )

    output_filename: str | None = Field(
        default=None,
        description='Name of the HDF5 file where intermediate output are stored.',
        title='Output File Name'
    )
