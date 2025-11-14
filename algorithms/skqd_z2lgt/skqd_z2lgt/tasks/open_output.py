"""Open the output HDF5 file."""
import os
import logging
from typing import Optional
import numpy as np
import h5py
from skqd_z2lgt.parameters import Parameters


def open_output(parameters: Parameters, logger: Optional[logging.Logger] = None) -> str:
    """Open a new output HDF5 file and set it up for the workflow, or validate an existing file.

    Args:
        parameters: Workflow parameters.

    Returns:
        The name of the output file.
    """
    logger = logger or logging.getLogger(__name__)

    attrs = [
        ('lattice', parameters.lgt.lattice),
        ('plaquette_energy', parameters.lgt.plaquette_energy),
        ('charged_vertices', parameters.lgt.charged_vertices),
        ('basis_2q', parameters.circuit.basis_2q),
        ('num_steps', parameters.skqd.n_trotter_steps),
        ('delta_t', parameters.skqd.dt),
        ('shots', parameters.runtime.shots)
    ]

    if os.path.exists(parameters.output_filename):
        logger.info('Validating configurations in existing file %s', parameters.output_filename)
        with h5py.File(parameters.output_filename, 'r') as source:
            for key, value in attrs:
                if ((isinstance(value, float) and not np.isclose(source.attrs[key], value))
                        or (isinstance(value, (int, str)) and source.attrs[key] != value)):
                    raise RuntimeError(f'Recorded {key} does not match the flow parameter')

    else:
        logger.info('Creating a new file %s', parameters.output_filename)
        with h5py.File(parameters.output_filename, 'w') as out:
            for key, value in attrs:
                out.attrs[key] = value
