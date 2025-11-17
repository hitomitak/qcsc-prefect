"""Open the output HDF5 file."""
import os
import logging
from typing import Optional
from pathlib import Path
from skqd_z2lgt.parameters import Parameters


def open_output(parameters: Parameters, logger: Optional[logging.Logger] = None) -> str:
    """Open a new output HDF5 file and set it up for the workflow, or validate an existing file.

    Args:
        parameters: Workflow parameters.

    Returns:
        The name of the output file.
    """
    logger = logger or logging.getLogger(__name__)

    path = Path(parameters.pkgpath) / 'parameters.json'
    if os.path.isdir(parameters.pkgpath):
        logger.info('Validating configurations in existing file %s', parameters.pkgpath)
        with open(path, 'r', encoding='utf-8') as source:
            params = Parameters.model_validate_json(source.read())

        if params != parameters:
            raise RuntimeError('Saved parameters do not match')  # Should show where

    else:
        logger.info('Creating a new file %s', parameters.pkgpath)
        os.makedirs(parameters.pkgpath)
        path = Path(parameters.pkgpath) / 'parameters.json'
        with open(path, 'w', encoding='utf-8') as out:
            out.write(parameters.model_dump_json())
