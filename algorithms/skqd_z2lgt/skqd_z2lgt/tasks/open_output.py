"""Open the output HDF5 file."""
import os
import logging
from typing import Optional
from pathlib import Path
from pydantic import BaseModel
from skqd_z2lgt.parameters import Parameters


def compare_models(model1: BaseModel, model2: BaseModel, _current: str = ''):
    for field in type(model1).model_fields:
        value1 = getattr(model1, field)
        value2 = getattr(model2, field)
        if isinstance(value1, BaseModel):
            if (mismatch := compare_models(value1, value2, f'{_current}{field}.')):
                return mismatch
        elif value1 != value2:
            return f'{_current}{field}', value1, value2


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

        if (mismatch := compare_models(params, parameters)):
            raise RuntimeError(f'Saved parameters do not match at {mismatch[0]}:'
                               f' {mismatch[1]} != {mismatch[2]}')

    else:
        logger.info('Creating a new file %s', parameters.pkgpath)
        os.makedirs(parameters.pkgpath)
        path = Path(parameters.pkgpath) / 'parameters.json'
        with open(path, 'w', encoding='utf-8') as out:
            out.write(parameters.model_dump_json())
