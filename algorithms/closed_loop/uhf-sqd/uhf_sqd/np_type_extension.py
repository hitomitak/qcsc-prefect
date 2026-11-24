"""Strict NumPy array types of the walker.

From kunalroy332/qcsc-prefect at 3968e58, ``algorithms/sbd/sbd/np_type_extension.py``.
"""

from typing import Annotated

import numpy as np
from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation

NpStrict2DArrayBool = Annotated[
    np.ndarray[tuple[int, int], np.dtype[np.bool]],
    NpArrayPydanticAnnotation.factory(data_type=np.bool, dimensions=2, strict_data_typing=True),
]

NpStrict1DArrayLL = Annotated[
    np.ndarray[tuple[int], np.dtype[np.longlong]],
    NpArrayPydanticAnnotation.factory(data_type=np.longlong, dimensions=1, strict_data_typing=True),
]
