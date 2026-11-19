"""pytest plugin: run prefect_test_harness tests on the in-process Prefect API (no sockets).

Local verification only (the sandbox forbids bind()); CI uses the real prefect_test_harness.
Load with: python -I -c "import sys; sys.path[:0] = [PR0_HARNESS, THIS_DIR]; import pytest;
sys.exit(pytest.main(['-p', 'inprocess_plugin', ...]))"
"""

import contextlib

import inprocess_prefect  # noqa: F401  (must precede the Prefect imports of the tests)
import prefect.testing.utilities as _utilities

_utilities.prefect_test_harness = lambda *args, **kwargs: contextlib.nullcontext()
