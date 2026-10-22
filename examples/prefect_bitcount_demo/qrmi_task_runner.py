"""Legacy CommandBlock path forwarding to the installed QRMI sampler worker."""

from __future__ import annotations

import sys
from pathlib import Path

# Keep the SQD example hook importable when this file is run by absolute path
# from a shared job directory. Installed library code needs no checkout path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qcsc_prefect.integrations.qiskit.qrmi.worker import main  # noqa: E402

if __name__ == "__main__":
    main()
