"""PR3: a sample pool written by the port passes the source's own loader check (3968e58).

Step 1 (port venv):   python -I pool_loader_compat.py write TESTS_DIR OUT_DIR
Step 2 (PR0 venv, where the source package `sbd` is installed):
                      python -I pool_loader_compat.py check OUT_DIR
Step 2 reads the file with the source's load_ndarray / load_ndarray_optional and runs
sbd.sqd._check_saved_pool, unchanged, with the source's ElectronicProperties fields for an
FCIDUMP input (fcidump_sha256 and reference_tag None). It must accept the pool, and still refuse
it for a different electron number (to show that the check is active).
"""

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

mode = sys.argv[1]
if mode == "write":
    tests, out = Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(tests))
    import os

    from uhf_sqd import data_io, sqd

    data_io.S3Bucket = None
    os.environ["SBD_LOCAL_STORAGE_DIR"] = str(out)
    from test_uhf_sqd_pool import DeviceSource, device_samples
    from uhf_sqd_helpers import NA, NB, NORB, FakeSolver, run_walker

    class MP:  # minimal monkeypatch
        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    from uhf_sqd_helpers import prepare_walker

    sqd.create_lucj_circuit = lambda **kwargs: "lucj-circuit"
    prepare_walker(MP(), solver=FakeSolver(), source=DeviceSource(*device_samples()))
    _, telemetry = run_walker()
    path = telemetry["raw_samples_path"]
    (out / "pool.json").write_text(json.dumps({"path": path, "norb": NORB, "nelec": [NA, NB]}))
    print("written", path)
else:
    out = Path(sys.argv[2])
    info = json.loads((out / "pool.json").read_text())
    from sbd import sqd as source_sqd
    from sbd.data_io import load_ndarray, load_ndarray_optional

    print("source module:", source_sqd.__file__)
    path = info["path"]
    num_bits = int(load_ndarray(path, "num_bits")[0])
    ref_meta = load_ndarray_optional(path, "ref_meta")
    logger = logging.getLogger("check")
    logging.basicConfig(level=logging.INFO)

    def props(nelec):
        return SimpleNamespace(
            num_orbitals=info["norb"],
            num_electrons=tuple(nelec),
            reference_tag=None,
            reference_source=None,
            fcidump_sha256=None,
            unrestricted=True,
        )

    source_sqd._check_saved_pool(path, num_bits, ref_meta, props(info["nelec"]), logger)
    print("ACCEPTED by the source loader:", json.loads(str(ref_meta.reshape(-1)[0])))
    try:
        source_sqd._check_saved_pool(path, num_bits, ref_meta, props([info["nelec"][0], 0]), logger)
    except ValueError as exc:
        print("REFUSED for a different nelec, as expected:", str(exc)[:120])
    else:
        sys.exit("the source loader did not refuse a different reference")
