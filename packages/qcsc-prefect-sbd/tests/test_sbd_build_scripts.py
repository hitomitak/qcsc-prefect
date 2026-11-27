"""Build presets must select the requested API revision in reused SBD checkouts."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

NATIVE = Path(__file__).resolve().parents[1] / "native"
SCRIPTS = ["build_sbd.sh", "build_sbd_fugaku.sh", "build_sbd_gpu.sh", "build_sbd_slurm.sh"]
CPU_SCRIPTS = ["build_sbd.sh", "build_sbd_fugaku.sh", "build_sbd_slurm.sh"]
# The drivers each script builds: (restricted, UHF=1).
DRIVERS = {script: ("diag", "diag_uhf") for script in CPU_SCRIPTS}
DRIVERS["build_sbd_gpu.sh"] = ("diag-gpu", "diag-gpu_uhf")


def git(directory, *arguments):
    return subprocess.check_output(
        [
            "git",
            "-c",
            "user.name=SBD test",
            "-c",
            "user.email=sbd-test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(directory),
            *arguments,
        ],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("missing_revision", [False, True])
def test_build_reused_checkout_at_requested_revision(tmp_path, script, missing_revision):
    run_build(tmp_path, script, missing_revision=missing_revision)


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("uhf", [False, True])
def test_uhf_builds_the_unrestricted_driver_only_when_requested(tmp_path, script, uhf):
    native, calls = run_build(tmp_path, script, uhf=uhf)
    compile_call = next(call for call in calls if "-c" in call)
    assert ("-D_UHF" in compile_call) == uhf
    # The recording compiler is not nvc++, so no nvc++ option is added.
    assert "--diag_suppress" not in compile_call
    restricted, unrestricted = DRIVERS[script]
    assert (native / unrestricted).exists() == uhf
    assert (native / restricted).exists() == (not uhf)


def test_gpu_build_points_an_open_mpi_wrapper_at_nvcpp(tmp_path):
    # hpcx's mpic++ calls g++ unless OMPI_CXX says otherwise; with nvc++ on PATH the script
    # sets OMPI_CXX=nvc++, and the wrapper then calls nvc++.
    native, calls = run_build(tmp_path, "build_sbd_gpu.sh", uhf=True, wrapper="open-mpi")
    wrapper_calls = [call for call in calls if call[0].startswith("OMPI_CXX=")]
    assert [call[0] for call in wrapper_calls] == ["OMPI_CXX=nvc++"] * 2  # compile and link
    assert "-c" in wrapper_calls[0]
    # With nvc++ behind the wrapper, the harmless nvc++ warnings are suppressed.
    assert "--diag_suppress" in wrapper_calls[0]
    assert (native / "diag-gpu_uhf").exists()


def test_gpu_build_stops_when_the_wrapper_does_not_call_nvcpp(tmp_path):
    run = run_build(tmp_path, "build_sbd_gpu.sh", uhf=True, wrapper="g++", expect_failure=True)
    assert "'mpic++' calls 'g++', but CCFLAGS are NVIDIA HPC SDK flags" in run.stderr
    assert "Using SBD commit" not in run.stdout


# The files of the pinned SBD that the multi-GPU build checks for.
MULTI_GPU_SBD = {
    "include/sbd/framework/nccl_utility.h": "// NCCL helpers\n",
    "include/sbd/chemistry/basic/davidson_thrust.h": "if( h_size == 1 ) {}\n",
    "include/sbd/chemistry/tpb/helper.h": "cr.reserve(2);\n",
}
NCCL_DEFINES = [
    "-DSBD_USE_NCCL",
    "-DSBD_NON_CUDA_AWARE_MPI",
    "-DSBD_THRUST_SAFE_MPI_ALLREDUCE",
    "-DSBD_USE_RANK_DISTRIBUTION",
    "-DSBD_USE_BLOCK_RANK_DISTRIBUTION",
]


def nvhpc_with_nccl(tmp_path):
    """An NVIDIA HPC SDK tree with NCCL's header and library, as MPI=1 looks them up."""
    root = tmp_path / "nvhpc"
    include = root / "comm_libs/13.0/nccl/include"
    lib = root / "comm_libs/13.0/nccl/lib"
    include.mkdir(parents=True)
    lib.mkdir(parents=True)
    (include / "nccl.h").write_text("")
    (lib / "libnccl.so").write_text("")
    return root, include, lib


@pytest.mark.parametrize("uhf", [False, True])
def test_mpi_builds_the_multi_gpu_driver_with_nccl(tmp_path, uhf):
    root, include, lib = nvhpc_with_nccl(tmp_path)
    native, calls = run_build(
        tmp_path,
        "build_sbd_gpu.sh",
        uhf=uhf,
        wrapper="open-mpi",
        sbd_files=MULTI_GPU_SBD,
        environment_overrides={"MPI": "1", "NVHPC_ROOT": str(root)},
    )
    compile_call, link_call = [call[1:] for call in calls if call[0].startswith("OMPI_CXX=")]
    assert "-c" in compile_call
    for define in NCCL_DEFINES:
        assert define in compile_call
    assert f"-I{include}" in compile_call
    assert ("-D_UHF" in compile_call) == uhf
    assert "-DSBD_USE_CUBLAS" not in compile_call
    assert f"-L{lib}" in link_call and "-lnccl" in link_call
    assert f"-Wl,-rpath,{lib}" in link_call
    driver = "diag-gpu_uhf-mpi" if uhf else "diag-gpu-mpi"
    assert sorted(p.name for p in native.iterdir() if p.name.startswith("diag")) == [driver]


def test_mpi_build_stops_without_nccl(tmp_path):
    run = run_build(
        tmp_path,
        "build_sbd_gpu.sh",
        uhf=True,
        wrapper="open-mpi",
        sbd_files=MULTI_GPU_SBD,
        environment_overrides={"MPI": "1", "NVHPC_ROOT": str(tmp_path / "no-nccl")},
        expect_failure=True,
    )
    assert "MPI=1: NCCL not found" in run.stderr
    assert "Using SBD commit" not in run.stdout


def test_mpi_build_stops_on_an_sbd_without_the_multi_gpu_fixes(tmp_path):
    root, _, _ = nvhpc_with_nccl(tmp_path)
    run = run_build(
        tmp_path,
        "build_sbd_gpu.sh",
        uhf=True,
        wrapper="open-mpi",
        environment_overrides={"MPI": "1", "NVHPC_ROOT": str(root)},
        expect_failure=True,
    )
    for missing in ("framework/nccl_utility.h", "PR79", "PR84"):
        assert missing in run.stderr
    assert not list((tmp_path / "native").glob("diag*"))


# A compiler whose object records the build (uhf or restricted) and whose link copies the
# object. The UHF link waits until the restricted compile has finished, which is the order in
# which a shared main.o makes the UHF driver link the restricted object.
ORDERING_COMPILER = """#!/bin/sh
[ "$1" = "-show" ] && exit 0
compile=0; uhf=0; out=""; obj=""; prev=""
for a in "$@"; do
  case "$a" in -c) compile=1 ;; -D_UHF) uhf=1 ;; esac
  if [ "$prev" = "-o" ]; then out="$a"; else case "$a" in *.o) obj="$a" ;; esac; fi
  prev="$a"
done
kind=restricted; [ "$uhf" = 1 ] && kind=uhf
if [ "$compile" = 1 ]; then
  echo "$kind" > "$out"; touch "$SIGNALS/$kind-compiled"
else
  if [ "$kind" = uhf ]; then
    i=0
    while [ ! -e "$SIGNALS/restricted-compiled" ] && [ "$i" -lt 400 ]; do
      sleep 0.05; i=$((i + 1))
    done
  fi
  cp "$obj" "$out"
fi
"""


@pytest.mark.parametrize("script", SCRIPTS)
def test_concurrent_builds_link_their_own_objects(tmp_path, script):
    # UHF=1 and UHF=0 builds started together, in an order that a shared main.o would mix up.
    remote = tmp_path / "remote"
    remote.mkdir()
    git(remote, "init", "-q")
    (remote / "api.txt").write_text("API\n")
    git(remote, "add", "api.txt")
    git(remote, "commit", "-qm", "API")
    dependency = tmp_path / "dependency"
    subprocess.run(["git", "clone", "-q", str(remote), str(dependency)], check=True)
    native = tmp_path / "native"
    native.mkdir()
    shutil.copy(NATIVE / script, native / script)
    shutil.copy(NATIVE / "main.cc", native / "main.cc")
    (native / "SBD_REVISION").write_text(git(remote, "rev-parse", "HEAD") + "\n")
    tools, signals = tmp_path / "bin", tmp_path / "signals"
    tools.mkdir()
    signals.mkdir()
    (tools / "module").write_text("#!/bin/sh\nexit 0\n")
    (tools / "module").chmod(0o755)
    (tools / "compiler").write_text(ORDERING_COMPILER)
    (tools / "compiler").chmod(0o755)
    for name in ("mpiicpc", "mpiicpx", "mpiFCCpx", "mpic++", "mpicxx"):
        (tools / name).symlink_to(tools / "compiler")
    environment = dict(
        os.environ,
        PATH=f"{tools}:{os.environ['PATH']}",
        SBD_DIR=str(dependency),
        SIGNALS=str(signals),
    )
    for name in ("SBD_COMMIT", "CCCOM", "UHF", "OMPI_CXX", "OMPI_CC", "WARNFLAGS"):
        environment.pop(name, None)

    def start(uhf):
        return subprocess.Popen(
            ["bash", str(native / script)],
            env={**environment, "UHF": "1" if uhf else "0"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    first = start(uhf=True)
    for _ in range(400):  # the UHF build has compiled; it now waits to link
        if (signals / "uhf-compiled").exists():
            break
        time.sleep(0.05)
    second = start(uhf=False)
    outputs = [first.communicate(timeout=60)[0], second.communicate(timeout=60)[0]]
    assert (first.returncode, second.returncode) == (0, 0), outputs
    restricted, unrestricted = DRIVERS[script]
    assert (native / unrestricted).read_text() == "uhf\n"
    assert (native / restricted).read_text() == "restricted\n"
    # Only the drivers are left in the native directory.
    assert sorted(p.name for p in native.iterdir()) == sorted(
        ["SBD_REVISION", script, "main.cc", restricted, unrestricted]
    )


def run_build(
    tmp_path,
    script,
    *,
    missing_revision=False,
    uhf=None,
    wrapper=None,
    expect_failure=False,
    sbd_files=None,
    environment_overrides=None,
):
    """Run ``script`` with a recording compiler; return the build dir and the compiler calls.

    ``wrapper`` replaces mpic++ by a wrapper that answers ``-show``: ``"open-mpi"`` calls
    ``$OMPI_CXX`` (g++ by default) and puts an nvc++ on PATH; ``"g++"`` always calls g++.
    ``sbd_files`` (path -> text) are added to the requested SBD revision, and
    ``environment_overrides`` to the build's environment.
    With ``expect_failure`` the failed run is returned instead.
    """
    remote = tmp_path / "remote"
    remote.mkdir()
    git(remote, "init", "-q")
    marker = remote / "api.txt"
    marker.write_text("old API\n")
    git(remote, "add", "api.txt")
    git(remote, "commit", "-qm", "old API")
    old = git(remote, "rev-parse", "HEAD")
    marker.write_text("current API\n")
    for relative, text in (sbd_files or {}).items():
        (remote / relative).parent.mkdir(parents=True, exist_ok=True)
        (remote / relative).write_text(text)
    git(remote, "add", "-A")
    git(remote, "commit", "-qm", "current API")
    requested = git(remote, "rev-parse", "HEAD")

    dependency = tmp_path / "dependency"
    dependency.mkdir()
    git(dependency, "init", "-q")
    git(dependency, "remote", "add", "origin", str(remote))
    if missing_revision:
        git(dependency, "fetch", "--depth=1", "origin", old)
        git(dependency, "checkout", "--detach", "FETCH_HEAD")
    else:
        git(dependency, "fetch", "origin")
        git(dependency, "checkout", "--detach", old)

    native = tmp_path / "native"
    native.mkdir()
    shutil.copy(NATIVE / script, native / script)
    shutil.copy(NATIVE / "main.cc", native / "main.cc")
    # Set the default revision file, rather than overriding SBD_COMMIT, so the test
    # checks that every preset uses the common default as well as checking out it.
    (native / "SBD_REVISION").write_text(requested + "\n")
    tools = tmp_path / "bin"
    tools.mkdir()
    (tools / "module").write_text("#!/bin/sh\nexit 0\n")
    compiler = tools / "compiler"
    calls = tmp_path / "compiler_calls.txt"
    compiler.write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> "{calls}"\n'
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = "-o" ]; then shift; touch "$1"; fi\n'
        "  shift\n"
        "done\n"
    )
    for executable in (tools / "module", compiler):
        executable.chmod(0o755)
    for name in ("mpiicpc", "mpiFCCpx", "mpic++", "mpicxx"):
        if not (wrapper and name == "mpic++"):
            (tools / name).symlink_to(compiler)
    if wrapper:
        backend = "${OMPI_CXX:-g++}" if wrapper == "open-mpi" else "g++"
        (tools / "mpic++").write_text(
            "#!/bin/sh\n"
            f'if [ "$1" = "-show" ]; then echo "{backend} -I/opt/mpi/include"; exit 0; fi\n'
            f'echo "OMPI_CXX=$OMPI_CXX $*" >> "{calls}"\n'
            f'exec "{compiler}" "$@"\n'
        )
        (tools / "mpic++").chmod(0o755)
        if wrapper == "open-mpi":
            (tools / "nvc++").symlink_to(compiler)
    environment = dict(os.environ, PATH=f"{tools}:{os.environ['PATH']}", SBD_DIR=str(dependency))
    environment.pop("SBD_COMMIT", None)
    environment.pop("CCCOM", None)
    for name in ("UHF", "MPI", "OMPI_CXX", "OMPI_CC", "NVHPC_ROOT", "NCCL_INCLUDE", "NCCL_LIB"):
        environment.pop(name, None)
    if uhf:
        environment["UHF"] = "1"
    environment.update(environment_overrides or {})
    run = subprocess.run(
        ["bash", str(native / script)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if expect_failure:
        assert run.returncode != 0, run.stdout
        return run
    assert run.returncode == 0, run.stdout + run.stderr
    assert git(dependency, "rev-parse", "HEAD") == requested
    assert (dependency / "api.txt").read_text() == "current API\n"
    assert f"Using SBD commit: {requested}" in run.stdout
    if uhf is None:
        expected_binary = "diag-gpu" if script == "build_sbd_gpu.sh" else "diag"
        assert (native / expected_binary).exists()
    return native, [line.split() for line in calls.read_text().splitlines()]
