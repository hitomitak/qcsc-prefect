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


def test_concurrent_gpu_builds_link_their_own_objects(tmp_path):
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
    shutil.copy(NATIVE / "build_sbd_gpu.sh", native / "build_sbd_gpu.sh")
    shutil.copy(NATIVE / "main.cc", native / "main.cc")
    (native / "SBD_REVISION").write_text(git(remote, "rev-parse", "HEAD") + "\n")
    tools, signals = tmp_path / "bin", tmp_path / "signals"
    tools.mkdir()
    signals.mkdir()
    (tools / "mpic++").write_text(ORDERING_COMPILER)
    (tools / "mpic++").chmod(0o755)
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
            ["bash", str(native / "build_sbd_gpu.sh")],
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
    assert (native / "diag-gpu_uhf").read_text() == "uhf\n"
    assert (native / "diag-gpu").read_text() == "restricted\n"
    # Only the drivers are left in the native directory.
    assert sorted(p.name for p in native.iterdir()) == [
        "SBD_REVISION",
        "build_sbd_gpu.sh",
        "diag-gpu",
        "diag-gpu_uhf",
        "main.cc",
    ]


def run_build(
    tmp_path, script, *, missing_revision=False, uhf=None, wrapper=None, expect_failure=False
):
    """Run ``script`` with a recording compiler; return the build dir and the compiler calls.

    ``wrapper`` replaces mpic++ by a wrapper that answers ``-show``: ``"open-mpi"`` calls
    ``$OMPI_CXX`` (g++ by default) and puts an nvc++ on PATH; ``"g++"`` always calls g++.
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
    git(remote, "commit", "-qam", "current API")
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
    for name in ("UHF", "OMPI_CXX", "OMPI_CC"):
        environment.pop(name, None)
    if uhf:
        environment["UHF"] = "1"
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
