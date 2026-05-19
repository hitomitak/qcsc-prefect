# PyPI Release Guide

This guide describes the maintainer release process for the `qiskit-community/qcsc-prefect`
monorepo.

The repository publishes multiple PyPI packages from `packages/`. The root `pyproject.toml`
is only a workspace coordinator and is not published.

## Release Model

All `qcsc-prefect` packages are released together with the same version.

Publishable packages:

- `qcsc-prefect`
- `qcsc-prefect-core`
- `qcsc-prefect-blocks`
- `qcsc-prefect-adapters`
- `qcsc-prefect-executor`
- `qcsc-prefect-qiskit`
- `qcsc-prefect-dice`

The `qcsc-prefect` package is a meta-package. A normal install includes the core packages:

```bash
python -m pip install "qcsc-prefect==<version>"
```

Optional integrations are installed with extras:

```bash
python -m pip install "qcsc-prefect[qiskit]==<version>"
python -m pip install "qcsc-prefect[dice]==<version>"
python -m pip install "qcsc-prefect[all]==<version>"
```

The DICE/SBD executable is not installed by `pip`. The `qcsc-prefect-dice` package only
installs the Python-side integration. The solver executable must be built separately for
the target HPC environment.

## Maintainer Prerequisites

Before starting a release, confirm the following repository settings and PyPI project
settings are in place.

1. GitHub Actions can push release branches.

   In the GitHub repository, check:

   `Settings` -> `Actions` -> `General` -> `Workflow permissions`

   The workflow needs read and write permission for repository contents because
   `prepare-release.yml` creates and pushes a `release/v<version>` branch.

   The setting that allows GitHub Actions to create pull requests is not required for the
   current workflow. Pull requests are opened manually.

2. PyPI Trusted Publishing is configured.

   Each PyPI project must have a Trusted Publisher entry for this repository and workflow.
   Normal releases use OIDC through `pypa/gh-action-pypi-publish`; do not add PyPI API tokens,
   username/password credentials, or repository secrets for publishing.

3. The `pypi` GitHub environment is configured.

   The PyPI publish job uses:

   ```yaml
   environment:
     name: pypi
   ```

   If the environment requires manual approval, a maintainer must approve it before the
   publish step runs.

## Step 1: Choose the Release Version

Pick the next PEP 440 version, for example `0.1.1`.

All package versions and all internal `qcsc-prefect-*` dependency pins must use this same
version for the release.

Use a normal final version for PyPI releases:

```text
0.1.1
```

Use a post-release version only for temporary TestPyPI validation branches:

```text
0.1.1.post1
```

## Step 2: Run a Build-Only TestPyPI Dry Run

This step is optional but recommended before preparing a release PR.

1. Open the `Publish packages to TestPyPI` workflow in GitHub Actions.
2. Click `Run workflow`.
3. Leave `publish` set to the default value:

   ```text
   false
   ```

4. Start the workflow.

With `publish=false`, the workflow builds distributions, runs `twine check`, runs smoke tests,
and uploads the `dist/` artifact. It does not publish anything to TestPyPI.

Use this as a safe CI dry run when you want to validate the packaging workflow without touching
TestPyPI.

## Step 3: Run the Prepare Release Workflow

Run the `Prepare release` workflow manually.

Inputs:

- `old_version`: the current version in `packages/*/pyproject.toml`
- `new_version`: the version you want to release

Example:

```text
old_version: 0.1.0
new_version: 0.1.1
```

The workflow performs these actions:

1. Checks out `main`.
2. Runs `scripts/bump-version.py <old_version> <new_version>`.
3. Updates every `packages/*/pyproject.toml` version.
4. Updates internal `qcsc-prefect-*` dependency pins.
5. Builds all packages with `scripts/build-all-packages.sh`.
6. Runs `twine check` on all generated distributions.
7. Creates clean virtual environments.
8. Installs `qcsc-prefect==<new_version>` from local distributions and runs smoke tests.
9. Installs `qcsc-prefect[qiskit]==<new_version>` from local distributions and runs smoke tests.
10. Creates and pushes a release branch:

    ```text
    release/v<new_version>
    ```

The workflow does not publish to PyPI and does not create a pull request.

## Step 4: Open the Release Pull Request Manually

After the prepare workflow completes, open a pull request manually from:

```text
release/v<new_version>
```

into:

```text
main
```

Use this title:

```text
Release v<new_version>
```

Recommended PR body:

```markdown
Release preparation for `v<new_version>`.

- Updated all qcsc-prefect package versions.
- Updated internal dependency pins.
- Built all distributions.
- Ran twine check.
- Ran smoke tests.

After this PR is merged, publish by pushing tag `v<new_version>`.
```

Review the PR before merging. The PR should contain version and internal dependency pin changes
only, unless the release intentionally includes other already-reviewed changes from `main`.

## Step 5: Merge the Release Pull Request

Merge the release PR into `main`.

After merging, make sure your local `main` is up to date:

```bash
git switch main
git pull --ff-only origin main
```

## Step 6: Create and Push the Release Tag

Create a tag that exactly matches the release version with a leading `v`.

Example:

```bash
git tag v0.1.1
git push origin v0.1.1
```

The tag version must match every package version. For example, tag `v0.1.1` requires:

```toml
version = "0.1.1"
```

in every `packages/*/pyproject.toml`.

The `Publish packages to PyPI` workflow runs automatically when the tag is pushed.

## Step 7: Approve and Monitor the PyPI Publish Workflow

The `Publish packages to PyPI` workflow performs these checks before publishing:

1. Runs `scripts/check-release-version.py --tag "${GITHUB_REF_NAME}"`.
2. Builds all packages with `scripts/build-all-packages.sh`.
3. Runs `twine check dist/*`.
4. Installs `qcsc-prefect==<version>` from the generated distributions.
5. Runs `scripts/smoke-test-packages.py`.
6. Installs `qcsc-prefect[qiskit]==<version>` from the generated distributions.
7. Runs `scripts/smoke-test-packages.py` again.
8. Uploads the `dist/` artifact.
9. Publishes with PyPI Trusted Publishing after the `pypi` environment gate is approved.

If the `pypi` environment requires approval, approve the deployment only after the build job
has passed.

No PyPI API token is used. The publish job uses:

```yaml
permissions:
  id-token: write
```

and `pypa/gh-action-pypi-publish`.

## Step 8: Verify the Published Packages

After the publish job succeeds, verify installation from PyPI in a clean environment.

Core install:

```bash
python -m venv /tmp/qcsc-prefect-release-core
/tmp/qcsc-prefect-release-core/bin/python -m pip install --upgrade pip
/tmp/qcsc-prefect-release-core/bin/python -m pip install "qcsc-prefect==<version>"
/tmp/qcsc-prefect-release-core/bin/python -c "import qcsc_prefect_core, qcsc_prefect_blocks, qcsc_prefect_adapters, qcsc_prefect_executor"
```

All extras install:

```bash
python -m venv /tmp/qcsc-prefect-release-all
/tmp/qcsc-prefect-release-all/bin/python -m pip install --upgrade pip
/tmp/qcsc-prefect-release-all/bin/python -m pip install "qcsc-prefect[all]==<version>"
/tmp/qcsc-prefect-release-all/bin/python -c "import qcsc_prefect.integrations.qiskit"
```

The DICE integration import is checked by `scripts/smoke-test-packages.py` when
`qcsc-prefect-dice` is installed. This smoke test does not run DICE/SBD and does not require
the external executable.

## Local Validation Commands

Use these commands when validating a release locally before or after opening the release PR.

Install build tooling:

```bash
python -m pip install --upgrade build twine
```

Build all distributions:

```bash
bash scripts/build-all-packages.sh
```

Run `twine check`:

```bash
python -m twine check dist/*
```

Install from local distributions:

```bash
python -m venv /tmp/qcsc-prefect-local-release
/tmp/qcsc-prefect-local-release/bin/python -m pip install --upgrade pip
/tmp/qcsc-prefect-local-release/bin/python -m pip install --find-links dist "qcsc-prefect[all]==<version>"
/tmp/qcsc-prefect-local-release/bin/python scripts/smoke-test-packages.py
```

## TestPyPI Upload Validation

Use TestPyPI only when you need to validate Trusted Publishing upload behavior.

1. Create a temporary validation branch.
2. Bump versions to a TestPyPI-only version, such as:

   ```text
   0.1.1.post1
   ```

3. Run the `Publish packages to TestPyPI` workflow.
4. Set `publish` to:

   ```text
   true
   ```

5. Confirm the workflow passes.
6. Do not merge the temporary TestPyPI branch into the release PR.

TestPyPI and PyPI versions are immutable. If a TestPyPI upload already used a version, choose a
new post-release version for the next validation attempt.

## Troubleshooting

### The prepare workflow cannot push the release branch

Check the GitHub Actions workflow permission:

`Settings` -> `Actions` -> `General` -> `Workflow permissions`

The workflow needs read and write permission for repository contents.

Also check whether the branch already exists:

```text
release/v<version>
```

If it exists, delete or rename the stale branch before rerunning the workflow.

### GitHub Actions is not permitted to create or approve pull requests

The current release workflow does not create pull requests. It only pushes the release branch.

If this error appears, confirm that the workflow being run is the latest `prepare-release.yml`
from `main`, and that it no longer uses a pull-request creation action.

### The publish workflow fails with a version mismatch

The tag does not match one or more package versions or internal dependency pins.

Run:

```bash
python scripts/check-release-version.py --tag v<version>
```

Fix the package versions through the prepare release flow, merge the fix, and create a matching
tag.

### PyPI reports that a file or version already exists

PyPI package files and versions are immutable. Do not delete and recreate releases as the normal
fix.

For a broken release, prefer yanking the release on PyPI and publishing a new version.

### Trusted Publishing fails

Confirm that every PyPI project has a Trusted Publisher entry for this repository, workflow, and
environment.

For the production workflow, the environment name is:

```text
pypi
```

For the TestPyPI workflow, the environment name is:

```text
testpypi
```

Do not fix Trusted Publishing failures by adding API tokens to the repository.
