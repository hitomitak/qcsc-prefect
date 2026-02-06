MIYABI_INTEGRATION=1 \
MIYABI_PBS_QUEUE=regular-c \
MIYABI_PBS_PROJECT=gz09 \
MIYABI_TEST_WORK_DIR=/work/gz09/z30541/test \
PYTHONPATH=packages/hpc-prefect-core/src:packages/hpc-prefect-adapters/src:packages/hpc-prefect-executor/src \
pytest -q -m miyabi_integration \
  packages/hpc-prefect-executor/tests/test_run_miyabi_job_miyabi_integration.py
