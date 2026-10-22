ssh -i ~/.ssh/id_ed25519 hitomi@slurm-vm1.vpc.cloud9.ibm.com 'mkdir -p /shared/qcsc-prefect'
rsync -avR --exclude='__pycache__/' --exclude='.pytest_cache/' --exclude='.env' \
  packages/qcsc-prefect-core \
  packages/qcsc-prefect-blocks \
  packages/qcsc-prefect-adapters \
  packages/qcsc-prefect-executor \
  examples/prefect_bitcount_demo \
  hitomi@slurm-vm1.vpc.cloud9.ibm.com:/shared/qcsc-prefect/

