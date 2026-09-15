# Sourced by every job script: the modules and virtual environment built by slurm/setup_env.sh.
module load StdEnv/2023 python/3.11
source "${DSSL_VENV:-$HOME/venvs/dssl}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 MPLBACKEND=Agg
