#!/bin/bash
# One-time environment. Run on a Narval LOGIN node from the repository root (compute nodes
# have no internet):   bash slurm/setup_env.sh      (DSSL_VENV overrides ~/venvs/dssl)
module load StdEnv/2023 python/3.11
set -euo pipefail
venv="${DSSL_VENV:-$HOME/venvs/dssl}"
if [ -e "$venv" ]; then echo "$venv already exists; remove it or set DSSL_VENV" >&2; exit 1; fi
virtualenv --no-download "$venv"
source "$venv/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy pandas scipy scikit-learn matplotlib joblib
for package in statsmodels seaborn; do pip install --no-index "$package" || pip install "$package"; done
# Not in the Alliance wheelhouse. Its skopt/openpyxl requirements are unused by cosinor.fit_me.
pip install --no-deps CosinorPy==3.1
python -c "import torch, tasks.yan_cosinor; from CosinorPy import cosinor; print('environment ok: torch', torch.__version__)"
mkdir -p logs
pip freeze > logs/pip_freeze.txt
echo "Created $venv; package versions in logs/pip_freeze.txt"
