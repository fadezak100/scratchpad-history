# Source this inside a Slurm job, from the project folder:
#     source scripts/cluster/setup_env.sh
# Builds the virtualenv on the compute node's local disk ($SLURM_TMPDIR), which is much faster than
# the network file systems (home, scratch), and installs from the cluster's own wheels (--no-index),
# so no internet access is needed. The environment is rebuilt in each job (about a minute) and is
# deleted with the job.
ENV_DIR="${SLURM_TMPDIR:?run this inside a Slurm job}/env"
module load python/3.11
virtualenv --no-download "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install --no-index --quiet --upgrade pip
pip install --no-index --quiet torch transformers numpy matplotlib pytest
# The code itself is used straight from the project folder; nothing is installed from it.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"
