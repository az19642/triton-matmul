#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000M
#SBATCH --time=00-12:00:00
#SBATCH --account=rrg-mmehride

PROJ_DIR="$SCRATCH/triton-matmul"

nvidia-smi > nvidia-smi.txt

module load python/3.11.5
module load cuda
module load cudnn

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# install cutlass from pre-built wheel
pip install --no-index $PROJ_DIR/nvidia_cutlass-3.6.0.0-py3-none-any.whl

# install other dependencies from wheelhouse
pip install --no-index -r $PROJ_DIR/requirements.txt

python experiment1.py