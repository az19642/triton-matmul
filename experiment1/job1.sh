#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10000M
#SBATCH --array=1,2,4,8,16,32,64,128
#SBATCH --time=00-03:00:00
#SBATCH --account=rrg-mmehride

nvidia-smi > nvidia-smi.txt

module load python/3.11.5
module load cuda
module load cudnn

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index $SCRATCH/triton-matmul/nvidia_cutlass-3.6.0.0-py3-none-any.whl
pip install --no-index -r $SCRATCH/triton-matmul/requirements.txt

python experiment1.py $SLURM_ARRAY_TASK_ID

cat $SLURM_TMPDIR/autotuning.out >> ./autotuning.out

mkdir -p ./gsm-k-autotuned_matmul_perf
cp -r $SLURM_TMPDIR/gsm-k-autotuned_matmul_perf/* ./gsm-k-autotuned_matmul_perf/
