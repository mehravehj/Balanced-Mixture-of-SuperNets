#!/usr/bin/env bash

#SBATCH --account=rrg-mpederso 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:1  
#SBATCH --cpus-per-task=8   
#SBATCH --mem=15G       
#SBATCH --time=04:00:00  
#SBATCH --job-name c3 
#SBATCH --output=output_base__%j.txt 

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURM_TMPDIR
echo "SLURM_ARRAYID="$SLURM_ARRAYID
echo "SLURM_ARRAYID"=$SLURM_ARRAYID
echo "SLURM_ARRAY_JOB_ID"=$SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID"=$SLURM_ARRAY_TASK_ID
echo "working directory = "$SLURM_SUBMIT_DIR


module load python/3.9
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch torchvision --no-index
pip install pandas numpy --no-index
pip install torchmetrics==0.10.3
pip install opencv-python-headless
pip install numba
pip install cupy
pip install ffcv

cp cifar100_train.beton $SLURM_TMPDIR
cp cifar100_test.beton $SLURM_TMPDIR

python train_original_baseline_ffcv100.py -t $SLURM_TMPDIR/cifar100_train.beton  -v $SLURM_TMPDIR/cifar100_test.beton -j 8 --epochs 400 -pn 46

