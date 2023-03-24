#!/usr/bin/env bash

#SBATCH --account=rrg-mpederso 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:1  
#SBATCH --cpus-per-task=8   
#SBATCH --mem=15G       
#SBATCH --time=3:00:00  
#SBATCH --job-name mm450
#SBATCH --output=output_mm_%j.txt            

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

cp cifar100_train_50.beton $SLURM_TMPDIR
cp cifar100_val_50.beton $SLURM_TMPDIR

python main.py -dd $SLURM_TMPDIR -nm 1 -b 256 -e 401 -wr 8 --seed 12 -mtp 200 -lr 0.05 -mlr 0.0001 -wd 0.0001 -tn 1


