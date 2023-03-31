#!/usr/bin/env bash

#SBATCH --account=def-mpederso       
#SBATCH --cpus-per-task=6                
#SBATCH --gres=gpu:1                     
#SBATCH --mem=25G                        
#SBATCH --time=8:00:00               

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURM_TMPDIR
echo "SLURM_ARRAYID="$SLURM_ARRAYID
echo "SLURM_ARRAYID"=$SLURM_ARRAYID
echo "SLURM_ARRAY_JOB_ID"=$SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID"=$SLURM_ARRAY_TASK_ID
echo "working directory = "$SLURM_SUBMIT_DIR

cp ../data/cifar-10-python.tar.gz $SLURM_TMPDIR
tar -xf $SLURM_TMPDIR/cifar-10-python.tar.gz -C $SLURM_TMPDIR
python train_multires.py -b 256 -tn 62 --seed 501 -e 800 -lr 0.1 -mlr 0.00001 -wd 0.0001 -wr 6 -vp 0.5 -alr 0.01 -awd 0.001 -dd $SLURM_TMPDIR/

