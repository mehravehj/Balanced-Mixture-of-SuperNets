#!/usr/bin/env bash

#SBATCH --account=def-mpederso       
#SBATCH --cpus-per-task=6                
#SBATCH --gres=gpu:1                     
#SBATCH --mem=25G                        
#SBATCH --time=1:00:00               

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURM_TMPDIR
echo "SLURM_ARRAYID="$SLURM_ARRAYID
echo "SLURM_ARRAYID"=$SLURM_ARRAYID
echo "SLURM_ARRAY_JOB_ID"=$SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID"=$SLURM_ARRAY_TASK_ID
echo "working directory = "$SLURM_SUBMIT_DIR

cp val.tar.gz $SLURM_TMPDIR
tar -xf $SLURM_TMPDIR/val.tar.gz -C $SLURM_TMPDIR
rm $SLURM_TMPDIR/val.tar.gz
cd $SLURM_TMPDIR/val
ls

cd $SLURM_SUBMIT_DIR

export IMAGENET_DIR=$SLURM_TMPDIR
export WRITE_DIR=$SLURM_TMPDIR

./write_imagenet.sh 500 0.5 90
cp val_500_0.5_90.ffcv $SLURM_SUBMIT_DIR


