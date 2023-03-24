#!/usr/bin/env bash

#SBATCH --account=rrg-mpederso 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1 
#SBATCH --cpus-per-task=8   
#SBATCH --mem=15G       
#SBATCH --time=05:00:00  
#SBATCH --job-name b4 
#SBATCH --output=output_base__%j.txt 

python write_dataset_50.py


