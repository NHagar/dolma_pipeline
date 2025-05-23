#!/bin/bash
#SBATCH --account=p32491  ## YOUR ACCOUNT pXXXX or bXXXX
#SBATCH --partition=normal  ### PARTITION (buyin, short, normal, etc)
#SBATCH --nodes=1 ## how many computers do you need
#SBATCH --ntasks-per-node=8 ## how many cpus or processors do you need on each computer
#SBATCH --time=48:00:00 ## how long does this need to run (remember different partitions have restrictions on this parameter)
#SBATCH --mem-per-cpu=8G ## how much RAM do you need per node (this effects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=redpj  ## When you run squeue -u NETID this is how you can identify the job
#SBATCH --mail-user=nicholas.hagar@northwestern.edu
#SBATCH --mail-type=ALL

module purge all

module load jq

uv run process_pipeline.py redpajama-data-v2