#!/bin/bash
#SBATCH --job-name=TRM_FIN
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16       # Match the 16 cores on your boston-lab node
#SBATCH --gres=gpu:1             # Request the single L40S GPU
#SBATCH --mem=60G                # Request almost all the 64GB RAM
#SBATCH --time=48:00:00          # Maximum allowed time

python trm_model.py
