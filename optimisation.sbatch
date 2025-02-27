#!/bin/bash

#SBATCH --error=opt_%j.log
#SBATCH --output=opt_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=168:00:00

python run_optimisation.py
