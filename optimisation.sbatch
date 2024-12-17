#!/bin/bash

#SBATCH --error=opt_%j.log
#SBATCH --output=opt_%j.log
#SBATCH --nodes=1
#SBATCH --time=168:00:00

python run_optimisation.py
