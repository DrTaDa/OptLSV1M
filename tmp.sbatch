#!/bin/bash

#SBATCH --error=opt_%j.log
#SBATCH --output=opt_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00

python 'Plot spontaneous, OT and ST.py'
