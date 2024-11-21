#!/bin/bash -l   
# Request 1 hour wallclock time
#SBATCH -t 01:00:00

# Tell SLURM that this is an array job, with tasks numbered from 1 to 250
#SBATCH -a 1-250

# Load your software to run
module load miniforge
conda activate testing-pytesimint

# Run the application, passing in the input and output filenames
python task_array_script_ellipse_gradient_spvr.py $SLURM_ARRAY_TASK_ID "20220713_091851_randomised_params.csv" "/mnt/scratch/users/earmmu/first_tests/"