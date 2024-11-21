# single core submission script

#Run with current environment (-V) and in the current directory (-cwd)
#$ -V -cwd

#Request some time- min 15 mins - max 48 hours
#$ -l h_rt=40:00:00

#Request some memory per core
#$ -l h_vmem=8G

#Get email at start and end of the job
#$ -m be

# Tell SGE that this is an array job, with "tasks" numbered from 1 to 150
#$ -t 1-250

#Now run the job
module load anaconda
source activate example_env_1
python task_array_script_ellipse_gradient_spvr.py $SGE_TASK_ID "20220713_091851_randomised_params.csv" "/nobackup/eememq/spvr00/random00/"