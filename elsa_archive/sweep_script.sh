#!/bin/sh
# SBATCH directives
#SBATCH -J elsa-rebuttal
#SBATCH -o ./out/%j.out  # Output file
##SBATCH -o ./out/%j.out
#SBATCH -t 3-00:00:00  # Run time (D-HH:MM:SS)

#### Select GPU
#SBATCH -p A100              # Partition
##SBATCH -p 3090              # Partition
##SBATCH -p A6000
#SBATCH --nodes=1            # Number of nodes
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2    # Number of CPUs
#SBATCH --gres=gpu:4         # Number of GPUs

cd /scratch/$USER/gpa

git pull

srun -I /bin/hostname
srun -I /bin/pwd
srun -I /bin/date

## Load modules
module purge
module load cuda/13.0.2

## Python Virtual Environment
echo "Start"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export project_name="rebuttal_elsa" # W&B Project Name
export agent="2zhx5x01" # W&B Sweep Agent ID
export env="gpa" # conda environment name

echo "source $HOME/anaconda3/etc/profile.d/conda.sh"
source /opt/anaconda3/2022.05/etc/profile.d/conda.sh    # Anaconda path


echo "conda activate $env"
conda activate $env    # Activate conda environment

# Run W&B Sweep Agent
srun wandb agent kwanheelee-postech/$project_name/$agent

date

echo "conda deactivate $env"
conda deactivate    # Deactivate environment

squeue --job $SLURM_JOBID

echo "##### END #####"
