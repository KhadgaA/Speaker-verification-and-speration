#!/bin/bash
#SBATCH --job-name=wavlm_large 	# Job name
#SBATCH --partition=small 		#Partition name can be test/small/medium/large/gpu/gpu2 #Partition “gpu or gpu2” should be used only for gpu jobs
#SBATCH --nodes=1 				# Run all processes on a single node
#SBATCH --ntasks=1 				# Run a single task
#SBATCH --cpus-per-task=4 		# Number of CPU cores per task
#SBATCH --mem=16gb 				# Total memory limit (optional)
#SBATCH --output=./logs/first_%j.log 	# Standard output and error log
date;hostname;pwd
# which gpu node was used
echo "Running on host" $(hostname)

# print the slurm environment variables sorted by name
printenv | grep -i slurm | sort

module load anaconda/3
eval "$(conda shell.bash hook)"
conda activate speech_env
export TORCHAUDIO_USE_BACKEND_DISPATCHER=1 

python ran.py &> ran.txt 
