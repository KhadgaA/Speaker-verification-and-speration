#!/bin/bash
#SBATCH --job-name=process_kathbadh 	# Job name
#SBATCH --partition=small 		#Partition name can be test/small/medium/large/gpu/gpu2 #Partition “gpu or gpu2” should be used only for gpu jobs
#SBATCH --nodes=1 				# Run all processes on a single node
#SBATCH --ntasks=1 				# Run a single task
#SBATCH --cpus-per-task=8 		# Number of CPU cores per task
#SBATCH --gres=gpu  			# Include gpu for the task (only for GPU jobs)
#SBATCH --mem=6gb 				# Total memory limit (optional)
#SBATCH --output=./logs/first_%j.log 	# Standard output and error log
date;hostname;pwd
# which gpu node was used
echo "Running on host" $(hostname)
# print the slurm environment variables sorted by name
printenv | grep -i slurm | sort
module load anaconda/3
eval "$(conda shell.bash hook)"
conda activate speech_env

python structure.py "/scratch/data/m23csa003/kathbadh/kb_data_clean_m4a" "/scratch/data/m23csa003/kathbadh/kb_data_clean_wav" telugu