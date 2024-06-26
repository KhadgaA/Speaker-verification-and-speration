#!/bin/bash
#SBATCH --job-name=wavlm_large 	# Job name
#SBATCH --partition=gpu2 		#Partition name can be test/small/medium/large/gpu/gpu2 #Partition “gpu or gpu2” should be used only for gpu jobs
#SBATCH --nodes=1 				# Run all processes on a single node
#SBATCH --ntasks=1 				# Run a single task
#SBATCH --cpus-per-task=4 		# Number of CPU cores per task
#SBATCH --gres=gpu:1  			# Include gpu for the task (only for GPU jobs)
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

srun -n 1 -c 1 --exclusive python eval_model.py --model wavlm_large --dataset kathbadh &> wavlm_large_kathbadh_full.txt &
srun -n 1 -c 1 --exclusive python eval_model.py --model hubert_large --dataset kathbadh &> hubert_large_kathbadh_full.txt &
srun -n 1 -c 1 --exclusive python eval_model.py --model wavlm_base_plus --dataset kathbadh &> wavlm_base_plus_kathbadh_full.txt &
wait
