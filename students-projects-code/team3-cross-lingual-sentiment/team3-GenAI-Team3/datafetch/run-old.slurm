#!/bin/bash

#SBATCH --time=32:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=standard
#SBATCH -J "get-nlp-dataset:fr"
#SBATCH --constraint=rivanna
#SBATCH --account=qian-iot-lab
#SBATCH --mem=32G
#SBATCH --output=slurm-logs/%x-%a.log 

# Load Apptainer module
module load apptainer

# Define working directory and container
WORK_DIR=/home/mpp7ez/scratch/nlp-project
CONTAINER=${WORK_DIR}/news_crawler.sif
BIND_DIRS="--bind ${WORK_DIR}:/mnt"

# Set environment variable for cache (if needed)
export NEWSPAPER_CACHE_DIR=/home/mpp7ez/scratch/newspaper_cache

# Define languages array
language="fr"


# Run the container with the specified language
apptainer run ${BIND_DIRS} ${CONTAINER} --lang ${language}