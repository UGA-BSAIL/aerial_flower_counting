#!/bin/bash

# Submission script that adds a job for model training.
#
# This script should be submitted from the root of this repository on Sapelo.
# It expects that a valid virtualenv has already been created with
# `poetry install`.

#SBATCH --partition=patterli_p
#SBATCH -J cotton_count_model_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=16gb
#SBATCH --mail-user=daniel.petti@uga.edu
#SBATCH --mail-type=END,FAIL

set -e

# Base directory we use for job output.
OUTPUT_BASE_DIR="/scratch/$(whoami)"
# Directory where our data and venv are located.
LARGE_FILES_DIR="/work/cylilab/cotton_counter"

function prepare_environment() {
  # Create the working directory for this job.
  job_dir="${OUTPUT_BASE_DIR}/job_${SLURM_JOB_ID}"
  mkdir "${job_dir}"
  echo "Job directory is ${job_dir}."

  # Copy the code.
  cp -Rd "${SLURM_SUBMIT_DIR}/"* "${job_dir}/"
  # Manually copy the config file too.
  cp "${SLURM_SUBMIT_DIR}/.kedro.yml" "${job_dir}/"

  # Link to the input data directory and venv.
  rm -rf "${job_dir}/data"
  ln -s "${LARGE_FILES_DIR}/data" "${job_dir}/data"
  ln -s "${LARGE_FILES_DIR}/.venv" "${job_dir}/.venv"

  # Create output directories.
  mkdir "${job_dir}/output_data"

  # Set the working directory correctly for Kedro.
  cd "${job_dir}"
}

# Prepare the environment.
prepare_environment

# Load needed modules.
ml Python/3.8.2-GCCcore-8.3.0
ml CUDA/10.1.243-GCC-8.3.0
ml cuDNN/7.6.4.38-gcccuda-2019b

# Set this for deterministic runs. For more info, see
# https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
export PYTHONHASHSEED=0

# Run the training.
poetry run kedro run -e categorical "$@"
