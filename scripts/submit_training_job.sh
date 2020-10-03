#!/bin/bash

# Submission script that adds a job for model training.
#
# This script should be submitted from the root of this repository on Sapelo.
# It expects that a valid virtualenv has already been created with
# `poetry install`.

#PBS -S /bin/bash
#PBS -q patterli_q
#PBS -N cotton_count_model_train
#PBS -l nodes=1:ppn=10:gpus=1
#PBS -l walltime=8:00:00
#PBS -l mem=12gb
#PBS -M daniel.petti@uga.edu
#PBS -m ae

set -e

# Base directory we use for job output.
OUTPUT_BASE_DIR="/scratch/${PBS_O_LOGNAME}"
# Directory where our data and venv are located.
LARGE_FILES_DIR="/work/cylilab/cotton_counter"

function prepare_environment() {
  # Create the working directory for this job.
  job_dir="${OUTPUT_BASE_DIR}/job_${PBS_JOBID}"
  mkdir "${job_dir}"
  echo "Job directory is ${job_dir}."

  # Copy the code.
  cp -Rd "${PBS_O_WORKDIR}/"* "${job_dir}/"
  # Manually copy the config file too.
  cp "${PBS_O_WORKDIR}/.kedro.yml" "${job_dir}/"

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
ml Python/3.7.4-GCCcore-8.3.0
ml CUDA/10.0.130
ml cuDNN/7.6.5.32-CUDA-10.0.130

# Run the training.
poetry run kedro run --pipeline model_training -e sapelo -e categorical "$@"
