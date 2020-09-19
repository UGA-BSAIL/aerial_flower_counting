#!/bin/bash

# Submission script that adds a job for model training.
#
# This script should be submitted from the root of this repository on Sapelo.
# It expects that a valid virtualenv has already been created with
# `poetry install`.

#PBS -S /bin/bash
#PBS -q patterli_q
#PBS -N cotton_count_model_train
#PBS -l nodes=1:ppn=6:gpus=1
#PBS -l walltime=2:00:00
#PBS -l mem=10gb
#PBS -M daniel.petti@uga.edu
#PBS -m ae

cd $PBS_O_WORKDIR

# Load needed modules.
ml Python/3.7.4-GCCcore-8.3.0
ml CUDA/10.0.130
ml cuDNN/7.6.5.32-CUDA-10.0.130

# Run the training.
poetry run kedro run --pipeline model_training
