#!/bin/bash

# Helper script that packages artifacts from a training run.

mkdir artifacts

# Grab the models and reports
zip -r artifacts/models.zip data/06_models/
zip -r artifacts/reports.zip data/08_reporting/

# Grab the logs.
zip -r artifacts/logs.zip logs/

# Grab the job output.
zip artifacts/output.zip cotton_count_model_train*
