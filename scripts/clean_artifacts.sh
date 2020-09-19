#!/bin/bash

# Cleans up any old artifacts that are sitting on the disk.

# Remove zipped artifacts.
rm -rf artifacts/

# Remove old models and reports.
rm -rf data/06_models/
rm -rf data/08_reporting/

# Remove old logs.
rm -rf logs/*

# Remove old job output.
rm cotton_count_model_train*
