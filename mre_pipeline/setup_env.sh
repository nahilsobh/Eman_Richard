#!/usr/bin/env bash
set -e
conda env create -f environment.yml
conda activate mre_pipeline
echo "Environment ready. Run: conda activate mre_pipeline"
