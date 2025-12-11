#!/bin/bash
# runs all configurations of experiments.
python run_exp.py --normalization per-image
python run_exp.py --normalization per-image --col_normalize
python run_exp.py --normalization per-image --use_orth_proj
python run_exp.py --normalization per-image --use_orth_proj --col_normalize

echo "DONE"
