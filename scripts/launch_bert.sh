#!/usr/bin/env bash

mkdir results
sbatch --partition=m40-short --gres=gpu:4 --mem=16384 --output=results/bert_%j.log scripts/bert.sh