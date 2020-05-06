#!/usr/bin/env bash

mkdir results
sbatch --partition=2080ti-short --gres=gpu:1 --mem=8192 --output=results/bert_%j.log scripts/bert.sh