#!/usr/bin/env bash

mkdir results
sbatch --partition=1080ti-short --gres=gpu:4 --mem=16384 --output=results/bert_%j.log scripts/bert.sh
