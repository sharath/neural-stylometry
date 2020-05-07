#!/usr/bin/env bash

mkdir output

configs=(0 7 8)

for config in ${configs[@]}
do
  sbatch --partition=m40-short --gres=gpu:4 --mem=16384 --output=output/%j.log scripts/bert.sh $config
done
