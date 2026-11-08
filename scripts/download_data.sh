#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " Starting Dataset Acquisition for Paper Reproduction Suite"
echo " Datasets: CIFAR-10, CIFAR-100, SVHN"
echo "=========================================================="

DATA_DIR=${1:-"./data/cache"}
mkdir -p "$DATA_DIR"

python3 data/download_data.py --dataset all --data_dir "$DATA_DIR"

echo "=========================================================="
echo " Dataset preparation successfully verified!"
echo "=========================================================="
