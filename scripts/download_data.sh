#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " Starting Dataset Acquisition for Paper Reproduction Suite"
echo " Datasets: CIFAR-10, CIFAR-100, SVHN"
echo "=========================================================="

DATA_DIR=${1:-"./data/cache"}
mkdir -p "$DATA_DIR"

PYTHON_BIN="python3"
if ! command -v python3 &> /dev/null; then
    if command -v python &> /dev/null; then
        PYTHON_BIN="python"
    else
        echo "Error: Neither python3 nor python found in PATH." >&2
        exit 1
    fi
fi

echo "Using Python binary: $($PYTHON_BIN --version 2>&1)"
$PYTHON_BIN data/download_data.py --dataset all --data_dir "$DATA_DIR"

echo "=========================================================="
echo " Dataset preparation successfully verified!"
echo "=========================================================="
