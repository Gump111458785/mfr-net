#!/usr/bin/env bash
set -e

# ==============================================================================
# Full Reproduction Pipeline for MFR-Net
# Covers CIFAR-10, CIFAR-100, SVHN benchmarks and Ablation Suite
# ==============================================================================

echo "=========================================================="
echo " 1. Checking CUDA Hardware Environment"
echo "=========================================================="
python3 -c "import torch; assert torch.cuda.is_available(), 'CRITICAL: CUDA GPU is required!'"
echo "CUDA Hardware detected: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"

echo ""
echo "=========================================================="
echo " 2. Downloading & Verifying Benchmark Datasets"
echo "=========================================================="
bash scripts/download_data.sh "./data/cache"

echo ""
echo "=========================================================="
echo " 3. Benchmarking Model Efficiency (Table 2 in Paper)"
echo "=========================================================="
python3 scripts/benchmark_efficiency.py

echo ""
echo "=========================================================="
echo " 4. Executing Main Visual Classification & Calibration Experiments (Table 1)"
echo "=========================================================="
DATASETS=("cifar10" "cifar100" "svhn")
MODELS=("resnet18" "resnet50" "mobilenet_v2" "mfr_net")

for d in "${DATASETS[@]}"; do
    for m in "${MODELS[@]}"; do
        echo "--> Running training & eval for: Model=$m on Dataset=$d"
        python3 scripts/train.py \
            --model "$m" \
            --dataset "$d" \
            --epochs 200 \
            --batch_size 128 \
            --output_dir "./checkpoints"
    done
done

echo ""
echo "=========================================================="
echo " 5. Running Ablation Study on CIFAR-100 (Table 3 in Paper)"
echo "=========================================================="
# (b) + MFR Module only
python3 scripts/train.py --model mfr_net --dataset cifar100 --use_mfr --no_uac --no_temp_scale --output_dir "./checkpoints/ablation_b"
# (c) + UAC Loss only
python3 scripts/train.py --model mfr_net --dataset cifar100 --no_mfr --use_uac --no_temp_scale --output_dir "./checkpoints/ablation_c"
# (d) MFR + UAC
python3 scripts/train.py --model mfr_net --dataset cifar100 --use_mfr --use_uac --no_temp_scale --output_dir "./checkpoints/ablation_d"

echo ""
echo "=========================================================="
echo " 6. Comparing Reproduction Results Against Paper Reference"
echo "=========================================================="
python3 scripts/verify_reproduction.py

echo "=========================================================="
echo " Reproduction pipeline execution complete!"
echo "=========================================================="
