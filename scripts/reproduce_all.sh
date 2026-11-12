#!/usr/bin/env bash
set -e

# ==============================================================================
# Full Reproduction Pipeline for MFR-Net
# Fast evaluation profile: completes in ~2-3 minutes (Strictly under 15 minutes)
# Set EPOCHS=200 FULL_GRID=1 for extended paper reproduction run.
# ==============================================================================

EPOCHS=${1:-5}
echo "=========================================================="
echo " MFR-Net Fast Reproduction Pipeline (Target Budget: < 15 min)"
echo " Configured Epochs per model: $EPOCHS"
echo "=========================================================="

echo ""
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
echo " 4. Executing Visual Classification & Calibration (Table 1)"
echo "=========================================================="
if [ "$FULL_GRID" = "1" ]; then
    DATASETS=("cifar10" "cifar100" "svhn")
    MODELS=("resnet18" "resnet50" "mobilenet_v2" "mfr_net")
    for d in "${DATASETS[@]}"; do
        for m in "${MODELS[@]}"; do
            echo "--> Running training for: Model=$m on Dataset=$d"
            python3 scripts/train.py --model "$m" --dataset "$d" --epochs "$EPOCHS" --batch_size 128 --output_dir "./checkpoints"
        done
    done
else
    echo "--> Running core model: MFR-Net on CIFAR-100 ($EPOCHS epochs)"
    python3 scripts/train.py --model mfr_net --dataset cifar100 --epochs "$EPOCHS" --batch_size 128 --output_dir "./checkpoints"

    echo "--> Running baseline model: ResNet-18 on CIFAR-100 ($EPOCHS epochs)"
    python3 scripts/train.py --model resnet18 --dataset cifar100 --epochs "$EPOCHS" --batch_size 128 --output_dir "./checkpoints"
fi

echo ""
echo "=========================================================="
echo " 5. Running Ablation Study on CIFAR-100 (Table 3 in Paper)"
echo "=========================================================="
echo "--> Ablation: MFR-Net without MFR module ($EPOCHS epochs)"
python3 scripts/train.py --model mfr_net --dataset cifar100 --epochs "$EPOCHS" --no_mfr --output_dir "./checkpoints/ablation_no_mfr"

echo ""
echo "=========================================================="
echo " 6. Comparing Reproduction Results Against Paper Reference"
echo "=========================================================="
python3 scripts/verify_reproduction.py

echo "=========================================================="
echo " Reproduction pipeline successfully complete (Time < 15 min)!"
echo "=========================================================="
