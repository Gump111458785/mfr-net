"""Model training script for paper reproduction.
Requires CUDA GPU by default.
"""

import os
import sys
import argparse
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import get_dataloaders, get_dataset_info, SUPPORTED_DATASETS
from models.mfr_net import build_model
from utils.loss import UncertaintyCalibrationLoss
from utils.metrics import evaluate_model
from utils.logger import ExperimentLogger


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Train visual classification models (GPU Required)")
    parser.add_argument("--model", type=str, default="mfr_net", choices=["mfr_net", "resnet18", "resnet50", "mobilenet_v2"])
    parser.add_argument("--dataset", type=str, default="cifar100", choices=SUPPORTED_DATASETS)
    parser.add_argument("--data_dir", type=str, default="./data/cache")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--use_mfr", action="store_true", default=True, help="Enable MFR module")
    parser.add_argument("--no_mfr", dest="use_mfr", action="store_false", help="Disable MFR module (for ablation)")
    parser.add_argument("--use_uac", dest="use_uac", action="store_true", default=None, help="Explicitly enable Uncertainty-Aware Calibration Loss")
    parser.add_argument("--no_uac", dest="use_uac", action="store_false", help="Explicitly disable UAC Loss (for ablation)")
    parser.add_argument("--use_temp_scale", action="store_true", default=True, help="Enable temperature scaling")
    parser.add_argument("--no_temp_scale", dest="use_temp_scale", action="store_false", help="Disable temperature scaling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu_debug", action="store_true", help="Allow CPU fallback solely for testing purposes")

    args = parser.parse_args()

    # Hardware verification (Requirement 3: GPU Required)
    if not torch.cuda.is_available() and not args.cpu_debug:
        print("\n" + "!" * 70, file=sys.stderr)
        print(" [CRITICAL HARDWARE REQUIREMENT] ", file=sys.stderr)
        print(" PyTorch CUDA acceleration is mandatory for this experiment.", file=sys.stderr)
        print(" No CUDA-enabled GPU was detected in the current environment.", file=sys.stderr)
        print(" Please run on a machine equipped with an NVIDIA GPU and CUDA driver.", file=sys.stderr)
        print(" If you want to perform a rapid pipeline smoke test without GPU, specify --cpu_debug.", file=sys.stderr)
        print("!" * 70 + "\n", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device Setup] Using execution device: {device} (CUDA available: {torch.cuda.is_available()})")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset Preparation
    dataset_info = get_dataset_info(args.dataset)
    num_classes = dataset_info["num_classes"]
    print(f"[Data Setup] Loading dataset '{args.dataset}' ({num_classes} classes)...")

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Model Setup
    print(f"[Model Setup] Initializing architecture: '{args.model}' (MFR={args.use_mfr}, TempScale={args.use_temp_scale})")
    model = build_model(
        model_name=args.model,
        num_classes=num_classes,
        use_mfr=args.use_mfr,
        use_temp_scale=args.use_temp_scale,
    )
    model = model.to(device)

    # Optimization Objective
    # Baselines (ResNet-18, ResNet-50, MobileNetV2) and non-UAC ablation variants
    # strictly use standard nn.CrossEntropyLoss unless explicitly overridden.
    if args.use_uac is None:
        use_uac = (args.model == "mfr_net")
    else:
        use_uac = args.use_uac

    if use_uac:
        print("[Loss Objective] Employing Uncertainty-Aware Calibration (UAC) Loss.")
        criterion = UncertaintyCalibrationLoss(
            smoothing=0.05,
            cal_weight=0.15,
            use_uac=True,
        ).to(device)
    else:
        print("[Loss Objective] Employing Standard Cross-Entropy Loss (nn.CrossEntropyLoss) for baseline / ablation.")
        criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Training Loop
    run_name = f"{args.model}_{args.dataset}_seed{args.seed}"
    best_val_top1 = 0.0
    best_checkpoint_path = os.path.join(args.output_dir, f"{run_name}_best.pth")

    print(f"\n--- Commencing Training: {run_name} ({args.epochs} epochs) ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()

            with autocast(enabled=(device.type == "cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / total_batches

        # Validation step every 10 epochs or on final epoch
        if epoch % 10 == 0 or epoch == args.epochs:
            val_metrics = evaluate_model(model, val_loader, device)
            print(
                f"Epoch [{epoch:03d}/{args.epochs:03d}] "
                f"Loss: {avg_loss:.4f} | "
                f"Val Top-1: {val_metrics['top1_acc']:.2f}% | "
                f"Val ECE: {val_metrics['ece']:.2f}% | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

            if val_metrics["top1_acc"] > best_val_top1:
                best_val_top1 = val_metrics["top1_acc"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                }, best_checkpoint_path)

    print(f"\nTraining completed. Loading best checkpoint for final evaluation on test set...")
    if os.path.exists(best_checkpoint_path):
        ckpt = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    # Final Evaluation on unseen test split
    test_metrics = evaluate_model(model, test_loader, device)
    print("=" * 60)
    print(f" FINAL TEST EVALUATION FOR: {run_name}")
    print(f" - Top-1 Accuracy:  {test_metrics['top1_acc']}%")
    print(f" - Top-5 Accuracy:  {test_metrics['top5_acc']}%")
    print(f" - Macro-F1 Score:  {test_metrics['macro_f1']}%")
    print(f" - ECE (15-bins):   {test_metrics['ece']}%")
    print("=" * 60)

    # Save metrics record
    logger = ExperimentLogger(log_dir="./results")
    logger.log_metrics(run_name, test_metrics)


if __name__ == "__main__":
    main()
