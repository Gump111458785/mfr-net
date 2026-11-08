"""Evaluation and test verification script.
Evaluates model on specified dataset, computing Top-1, Top-5, Macro-F1, and ECE.
Requires GPU by default.
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import get_dataloaders, get_dataset_info, SUPPORTED_DATASETS
from models.mfr_net import build_model
from utils.metrics import evaluate_model
from utils.logger import ExperimentLogger


def main():
    parser = argparse.ArgumentParser(description="Evaluate visual classification models on benchmark test sets")
    parser.add_argument("--model", type=str, default="mfr_net", choices=["mfr_net", "resnet18", "resnet50", "mobilenet_v2"])
    parser.add_argument("--dataset", type=str, default="cifar100", choices=SUPPORTED_DATASETS)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--data_dir", type=str, default="./data/cache")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--use_mfr", action="store_true", default=True)
    parser.add_argument("--no_mfr", dest="use_mfr", action="store_false")
    parser.add_argument("--use_temp_scale", action="store_true", default=True)
    parser.add_argument("--no_temp_scale", dest="use_temp_scale", action="store_false")
    parser.add_argument("--cpu_debug", action="store_true", help="Allow CPU fallback solely for testing purposes")

    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.cpu_debug:
        print("\n" + "!" * 70, file=sys.stderr)
        print(" [CRITICAL HARDWARE REQUIREMENT] ", file=sys.stderr)
        print(" CUDA-enabled GPU is required to run benchmark evaluation.", file=sys.stderr)
        print("!" * 70 + "\n", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Evaluation running on: {device}")

    dataset_info = get_dataset_info(args.dataset)
    num_classes = dataset_info["num_classes"]

    _, _, test_loader = get_dataloaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    model = build_model(
        model_name=args.model,
        num_classes=num_classes,
        use_mfr=args.use_mfr,
        use_temp_scale=args.use_temp_scale,
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"[Model] Loading checkpoint weights from '{args.checkpoint}'...")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        print("[Model] Note: No checkpoint specified or found; evaluating with initialized model weights.")

    model = model.to(device)

    print(f"\n[Evaluation] Benchmarking '{args.model}' on '{args.dataset}' test set...")
    metrics = evaluate_model(model, test_loader, device)

    print("\n" + "=" * 55)
    print(f" BENCHMARK EVALUATION SUMMARY: {args.model.upper()} on {args.dataset.upper()}")
    print("=" * 55)
    print(f"  • Top-1 Accuracy:       {metrics['top1_acc']:.2f} %")
    print(f"  • Top-5 Accuracy:       {metrics['top5_acc']:.2f} %")
    print(f"  • Macro-F1 Score:       {metrics['macro_f1']:.2f} %")
    print(f"  • Calibration ECE:      {metrics['ece']:.2f} %")
    print(f"  • Total Test Samples:   {metrics['total_samples']}")
    print("=" * 55 + "\n")

    logger = ExperimentLogger()
    run_name = f"eval_{args.model}_{args.dataset}"
    logger.log_metrics(run_name, metrics)


if __name__ == "__main__":
    main()
