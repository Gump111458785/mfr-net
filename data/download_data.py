"""Dataset downloading and verification utility for MFR-Net benchmarks.
Supports CIFAR-10, CIFAR-100, and SVHN datasets.
"""

import os
import argparse
import sys
from torchvision import datasets

SUPPORTED_DATASETS = ["cifar10", "cifar100", "svhn"]


def download_dataset(dataset_name: str, root_dir: str = "./data/cache") -> bool:
    """Download a single benchmark dataset to the designated cache directory.
    
    Args:
        dataset_name: Name of dataset ('cifar10', 'cifar100', 'svhn').
        root_dir: Destination folder path.
        
    Returns:
        bool: True if download/verification succeeded.
    """
    dataset_name = dataset_name.lower()
    os.makedirs(root_dir, exist_ok=True)
    target_dir = os.path.join(root_dir, dataset_name)
    os.makedirs(target_dir, exist_ok=True)

    print(f"[Dataset] Downloading & verifying '{dataset_name}' into '{target_dir}'...")

    try:
        if dataset_name == "cifar10":
            datasets.CIFAR10(root=target_dir, train=True, download=True)
            datasets.CIFAR10(root=target_dir, train=False, download=True)
        elif dataset_name == "cifar100":
            datasets.CIFAR100(root=target_dir, train=True, download=True)
            datasets.CIFAR100(root=target_dir, train=False, download=True)
        elif dataset_name == "svhn":
            datasets.SVHN(root=target_dir, split="train", download=True)
            datasets.SVHN(root=target_dir, split="test", download=True)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Expected one of {SUPPORTED_DATASETS}")

        print(f"[Dataset] Successfully prepared '{dataset_name}'.")
        return True
    except Exception as e:
        print(f"[Dataset Error] Failed to prepare dataset '{dataset_name}': {e}", file=sys.stderr)
        return False


def verify_dataset(dataset_name: str, root_dir: str = "./data/cache") -> bool:
    """Check if the dataset files exist locally without triggering re-download."""
    dataset_name = dataset_name.lower()
    target_dir = os.path.join(root_dir, dataset_name)
    if not os.path.isdir(target_dir):
        return False

    try:
        if dataset_name == "cifar10":
            datasets.CIFAR10(root=target_dir, train=True, download=False)
            datasets.CIFAR10(root=target_dir, train=False, download=False)
        elif dataset_name == "cifar100":
            datasets.CIFAR100(root=target_dir, train=True, download=False)
            datasets.CIFAR100(root=target_dir, train=False, download=False)
        elif dataset_name == "svhn":
            datasets.SVHN(root=target_dir, split="train", download=False)
            datasets.SVHN(root=target_dir, split="test", download=False)
        return True
    except Exception:
        return False


def download_all_datasets(root_dir: str = "./data/cache"):
    """Download and prepare all 3 benchmark datasets required by the paper."""
    print("=" * 60)
    print("Preparing All Experimental Datasets (CIFAR-10, CIFAR-100, SVHN)")
    print("=" * 60)
    success = {}
    for d in SUPPORTED_DATASETS:
        res = download_dataset(d, root_dir)
        success[d] = res

    print("\n[Download Summary]")
    for d, s in success.items():
        status = "READY" if s else "FAILED"
        print(f" - {d.upper()}: {status}")

    all_ok = all(success.values())
    if not all_ok:
        sys.exit(1)
    print("All benchmark datasets are ready for training and evaluation.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "cifar10", "cifar100", "svhn"],
        help="Dataset name to download or 'all' for complete paper benchmark suite",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/cache",
        help="Root directory to store dataset caches",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        download_all_datasets(args.data_dir)
    else:
        ok = download_dataset(args.dataset, args.data_dir)
        if not ok:
            sys.exit(1)
