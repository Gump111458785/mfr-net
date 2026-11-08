"""Dataset loading and data transformation pipelines for multi-dataset experiments.
Supported datasets: CIFAR-10, CIFAR-100, SVHN.
"""

import os
from typing import Tuple, Dict, Any
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from .download_data import verify_dataset, download_dataset

SUPPORTED_DATASETS = ["cifar10", "cifar100", "svhn"]

# Standard dataset statistics
DATASET_CONFIGS = {
    "cifar10": {
        "num_classes": 10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "img_size": 32,
    },
    "cifar100": {
        "num_classes": 100,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "img_size": 32,
    },
    "svhn": {
        "num_classes": 10,
        "mean": (0.4377, 0.4438, 0.4728),
        "std": (0.1980, 0.2010, 0.1970),
        "img_size": 32,
    },
}


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {SUPPORTED_DATASETS}")
    return DATASET_CONFIGS[dataset_name]


def get_transforms(dataset_name: str) -> Tuple[transforms.Compose, transforms.Compose]:
    """Build data augmentation and validation transforms."""
    cfg = get_dataset_info(dataset_name)
    mean, std = cfg["mean"], cfg["std"]

    if dataset_name in ["cifar10", "cifar100"]:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    elif dataset_name == "svhn":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=2, padding_mode="edge"),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        raise NotImplementedError

    return train_transform, test_transform


def get_dataloaders(
    dataset_name: str,
    data_dir: str = "./data/cache",
    batch_size: int = 128,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    download: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Construct PyTorch DataLoaders for train, validation, and test splits.
    
    Args:
        dataset_name: Target dataset name ('cifar10', 'cifar100', 'svhn').
        data_dir: Directory where datasets are stored/cached.
        batch_size: Batch size for loaders.
        num_workers: Number of DataLoader worker processes.
        val_ratio: Fraction of training set reserved for validation.
        download: Whether to automatically trigger download if dataset is missing.
        seed: Random seed for train/val splitting.
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Dataset {dataset_name} not supported. Must be in {SUPPORTED_DATASETS}")

    target_dir = os.path.join(data_dir, dataset_name)
    if not verify_dataset(dataset_name, data_dir):
        if download:
            print(f"[Dataset] Missing cached data for '{dataset_name}'. Initiating download...")
            download_dataset(dataset_name, data_dir)
        else:
            raise FileNotFoundError(
                f"Dataset '{dataset_name}' not found in '{target_dir}'. "
                f"Please run 'python data/download_data.py --dataset {dataset_name}' first."
            )

    train_tf, test_tf = get_transforms(dataset_name)

    if dataset_name == "cifar10":
        full_train = datasets.CIFAR10(root=target_dir, train=True, transform=train_tf, download=False)
        test_dataset = datasets.CIFAR10(root=target_dir, train=False, transform=test_tf, download=False)
    elif dataset_name == "cifar100":
        full_train = datasets.CIFAR100(root=target_dir, train=True, transform=train_tf, download=False)
        test_dataset = datasets.CIFAR100(root=target_dir, train=False, transform=test_tf, download=False)
    elif dataset_name == "svhn":
        full_train = datasets.SVHN(root=target_dir, split="train", transform=train_tf, download=False)
        test_dataset = datasets.SVHN(root=target_dir, split="test", transform=test_tf, download=False)

    # Split full_train into train and validation
    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
