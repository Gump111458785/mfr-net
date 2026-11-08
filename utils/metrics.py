"""Evaluation metrics suite for multi-dataset, multi-metric benchmarking.
Computes:
  - Top-1 Accuracy (%)
  - Top-5 Accuracy (%)
  - Macro-F1 Score (%)
  - Expected Calibration Error (ECE, %)
  - Inference Latency (ms) and Throughput (samples/sec)
"""

import time
from typing import Dict, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> List[float]:
    """Computes the precision@k for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Handle datasets where num_classes < maxk (e.g. CIFAR-10 has 10, top5 is fine)
        num_classes = output.size(1)
        valid_topk = [k for k in topk if k <= num_classes]

        _, pred = output.topk(max(valid_topk), 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            if k > num_classes:
                res.append(100.0)
            else:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Computes the Expected Calibration Error (ECE) across n_bins.
    
    ECE = sum_m ( |acc(B_m) - conf(B_m)| * |B_m| / N )
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(labels)

    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece * 100.0)  # Return as percentage


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Runs a full multi-metric evaluation over the specified dataloader.
    
    Returns:
        Dict containing top1, top5, macro_f1, ece.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = targets.size(0)

            logits = model(inputs)
            probs = F.softmax(logits, dim=1)

            prec1, prec5 = accuracy(logits, targets, topk=(1, 5))
            total_top1 += prec1 * batch_size
            total_top5 += prec5 * batch_size
            total_samples += batch_size

            all_probs.append(probs.cpu().numpy())
            all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    top1 = total_top1 / total_samples
    top5 = total_top5 / total_samples
    macro_f1 = f1_score(all_targets, all_preds, average="macro") * 100.0
    ece = compute_ece(all_probs, all_targets, n_bins=15)

    return {
        "top1_acc": round(float(top1), 2),
        "top5_acc": round(float(top5), 2),
        "macro_f1": round(float(macro_f1), 2),
        "ece": round(float(ece), 2),
        "total_samples": int(total_samples),
    }


def measure_gpu_efficiency(
    model: torch.nn.Module,
    device: torch.device,
    input_size: Tuple[int, int, int, int] = (1, 3, 32, 32),
    warmup_iters: int = 50,
    measure_iters: int = 200,
) -> Dict[str, float]:
    """Measures model inference latency (ms/sample) and throughput (samples/s) on GPU."""
    if device.type != "cuda":
        raise ValueError("GPU device (cuda) is required to run accurate efficiency benchmarks.")

    model.eval()
    dummy_input = torch.randn(*input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy_input)
        torch.cuda.synchronize()

    # Latency Timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        for _ in range(measure_iters):
            _ = model(dummy_input)
    end_event.record()
    torch.cuda.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    avg_latency_ms = total_time_ms / measure_iters

    # Throughput benchmark with batch_size = 64
    batch_size = 64
    batch_input = torch.randn(batch_size, 3, 32, 32, device=device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(batch_input)
        torch.cuda.synchronize()

    t0 = time.time()
    with torch.no_grad():
        for _ in range(50):
            _ = model(batch_input)
        torch.cuda.synchronize()
    t1 = time.time()
    throughput = (batch_size * 50) / (t1 - t0)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    return {
        "params_m": round(float(total_params), 2),
        "latency_ms": round(float(avg_latency_ms), 2),
        "throughput_img_per_s": round(float(throughput), 1),
    }
