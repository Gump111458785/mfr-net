"""Model efficiency benchmark script.
Measures Parameters (M), FLOPs estimation (G), GPU Latency (ms), and Throughput (img/s).
GPU is required.
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mfr_net import build_model
from utils.metrics import measure_gpu_efficiency
from utils.logger import ExperimentLogger


def estimate_flops(model: torch.nn.Module, input_size=(1, 3, 32, 32)) -> float:
    """Rough analytical FLOPs estimation for convolution and linear layers (in GFLOPs)."""
    total_flops = 0
    def conv_hook(self, input, output):
        batch_size, input_channels, height, width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels // self.groups)
        flops = batch_size * output_channels * output_height * output_width * kernel_ops * 2
        nonlocal total_flops
        total_flops += flops

    def linear_hook(self, input, output):
        batch_size = input[0].size(0)
        weight_ops = self.in_features * self.out_features * 2
        nonlocal total_flops
        total_flops += batch_size * weight_ops

    hooks = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    dummy = torch.randn(*input_size)
    with torch.no_grad():
        _ = model(dummy)

    for h in hooks:
        h.remove()

    return round(total_flops / 1e9, 2)


def main():
    parser = argparse.ArgumentParser(description="Model efficiency evaluation on GPU")
    parser.add_argument("--cpu_debug", action="store_true", help="Allow CPU fallback for debugging")
    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.cpu_debug:
        print("\n" + "!" * 70, file=sys.stderr)
        print(" [CRITICAL HARDWARE REQUIREMENT] ", file=sys.stderr)
        print(" GPU is required to obtain precise hardware latency/throughput metrics.", file=sys.stderr)
        print("!" * 70 + "\n", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Hardware Efficiency Profiler running on: {device}")
    if device.type == "cuda":
        print(f"[Device Name] {torch.cuda.get_device_name(0)}")

    models_to_test = ["mobilenet_v2", "resnet18", "resnet50", "mfr_net"]
    results = []

    for name in models_to_test:
        print(f"[Benchmarking] Profiling {name}...")
        model = build_model(name, num_classes=100)

        # Estimate FLOPs on CPU/dummy
        flops_g = estimate_flops(model, input_size=(1, 3, 32, 32))

        # Measure GPU timing
        model = model.to(device)
        if device.type == "cuda":
            eff = measure_gpu_efficiency(model, device)
            latency = eff["latency_ms"]
            throughput = eff["throughput_img_per_s"]
            params = eff["params_m"]
        else:
            params = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
            latency = "N/A (CPU)"
            throughput = "N/A (CPU)"

        results.append({
            "Architecture": name.upper(),
            "Params (M)": params,
            "FLOPs (G)": flops_g,
            "Latency (ms)": latency,
            "Throughput (img/s)": throughput,
        })

    ExperimentLogger.print_comparison_table(results, title="Hardware Complexity and Inference Efficiency (Table 2)")


if __name__ == "__main__":
    main()
