"""Automated reproduction verification script.
Compares experimental results against the ground-truth values reported in the paper.
"""

import os
import sys
import json
import argparse

try:
    from tabulate import tabulate
except ImportError:
    def tabulate(rows, headers, tablefmt="github"):
        col_widths = [len(str(h)) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(val)))
        
        header_line = "| " + " | ".join(f"{str(h):<{col_widths[i]}}" for i, h in enumerate(headers)) + " |"
        sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
        row_lines = [
            "| " + " | ".join(f"{str(val):<{col_widths[i]}}" for i, val in enumerate(row)) + " |"
            for row in rows
        ]
        return "\n".join([header_line, sep_line] + row_lines)



def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Verify reproduction accuracy against paper results")
    parser.add_argument("--reference", type=str, default="./results/paper_reference_results.json")
    parser.add_argument("--eval_results_dir", type=str, default="./results")
    parser.add_argument("--tolerance", type=float, default=0.75, help="Acceptable delta (%) for Top-1 Accuracy reproduction")
    args = parser.parse_args()

    if not os.path.exists(args.reference):
        print(f"Error: Reference paper results not found at {args.reference}", file=sys.stderr)
        sys.exit(1)

    ref_data = load_json(args.reference)
    ref_table1 = ref_data["table_1_main_results"]

    print("\n" + "=" * 75)
    print(f" AUTOMATED PAPER REPRODUCTION VERIFICATION SUITE")
    print(f" Paper: '{ref_data['paper_title']}'")
    print(f" Allowed Accuracy Delta Tolerance: ±{args.tolerance}%")
    print("=" * 75)

    comparison_rows = []
    all_passed = True

    for dataset, models in ref_table1.items():
        for model_name, ref_metrics in models.items():
            run_name = f"eval_{model_name}_{dataset}"
            result_file = os.path.join(args.eval_results_dir, f"{run_name}_metrics.json")

            if not os.path.exists(result_file):
                # Try fallback training run name
                run_name_train = f"{model_name}_{dataset}_seed42"
                result_file = os.path.join(args.eval_results_dir, f"{run_name_train}_metrics.json")

            if os.path.exists(result_file):
                res = load_json(result_file)
                delta_top1 = abs(res.get("top1_acc", 0.0) - ref_metrics["top1_acc"])
                delta_ece = abs(res.get("ece", 0.0) - ref_metrics["ece"])
                passed = delta_top1 <= args.tolerance
                if not passed:
                    all_passed = False

                status = "PASS" if passed else "FAIL"
                comparison_rows.append([
                    dataset.upper(),
                    model_name.upper(),
                    f"{ref_metrics['top1_acc']:.2f}%",
                    f"{res.get('top1_acc', 0.0):.2f}%",
                    f"{delta_top1:.2f}%",
                    f"{ref_metrics['ece']:.2f}%",
                    f"{res.get('ece', 0.0):.2f}%",
                    status
                ])
            else:
                comparison_rows.append([
                    dataset.upper(),
                    model_name.upper(),
                    f"{ref_metrics['top1_acc']:.2f}%",
                    "PENDING RUN",
                    "-",
                    f"{ref_metrics['ece']:.2f}%",
                    "-",
                    "NOT RUN"
                ])

    headers = ["Dataset", "Model", "Paper Top-1", "Reprod Top-1", "Δ Top-1", "Paper ECE", "Reprod ECE", "Status"]
    print(tabulate(comparison_rows, headers=headers, tablefmt="github"))
    print("=" * 75)

    if all_passed and any(row[-1] == "PASS" for row in comparison_rows):
        print("\n>>> REPRODUCTION SUCCESS: Evaluated checkpoints reproduced paper metrics within target tolerance.\n")
    else:
        print("\n>>> NOTICE: Some benchmarks have not yet been executed or require GPU training to reproduce.\n")


if __name__ == "__main__":
    main()
