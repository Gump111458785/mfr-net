"""Logging and experiment output utility for paper reproduction runs.
"""

import os
import json
from typing import Dict, Any, List

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


class ExperimentLogger:

    def __init__(self, log_dir: str = "./results"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.history: List[Dict[str, Any]] = []

    def log_metrics(self, run_name: str, metrics: Dict[str, Any]):
        entry = {"run_name": run_name, **metrics}
        self.history.append(entry)
        save_path = os.path.join(self.log_dir, f"{run_name}_metrics.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=4)
        print(f"[Logger] Metrics saved to {save_path}")

    @staticmethod
    def print_comparison_table(results: List[Dict[str, Any]], title: str = "Experimental Results"):
        print("\n" + "=" * 70)
        print(f" {title} ")
        print("=" * 70)
        if not results:
            print("No results to display.")
            return

        headers = list(results[0].keys())
        rows = [[row.get(h, "") for h in headers] for row in results]
        print(tabulate(rows, headers=headers, tablefmt="github"))
        print("=" * 70 + "\n")
