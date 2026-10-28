"""End-to-end experiment runner for ChurnLens."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from churnlens.config import load_config
from churnlens.main import run_full_pipeline


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete ChurnLens training, validation threshold tuning, "
            "held-out test evaluation, and plotting workflow."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Run using the 21-column synthetic Telco fixture (for offline verification).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional override path to a local Telco-Customer-Churn.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional override output directory for metrics, tables, and plots.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(
        config_path=args.config,
        use_synthetic=True if args.use_synthetic else None,
        output_dir=args.output_dir,
        raw_csv_path=args.data_path,
    )

    metadata = run_full_pipeline(cfg)
    md_path = Path(metadata["artifacts"]["comparison_markdown"])
    n_num = len(metadata["split_sizes"]["numeric_features"])
    n_cat = len(metadata["split_sizes"]["categorical_features"])
    pos_rate_pct = metadata["dataset_summary"]["positive_class_rate"] * 100.0

    print(f"\n=== ChurnLens Experiment Summary (data_source: {metadata['data_source']}) ===")
    print(
        f"Rows: {metadata['dataset_summary']['n_rows']} | "
        f"Features: {n_num + n_cat} | "
        f"Positive Churn Rate: {pos_rate_pct:.2f}%"
    )
    print(
        f"Splits -> Train (Full): {metadata['split_sizes']['train_full']} "
        f"(Subtrain: {metadata['split_sizes']['subtrain']}, "
        f"Val: {metadata['split_sizes']['validation']}) | "
        f"Test: {metadata['split_sizes']['test']}\n"
    )
    print(md_path.read_text(encoding="utf-8"))
    print("Generated Artifacts:")
    print(json.dumps(metadata["artifacts"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
