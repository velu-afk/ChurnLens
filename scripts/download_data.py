"""Download the IBM Telco Customer Churn dataset from the official IBM GitHub repository."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from churnlens.data import inspect_dataframe

DEFAULT_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DEFAULT_OUTPUT = "data/raw/Telco-Customer-Churn.csv"


def download_telco_csv(url: str = DEFAULT_URL, output_path: str = DEFAULT_OUTPUT) -> Path:
    """Download the Telco Customer Churn CSV and verify its schema."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading IBM Telco Customer Churn dataset from:\n  {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ChurnLens/0.1.0"})
        with urllib.request.urlopen(req, timeout=25) as response:
            content = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            f"Failed to download dataset from '{url}'.\n"
            "If you are in an offline environment or behind a firewall:\n"
            f"  1. Manually place 'Telco-Customer-Churn.csv' at '{out}', OR\n"
            "  2. Run the pipeline in synthetic verification mode via `--use-synthetic`.\n"
            f"Underlying network error: {exc}"
        ) from exc

    out.write_bytes(content)
    sha256 = hashlib.sha256(content).hexdigest()
    df = pd.read_csv(out)
    summary = inspect_dataframe(df)

    print(f"Saved dataset to: {out}")
    print(f"SHA-256 checksum : {sha256}")
    print(
        f"Verified schema  : {summary['n_rows']} rows x {summary['n_columns']} columns | "
        f"Churn rate = {summary['positive_class_rate'] * 100:.2f}%"
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and verify the IBM Telco Customer Churn CSV dataset."
    )
    parser.add_argument("--url", type=str, default=DEFAULT_URL)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        download_telco_csv(url=args.url, output_path=args.output)
        return 0
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
