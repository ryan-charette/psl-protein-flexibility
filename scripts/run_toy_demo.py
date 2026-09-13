#!/usr/bin/env python3
"""Run the synthetic legacy graph smoke demo (not a research benchmark)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psl_flexibility.demo import run_toy_demo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "data/toy/processed",
        help="Directory for the synthetic feature table and smoke-test metrics",
    )
    args = parser.parse_args()
    print(json.dumps(run_toy_demo(args.out_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
