#!/usr/bin/env python3
"""Generate PSL features for a directory of PDB files."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psl_flexibility.cli import main  # noqa: E402


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("features")
    elif sys.argv[1] != "features":
        sys.argv.insert(1, "features")
    main()
