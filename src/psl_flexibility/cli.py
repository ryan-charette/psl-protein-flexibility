"""Command-line interface for psl-protein-flexibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from psl_flexibility.demo import run_toy_demo
from psl_flexibility.features import (
    FeatureConfig,
    feature_names,
    features_for_residues,
    parse_float_list,
    parse_int_list,
    write_config,
    write_feature_csv,
)
from psl_flexibility.structure import parse_ca_pdb, pdb_files


def main() -> None:
    parser = argparse.ArgumentParser(prog="psl-flexibility")
    subparsers = parser.add_subparsers(dest="command", required=True)

    features_parser = subparsers.add_parser(
        "features",
        help="Generate PSL features for a directory of PDB files.",
    )
    features_parser.add_argument("--pdb-dir", type=Path, required=True, help="Directory containing PDB files.")
    features_parser.add_argument("--pattern", default="*.pdb", help="Glob pattern inside --pdb-dir.")
    features_parser.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    features_parser.add_argument("--radii", default="6,9,12", help="Comma-separated neighborhood radii.")
    features_parser.add_argument(
        "--sheaf",
        choices=["constant", "center_labeled", "atom_centered"],
        default="center_labeled",
    )
    features_parser.add_argument("--stats", choices=["median", "std", "both"], default="both")
    features_parser.add_argument("--degrees", default="0", help="Comma-separated PSL degrees: 0, 1, and/or 2.")
    features_parser.add_argument("--p-widths", default="0.0", help="Comma-separated persistence-width weights.")
    features_parser.add_argument("--scale-labels", action="store_true")

    demo_parser = subparsers.add_parser("demo", help="Run the bundled synthetic-data demo.")
    demo_parser.add_argument("--out-dir", type=Path, default=Path("data") / "toy" / "processed")

    args = parser.parse_args()
    if args.command == "features":
        run_features_command(args)
    elif args.command == "demo":
        summary = run_toy_demo(args.out_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:  # pragma: no cover - argparse enforces choices.
        raise ValueError(args.command)


def run_features_command(args: argparse.Namespace) -> None:
    config = FeatureConfig(
        radii=parse_float_list(args.radii),
        sheaf=str(args.sheaf),
        stats=str(args.stats),
        degrees=parse_int_list(args.degrees),
        p_widths=parse_float_list(args.p_widths),
        scale_labels=bool(args.scale_labels),
    )
    names = feature_names(config)
    records_and_features = []
    for pdb_path in pdb_files(args.pdb_dir, args.pattern):
        records = parse_ca_pdb(pdb_path)
        features = features_for_residues(records, config)
        records_and_features.append((pdb_path.stem, records, features))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    row_count = write_feature_csv(args.out_dir / "psl_features.csv", records_and_features, names)
    (args.out_dir / "feature_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    write_config(args.out_dir / "feature_config.json", config)
    print(f"Wrote {row_count} residue rows and {len(names)} PSL features to {args.out_dir}")


if __name__ == "__main__":
    main()
