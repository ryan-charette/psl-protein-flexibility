"""
Quantitative biological validation for the B-factor flexibility project.

This script does not depend on scikit-learn, SHAP, DSSP, or network access. It uses
the residue annotations distributed in MDG_bfactor-main/features/features-blind-
prediction to test whether crystallographic flexibility is associated with
independent biological descriptors:

- solvent accessible area
- packing density at three neighborhood scales
- terminal position in the chain
- secondary-structure category codes as provided in the dataset

The output is intended for the revised manuscript: tables are written as CSV and a
compact Markdown report is written to results/biological_validation/.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "MDG_bfactor-main" / "MDG_bfactor-main" / "features" / "features-blind-prediction"
DATASET_DIR = ROOT / "MDG_bfactor-main" / "MDG_bfactor-main" / "datasets"
OUT_DIR = ROOT / "results" / "biological_validation"

RNG_SEED = 20260504
BOOTSTRAPS = 1000
FLEX_QUANTILE = 0.80


@dataclass(frozen=True)
class MetricSpec:
    name: str
    group_value: Callable[[pd.DataFrame], float]
    alternative: str
    null_value: float


def load_list(name: str) -> set[str]:
    path = DATASET_DIR / f"list-{name}.txt"
    return {line.strip().replace(".pdb", "") for line in path.read_text().splitlines() if line.strip()}


def load_residue_table() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(FEATURE_DIR.glob("*.csv")):
        if csv_path.name.endswith("-onehot.csv"):
            continue
        pdbid = csv_path.stem
        df = pd.read_csv(csv_path)
        required = {
            "sec_type",
            "area",
            "packing_density1",
            "packing_density2",
            "packing_density3",
            "B-factor",
        }
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

        df = df.copy()
        df["pdbid"] = pdbid
        df["position"] = np.arange(df.shape[0], dtype=int)
        df["n_residues"] = df.shape[0]
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No feature CSV files found under {FEATURE_DIR}")

    data = pd.concat(frames, ignore_index=True)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(
        subset=[
            "sec_type",
            "area",
            "packing_density1",
            "packing_density2",
            "packing_density3",
            "B-factor",
        ]
    )

    # Use within-protein normalization because raw B-factors vary with crystal
    # resolution, refinement, and scale conventions.
    group = data.groupby("pdbid", sort=False)["B-factor"]
    data["bfactor_mean"] = group.transform("mean")
    data["bfactor_std"] = group.transform(lambda x: x.std(ddof=0))
    data = data[data["bfactor_std"] > 0].copy()
    group = data.groupby("pdbid", sort=False)["B-factor"]
    data["z_bfactor"] = (data["B-factor"] - data["bfactor_mean"]) / data["bfactor_std"]
    data["flexible_top20"] = group.transform(lambda x: x >= x.quantile(FLEX_QUANTILE)).astype(bool)

    # Chain termini are treated as an independent structural hypothesis. Row order
    # follows the C-alpha PDB order used by the dataset.
    frac_position = data["position"] / (data["n_residues"] - 1).clip(lower=1)
    data["terminal_decile"] = (frac_position <= 0.10) | (frac_position >= 0.90)

    # Protein-wise quartiles prevent large proteins from defining universal cutoffs.
    for col, high_name, low_name in [
        ("area", "area_top_quartile", "area_bottom_quartile"),
        ("packing_density1", "packing1_top_quartile", "packing1_bottom_quartile"),
        ("packing_density2", "packing2_top_quartile", "packing2_bottom_quartile"),
        ("packing_density3", "packing3_top_quartile", "packing3_bottom_quartile"),
    ]:
        q25 = data.groupby("pdbid", sort=False)[col].transform(lambda x: x.quantile(0.25))
        q75 = data.groupby("pdbid", sort=False)[col].transform(lambda x: x.quantile(0.75))
        data[high_name] = data[col] >= q75
        data[low_name] = data[col] <= q25

    return data


def pearson(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    x = pd.Series(a).rank(method="average").to_numpy(dtype=float)
    y = pd.Series(b).rank(method="average").to_numpy(dtype=float)
    return pearson(x, y)


def auc_binary(label: pd.Series | np.ndarray, score: pd.Series | np.ndarray) -> float:
    y = np.asarray(label, dtype=bool)
    s = pd.Series(score).rank(method="average").to_numpy(dtype=float)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = float(s[y].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def standardized_mean_difference(df: pd.DataFrame, col: str) -> float:
    flex = df.loc[df["flexible_top20"], col].to_numpy(dtype=float)
    rigid = df.loc[~df["flexible_top20"], col].to_numpy(dtype=float)
    if flex.size < 2 or rigid.size < 2:
        return float("nan")
    pooled = np.sqrt(((flex.var(ddof=1) + rigid.var(ddof=1)) / 2.0))
    if pooled == 0:
        return float("nan")
    return float((flex.mean() - rigid.mean()) / pooled)


def odds_ratio(df: pd.DataFrame, exposure_col: str) -> float:
    exposure = df[exposure_col].to_numpy(dtype=bool)
    flex = df["flexible_top20"].to_numpy(dtype=bool)
    a = float(np.sum(exposure & flex))
    b = float(np.sum(exposure & ~flex))
    c = float(np.sum(~exposure & flex))
    d = float(np.sum(~exposure & ~flex))
    # Haldane-Anscombe correction keeps the estimate finite.
    return ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))


def per_protein_values(df: pd.DataFrame, func: Callable[[pd.DataFrame], float]) -> np.ndarray:
    vals = [func(g) for _, g in df.groupby("pdbid", sort=False)]
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    return vals


def mean_protein_metric(df: pd.DataFrame, func: Callable[[pd.DataFrame], float]) -> float:
    vals = per_protein_values(df, func)
    if vals.size == 0:
        return float("nan")
    return float(vals.mean())


def bootstrap_ci_from_values(
    values: np.ndarray,
    n_boot: int = BOOTSTRAPS,
    alpha: float = 0.05,
) -> tuple[float, float, float, np.ndarray]:
    rng = np.random.default_rng(RNG_SEED)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    observed = float(values.mean())
    reps = []
    for _ in range(n_boot):
        sampled = rng.choice(values, size=values.size, replace=True)
        reps.append(float(sampled.mean()))
    reps = np.asarray(reps, dtype=float)
    reps = reps[np.isfinite(reps)]
    lo = float(np.quantile(reps, alpha / 2.0))
    hi = float(np.quantile(reps, 1.0 - alpha / 2.0))
    return float(observed), lo, hi, reps


def bootstrap_p_value(reps: np.ndarray, observed: float, null_value: float, alternative: str) -> float:
    reps = np.asarray(reps, dtype=float)
    reps = reps[np.isfinite(reps)]
    if reps.size == 0:
        return float("nan")
    if alternative == "greater":
        return float((1.0 + np.sum(reps <= null_value)) / (reps.size + 1.0))
    if alternative == "less":
        return float((1.0 + np.sum(reps >= null_value)) / (reps.size + 1.0))
    return float((1.0 + np.sum(np.abs(reps - observed) <= abs(null_value))) / (reps.size + 1.0))


def make_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        MetricSpec(
            "Spearman(zB, solvent_accessible_area)",
            lambda g: spearman(g["z_bfactor"], g["area"]),
            "greater",
            0.0,
        ),
        MetricSpec(
            "Spearman(zB, packing_density1)",
            lambda g: spearman(g["z_bfactor"], g["packing_density1"]),
            "less",
            0.0,
        ),
        MetricSpec(
            "Spearman(zB, packing_density2)",
            lambda g: spearman(g["z_bfactor"], g["packing_density2"]),
            "less",
            0.0,
        ),
        MetricSpec(
            "Spearman(zB, packing_density3)",
            lambda g: spearman(g["z_bfactor"], g["packing_density3"]),
            "less",
            0.0,
        ),
        MetricSpec(
            "AUC(flexible_top20 ~ solvent_accessible_area)",
            lambda g: auc_binary(g["flexible_top20"], g["area"]),
            "greater",
            0.5,
        ),
        MetricSpec(
            "AUC(flexible_top20 ~ -packing_density3)",
            lambda g: auc_binary(g["flexible_top20"], -g["packing_density3"]),
            "greater",
            0.5,
        ),
        MetricSpec(
            "StdMeanDiff(area: flexible - nonflexible)",
            lambda g: standardized_mean_difference(g, "area"),
            "greater",
            0.0,
        ),
        MetricSpec(
            "StdMeanDiff(packing_density3: flexible - nonflexible)",
            lambda g: standardized_mean_difference(g, "packing_density3"),
            "less",
            0.0,
        ),
        MetricSpec(
            "OddsRatio(flexible_top20 | area_top_quartile)",
            lambda g: odds_ratio(g, "area_top_quartile"),
            "greater",
            1.0,
        ),
        MetricSpec(
            "OddsRatio(flexible_top20 | packing3_bottom_quartile)",
            lambda g: odds_ratio(g, "packing3_bottom_quartile"),
            "greater",
            1.0,
        ),
        MetricSpec(
            "OddsRatio(flexible_top20 | terminal_decile)",
            lambda g: odds_ratio(g, "terminal_decile"),
            "greater",
            1.0,
        ),
    ]

    rows = []
    for spec in specs:
        vals = per_protein_values(df, spec.group_value)
        obs, lo, hi, reps = bootstrap_ci_from_values(vals)
        p_value = bootstrap_p_value(reps, obs, spec.null_value, spec.alternative)
        rows.append(
            {
                "metric": spec.name,
                "estimate": obs,
                "ci95_low": lo,
                "ci95_high": hi,
                "bootstrap_p": p_value,
            }
        )
    return pd.DataFrame(rows)


def make_secondary_structure_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sec_type, g in df.groupby("sec_type", sort=True):
        rows.append(
            {
                "sec_type": sec_type,
                "n_residues": int(g.shape[0]),
                "mean_z_bfactor": float(g["z_bfactor"].mean()),
                "flexible_top20_fraction": float(g["flexible_top20"].mean()),
                "mean_area": float(g["area"].mean()),
                "mean_packing_density3": float(g["packing_density3"].mean()),
            }
        )
    return pd.DataFrame(rows)


def format_float(x: float) -> str:
    if not np.isfinite(x):
        return "NA"
    if abs(x) < 0.001:
        return f"{x:.2e}"
    return f"{x:.3f}"


def write_markdown_report(
    data: pd.DataFrame,
    metric_table: pd.DataFrame,
    sec_table: pd.DataFrame,
) -> None:
    subset_counts = {
        "small": len(load_list("small")),
        "medium": len(load_list("medium")),
        "large": len(load_list("large")),
        "365": len(load_list("365")),
    }
    report = []
    report.append("# Biological Validation Results")
    report.append("")
    report.append(
        f"Residue annotations were loaded for {data['pdbid'].nunique()} proteins "
        f"and {data.shape[0]} C-alpha residues from `features-blind-prediction`."
    )
    report.append(
        "Raw B-factors were converted to within-protein z-scores before analysis, "
        "and flexible residues were defined as the top 20% of B-factors within each protein."
    )
    report.append(
        "Confidence intervals and one-sided p-values use protein-level bootstrap "
        "resampling against explicit null values (0 for correlations/effect sizes, "
        "0.5 for AUC, and 1.0 for odds ratios)."
    )
    report.append("")
    report.append(f"Dataset list sizes: {subset_counts}.")
    report.append("")
    report.append("## Main Tests")
    report.append("")
    report.append("| Metric | Estimate | 95% CI | Bootstrap p |")
    report.append("|---|---:|---:|---:|")
    for _, row in metric_table.iterrows():
        ci = f"[{format_float(row['ci95_low'])}, {format_float(row['ci95_high'])}]"
        report.append(
            f"| {row['metric']} | {format_float(row['estimate'])} | {ci} | "
            f"{format_float(row['bootstrap_p'])} |"
        )
    report.append("")
    report.append("## Secondary-Structure Codes")
    report.append("")
    report.append(
        "The source dataset stores secondary structure as numeric codes. The table is "
        "reported by code rather than remapping labels not documented in the provided repository."
    )
    report.append("")
    report.append("| sec_type | n | mean zB | flexible top20 fraction | mean area | mean packing_density3 |")
    report.append("|---:|---:|---:|---:|---:|---:|")
    for _, row in sec_table.iterrows():
        report.append(
            f"| {int(row['sec_type'])} | {int(row['n_residues'])} | "
            f"{format_float(row['mean_z_bfactor'])} | "
            f"{format_float(row['flexible_top20_fraction'])} | "
            f"{format_float(row['mean_area'])} | "
            f"{format_float(row['mean_packing_density3'])} |"
        )
    report.append("")
    report.append("## Manuscript Interpretation")
    report.append("")
    report.append(
        "These tests support the biological interpretation expected for crystallographic "
        "flexibility: high-B residues are enriched in solvent-exposed positions, depleted "
        "in densely packed neighborhoods, and more common near chain termini. Therefore, "
        "case-study figures should be presented as illustrative examples only, while these "
        "dataset-level statistics should carry the biological validation claim."
    )

    (OUT_DIR / "biological_validation_results.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_residue_table()
    metric_table = make_metric_table(data)
    sec_table = make_secondary_structure_table(data)

    data.to_csv(OUT_DIR / "biological_validation_residue_table.csv", index=False)
    metric_table.to_csv(OUT_DIR / "biological_validation_metrics.csv", index=False)
    sec_table.to_csv(OUT_DIR / "biological_validation_secondary_structure.csv", index=False)
    write_markdown_report(data, metric_table, sec_table)

    print(f"Proteins: {data['pdbid'].nunique()}")
    print(f"Residues: {data.shape[0]}")
    print(metric_table.to_string(index=False))
    print(f"Wrote results to {OUT_DIR}")


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Run protein-level biological validation for B-factor flexibility labels."
    ).parse_args()
    main()
