"""Auditable structure records and fixed evaluation partitions for the study.

Sequence grouping uses observed C-alpha sequences, never B-factor values. These
groups control a specified sequence-similarity criterion; they are not a claim
that all evolutionary relationships have been identified.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices


SEED = 20260913
ALIGNMENT_CONFIG = {
    "library": "Bio.Align.PairwiseAligner",
    "mode": "local",
    "substitution_matrix": "BLOSUM62",
    "open_gap_score": -10.0,
    "extend_gap_score": -0.5,
    "identity_threshold": 0.30,
    "reciprocal_sequence_coverage_threshold": 0.80,
    "minimum_aligned_pairs": 50,
    "identity_denominator": "aligned residue pairs, excluding gap columns",
    "coverage_numerator": "aligned residue pairs",
    "unknown_identity": "X-X is not counted as an identity",
    "sequence_source": "observed first-model C-alpha residues, separately by chain",
    "tie_policy": "first optimal alignment returned by the pinned Biopython version",
    "grouping": "connected components of protein pairs with any qualifying chain pair",
    "exact_duplicate_control": "identical complete-record multisets of observed chain sequences grouped regardless length",
    "limitation": "short proteins and fragmentary homologs can remain undetected; not all-family independence",
}
AA_CODES = dict(zip(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
    "ARNDCQEGHILKMFPSTWYV",
    strict=True,
))
AA_CODES.update({"MSE": "M", "SEC": "X", "PYL": "X", "UNK": "X"})
RESIDUE_COLUMNS = [
    "protein_id", "row_index", "residue_key", "model_id", "chain_id", "residue_number",
    "insertion_code", "residue_name", "amino_acid", "altloc", "occupancy", "atom_serial",
    "x", "y", "z", "b_factor", "z_b_factor",
]


def stable_hash(value: str, seed: int = SEED) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protein_id_from_path(path: Path) -> str:
    return Path(path).stem.removesuffix("_CA_A2").upper()


def read_structure(path: Path, protein_id: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Read the first PDB model with explicit deterministic altloc selection.

    Alternate conformers are selected by highest occupancy, then blank, A, and
    lexical altloc order. A repeated identical residue/altloc key is an error.
    HETATM calcium ions are excluded through the recognized amino-acid names.
    """
    path = Path(path)
    protein_id = protein_id or protein_id_from_path(path)
    selected_model: int | None = None
    current_model = 1
    saw_model = False
    candidates: dict[tuple, list[dict]] = {}
    seen_altloc: set[tuple] = set()
    discarded_models = 0
    unsupported = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="strict").splitlines(), 1):
        if line.startswith("MODEL"):
            saw_model = True
            current_model = int(line[10:14].strip() or "1")
            if selected_model is None:
                selected_model = current_model
            continue
        if not line.startswith(("ATOM  ", "HETATM")) or line[12:16].strip() != "CA":
            continue
        if selected_model is None:
            selected_model = current_model
        if current_model != selected_model:
            discarded_models += 1
            continue
        name = line[17:20].strip().upper()
        if name not in AA_CODES:
            unsupported += 1
            continue
        try:
            chain = line[21].strip() or "_"
            number = int(line[22:26])
            insertion = line[26].strip() or "."
            altloc = line[16].strip() or "."
            occupancy = float(line[54:60].strip() or "1")
            values = [float(line[start:start + 8]) for start in (30, 38, 46)]
            b_factor = float(line[60:66])
            serial = int(line[6:11])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid C-alpha record") from exc
        if not np.isfinite([*values, occupancy, b_factor]).all() or occupancy < 0 or b_factor < 0:
            raise ValueError(f"{path}:{line_number}: invalid coordinate, occupancy or B-factor")
        key = (selected_model, chain, number, insertion)
        alternate_key = (*key, altloc)
        if alternate_key in seen_altloc:
            raise ValueError(f"{path}:{line_number}: duplicate residue/altloc key {alternate_key}")
        seen_altloc.add(alternate_key)
        record = {
            "protein_id": protein_id, "residue_key": "|".join(map(str, key)),
            "model_id": selected_model, "chain_id": chain, "residue_number": number,
            "insertion_code": insertion, "residue_name": name, "amino_acid": AA_CODES[name],
            "altloc": altloc, "occupancy": occupancy, "atom_serial": serial,
            "x": values[0], "y": values[1], "z": values[2], "b_factor": b_factor,
        }
        candidates.setdefault(key, []).append(record)
    records = []
    for alternatives in candidates.values():
        alternatives.sort(key=lambda r: (
            -r["occupancy"], 0 if r["altloc"] == "." else 1 if r["altloc"] == "A" else 2,
            r["altloc"],
        ))
        records.append(alternatives[0])
    if len(records) < 3:
        raise ValueError(f"{path}: fewer than three eligible C-alpha residues")
    frame = pd.DataFrame(records)
    if frame[["x", "y", "z"]].duplicated().any():
        raise ValueError(f"{path}: duplicate C-alpha coordinates across distinct residue keys")
    frame["row_index"] = np.arange(len(frame), dtype=int)
    values = frame["b_factor"].to_numpy(dtype=float)
    scale = float(values.std(ddof=0))
    if scale <= 0:
        raise ValueError(f"{path}: constant B-factor target")
    frame["z_b_factor"] = (values - values.mean()) / scale
    audit = {
        "model_id": selected_model, "explicit_model_records": saw_model,
        "excluded_other_model_ca": discarded_models, "excluded_nonprotein_ca": unsupported,
        "discarded_altloc_ca": len(seen_altloc) - len(records),
        "bfactor_mean": float(values.mean()), "bfactor_std": scale,
    }
    return frame[RESIDUE_COLUMNS], audit


def make_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    return aligner


def sequence_similarity(a: str, b: str, aligner: PairwiseAligner | None = None) -> dict:
    """Measure one optimal local alignment; unknown X-X pairs are nonidentical."""
    if not a or not b:
        return {"identity": 0.0, "coverage_shorter": 0.0, "coverage_a": 0.0, "coverage_b": 0.0,
                "aligned_pairs": 0, "score": 0.0}
    aligner = aligner or make_aligner()
    alignments = aligner.align(a, b)
    try:
        alignment = alignments[0]
    except IndexError:
        return {"identity": 0.0, "coverage_shorter": 0.0, "coverage_a": 0.0, "coverage_b": 0.0,
                "aligned_pairs": 0, "score": 0.0}
    matched = 0
    paired = 0
    for (a0, a1), (b0, b1) in zip(*alignment.aligned, strict=True):
        left, right = a[a0:a1], b[b0:b1]
        paired += len(left)
        matched += sum(x == y and x != "X" for x, y in zip(left, right, strict=True))
    return {
        "identity": matched / paired if paired else 0.0,
        "coverage_shorter": paired / min(len(a), len(b)),
        "coverage_a": paired / len(a), "coverage_b": paired / len(b),
        "aligned_pairs": paired, "score": float(alignment.score),
    }


def cluster_sequences(
    chains: dict[str, dict[str, str]],
    identity: float = 0.30,
    coverage: float = 0.80,
    minimum_aligned_pairs: int = 50,
    reciprocal: bool = True,
    exact_duplicates: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, str], pd.DataFrame]:
    """Connect whole proteins if any observed-chain pair meets both cutoffs."""
    ids = sorted(chains)
    parent = {item: item for item in ids}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(a: str, b: str) -> None:
        x, y = sorted([find(a), find(b)])
        parent[y] = x

    aligner = make_aligner()
    rows = []
    signatures: dict[tuple[str, ...], str] = {}
    if exact_duplicates:
        for protein in ids:
            signature = tuple(sorted(chains[protein].values()))
            if signature in signatures and not any("X" in sequence for sequence in signature):
                other = signatures[signature]
                union(other, protein)
                length = sum(map(len, signature))
                rows.append({"protein_a": other, "protein_b": protein, "chain_a": "*", "chain_b": "*",
                             "length_a": length, "length_b": length, "identity": 1.0,
                             "coverage_shorter": 1.0, "coverage_a": 1.0, "coverage_b": 1.0,
                             "aligned_pairs": length, "score": np.nan, "edge_type": "exact_complete_record"})
            else:
                signatures[signature] = protein
    completed = 0
    total = len(ids) * (len(ids) - 1) // 2
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            best = None
            for chain_a, seq_a in sorted(chains[a].items()):
                for chain_b, seq_b in sorted(chains[b].items()):
                    if min(len(seq_a), len(seq_b)) < minimum_aligned_pairs:
                        continue
                    if reciprocal and min(len(seq_a), len(seq_b)) / max(len(seq_a), len(seq_b)) < coverage:
                        continue
                    result = sequence_similarity(seq_a, seq_b, aligner)
                    measured_coverage = min(result["coverage_a"], result["coverage_b"]) if reciprocal else result["coverage_shorter"]
                    if (result["identity"] + 1e-12 >= identity and measured_coverage + 1e-12 >= coverage
                            and result["aligned_pairs"] >= minimum_aligned_pairs):
                        candidate = {"protein_a": a, "protein_b": b, "chain_a": chain_a,
                                     "chain_b": chain_b, "length_a": len(seq_a), "length_b": len(seq_b),
                                     "edge_type": "sequence_alignment", **result}
                        if best is None or (result["identity"], result["coverage_shorter"]) > (
                            best["identity"], best["coverage_shorter"]
                        ):
                            best = candidate
            if best is not None:
                rows.append(best)
                union(a, b)
            completed += 1
        if progress:
            progress(completed, total)
    columns = ["protein_a", "protein_b", "chain_a", "chain_b", "length_a", "length_b", "identity",
               "coverage_shorter", "coverage_a", "coverage_b", "aligned_pairs", "score", "edge_type"]
    return {item: "cluster_" + find(item) for item in ids}, pd.DataFrame(rows, columns=columns)


def balanced_folds(
    sizes: dict[str, int], groups: dict[str, str], n_folds: int = 5, seed: int = SEED,
) -> dict[str, int]:
    """Greedily balance residue counts while keeping each group intact."""
    if set(sizes) != set(groups):
        raise ValueError("Protein sizes and groups must contain exactly the same IDs")
    group_sizes: dict[str, int] = {}
    for protein, size in sizes.items():
        group_sizes[groups[protein]] = group_sizes.get(groups[protein], 0) + size
    if len(group_sizes) < n_folds or n_folds < 2:
        raise ValueError(f"Need at least {n_folds} independent groups; got {len(group_sizes)}")
    totals = [0] * n_folds
    assignments = {}
    for group in sorted(group_sizes, key=lambda g: (-group_sizes[g], stable_hash(g, seed))):
        fold = min(range(n_folds), key=lambda f: (totals[f], f))
        assignments[group] = fold + 1
        totals[fold] += group_sizes[group]
    return {protein: assignments[group] for protein, group in groups.items()}


def nested_subcohorts(proteins: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Predeclare nested size-stratified 60/20-protein computational subsets."""
    result = proteins[["protein_id", "n_residues"]].copy()
    ordered = result.sort_values(["n_residues", "protein_id"])["protein_id"].tolist()
    stratum = {protein: i for i, group in enumerate(np.array_split(ordered, 3)) for protein in group}
    required = {"1ULR", "1X3O"}.intersection(ordered)

    def select(pool: set[str], count: int) -> set[str]:
        selected = required.intersection(pool)
        count = min(count, len(pool))
        quotas = [count // 3 + int(i < count % 3) for i in range(3)]
        for i, quota in enumerate(quotas):
            candidates = sorted((p for p in pool - selected if stratum[p] == i), key=lambda p: stable_hash(p, seed))
            current = sum(stratum[p] == i for p in selected)
            selected.update(candidates[:max(0, quota - current)])
        remaining = sorted(pool - selected, key=lambda p: stable_hash(p, seed))
        selected.update(remaining[:max(0, count - len(selected))])
        return selected

    subset60 = select(set(ordered), 60)
    subset20 = select(subset60, 20)
    result["length_stratum"] = result["protein_id"].map(stratum)
    result["subset60"] = result["protein_id"].isin(subset60)
    result["subset20"] = result["protein_id"].isin(subset20)
    result["nested60"] = result["subset60"]
    result["nested20"] = result["subset20"]
    result["case_study"] = result["protein_id"].map({"1ULR": "hero", "1X3O": "companion"}).fillna("")
    return result


def prepare_dataset(
    data_root: Path,
    out_dir: Path,
    *,
    folds: int = 5,
    seed: int = SEED,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Create residue, protein, exclusion, similarity-edge and fold manifests."""
    import Bio

    data_root, out_dir = Path(data_root).resolve(), Path(out_dir).resolve()
    pdb_dir = data_root / "datasets" / "365"
    paths = sorted(pdb_dir.glob("*.pdb"))
    if not paths:
        raise FileNotFoundError(f"No benchmark PDB files under {pdb_dir}")
    protein_rows, residue_frames, audit_rows = [], [], []
    chains = {}
    seen_ids = set()
    for path in paths:
        protein = protein_id_from_path(path)
        if protein in seen_ids:
            raise ValueError(f"Duplicate normalized protein ID: {protein}")
        seen_ids.add(protein)
        file_hash = sha256_file(path)
        try:
            residues, audit = read_structure(path, protein)
        except ValueError as exc:
            audit_rows.append({"protein_id": protein, "included": False, "reason": str(exc),
                               "source_path": str(path), "source_sha256": file_hash})
            continue
        chain_sequences = {chain: "".join(group["amino_acid"])
                           for chain, group in residues.groupby("chain_id", sort=False)}
        chains[protein] = chain_sequences
        annotation_path = data_root / "features" / "features-blind-prediction" / f"{protein}.csv"
        annotation_status = "absent"
        annotation_columns = ""
        if annotation_path.exists():
            annotation_columns = ",".join(pd.read_csv(annotation_path, nrows=0).columns)
            annotation_status = "excluded_no_verified_chain_residue_identity"
        protein_rows.append({
            "protein_id": protein, "n_residues": len(residues), "n_chains": len(chain_sequences),
            "model_id": audit["model_id"], "source_path": str(path), "source_sha256": file_hash,
            "chain_sequences_json": json.dumps(chain_sequences, sort_keys=True),
            "n_unknown_residues": int((residues["amino_acid"] == "X").sum()),
            "annotation_status": annotation_status, **audit,
        })
        residue_frames.append(residues)
        audit_rows.append({"protein_id": protein, "included": True, "reason": "included",
                           "source_path": str(path), "source_sha256": file_hash,
                           "annotation_status": annotation_status, "annotation_columns": annotation_columns,
                           **audit})
    if not protein_rows:
        raise ValueError("No eligible proteins")
    proteins = pd.DataFrame(protein_rows).sort_values("protein_id").reset_index(drop=True)
    residues = pd.concat(residue_frames, ignore_index=True)
    cohort = nested_subcohorts(proteins, seed)
    proteins = proteins.merge(cohort.drop(columns="n_residues"), on="protein_id", validate="one_to_one")
    # These score-independent inputs can be generated while sequence grouping runs.
    out_dir.mkdir(parents=True, exist_ok=True)
    proteins.to_csv(out_dir / "proteins.csv", index=False)
    residues.to_csv(out_dir / "residues.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(out_dir / "audit.csv", index=False)
    return finalize_partitions(out_dir, folds=folds, seed=seed, progress=progress)


def finalize_partitions(
    out_dir: Path, *, folds: int = 5, seed: int = SEED,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Group already audited manifests without rewriting any coordinate data."""
    import Bio

    out_dir = Path(out_dir)
    proteins = pd.read_csv(out_dir / "proteins.csv", dtype={"protein_id": str})
    residues = pd.read_csv(out_dir / "residues.csv", dtype={"protein_id": str})
    audit_table = pd.read_csv(out_dir / "audit.csv", dtype={"protein_id": str})
    chains = {row.protein_id: json.loads(row.chain_sequences_json) for row in proteins.itertuples()}
    clusters, edges = cluster_sequences(chains, progress=progress)
    edges.to_csv(out_dir / "sequence_edges.csv", index=False)
    sizes = dict(zip(proteins["protein_id"], proteins["n_residues"], strict=True))
    ordinary = balanced_folds(sizes, {p: p for p in sizes}, folds, seed)
    grouped = balanced_folds(sizes, clusters, folds, seed)
    fold_table = pd.DataFrame({"protein_id": sorted(sizes)})
    fold_table["cluster_id"] = fold_table["protein_id"].map(clusters)
    fold_table["protein_fold"] = fold_table["protein_id"].map(ordinary)
    fold_table["cluster_fold"] = fold_table["protein_id"].map(grouped)
    metadata = {
        "seed": seed, "n_folds": folds, "n_source_files": len(audit_table), "n_proteins": len(proteins),
        "n_residues": len(residues), "n_clusters": len(set(clusters.values())), "n_edges": len(edges),
        "biopython_version": Bio.__version__, "alignment": ALIGNMENT_CONFIG,
        "fold_policy": "descending group residue count, seeded SHA256 tie order; least-loaded fold, numeric tie order",
        "target": "C-alpha B-factor standardized within the entire protein record, ddof=0; used as target only",
        "structure_policy": "first model; recognized protein C-alpha atoms; altloc highest occupancy then blank/A/lexical; duplicate coordinates excluded",
        "residue_key": "model_id|chain_id|residue_number|insertion_code; blank chain='_', blank insertion='.'",
        "subcohort_policy": "nested 60 then20, equal protein-count length terciles, seeded SHA256 within tercile; force1ULR/1X3O",
        "annotations": "legacy annotation columns do not establish residue identity; excluded from primary modeling",
        "protocol_amendment": {
            "timing": "before fitting any prediction model; based only on observed sequences",
            "original_rule": "local alignment identity>=30%, coverage>=80% of shorter chain, any chain pair",
            "observed_failure": "one connected component for all364 proteins, preventing five-fold grouped evaluation",
            "amended_rule": "identity>=30%, coverage>=80% of each chain,>=50 aligned pairs; exact whole-record sequence duplicates also grouped",
        },
    }
    lengths = np.array([len(sequence) for chain in chains.values() for sequence in chain.values()])
    metadata["chain_length_audit"] = {
        "n_chains": len(lengths), "n_multichain_records": int((proteins["n_chains"] > 1).sum()),
        "minimum": int(lengths.min()), "median": float(np.median(lengths)), "maximum": int(lengths.max()),
        "n_chains_under50": int((lengths < 50).sum()),
        "n_records_all_chains_under50": sum(max(map(len, chain.values())) < 50 for chain in chains.values()),
    }
    fold_table.to_csv(out_dir / "folds.csv", index=False)
    original = out_dir / "sequence_edges_original.csv"
    if original.exists():
        metadata["protocol_amendment"]["original_edge_file_sha256"] = sha256_file(original)
        metadata["protocol_amendment"]["original_n_edges"] = len(pd.read_csv(original))
    metadata["manifest_sha256"] = {name: sha256_file(out_dir / name)
                                   for name in ["proteins.csv", "residues.csv", "folds.csv", "sequence_edges.csv"]}
    (out_dir / "dataset_config.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata
