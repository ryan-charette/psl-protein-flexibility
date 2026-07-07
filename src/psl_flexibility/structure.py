"""Protein structure parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ResidueRecord:
    """C-alpha residue record parsed from a PDB file."""

    protein_id: str
    atom_serial: int
    residue_name: str
    chain_id: str
    residue_number: int
    insertion_code: str
    coord: tuple[float, float, float]
    b_factor: float


def parse_ca_pdb(path: Path, protein_id: str | None = None) -> list[ResidueRecord]:
    """Parse C-alpha coordinates and B-factors from a PDB file."""

    resolved = Path(path)
    parsed_protein_id = protein_id or resolved.stem
    records: list[ResidueRecord] = []
    for line in resolved.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if line[12:16].strip() != "CA":
            continue
        records.append(_parse_ca_line(line, parsed_protein_id))

    if not records:
        raise ValueError(f"No C-alpha atoms found in {resolved}.")
    return records


def coordinates(records: list[ResidueRecord]) -> np.ndarray:
    """Return an ``n x 3`` coordinate array from residue records."""

    return np.asarray([record.coord for record in records], dtype=float)


def bfactors(records: list[ResidueRecord]) -> np.ndarray:
    """Return a B-factor array from residue records."""

    return np.asarray([record.b_factor for record in records], dtype=float)


def pdb_files(pdb_dir: Path, pattern: str = "*.pdb") -> list[Path]:
    """Return sorted PDB files from a directory."""

    root = Path(pdb_dir)
    if not root.exists():
        raise FileNotFoundError(f"PDB directory not found: {root}")
    files = sorted(path for path in root.glob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} found in {root}")
    return files


def _parse_ca_line(line: str, protein_id: str) -> ResidueRecord:
    try:
        atom_serial = int(line[6:11])
        residue_name = line[17:20].strip() or "UNK"
        chain_id = line[21].strip() or "_"
        residue_number = int(line[22:26])
        insertion_code = line[26].strip()
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        b_factor = float(line[60:66])
    except (IndexError, ValueError):
        parts = line.split()
        if len(parts) < 11:
            raise ValueError(f"Could not parse PDB ATOM line: {line!r}") from None
        atom_serial = int(parts[1])
        residue_name = parts[3]
        chain_id = parts[4] if len(parts[4]) == 1 else "_"
        residue_number = int(parts[5] if chain_id != "_" else parts[4])
        insertion_code = ""
        offset = 6 if chain_id != "_" else 5
        x = float(parts[offset])
        y = float(parts[offset + 1])
        z = float(parts[offset + 2])
        b_factor = float(parts[offset + 4])

    return ResidueRecord(
        protein_id=protein_id,
        atom_serial=atom_serial,
        residue_name=residue_name,
        chain_id=chain_id,
        residue_number=residue_number,
        insertion_code=insertion_code,
        coord=(x, y, z),
        b_factor=b_factor,
    )


__all__ = ["ResidueRecord", "bfactors", "coordinates", "parse_ca_pdb", "pdb_files"]
