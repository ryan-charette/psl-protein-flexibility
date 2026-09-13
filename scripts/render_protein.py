"""Reproducible, keyed molecular cartoons for residue-level model explanations.

The cartoon follows a smoothed C-alpha backbone; HELIX/SHEET records determine
ribbon types, carbonyl vectors determine ribbon frames, and arrowheads indicate
the C-terminal direction of annotated beta strands. This is geometric rendering,
not a reconstructed atomistic surface. It never invents missing residue values.

Example (the values CSV must include benchmark C-alpha x/y/z coordinates)::

    python scripts/render_protein.py --pdb data/raw/figures/1ULR.pdb \
        --values results/figures/1ulr_values.csv --column shap_spectral \
        --output results/figures/1ulr_shap.png

Values are keyed by chain_id, residue_number and insertion_code. Every supplied
residue must match a full-PDB C-alpha coordinate within 0.02 A. Camera and map
audits are saved beside CLI output. Unmapped context residues are neutral gray.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from threadpoolctl import threadpool_limits


PALETTE = ("#007f86", "#f1f2ec", "#db6a32")
COORDINATE_TOLERANCE = 0.02


def _text(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return "" if text in {".", "?"} else text


def _key(chain: Any, number: Any, insertion: Any = "") -> tuple[str, int, str]:
    return _text(chain), int(number), _text(insertion)


def read_structure(pdb_path: str | Path) -> list[dict[str, Any]]:
    """Read first-model protein residues, retaining peptide atoms and annotation."""
    records: dict[tuple[str, int, str], dict[str, Any]] = {}
    annotations: list[tuple[str, int, str, int, str, str]] = []
    for line in Path(pdb_path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("ENDMDL"):
            break
        if line.startswith("HELIX "):
            annotations.append((line[19].strip(), int(line[21:25]), line[25].strip(), int(line[33:37]), line[37].strip(), "helix"))
        elif line.startswith("SHEET "):
            annotations.append((line[21].strip(), int(line[22:26]), line[26].strip(), int(line[33:37]), line[37].strip(), "sheet"))
        elif line.startswith("ATOM  "):
            atom = line[12:16].strip()
            if atom not in {"N", "CA", "C", "O"}:
                continue
            key = _key(line[21], line[22:26], line[26])
            record = records.setdefault(key, {"key": key, "name": line[17:20].strip(), "atoms": {}, "ranks": {}, "secondary": "loop"})
            alternate = line[16].strip()
            occupancy = float(line[54:60].strip() or 0)
            priority = (occupancy, alternate == "", alternate == "A", -ord(alternate[0]) if alternate else 0)
            if atom not in record["ranks"] or priority > record["ranks"][atom]:
                record["atoms"][atom] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                record["ranks"][atom] = priority
                if atom == "CA":
                    record["b_factor"] = float(line[60:66])
    result = [record for record in records.values() if "CA" in record["atoms"]]
    if not result:
        raise ValueError(f"No protein C-alpha atoms in {pdb_path}.")
    for record in result:
        chain, number, insertion = record["key"]
        for ann_chain, start, start_insert, end, end_insert, secondary in annotations:
            if chain == ann_chain and (start, start_insert) <= (number, insertion) <= (end, end_insert):
                record["secondary"] = secondary
    return result


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)
    return vector / length if length > 1e-12 else np.zeros(3)


def _segments(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    for record in records:
        if not chunks or record["key"][0] != chunks[-1][-1]["key"][0] or np.linalg.norm(record["atoms"]["CA"] - chunks[-1][-1]["atoms"]["CA"]) > 4.8:
            chunks.append([])
        chunks[-1].append(record)
    return chunks


def _cartoon_meshes(records: list[dict[str, Any]], pv: Any) -> list[Any]:
    meshes = []
    for chain in _segments(records):
        if len(chain) == 1:
            sphere = pv.Sphere(radius=.28, center=chain[0]["atoms"]["CA"], theta_resolution=16, phi_resolution=16)
            sphere["value"] = np.full(sphere.n_points, chain[0]["value"])
            meshes.append(sphere)
            continue
        ca = np.array([row["atoms"]["CA"] for row in chain])
        backbone = ca.copy()
        # Remove the peptide zigzag inside sheets, as conventional cartoons do.
        for i in range(1, len(chain) - 1):
            if chain[i]["secondary"] == "sheet":
                backbone[i] = .25 * ca[i-1] + .5 * ca[i] + .25 * ca[i+1]
        parameter = np.arange(len(chain), dtype=float)
        spline = CubicSpline(parameter, backbone, axis=0, bc_type="natural")
        frames = []
        for i, row in enumerate(chain):
            tangent = _normalize(spline(parameter[i], 1))
            if "O" in row["atoms"] and "C" in row["atoms"]:
                direction = row["atoms"]["O"] - row["atoms"]["C"]
            elif 0 < i < len(chain)-1:
                direction = np.cross(ca[i] - ca[i-1], ca[i+1] - ca[i])
            else:
                direction = np.eye(3)[np.argmin(np.abs(tangent))]
            direction = direction - np.dot(direction, tangent) * tangent
            if np.linalg.norm(direction) < 1e-8:
                direction = np.eye(3)[np.argmin(np.abs(tangent))]
                direction = direction - np.dot(direction, tangent) * tangent
            direction = _normalize(direction)
            if frames and np.dot(direction, frames[-1]) < 0:
                direction = -direction
            frames.append(direction)
        frames = np.asarray(frames)
        frame_spline = CubicSpline(parameter, frames, axis=0, bc_type="natural")
        groups = []
        for i, row in enumerate(chain):
            if not groups or groups[-1][2] != row["secondary"]:
                groups.append([i, i, row["secondary"]])
            else:
                groups[-1][1] = i
        values = np.array([row["value"] for row in chain], dtype=float)
        for first, last, secondary in groups:
            start, stop = max(0., first-.5), min(len(chain)-1., last+.5)
            samples = np.linspace(start, stop, max(10, int((stop-start)*20)+1))
            centers = spline(samples)
            tangents = spline(samples, 1)
            tangents /= np.linalg.norm(tangents, axis=1)[:, None]
            width_vectors = frame_spline(samples)
            width_vectors -= np.sum(width_vectors*tangents, axis=1)[:, None]*tangents
            width_vectors /= np.maximum(np.linalg.norm(width_vectors, axis=1)[:, None], 1e-12)
            for i in range(1, len(width_vectors)):
                if np.dot(width_vectors[i], width_vectors[i-1]) < 0:
                    width_vectors[i] *= -1
            normals = np.cross(tangents, width_vectors)
            if secondary == "helix":
                widths, thickness = np.full(len(samples), .82), .16
            elif secondary == "sheet":
                widths, thickness = np.full(len(samples), .86), .14
                arrow_length = min(1.2, (stop-start)*.45)
                arrow = samples >= stop-arrow_length
                widths[arrow] = np.maximum(.045, 1.48*(stop-samples[arrow])/max(arrow_length, 1e-8))
            else:
                widths, thickness = np.full(len(samples), .24), .24
            rings = 16
            angles = np.linspace(0, 2*np.pi, rings, endpoint=False)
            vertices = (centers[:, None, :] + widths[:, None, None]*np.cos(angles)[None, :, None]*width_vectors[:, None, :] + thickness*np.sin(angles)[None, :, None]*normals[:, None, :]).reshape(-1, 3)
            faces = []
            for i in range(len(samples)-1):
                for j in range(rings):
                    faces.extend([4, i*rings+j, i*rings+(j+1)%rings, (i+1)*rings+(j+1)%rings, (i+1)*rings+j])
            faces.extend([rings, *range(rings-1, -1, -1)])
            faces.extend([rings, *range((len(samples)-1)*rings, len(samples)*rings)])
            mesh = pv.PolyData(vertices, np.asarray(faces))
            lo = np.floor(samples).astype(int)
            hi = np.minimum(lo+1, len(values)-1)
            fraction = samples-lo
            sampled_values = values[lo]*(1-fraction) + values[hi]*fraction
            # Never interpolate across an unmeasured residue.
            sampled_values[~np.isfinite(values[lo]) | ~np.isfinite(values[hi])] = np.nan
            mesh["value"] = np.repeat(sampled_values, rings)
            meshes.append(mesh)
    return meshes


def _default_camera(coordinates: np.ndarray) -> dict[str, Any]:
    center = coordinates.mean(axis=0)
    _, _, axes = np.linalg.svd(coordinates-center, full_matrices=False)
    up = axes[0]
    if up[np.argmax(np.abs(up))] < 0:
        up = -up
    right = axes[1] if len(axes) > 1 else np.array([1., 0, 0])
    if right[np.argmax(np.abs(right))] < 0:
        right = -right
    direction = _normalize(np.cross(right, up) + .22*right + .10*up)
    right = _normalize(np.cross(up, direction))
    up = _normalize(np.cross(direction, right))
    vertical, horizontal = (coordinates-center)@up, (coordinates-center)@right
    # Center the projected bounds, which need not be centered on the centroid.
    center = center + .5*(vertical.max()+vertical.min())*up + .5*(horizontal.max()+horizontal.min())*right
    span = max(np.ptp(vertical), np.ptp(horizontal))
    distance = max(60., 4*span)
    return {"position": (center+distance*direction).tolist(), "focal_point": center.tolist(), "view_up": up.tolist(), "parallel_scale": float(.58*span+1.5)}


def render_protein(
    pdb_path: str | Path,
    residue_df: pd.DataFrame,
    value_column: str,
    out_png: str | Path,
    camera: dict[str, Any] | None = None,
    clim: tuple[float, float] | float | None = None,
    *,
    size: int = 1800,
    transparent: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Render one audited structure; return JSON-serializable camera and mapping audit.

    The caller should supply one protein's rows. Required columns are chain_id,
    residue_number, insertion_code, x, y, z and value_column. ``clim`` is either
    a positive magnitude or a symmetric (negative, positive) interval.
    """
    output = Path(out_png)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "data" / "processed" / "mpl-cache"))
    import pyvista as pv
    import vtk
    from matplotlib.colors import LinearSegmentedColormap

    required = {"chain_id", "residue_number", "insertion_code", "x", "y", "z", value_column}
    if not required.issubset(residue_df.columns):
        raise ValueError(f"Missing residue table columns: {sorted(required-set(residue_df.columns))}.")
    frame = residue_df.copy()
    if frame.empty:
        raise ValueError("The residue table is empty.")
    keys = [_key(row.chain_id, row.residue_number, row.insertion_code) for row in frame.itertuples()]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate residue identifiers in the supplied table.")
    coordinates = frame[["x", "y", "z"]].to_numpy(dtype=float)
    values = frame[value_column].to_numpy(dtype=float)
    if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(values)):
        raise ValueError("Mapped coordinates and residue values must all be finite.")
    records = read_structure(pdb_path)
    lookup = {record["key"]: record for record in records}
    missing = [key for key in keys if key not in lookup]
    if missing:
        raise ValueError(f"Residues absent from full structure: {missing[:10]}.")
    errors = np.array([np.linalg.norm(lookup[key]["atoms"]["CA"]-xyz) for key, xyz in zip(keys, coordinates, strict=True)])
    if np.any(errors > COORDINATE_TOLERANCE):
        worst = int(np.argmax(errors))
        raise ValueError(f"Benchmark/full-structure coordinate mismatch at {keys[worst]}: {errors[worst]:.4f} A (limit {COORDINATE_TOLERANCE} A).")
    if "residue_name" in frame.columns:
        mismatch = [key for key, name in zip(keys, frame["residue_name"], strict=True) if lookup[key]["name"] != _text(name)]
        if mismatch:
            raise ValueError(f"Residue-name mismatch: {mismatch[:10]}.")
    value_map = dict(zip(keys, values, strict=True))
    chains = {key[0] for key in keys}
    records = [record for record in records if record["key"][0] in chains]
    for record in records:
        record["value"] = value_map.get(record["key"], np.nan)
    if clim is None:
        magnitude = max(float(np.max(np.abs(values))), 1e-12)
    elif np.isscalar(clim):
        magnitude = float(clim)
    else:
        if not np.isclose(float(clim[0]), -float(clim[1])):
            raise ValueError("Diverging color limits must be symmetric around zero.")
        magnitude = float(clim[1])
    if not np.isfinite(magnitude) or magnitude <= 0:
        raise ValueError("Color range magnitude must be positive and finite.")
    camera = dict(camera) if camera is not None else _default_camera(np.array([row["atoms"]["CA"] for row in records]))
    vtk.vtkMultiThreader.SetGlobalMaximumNumberOfThreads(1)
    vtk.vtkSMPTools.Initialize(1)
    cmap = LinearSegmentedColormap.from_list("psl_diverging", PALETTE, N=256)
    with threadpool_limits(limits=1):
        plotter = pv.Plotter(off_screen=True, window_size=(size, size), lighting="none")
        try:
            plotter.set_background("white")
            plotter.enable_anti_aliasing("ssaa")
            for mesh in _cartoon_meshes(records, pv):
                plotter.add_mesh(mesh, scalars="value", clim=(-magnitude, magnitude), cmap=cmap, nan_color="#afb7b9", show_scalar_bar=False, smooth_shading=True, ambient=.26, diffuse=.74, specular=.24, specular_power=28)
            center = np.array(camera["focal_point"])
            direction = _normalize(np.array(camera["position"])-center)
            up = _normalize(np.array(camera["view_up"]))
            right = _normalize(np.cross(up, direction))
            for offset, intensity in ((60*direction-35*right+45*up, .82), (40*direction+45*right+10*up, .43), (-30*direction+45*up, .40)):
                plotter.add_light(pv.Light(position=center+offset, focal_point=center, color="white", intensity=intensity, positional=False, light_type="scene light"))
            plotter.camera_position = [camera["position"], camera["focal_point"], camera["view_up"]]
            plotter.enable_parallel_projection()
            plotter.camera.parallel_scale = camera["parallel_scale"]
            plotter.reset_camera_clipping_range()
            plotter.show(interactive=False, auto_close=False)
            plotter.screenshot(str(output), transparent_background=transparent)
        finally:
            plotter.close()
    audit = {
        "pdb": str(Path(pdb_path).resolve()), "column": value_column,
        "mapped_residues": len(keys), "rendered_residues": len(records),
        "unmapped_context_residues": sum(row["key"] not in value_map for row in records),
        "maximum_coordinate_error_angstrom": float(errors.max()),
        "coordinate_tolerance_angstrom": COORDINATE_TOLERANCE,
        "secondary_structure_counts": {kind: sum(row["secondary"] == kind for row in records) for kind in ("helix", "sheet", "loop")},
        "chains": sorted(chains), "color_limits": [-magnitude, magnitude],
        "clipped_value_count": int(np.count_nonzero(np.abs(values)>magnitude)),
        "image_size": [size, size], "transparent_background": transparent,
        "geometry": "Smoothed C-alpha cartoon; PDB HELIX/SHEET annotations; carbonyl ribbon frames; C-terminal beta arrowheads",
        "color_interpolation": "Linear between measured adjacent residues; unmapped context gray",
    }
    return camera, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--values", type=Path, required=True)
    parser.add_argument("--column", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", type=Path, help="Reuse a camera JSON from an earlier panel")
    parser.add_argument("--limit", type=float, help="Symmetric color range +/- this value")
    parser.add_argument("--size", type=int, default=1800)
    parser.add_argument("--opaque", action="store_true")
    args = parser.parse_args()
    frame = pd.read_csv(args.values, keep_default_na=False)
    camera = json.loads(args.camera.read_text()) if args.camera else None
    camera, audit = render_protein(args.pdb, frame, args.column, args.output, camera, args.limit, size=args.size, transparent=not args.opaque)
    args.output.with_suffix(".camera.json").write_text(json.dumps(camera, indent=2), encoding="utf-8")
    args.output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
