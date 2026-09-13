#!/usr/bin/env python3
"""Acquire the pinned benchmark inputs and the two verified rendering structures.

Existing files must match their expected contents; this script never overwrites
an inconsistent input. Only benchmark structures, annotation CSVs, dataset lists
and upstream attribution files are extracted. No downloaded code is executed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
import urllib.request
from urllib.parse import urlsplit
import zipfile


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "c987feb2c45c785799685c03d270ea8053e5706d"
UPSTREAM_URL = f"https://github.com/fenghon1/MDG_bfactor/archive/{UPSTREAM_COMMIT}.zip"
FIGURE_HASHES = {
    "1ULR": "9d8fa6b8dd35db1afd5267c6362405a8a05a153e89e7ed9a377f7b1d46fc8edc",
    "1X3O": "d2397b637afa3e4f73a94c372700c3c82d82e870dedf657a0039cfc2bc88cb20",
}
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
MAX_EXTRACTED_BYTES = 250 * 1024 * 1024


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_inputs(manifest: Path) -> tuple[dict[str, str], set[str]]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected, annotations = {}, set()
    for row in rows:
        filename = PurePosixPath(row["source_path"].replace("\\", "/")).name
        if not re.fullmatch(r"[A-Za-z0-9_]+\.pdb", filename):
            raise ValueError(f"Unsafe source filename in protein manifest: {filename!r}")
        relative = "datasets/365/" + filename
        if relative in expected or not re.fullmatch(r"[a-fA-F0-9]{64}", row["source_sha256"]):
            raise ValueError("Duplicate source file or invalid expected SHA256")
        expected[relative] = row["source_sha256"].lower()
        if row.get("annotation_status", "absent") != "absent":
            protein = row["protein_id"]
            if not re.fullmatch(r"[A-Za-z0-9_]+", protein):
                raise ValueError("Unsafe protein identifier")
            annotations.add(f"features/features-blind-prediction/{protein}.csv")
    if not expected:
        raise ValueError("The protein manifest contains no source hashes")
    return expected, annotations


def wanted_member(relative: str, expected: dict[str, str], annotations: set[str]) -> bool:
    return (relative in expected or relative in annotations or relative in {"README.md", "LICENSE"}
            or bool(re.fullmatch(r"datasets/list-(365|small|medium|large|blind-prediction)\.txt", relative)))


def safe_members(archive: zipfile.ZipFile, expected: dict[str, str], annotations: set[str]) -> dict[str, zipfile.ZipInfo]:
    selected = {}
    seen = set()
    for info in archive.infolist():
        name = info.filename
        if "\\" in name or "\x00" in name or ":" in name or name.startswith("/"):
            raise ValueError(f"Unsafe archive path: {name!r}")
        parts = PurePosixPath(name).parts
        if not parts or any(part in {"..", "."} for part in parts):
            raise ValueError(f"Unsafe archive path: {name!r}")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f"Archive symlink is not an input: {name!r}")
        if info.is_dir():
            continue
        if len(parts) < 2 or not parts[0].startswith("MDG_bfactor-"):
            raise ValueError(f"Unexpected archive root: {name!r}")
        relative = "/".join(parts[1:])
        if not wanted_member(relative, expected, annotations):
            continue
        if relative.casefold() in seen:
            raise ValueError(f"Duplicate archive destination: {relative}")
        seen.add(relative.casefold())
        selected[relative] = info
    absent = (set(expected) | annotations).difference(selected)
    if absent:
        raise ValueError(f"Pinned archive lacks required inputs: {sorted(absent)[:10]}")
    if sum(info.file_size for info in selected.values()) > MAX_EXTRACTED_BYTES:
        raise ValueError("Selected archive inputs exceed the extraction size limit")
    return selected


def extract_verified(archive_path: Path, destination: Path, expected: dict[str, str], annotations: set[str],
                     *, verify_only: bool = False) -> dict:
    destination = destination.resolve()
    written, existing = 0, 0
    with zipfile.ZipFile(archive_path) as archive:
        members = safe_members(archive, expected, annotations)
        # Validate every source hash and destination before writing the first file.
        for relative, info in members.items():
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Archive destination escapes input directory: {relative}")
            with archive.open(info) as handle:
                h = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(chunk)
            actual = h.hexdigest()
            if relative in expected and actual != expected[relative]:
                raise ValueError(f"Source hash mismatch in pinned archive: {relative}")
            if target.exists() and digest(target) != actual:
                raise ValueError(f"Existing input differs; left unchanged: {target}")
        for relative, info in members.items():
            target = (destination / relative).resolve()
            if target.exists():
                existing += 1
                continue
            if verify_only:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
                temporary = Path(output.name)
                try:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(chunk)
                except BaseException:
                    output.close()
                    temporary.unlink(missing_ok=True)
                    raise
            temporary.replace(target)
            written += 1
    return {"verified_protein_sources": len(expected), "selected_archive_members": len(members),
            "existing_files": existing, "written_files": written, "archive_sha256": digest(archive_path)}


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "protein-sheaf-study-reproduction/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        final = urlsplit(response.url)
        if final.scheme != "https" or final.hostname not in {"github.com", "codeload.github.com", "files.rcsb.org"}:
            raise ValueError("Unexpected download redirect")
        declared = int(response.headers.get("Content-Length", "0"))
        if declared > MAX_DOWNLOAD_BYTES:
            raise ValueError("Download exceeds the configured size limit")
        size = 0
        with target.open("wb") as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Download exceeds the configured size limit")
                output.write(chunk)


def acquire_figures(directory: Path, *, verify_only: bool = False) -> dict:
    report = {}
    for protein, expected in FIGURE_HASHES.items():
        target = directory / f"{protein}.pdb"
        if target.exists():
            if digest(target) != expected:
                raise ValueError(f"Rendering structure differs from the recorded source: {target}")
            report[protein] = {"status": "existing_verified", "sha256": expected}
            continue
        if verify_only:
            raise FileNotFoundError(f"Missing rendering structure: {target}")
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".pdb", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            download(f"https://files.rcsb.org/download/{protein}.pdb", temporary)
            if digest(temporary) != expected:
                raise ValueError(f"RCSB {protein} changed from the version used by the study; downloaded file not installed")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        report[protein] = {"status": "downloaded_verified", "sha256": expected}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/study/dataset/proteins.csv")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/MDG_bfactor-main")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "data/raw/figures")
    parser.add_argument("--archive", type=Path, help="Use an already downloaded archive instead of the network")
    parser.add_argument("--verify-only", action="store_true", help="Read-only validation; no download, extraction or report file")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    expected, annotations = expected_inputs(args.manifest)
    report = {"upstream_repository": "fenghon1/MDG_bfactor", "upstream_commit": UPSTREAM_COMMIT,
              "upstream_url": UPSTREAM_URL, "protein_manifest_sha256": digest(args.manifest)}
    existing_valid = all((args.data_root / relative).exists() and digest(args.data_root / relative) == expected_hash
                         for relative, expected_hash in expected.items())
    annotations_present = all((args.data_root / relative).exists() for relative in annotations)
    if args.archive is not None:
        report["benchmark"] = extract_verified(args.archive, args.data_root, expected, annotations, verify_only=args.verify_only)
    elif existing_valid and annotations_present:
        report["benchmark"] = {"status": "existing_verified", "verified_protein_sources": len(expected)}
    elif args.verify_only:
        raise ValueError("Benchmark inputs are missing or fail the committed source hashes")
    else:
        args.data_root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="fetch-benchmark-", dir=args.data_root.parent) as scratch:
            archive = Path(scratch) / "benchmark.zip"
            download(UPSTREAM_URL, archive)
            report["benchmark"] = extract_verified(archive, args.data_root, expected, annotations)
    if not args.skip_figures:
        report["figures"] = acquire_figures(args.figure_dir, verify_only=args.verify_only)
    report["status"] = "verified"
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        raise SystemExit(1) from error
