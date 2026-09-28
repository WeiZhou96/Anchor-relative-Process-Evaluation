#!/usr/bin/env python
"""Verify the derived-data package and install it into the layout the code reads.

The code resolves its inputs relative to the repository root, exactly as in the
original runs:

    <root>/data/manifest/manifest_real.csv ...      <- <data>/accident/manifest/
    <root>/outputs/answers/<system>/answers.csv     <- <data>/accident/outputs/answers/<system>/answers.csv.gz
    <root>/outputs/{r2b,r2c,r2_S,prefix_lists,run002,m3_*,stage2_*,...}
                                                    <- <data>/accident/outputs/...

Answer matrices are shipped gzip-compressed.  ``install`` decompresses them and checks
that every restored ``answers.csv`` has the SHA-256 recorded in
``accident/outputs/answers/ANSWERS_INDEX.csv`` (the hash of the file that the original
analyses read), so the restored files are byte-identical to the originals.

Usage
-----
    python scripts/release/install_data.py verify  --data /path/to/ape-data
    python scripts/release/install_data.py install --data /path/to/ape-data [--root .] [--link] [--jobs 8]

``verify`` checks every file against ``<data>/SHA256SUMS``.  ``install`` never
overwrites an existing file unless ``--force`` is given.  With ``--link`` the
non-answer output directories are symlinked instead of copied; runs that write new
results into ``outputs/`` would then write into the data package, so copying is the
default.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT_DEFAULT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(data: Path) -> int:
    sums = data / "SHA256SUMS"
    bad = 0
    n = 0
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        n += 1
        p = data / rel
        if not p.is_file() or sha256(p) != digest:
            bad += 1
            print("MISMATCH", rel, file=sys.stderr)
    print(f"checked {n} files, {bad} mismatches")
    return 1 if bad else 0


def _place(src: Path, dst: Path, link: bool, force: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not force:
            raise SystemExit(f"refusing to overwrite {dst} (use --force)")
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if link:
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
    elif src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _restore(job: tuple[str, str, str, bool]) -> str:
    gz, out, expected, force = job
    out_p = Path(out)
    if out_p.exists() and not force:
        if sha256(out_p) == expected:
            return "kept"
        raise SystemExit(f"{out} exists with a different hash (use --force)")
    tmp = out_p.with_suffix(".csv.part")
    with gzip.open(gz, "rb") as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 20)
    got = sha256(tmp)
    if got != expected:
        tmp.unlink()
        raise SystemExit(f"hash mismatch after decompression: {gz}")
    tmp.replace(out_p)
    return "restored"


def install(data: Path, root: Path, link: bool, force: bool, jobs: int) -> int:
    acc = data / "accident"
    # manifests and their reports
    for f in sorted((acc / "manifest").iterdir()):
        _place(f, root / "data" / "manifest" / f.name, False, force)
    # frozen protocol and preregistration: the repository already carries them; check identity
    for sub in ("protocol", "prereg"):
        for f in sorted((acc / sub).iterdir()):
            tracked = root / sub / f.name
            if not tracked.exists() or sha256(tracked) != sha256(f):
                raise SystemExit(f"{tracked} differs from the data package copy {f}")
    # outputs other than answers
    out = root / "outputs"
    for entry in sorted((acc / "outputs").iterdir()):
        if entry.name == "answers":
            continue
        _place(entry, out / entry.name, link, force)
    # answers: side files are copied, answer matrices are decompressed and hash-checked
    ans_src = acc / "outputs" / "answers"
    ans_dst = out / "answers"
    ans_dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(ans_src.iterdir()):
        if f.is_file():
            _place(f, ans_dst / f.name, False, force)
    jobs_list = []
    with open(ans_src / "ANSWERS_INDEX.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            sid = row["system_id"]
            (ans_dst / sid).mkdir(exist_ok=True)
            for f in sorted((ans_src / sid).iterdir()):
                if f.name != "answers.csv.gz":
                    _place(f, ans_dst / sid / f.name, False, force)
            jobs_list.append((str(ans_src / sid / "answers.csv.gz"), str(ans_dst / sid / "answers.csv"),
                              row["csv_sha256"], force))
    with ProcessPoolExecutor(max(1, jobs)) as ex:
        results = list(ex.map(_restore, jobs_list))
    print(f"answer matrices: {results.count('restored')} restored, {results.count('kept')} already present; "
          f"all {len(results)} match the recorded SHA-256")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["verify", "install"])
    ap.add_argument("--data", type=Path, default=Path(os.environ.get("APE_DATA", "")) if os.environ.get("APE_DATA") else None,
                    help="root of the derived-data package (default: $APE_DATA)")
    ap.add_argument("--root", type=Path, default=ROOT_DEFAULT, help="code repository root (default: this repository)")
    ap.add_argument("--link", action="store_true", help="symlink output directories instead of copying them")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    if a.data is None:
        ap.error("--data or APE_DATA is required")
    if a.command == "verify":
        return verify(a.data)
    return install(a.data, a.root.resolve(), a.link, a.force, a.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
