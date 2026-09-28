"""Decode-verify MM-AU image sequences and collect duplicate evidence.

This is a diagnostic pass over the released image directories. It answers three
questions the frame-index preflight (``mmau_preflight.py``) explicitly left open:

1. **Decodability.** The preflight only read file names and sizes. Here every
   frame is fully decoded (``PIL.Image.load`` at native resolution) so that a
   truncated or corrupt JPEG/PNG surfaces as a recorded exception instead of as
   a silent failure during training.
2. **Geometry consistency.** Width/height/mode are recorded per frame and
   collapsed to the set of distinct geometries per sequence. A sequence whose
   frame size changes mid-way is not a usable clip on a fixed grid.
3. **Duplicate evidence, separated by strength.** For a fixed set of probe
   positions the exact file digest (``blake2b`` of the raw bytes) and a 64-bit
   dHash are both recorded. Identical digests are *confirmed* byte-identical
   files. Equal or near-equal dHashes are only *candidates*: visually similar
   frames are not proof of one source event, and this script never merges
   anything on that basis.

The dataset is read-only. Nothing here writes into the dataset root, infers an
FPS, assigns a split, or produces an APE manifest.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import signal
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from hashlib import blake2b
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

# Pillow refuses very large images by default as a decompression-bomb guard.
# The released frames are ordinary dashcam resolutions; keep the guard.
Image.MAX_IMAGE_PIXELS = 80_000_000

#: How many probe frames per sequence carry byte digests and dHashes.
N_PROBES = 8

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _dhash(img: "Image.Image", size: int = 8) -> str:
    """64-bit difference hash of a decoded image, as 16 hex chars.

    Horizontal neighbour comparison on a ``(size+1, size)`` greyscale thumbnail.
    Returned as hex so that Hamming distance is computed downstream on ints.
    """
    small = img.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR)
    px = small.tobytes()
    bits = 0
    n = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits = (bits << 1) | int(px[base + col] < px[base + col + 1])
            n += 1
    return f"{bits:0{n // 4}x}"


def _probe_positions(count: int, anchor_idx: Optional[int]) -> List[int]:
    """Indices (into the sorted frame list) that get digests and hashes.

    Always includes the first and last frame and, when known, the anchor frame;
    the remainder are spread evenly so that a near-duplicate of a *different*
    segment of the same source video still collides on some probe.
    """
    if count <= 0:
        return []
    wanted = {0, count - 1}
    if anchor_idx is not None and 0 <= anchor_idx < count:
        wanted.add(anchor_idx)
    remaining = N_PROBES - len(wanted)
    if remaining > 0 and count > 2:
        for k in range(1, remaining + 1):
            wanted.add(min(count - 1, max(0, round(k * (count - 1) / (remaining + 1)))))
    return sorted(wanted)


def inspect_sequence(task: Dict[str, Any]) -> Dict[str, Any]:
    """Decode every frame of one sequence and summarise it.

    Runs in a worker process; returns a JSON-serialisable record and never
    raises for data problems -- a problem is part of the answer.
    """
    directory = Path(task["directory"])
    out: Dict[str, Any] = {
        "hashcode": task["hashcode"],
        "video_name": task["video_name"],
        "source": task.get("source"),
        "relative_image_directory": task.get("relative_image_directory"),
        "preflight_status": task.get("preflight_status"),
    }
    try:
        names = sorted(
            entry.name
            for entry in os.scandir(directory)
            if entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_SUFFIXES
        )
    except OSError as exc:
        out.update(decode_status="directory_error", error=f"{type(exc).__name__}: {exc}")
        return out

    # Frame order is the released numeric index, not lexicographic chance.
    def _key(name: str) -> Tuple[int, str]:
        stem = Path(name).stem
        return (int(stem) if stem.isdigit() else -1, name)

    names.sort(key=_key)
    count = len(names)
    anchor_frame = task.get("anchor_frame")
    anchor_idx: Optional[int] = None
    if isinstance(anchor_frame, int):
        for k, name in enumerate(names):
            stem = Path(name).stem
            if stem.isdigit() and int(stem) == anchor_frame:
                anchor_idx = k
                break

    probes = set(_probe_positions(count, anchor_idx))
    geometries: Dict[str, int] = {}
    failures: List[Dict[str, str]] = []
    probe_records: List[Dict[str, Any]] = []
    decoded = 0
    total_bytes = 0

    for k, name in enumerate(names):
        path = directory / name
        try:
            raw = path.read_bytes()
            total_bytes += len(raw)
            with Image.open(io.BytesIO(raw)) as img:
                img.load()  # full native-resolution decode, not a header peek
                geom = f"{img.width}x{img.height}:{img.mode}"
                geometries[geom] = geometries.get(geom, 0) + 1
                decoded += 1
                if k in probes:
                    probe_records.append(
                        {
                            "position": k,
                            "frame": Path(name).stem,
                            "is_anchor": anchor_idx is not None and k == anchor_idx,
                            "bytes": len(raw),
                            "file_digest": blake2b(raw, digest_size=16).hexdigest(),
                            "dhash": _dhash(img),
                        }
                    )
        except Exception as exc:  # noqa: BLE001 - the exception type is the datum
            if len(failures) < 20:
                failures.append({"frame": name, "error": f"{type(exc).__name__}: {exc}"})

    out.update(
        frame_files=count,
        decoded_ok=decoded,
        decode_failures=count - decoded,
        failure_examples=failures,
        geometries=geometries,
        n_distinct_geometries=len(geometries),
        total_bytes=total_bytes,
        anchor_position=anchor_idx,
        probes=probe_records,
        decode_status=("empty" if count == 0 else "all_frames_decoded" if decoded == count else "decode_failure"),
    )
    return out


def load_tasks(preflight: Path, root: Path) -> List[Dict[str, Any]]:
    """Build the work list from the frame-index preflight ledger."""
    tasks: List[Dict[str, Any]] = []
    with preflight.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rel = row.get("relative_image_directory")
            if not rel:
                continue
            anchor = row.get("anchor_frame")
            tasks.append(
                {
                    "hashcode": row["hashcode"],
                    "video_name": row["video_name"],
                    "source": row.get("source"),
                    "relative_image_directory": rel,
                    "preflight_status": row.get("status"),
                    "anchor_frame": int(anchor) if isinstance(anchor, (int, float)) and anchor >= 0 else None,
                    "directory": str(root / rel),
                }
            )
    return tasks


def done_keys(out_path: Path) -> set:
    """Recover a canonical ledger, archiving damaged or retryable records first."""
    if not out_path.exists():
        return set()
    original = out_path.read_bytes()
    complete = {}
    changed = bool(original and not original.endswith(b"\n"))
    terminal = {"all_frames_decoded", "decode_failure", "empty"}
    for line in original.splitlines():
        try:
            record = json.loads(line)
            key = record["hashcode"]
            if record.get("decode_status") not in terminal:
                changed = True
                continue
            if key in complete:
                changed = True
            complete[key] = record
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
            changed = True
    if changed:
        backup = out_path.with_name(out_path.name + f".before-recovery-{time.time_ns()}")
        backup.write_bytes(original)
        temporary = out_path.with_name(out_path.name + ".recovered.tmp")
        temporary.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in complete.values()), encoding="utf-8"
        )
        os.replace(temporary, out_path)
        logging.warning("Recovered ledger boundary/retry state; original evidence saved to %s", backup)
    return set(complete)


def run(root: Path, preflight: Path, out_dir: Path, workers: int, limit: Optional[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "decode.jsonl"
    tasks = load_tasks(preflight, root)
    already = done_keys(out_path)
    pending = [t for t in tasks if t["hashcode"] not in already]
    if limit is not None:
        pending = pending[:limit]
    logging.info(
        "sequences total=%d already=%d pending=%d workers=%d",
        len(tasks),
        len(already),
        len(pending),
        workers,
    )

    stop = {"flag": False}

    def _handle(signum: int, _frame: Any) -> None:
        logging.warning("signal %d received; finishing in-flight work then stopping", signum)
        stop["flag"] = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle)

    if workers < 1:
        raise ValueError("workers must be positive")
    written = 0
    remaining = iter(pending)
    with out_path.open("a", encoding="utf-8") as sink, ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}

        def submit_one() -> None:
            task = next(remaining, None)
            if task is not None:
                futures[pool.submit(inspect_sequence, task)] = task["hashcode"]

        for _ in range(2 * workers):
            submit_one()
        while futures:
            finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in finished:
                key = futures.pop(fut)
                try:
                    record = fut.result()
                except Exception as exc:  # noqa: BLE001 - task failure remains explicit evidence
                    record = {"hashcode": key, "decode_status": "worker_error", "error": str(exc)}
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                written += 1
                if written % 200 == 0:
                    logging.info("written %d/%d", written, len(pending))
                if not stop["flag"]:
                    submit_one()
    logging.info("done; wrote %d records to %s", written, out_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="MM-AU data root (read-only)")
    parser.add_argument("--preflight", type=Path, required=True, help="preflight.jsonl from mmau_preflight")
    parser.add_argument("--out", type=Path, required=True, help="output directory for decode.jsonl")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="inspect at most N pending sequences")
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args(argv)

    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if args.log is not None:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)

    run(args.root, args.preflight, args.out, args.workers, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
