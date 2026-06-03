"""Utilities for loading HyPER prediction outputs."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from collections.abc import Iterator

import pandas as pd


LOGGER = logging.getLogger(__name__)


def _natural_part_key(path: Path) -> tuple:
    parts: list[int | str] = []
    for piece in re.split(r"(\d+)", path.name):
        if not piece:
            continue
        parts.append(int(piece) if piece.isdigit() else piece.lower())
    return tuple(parts)


def _parts_dir(path: Path) -> Path:
    if path.exists() and path.is_dir():
        return path
    if path.name.endswith(".pkl.parts"):
        return path
    raise ValueError(f"Not a HyPER pickle-parts path: {path}")


def _part_files(parts_dir: Path) -> list[Path]:
    manifest_path = parts_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest_files = manifest.get("part_files") or []
        part_paths = [parts_dir / name for name in manifest_files]
        if part_paths and all(path.exists() for path in part_paths):
            return part_paths
        LOGGER.warning(
            "Manifest exists at %s but did not point to readable part files; "
            "falling back to directory scan.",
            manifest_path,
        )

    candidates = [
        path
        for path in parts_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".pkl", ".pickle"}
    ]
    return sorted(candidates, key=_natural_part_key)


def _manifest_n_events(parts_dir: Path) -> int | None:
    manifest_path = parts_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    n_events = manifest.get("n_events")
    try:
        return int(n_events) if n_events is not None else None
    except (TypeError, ValueError):
        return None


def _load_pickle_parts(parts_dir: Path, max_events: int | None = None) -> pd.DataFrame:
    part_files = _part_files(parts_dir)
    if not part_files:
        raise FileNotFoundError(f"No pickle part files found in {parts_dir}")

    LOGGER.info("Found %d HyPER pickle part files in %s", len(part_files), parts_dir)
    frames: list[pd.DataFrame] = []
    loaded_rows = 0
    truncated = False

    for part_file in part_files:
        if max_events is not None and loaded_rows >= max_events:
            truncated = True
            break

        frame = pd.read_pickle(part_file)
        part_rows = len(frame)
        LOGGER.info("Loaded %s with %d rows", part_file.name, part_rows)

        if max_events is not None and loaded_rows + part_rows > max_events:
            keep = max_events - loaded_rows
            frame = frame.iloc[:keep].copy()
            truncated = True
            LOGGER.info(
                "Truncated %s to %d rows to respect max_events=%d",
                part_file.name,
                keep,
                max_events,
            )

        frames.append(frame)
        loaded_rows += len(frame)

    if not frames:
        result = pd.DataFrame()
    else:
        result = pd.concat(frames, ignore_index=True)

    total_rows = _manifest_n_events(parts_dir)
    if total_rows is None:
        total_rows = loaded_rows

    result.attrs["hyper_total_rows"] = int(total_rows)
    result.attrs["hyper_loaded_rows"] = int(len(result))
    result.attrs["hyper_truncated"] = bool(truncated)
    result.attrs["hyper_parts_found"] = int(len(part_files))

    LOGGER.info(
        "Total HyPER rows loaded: %d%s",
        len(result),
        " (truncated by max_events)" if truncated else "",
    )
    return result


def load_hyper_prediction_output(
    path: str | Path,
    max_events: int | None = None,
) -> pd.DataFrame:
    """Load a HyPER prediction output from a single file or `.pkl.parts` directory.

    Supported inputs are `.pkl`, `.pickle`, `.csv`, line-delimited `.json` /
    `.jsonl`, and directories or prefixes ending in `.pkl.parts`.
    """
    output_path = Path(path)
    lower_name = output_path.name.lower()

    if max_events is not None and max_events < 0:
        raise ValueError("max_events must be non-negative or None")

    if lower_name.endswith(".pkl.parts") or (output_path.exists() and output_path.is_dir()):
        return _load_pickle_parts(_parts_dir(output_path), max_events=max_events)

    if not output_path.exists():
        raise FileNotFoundError(output_path)

    if lower_name.endswith((".pkl", ".pickle")):
        frame = pd.read_pickle(output_path)
    elif lower_name.endswith(".csv"):
        frame = pd.read_csv(output_path, nrows=max_events)
    elif lower_name.endswith((".json", ".jsonl")):
        frame = pd.read_json(output_path, lines=True)
    else:
        raise ValueError(f"Unsupported HyPER output format: {output_path}")

    total_rows = len(frame)
    truncated = False
    if max_events is not None and len(frame) > max_events:
        frame = frame.iloc[:max_events].copy()
        truncated = True

    frame.attrs["hyper_total_rows"] = int(total_rows)
    frame.attrs["hyper_loaded_rows"] = int(len(frame))
    frame.attrs["hyper_truncated"] = bool(truncated)
    frame.attrs["hyper_parts_found"] = 0

    LOGGER.info(
        "Loaded %s with %d rows%s",
        output_path,
        len(frame),
        " (truncated by max_events)" if truncated else "",
    )
    return frame


def iter_hyper_prediction_parts(
    path: str | Path,
    max_events: int | None = None,
    chunk_size: int | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield HyPER prediction rows in bounded chunks.

    `.pkl.parts` directories are streamed part-by-part. Other formats fall back
    to the existing loader and then yield slices, which is still useful for
    keeping downstream H5/evaluation memory bounded.
    """
    output_path = Path(path)
    lower_name = output_path.name.lower()

    if max_events is not None and max_events < 0:
        raise ValueError("max_events must be non-negative or None")
    if chunk_size is not None and int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive or None")

    if lower_name.endswith(".pkl.parts") or (output_path.exists() and output_path.is_dir()):
        parts_dir = _parts_dir(output_path)
        part_files = _part_files(parts_dir)
        if not part_files:
            raise FileNotFoundError(f"No pickle part files found in {parts_dir}")

        total_rows = _manifest_n_events(parts_dir)
        loaded_rows = 0
        buffer: list[pd.DataFrame] = []
        buffer_rows = 0

        def flush() -> pd.DataFrame | None:
            nonlocal buffer, buffer_rows
            if not buffer:
                return None
            frame = pd.concat(buffer, ignore_index=True) if len(buffer) > 1 else buffer[0].reset_index(drop=True)
            buffer = []
            buffer_rows = 0
            return frame

        for part_file in part_files:
            if max_events is not None and loaded_rows >= max_events:
                break
            frame = pd.read_pickle(part_file)
            if max_events is not None and loaded_rows + len(frame) > max_events:
                frame = frame.iloc[: max_events - loaded_rows].copy()
            loaded_rows += len(frame)

            if chunk_size is None:
                frame.attrs["hyper_total_rows"] = int(total_rows if total_rows is not None else loaded_rows)
                frame.attrs["hyper_loaded_rows"] = int(len(frame))
                frame.attrs["hyper_truncated"] = bool(max_events is not None and loaded_rows >= max_events)
                frame.attrs["hyper_parts_found"] = int(len(part_files))
                yield frame
                continue

            buffer.append(frame)
            buffer_rows += len(frame)
            while buffer_rows >= int(chunk_size):
                combined = pd.concat(buffer, ignore_index=True) if len(buffer) > 1 else buffer[0].reset_index(drop=True)
                out = combined.iloc[: int(chunk_size)].copy()
                remainder = combined.iloc[int(chunk_size):].reset_index(drop=True)
                buffer = [remainder] if len(remainder) else []
                buffer_rows = len(remainder)
                out.attrs["hyper_total_rows"] = int(total_rows if total_rows is not None else loaded_rows)
                out.attrs["hyper_loaded_rows"] = int(len(out))
                out.attrs["hyper_truncated"] = bool(max_events is not None and loaded_rows >= max_events)
                out.attrs["hyper_parts_found"] = int(len(part_files))
                yield out

        tail = flush()
        if tail is not None and len(tail):
            tail.attrs["hyper_total_rows"] = int(total_rows if total_rows is not None else loaded_rows)
            tail.attrs["hyper_loaded_rows"] = int(len(tail))
            tail.attrs["hyper_truncated"] = bool(max_events is not None and loaded_rows >= max_events)
            tail.attrs["hyper_parts_found"] = int(len(part_files))
            yield tail
        return

    frame = load_hyper_prediction_output(output_path, max_events=max_events)
    if chunk_size is None:
        yield frame
        return
    for start in range(0, len(frame), int(chunk_size)):
        chunk = frame.iloc[start : start + int(chunk_size)].copy()
        chunk.attrs.update(frame.attrs)
        yield chunk
