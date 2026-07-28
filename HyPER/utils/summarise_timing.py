#!/usr/bin/env python3
"""Print compact timing summaries for HyPER timing JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timing_json", nargs="+")
    return parser.parse_args()


def _get(payload, *path):
    cur = payload
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def main() -> int:
    args = parse_args()
    rows = []
    for path in args.timing_json:
        timing_path = Path(path)
        with timing_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows.append(
            {
                "run": timing_path.parent.name,
                "batch_size": payload.get("batch_size"),
                "num_workers": payload.get("num_workers"),
                "pin_memory": payload.get("pin_memory"),
                "persistent": payload.get("persistent_workers"),
                "prefetch": payload.get("prefetch_factor"),
                "mean_step_s": _get(payload, "train_step", "mean_s"),
                "median_step_s": _get(payload, "train_step", "median_s"),
                "epoch_s": _get(payload, "train_epoch", "mean_s"),
                "events_per_s": payload.get("events_per_second_epoch"),
                "file": str(timing_path),
            }
        )

    headers = ["run", "batch_size", "num_workers", "pin_memory", "persistent", "prefetch", "mean_step_s", "epoch_s", "events_per_s"]
    print("\t".join(headers))
    for row in rows:
        print("\t".join("" if row.get(name) is None else str(row.get(name)) for name in headers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
