"""Artifact-only validator for a completed standalone Stage-1 smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker", required=True, type=Path)
    parser.add_argument("--topology", required=True, choices=("ttbar1L", "ttH"))
    parser.add_argument("--required-completed-trials", type=int, default=2)
    args = parser.parse_args()
    marker = args.marker.resolve()
    if not marker.is_file():
        raise FileNotFoundError(marker)
    failure = marker.parent / "smoke_failure.json"
    if failure.exists():
        raise RuntimeError(f"Smoke failure marker exists: {failure}")
    data = json.loads(marker.read_text(encoding="utf-8"))
    required = {
        "schema_version", "topology", "completed_trials", "monitor", "direction",
        "best_objective_value", "best_objective_epoch", "final_objective_value",
        "final_objective_epoch", "gradients_finite", "test_overlap_count", "result_paths",
        "study_sqlite_path", "study_name", "graph_db_path", "graph_db_manifest_hash",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise RuntimeError(f"Smoke marker is missing fields: {missing}")
    if data["schema_version"] != SCHEMA_VERSION or data["topology"] != args.topology:
        raise RuntimeError("Smoke schema/topology mismatch.")
    if int(data["completed_trials"]) < args.required_completed_trials:
        raise RuntimeError("Smoke has too few completed trials.")
    if data["monitor"] != "val_reconstruction_loss" or data["direction"] != "min":
        raise RuntimeError("Smoke objective contract mismatch.")
    for name in ("best_objective_value", "final_objective_value"):
        if not math.isfinite(float(data[name])):
            raise RuntimeError(f"Smoke field {name} is not finite.")
    if data["best_objective_epoch"] is None or data["final_objective_epoch"] is None:
        raise RuntimeError("Smoke objective epochs are missing.")
    if data["gradients_finite"] is not True or int(data["test_overlap_count"]) != 0:
        raise RuntimeError("Smoke gradients or test-overlap gate failed.")
    for path in data["result_paths"]:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    sqlite = Path(data["study_sqlite_path"])
    if not sqlite.is_file():
        raise FileNotFoundError(sqlite)
    with sqlite3.connect(f"file:{sqlite.resolve()}?mode=ro", uri=True) as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM trials AS t JOIN studies AS s ON s.study_id=t.study_id "
            "WHERE s.study_name=? AND t.state='COMPLETE'",
            (data["study_name"],),
        ).fetchone()
    completed = int(row[0])
    if completed != int(data["completed_trials"]):
        raise RuntimeError(f"Study/marker completed-trial mismatch: {completed} != {data['completed_trials']}")
    graph_manifest = Path(str(data["graph_db_path"]) + ".manifest.json")
    if not graph_manifest.is_file() or file_hash(graph_manifest) != data["graph_db_manifest_hash"]:
        raise RuntimeError("Persistent graph manifest hash does not match the smoke's staged source.")
    print(json.dumps({"valid": True, "marker": str(marker), "completed_trials": completed}, sort_keys=True))


if __name__ == "__main__":
    main()
