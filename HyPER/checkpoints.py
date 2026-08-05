"""Shared checkpoint selection and metadata helpers."""

from __future__ import annotations

from pathlib import Path

import torch


def resolve_checkpoint(
    selector: str | None,
    model_directory: str | None,
    *,
    purpose: str = "Checkpoint",
) -> Path:
    if selector is None or not str(selector).strip():
        raise ValueError(
            f"{purpose} must be an explicit path or selector 'best'/'last'."
        )
    selector = str(selector).strip()
    direct = Path(selector).expanduser()
    if direct.is_file():
        return direct.resolve()
    if selector not in {"best", "last"}:
        raise FileNotFoundError(
            f"{purpose} must be an existing path or 'best'/'last', got {selector!r}."
        )
    if model_directory is None or not str(model_directory).strip():
        raise ValueError(
            f"Checkpoint selector {selector!r} requires a model directory."
        )
    directory = Path(str(model_directory)).expanduser() / "checkpoints"
    if selector == "last":
        candidate = directory / "last.ckpt"
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate.resolve()
    candidates = sorted(directory.glob("best-total*.ckpt"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Selector 'best' requires exactly one best-total checkpoint in "
            f"{directory}, found {len(candidates)}."
        )
    return candidates[0].resolve()


def checkpoint_metadata(path: str | Path) -> dict:
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu")
    callbacks = checkpoint.get("callbacks", {})
    monitor = score = None
    for state in callbacks.values():
        if isinstance(state, dict) and state.get("monitor") and "best_model_score" in state:
            monitor = state.get("monitor")
            value = state.get("best_model_score")
            if value is not None:
                score = float(value)
            break
    allowed = {"val_loss", "val_reconstruction_loss", "val_classification_loss"}
    if monitor not in allowed or score is None:
        raise RuntimeError(
            f"Checkpoint {path} does not contain a valid finite monitored callback score; "
            f"found monitor={monitor!r}, score={score!r}."
        )
    return {
        "checkpoint_path": str(path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "checkpoint_monitor": monitor,
        "checkpoint_score": score,
    }
