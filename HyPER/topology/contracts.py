"""Shared topology post-processing contracts.

Topology reconstruction functions accept either an in-memory prediction table or
one of HyPER's supported prediction outputs.  Classification inclusion is derived
from the canonical task mode unless an explicit override is supplied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .prediction_io import load_hyper_prediction_output

_CLASSIFICATION_TASKS = {"classification", "joint", "probe"}


def load_prediction_frame(value: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, (str, Path)):
        return load_hyper_prediction_output(str(value))
    raise TypeError(
        f"Prediction input must be a path or pandas.DataFrame, got {type(value).__name__}."
    )


def load_config_mapping(config: str | Path | dict[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        return config
    path = Path(config).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Config root must be a mapping: {path}")
    return loaded


def classification_enabled(
    config: str | Path | dict[str, Any] | None = None,
    explicit: bool | None = None,
) -> bool:
    """Resolve classifier-column inclusion from one canonical task mode.

    ``explicit`` is retained for callers operating directly on a prediction
    table.  Repository configs use ``task.mode``.  Legacy external configs are
    accepted only at this I/O boundary so old checkpoints can still be inspected.
    """
    if explicit is not None:
        return bool(explicit)
    cfg = load_config_mapping(config)
    mode = str(cfg.get("task", {}).get("mode", "")).strip().lower().replace("-", "_")
    if mode:
        if mode not in {"reconstruction", "classification", "joint", "probe"}:
            raise ValueError(f"Unsupported task.mode {mode!r}.")
        return mode in _CLASSIFICATION_TASKS
    legacy = cfg.get("classification", {}).get("enabled")
    return bool(legacy) if legacy is not None else False


def append_classifier_column(
    columns: list[str],
    predictions: pd.DataFrame,
    enabled: bool,
) -> list[str]:
    if not enabled:
        return columns
    if "HyPER_CLS_PROB" not in predictions.columns:
        raise KeyError(
            "The selected task includes classification, but HyPER_CLS_PROB is absent "
            "from the prediction output. Re-run prediction with a classification or joint model."
        )
    return [*columns, "HyPER_CLS_PROB"]
