"""Strict YAML-defined Optuna search spaces using ordinary config paths."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from omegaconf import DictConfig, OmegaConf


_ALLOWED_KEYS = {"type", "low", "high", "log", "step", "values"}


def _plain(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else deepcopy(value)


def _target(config: Mapping[str, Any], path: str) -> tuple[Mapping[str, Any], str, Any]:
    if not isinstance(path, str) or not path or path.startswith(".") or path.endswith("."):
        raise ValueError(f"Invalid config path {path!r}.")
    parts = path.split(".")
    cursor: Any = config
    for part in parts[:-1]:
        if not isinstance(cursor, Mapping) or part not in cursor:
            raise KeyError(f"Search-space config path does not exist: {path}")
        cursor = cursor[part]
    key = parts[-1]
    if not isinstance(cursor, Mapping) or key not in cursor:
        raise KeyError(f"Search-space config path does not exist: {path}")
    return cursor, key, cursor[key]


def validate_search_space(config: DictConfig | Mapping, search_space: Mapping) -> None:
    plain = _plain(config)
    if not isinstance(search_space, Mapping):
        raise TypeError("tuning.search_space must be a mapping keyed by config path.")
    names = list(search_space)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate search-space parameter names are not allowed.")
    for path, raw_spec in search_space.items():
        _, _, current = _target(plain, path)
        if not isinstance(raw_spec, Mapping):
            raise TypeError(f"Search-space specification for {path} must be a mapping.")
        unknown = set(raw_spec) - _ALLOWED_KEYS
        if unknown:
            raise ValueError(f"Unknown search-space keys for {path}: {sorted(unknown)}")
        kind = str(raw_spec.get("type", "")).lower()
        if kind not in {"float", "int", "categorical"}:
            raise ValueError(f"Unknown parameter type {kind!r} for {path}.")
        if isinstance(current, bool):
            raise TypeError(f"Boolean config field {path} cannot be tuned as a numeric parameter.")
        if kind == "categorical":
            values = list(raw_spec.get("values", []))
            if not values:
                raise ValueError(f"Categorical parameter {path} has no choices.")
            if isinstance(current, bool):
                compatible = all(isinstance(value, bool) for value in values)
            elif isinstance(current, int):
                compatible = all(isinstance(value, int) and not isinstance(value, bool) for value in values)
            elif isinstance(current, float):
                compatible = all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values)
            elif isinstance(current, str):
                compatible = all(isinstance(value, str) for value in values)
            else:
                compatible = all(isinstance(value, type(current)) for value in values)
            if not compatible:
                raise TypeError(f"Categorical choices for {path} are incompatible with {type(current).__name__}.")
        else:
            if "low" not in raw_spec or "high" not in raw_spec:
                raise ValueError(f"Numeric parameter {path} requires low and high bounds.")
            low, high = raw_spec["low"], raw_spec["high"]
            if isinstance(low, bool) or isinstance(high, bool) or float(low) >= float(high):
                raise ValueError(f"Invalid bounds for {path}: low={low}, high={high}.")
            if bool(raw_spec.get("log", False)) and float(low) <= 0:
                raise ValueError(f"Log-scaled parameter {path} requires low > 0.")
            if raw_spec.get("step") is not None and float(raw_spec["step"]) <= 0:
                raise ValueError(f"Parameter {path} requires a positive step.")
            if bool(raw_spec.get("log", False)) and raw_spec.get("step") is not None:
                raise ValueError(f"Optuna does not support log and step together for {path}.")
            if kind == "int" and not isinstance(current, int):
                raise TypeError(f"Integer parameter {path} targets non-integer field {type(current).__name__}.")
            if kind == "float" and not isinstance(current, (int, float)):
                raise TypeError(f"Float parameter {path} targets non-numeric field {type(current).__name__}.")


def sample_and_apply(trial, config: DictConfig | Mapping, search_space: Mapping) -> tuple[DictConfig, dict]:
    validate_search_space(config, search_space)
    plain = _plain(config)
    sampled = {}
    for path, spec_cfg in search_space.items():
        spec = dict(spec_cfg)
        kind = str(spec["type"]).lower()
        if kind == "float":
            kwargs = {"log": bool(spec.get("log", False))}
            if spec.get("step") is not None:
                kwargs["step"] = float(spec["step"])
            value = trial.suggest_float(path, float(spec["low"]), float(spec["high"]), **kwargs)
        elif kind == "int":
            kwargs = {"log": bool(spec.get("log", False))}
            if spec.get("step") is not None:
                kwargs["step"] = int(spec["step"])
            value = trial.suggest_int(path, int(spec["low"]), int(spec["high"]), **kwargs)
        else:
            value = trial.suggest_categorical(path, list(spec["values"]))
        parent, key, _ = _target(plain, path)
        parent[key] = value
        if parent[key] != value:
            raise RuntimeError(f"Sampled value for {path} was not applied.")
        sampled[path] = value
    return OmegaConf.create(plain), sampled
