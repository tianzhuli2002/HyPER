"""Canonical HyPER topology and task configuration helpers.

The public configuration surface uses exactly two topology names and four task
modes.  Legacy external configs are still readable so existing promoted
checkpoints can be evaluated, but current repository configs do not expose the
old pair of task booleans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf

CANONICAL_TOPOLOGIES = ("ttbar1L", "ttH")
CANONICAL_TASKS = ("reconstruction", "classification", "joint", "probe")

_TOPOLOGY_ALIASES = {
    "ttbar1l": "ttbar1L",
    "ttbar_single_lep": "ttbar1L",
    "tth": "ttH",
}


@dataclass(frozen=True)
class TaskSpec:
    mode: str
    classification_enabled: bool
    reconstruction_enabled: bool
    probe_enabled: bool
    prediction_product: str


_TASK_SPECS = {
    "reconstruction": TaskSpec("reconstruction", False, True, False, "selected"),
    "classification": TaskSpec("classification", True, False, False, "classifier"),
    "joint": TaskSpec("joint", True, True, False, "selected"),
    "probe": TaskSpec("probe", True, False, True, "classifier"),
}


def canonical_topology(value: Any) -> str:
    """Return the canonical topology name or raise a precise error."""
    if value is None or not str(value).strip():
        raise ValueError(
            "A topology is required. Choose exactly one of: ttbar1L, ttH."
        )
    raw = str(value).strip()
    if raw in CANONICAL_TOPOLOGIES:
        return raw
    normalised = raw.replace(" ", "_").lower()
    try:
        return _TOPOLOGY_ALIASES[normalised]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported topology {raw!r}. Choose exactly one of: "
            f"{', '.join(CANONICAL_TOPOLOGIES)}."
        ) from exc


def _legacy_task_mode(cfg: DictConfig) -> str | None:
    """Read old external configs without keeping the old toggles public."""
    classification = OmegaConf.select(cfg, "classification.enabled", default=None)
    reconstruction = OmegaConf.select(cfg, "reconstruction.enabled", default=None)
    probe = OmegaConf.select(cfg, "probe.enabled", default=False)
    if classification is None and reconstruction is None:
        return None
    classification = bool(classification)
    reconstruction = bool(reconstruction)
    if bool(probe):
        if not classification or reconstruction:
            raise ValueError(
                "Legacy probe configuration must enable classification and disable reconstruction."
            )
        return "probe"
    if classification and reconstruction:
        return "joint"
    if classification:
        return "classification"
    if reconstruction:
        return "reconstruction"
    raise ValueError("At least one HyPER task must be enabled.")


def task_spec(cfg_or_mode: DictConfig | str) -> TaskSpec:
    """Resolve one canonical task mode and its derived behaviour."""
    if isinstance(cfg_or_mode, str):
        mode = cfg_or_mode
    else:
        mode = OmegaConf.select(cfg_or_mode, "task.mode", default=None)
        if mode is None or not str(mode).strip():
            mode = _legacy_task_mode(cfg_or_mode)
    if mode is None or not str(mode).strip():
        raise ValueError(
            "A task is required. Choose exactly one of: reconstruction, "
            "classification, joint, probe."
        )
    mode = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "reconstruction_only": "reconstruction",
        "classification_only": "classification",
        "frozen_probe": "probe",
    }
    mode = aliases.get(mode, mode)
    try:
        return _TASK_SPECS[mode]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported task {mode!r}. Choose exactly one of: "
            f"{', '.join(CANONICAL_TASKS)}."
        ) from exc


def configured_topology(cfg: DictConfig) -> str:
    """Resolve the canonical topology from current or legacy config locations."""
    value = OmegaConf.select(cfg, "topology", default=None)
    if value is None:
        value = OmegaConf.select(cfg, "predicting.topology", default=None)
    return canonical_topology(value)


def validate_task_losses(cfg: DictConfig, spec: TaskSpec | None = None) -> None:
    """Reject task/loss combinations that silently optimise the wrong objective."""
    spec = task_spec(cfg) if spec is None else spec
    edge = float(OmegaConf.select(cfg, "loss.edge_weight", default=0.0))
    hyperedge = float(OmegaConf.select(cfg, "loss.hyperedge_weight", default=0.0))
    classification = float(
        OmegaConf.select(cfg, "loss.classification_weight", default=0.0)
    )
    if min(edge, hyperedge, classification) < 0:
        raise ValueError("Loss weights must be non-negative.")
    if spec.reconstruction_enabled and edge + hyperedge <= 0:
        raise ValueError(
            f"Task {spec.mode!r} requires a positive reconstruction loss weight."
        )
    if not spec.reconstruction_enabled and (edge != 0 or hyperedge != 0):
        raise ValueError(
            f"Task {spec.mode!r} must set edge_weight=0 and hyperedge_weight=0."
        )
    if spec.classification_enabled and classification <= 0:
        raise ValueError(
            f"Task {spec.mode!r} requires classification_weight > 0."
        )
    if not spec.classification_enabled and classification != 0:
        raise ValueError(
            f"Task {spec.mode!r} must set classification_weight=0."
        )


def validate_runtime_config(cfg: DictConfig) -> tuple[str, TaskSpec]:
    """Validate the small set of choices that define every production run."""
    topology = configured_topology(cfg)
    spec = task_spec(cfg)
    validate_task_losses(cfg, spec)
    if spec.probe_enabled and not bool(OmegaConf.select(cfg, "probe.enabled", default=True)):
        raise ValueError("Task 'probe' requires a probe configuration section.")
    return topology, spec


def set_task_mode(cfg: DictConfig, mode: str) -> TaskSpec:
    """Set a canonical task in mutable tuning configs and remove old toggles."""
    spec = task_spec(mode)
    OmegaConf.update(cfg, "task.mode", spec.mode, merge=False, force_add=True)
    if "classification" in cfg and isinstance(cfg.classification, DictConfig):
        cfg.classification.pop("enabled", None)
    if "reconstruction" in cfg and isinstance(cfg.reconstruction, DictConfig):
        cfg.reconstruction.pop("enabled", None)
    return spec
