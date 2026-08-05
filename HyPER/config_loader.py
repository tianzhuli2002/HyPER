"""Compose one resolved HyPER runtime configuration."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from HyPER.configuration import canonical_topology, task_spec, validate_runtime_config


def compose_runtime_config(
    topology: str,
    task: str,
    overrides: list[str] | tuple[str, ...] = (),
) -> DictConfig:
    """Compose the canonical Hydra config without changing the working directory."""
    from hydra import compose, initialize_config_dir

    topology = canonical_topology(topology)
    task = task_spec(task).mode
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="default",
            overrides=[f"topology={topology}", f"task={task}", *list(overrides)],
        )
    validate_runtime_config(cfg)
    return cfg


def write_resolved_config(cfg: DictConfig, output: str | Path) -> Path:
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    temporary.replace(path)
    return path
