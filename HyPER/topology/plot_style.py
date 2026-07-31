"""Small shared plotting style for HyPER classification and reconstruction studies."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

BACKGROUND_COLOUR = "#4C566A"
SIGNAL_COLOUR = "#3B82F6"
FM_COLOUR = "#2A9D8F"
NONFM_COLOUR = "#E07A5F"
REFERENCE_COLOUR = "#8A8F98"
JOINT_COLOUR = "#7C3AED"
ZERO_SHOT_COLOUR = "#228B60"
DIRECT_COLOUR = "#7A7A7A"
ALIGNED_COLOUR = "#C44E52"
SHUFFLED_COLOUR = "#B8B8B8"
RANDOM_COLOUR = "#9A7D0A"

METHOD_COLOURS = {
    "native_classification_only_score": SIGNAL_COLOUR,
    "native_joint_score": JOINT_COLOUR,
    "reconstruction_zero_shot_score": ZERO_SHOT_COLOUR,
    "joint_reconstruction_zero_shot_score": FM_COLOUR,
    "reconstruction_to_classification_direct_score": DIRECT_COLOUR,
    "reconstruction_to_classification_paired_score": ALIGNED_COLOUR,
    "reconstruction_to_joint_direct_score": DIRECT_COLOUR,
    "reconstruction_to_joint_paired_score": ALIGNED_COLOUR,
    "joint_to_classification_direct_score": DIRECT_COLOUR,
    "joint_to_classification_paired_score": ALIGNED_COLOUR,
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (7.4, 5.5),
            "figure.dpi": 130,
            "savefig.dpi": 250,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "font.size": 11.5,
            "axes.labelsize": 12.5,
            "axes.titlesize": 14,
            "axes.titleweight": "normal",
            "axes.linewidth": 1.05,
            "axes.grid": False,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 5.5,
            "ytick.major.size": 5.5,
            "xtick.minor.size": 2.8,
            "ytick.minor.size": 2.8,
            "legend.frameon": False,
            "legend.fontsize": 10.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def decorate_axis(ax, *, title: str | None = None, minor_ticks: bool = True) -> None:
    ax.tick_params(which="both", direction="in", top=True, right=True)
    if minor_ticks:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    if title:
        ax.set_title(title, pad=12)


def save_figure(fig, output_dir: str | Path, stem: str, formats: Iterable[str] = ("pdf", "png")) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt = str(fmt).lower().lstrip(".")
        kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            kwargs["dpi"] = 250
        fig.savefig(output / f"{stem}.{fmt}", **kwargs)
    plt.close(fig)
