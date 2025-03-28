from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PAPER_TEXTWIDTH = 5.1483  # inches
NICE_METRIC_MAPPING = {
    "Dice_TC": "DSC (TC)",
    "Dice_WT": "DSC (WT)",
    "Dice_ET": "DSC (ET)",
    "Dice_mean": "Mean DSC",
    "Hausdorff95_WT": "Hausdorff-95 (WT)",
    "Hausdorff95_TC": "Hausdorff-95 (TC)",
    "Hausdorff95_ET": "Hausdorff-95 (ET)",
    "Hausdorff95_mean": "Mean Hausdorff-95",
    "communication_metric": "Convergence score",
    "Ranking score": "Rank score",
}
REGION_LIST = ["WT", "TC", "ET"]
DICE_METRICS = ["Dice_WT", "Dice_TC", "Dice_ET"]
HAUSD_METRICS = ["Hausdorff95_WT", "Hausdorff95_TC", "Hausdorff95_ET"]


def plt_save_and_close(fig, output_file: str, save_as=None, **kwargs):
    output_file = Path(output_file)
    default_kwargs = {
        "bbox_inches": "tight",
        "dpi": 300,
        "pad_inches": 0.005 * PAPER_TEXTWIDTH,
    }
    default_kwargs.update(kwargs)
    if save_as is None:
        save_as = [".png", ".pdf"]
    if len(output_file.suffix) > 0 and output_file.suffix not in save_as:
        save_as.append(output_file.suffix)
    for ext in save_as:
        fig.savefig(output_file.with_suffix(ext), **default_kwargs)
    plt.close(fig)


def get_figsize(textwidth_factor=1.0, aspect_ratio=None):
    width = PAPER_TEXTWIDTH * textwidth_factor
    if aspect_ratio is None:
        aspect_ratio = (np.sqrt(5) - 1.0) / 2.0  # because it looks good
    height = width * aspect_ratio  # figure height in inches
    return width, height
