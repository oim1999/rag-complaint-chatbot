from pathlib import Path
import matplotlib.pyplot as plt
from src.config import FIGURES_DIR


def save_figure(filename: str, dpi: int = 150) -> Path:
    """
    Save the active matplotlib figure to FIGURES_DIR/<filename>.
    Directory is created if it does not exist.

    Parameters
    ----------
    filename : str
        File name including extension, e.g. 'product_distribution.png'.
    dpi : int
        Resolution of saved image.

    Returns
    -------
    Path
        Absolute path where the figure was saved.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Figure saved → {path}")
    return path