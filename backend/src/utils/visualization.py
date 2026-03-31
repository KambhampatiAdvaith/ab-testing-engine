import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def setup_style() -> None:
    """Set a consistent, clean plot style across all visualizations."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        try:
            plt.style.use('seaborn-whitegrid')
        except Exception:
            pass  # fall back to default style


def save_figure(fig: plt.Figure, path: str, dpi: int = 100) -> None:
    """
    Save a matplotlib figure to disk, creating directories as needed.

    Args:
        fig: Matplotlib Figure object
        path: Output file path
        dpi: Resolution in dots per inch
    """
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {path}")


def create_figure(figsize: tuple = (10, 6)) -> tuple:
    """
    Create a new matplotlib Figure and Axes pair.

    Args:
        figsize: Figure dimensions as (width, height) in inches

    Returns:
        Tuple of (fig, ax)
    """
    return plt.subplots(figsize=figsize)
