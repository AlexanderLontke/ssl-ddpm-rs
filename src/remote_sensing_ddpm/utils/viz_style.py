import seaborn as sns
import matplotlib as mpl


def set_matplotlib_style():
    # Theme
    sns.set_style("whitegrid")
    # Context
    sns.set_context("paper", font_scale=2)
    # Errorbars
    mpl.rcParams["errorbar.capsize"] = 5.0

    mpl.rcParams["legend.fontsize"] = "xx-small"  # using a named size
    mpl.rcParams["image.cmap"] = "viridis"

    # Set the font family to Times New Roman
    # mpl.rcParams['font.family'] = 'serif'
    # mpl.rcParams['font.serif'] = ['Times New Roman']
