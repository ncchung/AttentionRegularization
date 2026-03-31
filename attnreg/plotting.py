import matplotlib.pyplot as plt
import cmasher as cmr
import matplotlib.colors as clr
import matplotlib.patches as patches
import seaborn as sns
from typing import Optional
import numpy as np
import pandas as pd
from pathlib import Path

# Define custom diverging color map
MY_CMAP = clr.LinearSegmentedColormap.from_list(
    "Random gradient 1030",
    (
        (0.000, (0.000, 0.890, 1.000)),
        (0.370, (0.263, 0.443, 0.671)),
        (0.500, (0.000, 0.000, 0.000)),
        (0.630, (0.545, 0.353, 0.267)),
        (1.000, (1.000, 0.651, 0.000)),
    ),
)

def get_cmap(heatmap):
    """Return a diverging colormap, such that 0 is at the center(black)"""
    if heatmap.min() > 0 and heatmap.max() > 0:
        bottom = 0.5
        top = 1.0
    elif heatmap.min() < 0 and heatmap.max() < 0:
        bottom = 0.0
        top = 0.5
    else:
        bottom = 0.5 - abs((heatmap.min() / abs(heatmap).max()) / 2)
        top = 0.5 + abs((heatmap.max() / abs(heatmap).max()) / 2)
    return cmr.get_sub_cmap(MY_CMAP, bottom, top)

def get_attention_cmap(heatmap):
    """Return a diverging colormap, such that 0 is at the center(black)"""
    if heatmap.min() > 0 and heatmap.max() > 0:
        bottom = 0.5
        top = 1.0
    elif heatmap.min() < 0 and heatmap.max() < 0:
        bottom = 0.0
        top = 0.5
    else:
        bottom = 0.5 - abs((heatmap.min() / abs(heatmap).max()) / 2)
        top = 0.5 + abs((heatmap.max() / abs(heatmap).max()) / 2)
    return cmr.get_sub_cmap(MY_CMAP, bottom, top)


def plot_attention(
    original,
    maps,
    stats,
    titles=None,
    cmap_fixed=True,
    savename=None,
    figsize=(18, 7),
):
    """Plot the original image, attention maps, and statistic maps."""
    
    total_maps = maps + stats
    num_plots = 1 + len(total_maps)

    if titles is None:
        titles = [""] * len(total_maps)

    fig, axs = plt.subplots(
        1, num_plots, figsize=figsize, layout="constrained", squeeze=False
    )
    axs = axs[0]

    # Plot original image
    axs[0].imshow(original)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    # Fixed colormap for attention maps if requested
    fixed_cmap = get_attention_cmap(maps[0]) if (cmap_fixed and maps) else None

    for i, m in enumerate(total_maps):
        ax = axs[i + 1]

        if i < len(maps):
            # Attention maps
            cmap = fixed_cmap if cmap_fixed else get_attention_cmap(m)
        else:
            # Stats maps
            cmap = "plasma" if (m.min() >= 0 and m.max() <= 1) else get_attention_cmap(m)

        im = ax.imshow(m, cmap=cmap)
        ax.set_title(titles[i])
        ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.3)

    if savename:
        plt.savefig(
            f"relevance_maps/attention_{savename}.pdf",
            bbox_inches="tight",
            pad_inches=0,
        )

    plt.show()


def plot_statistics(
    z,
    p,
    l,
    null_z,
    z_roi=None,
    bins_l=50,
    bins_p=50,
    bins_z=100,
    range_l=(0.0,1.0),
    range_p=(0.0,1.0),
    range_z=(-7.0,7.0),
    savename=None,
    figsize=(18, 5),
    fontsize=12,
)->None:
    
    with plt.rc_context({'font.size': fontsize}):  
        fig, axs = plt.subplots(1, 3, figsize=figsize)
        
        axs[0].hist(l.flatten(), bins=bins_l, range=range_l, color='blue', alpha=0.7)
        axs[0].set_title('Histogram of LFDR')
        axs[0].set_xlabel('l values')
        axs[0].set_ylabel('Frequency')
        
        # Histogram for p_roi
        axs[1].hist(p.flatten(), bins=bins_p, range=range_p, color='red', alpha=0.7)
        axs[1].set_title('Histogram of p-values')
        axs[1].set_xlabel('p values')
        axs[1].set_ylabel('Frequency')
        
        # Histogram for z
        axs[2].hist(z.flatten(), bins=bins_z, range=range_z, color='green', alpha=0.5, label='Observed', density=True)
        axs[2].hist(null_z.flatten(), bins=bins_z, color='grey', alpha=0.5, label='Bootstrap', density=True)
        
        axs[2].set_title('Histogram of z-scores in the image')
        axs[2].set_xlabel('Values')
        axs[2].set_ylabel('Frequency')
        axs[2].legend()
        
        plt.tight_layout()
        
        if savename:
            plt.savefig(
                f"relevance_maps/stats_histograms_{savename}.pdf", 
                format="pdf", 
                bbox_inches="tight"
            )
            
        plt.show()


def plot_statistics_in_ROI(
    z,
    z_roi,
    p_roi,
    l_roi,
    null_z,
    bins_l=50,
    bins_p=50,
    bins_z=100,
    range_l=(0.0,1.0),
    ylim_l: float | None = None,
    range_p=(0.0,1.0),
    ylim_p: float | None = None,
    range_z=(-7.0,7.0),
    ylim_z: float | None = None,
    savename=None,
    figsize=(18, 5),
    fontsize=12,
)->None:
    
    with plt.rc_context({'font.size': fontsize}):  
        fig, axs = plt.subplots(1, 3, figsize=figsize)
        
        axs[0].hist(l_roi.flatten(), bins=bins_l, range=range_l, color='blue', alpha=0.7)
        axs[0].set_title('Histogram of LFDR in ROI')
        axs[0].set_xlabel('l values')
        axs[0].set_ylabel('Frequency')
        if ylim_l:
            axs[0].set_ylim(0,ylim_l)
        
        # Histogram for p_roi
        axs[1].hist(p_roi.flatten(), bins=bins_p, range=range_p, color='red', alpha=0.7)
        axs[1].set_title('Histogram of p-values in ROI')
        axs[1].set_xlabel('p values')
        axs[1].set_ylabel('Frequency')
        if ylim_p:
            axs[1].set_ylim(0,ylim_p)
        
        # Histogram for z
        axs[2].hist(z.flatten(), bins=bins_z, color='green', alpha=0.5, label='Observed', density=True)
        axs[2].hist(null_z.flatten(), bins=bins_z, color='grey', alpha=0.5, label='Bootstrap', density=True)
        axs[2].hist(z_roi.flatten(), bins=bins_z, color='red', alpha=0.2, label='Observed in ROI', density=True, range=(z.min(), z.max()))
        if ylim_z:
            axs[2].set_ylim(0,ylim_z)
        
        axs[2].set_title('Histogram of z-scores in the image')
        axs[2].set_xlabel('Values')
        axs[2].set_ylabel('Frequency')
        axs[2].legend()
        
        #plt.tight_layout()
        
        if savename:
            plt.savefig(
                f"../results/plots/stats_histograms_{savename}.pdf", 
                format="pdf", 
                bbox_inches="tight"
            )
            
        plt.show()
        

def plot_attention_cdam(
    original_img, 
    attention_map, 
    cdam, 
    preds, 
    save_name: Optional[str] = None,
):
    # Create a grid of subplots: 1st row with 5 plots, 2nd row with 1 plot spanning all columns
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')

    # Original img:
    cmap = "grey" if original_img.mode == "L" else None
    axs[0][0].imshow(original_img, cmap=cmap)
    axs[0][0].set_title("Original")
    axs[0][0].axis('off')  # Hide axis

    # Attention map:
    axs[0][1].imshow(attention_map, cmap=get_cmap(attention_map))
    axs[0][1].set_title("Attention Map")
    axs[0][1].axis('off')  # Hide axis
    # Calculate and display the sum of pixel values for the attention map
    axs[0][1].text(0.5, -0.05, f"Sum: {attention_map.sum():.2f}", ha='center', va='top', transform=axs[0][1].transAxes, fontsize=10, color='black')

    # Raw CDAM Map:
    map_ = cdam
    cdam1 = axs[0][2].imshow(map_, cmap=get_cmap(map_))
    axs[0][2].set_title(f"Raw CDAM map for {preds}")
    axs[0][2].axis('off')  # Hide axis
    # Calculate and display the sum of pixel values for the Raw CDAM map
    axs[0][2].text(0.5, -0.05, f"Sum: {cdam.sum():.2f}", ha='center', va='top', transform=axs[0][2].transAxes, fontsize=10, color='black')
    # add colorbar
    fig.colorbar(cdam1, ax=axs[0][2], shrink=0.6)

    # attention map histogram
    #pixel_values = attention_map #.flatten()
    sns.histplot(attention_map.flatten(), kde=False, bins=100, color="grey", edgecolor="black", ax=axs[1][1])
    axs[1][1].set_title("Attention Map histogram")
    axs[1][1].set_xlabel("AM Values")
    axs[1][1].set_ylabel("Frequency")
    # Calculate and display the mean and variance of the pixel values
    axs[1][1].text(0.5, -0.15, f"Mean: {attention_map.flatten().mean():.3e} Var: {attention_map.flatten().var():.3e}", ha='center', va='top', transform=axs[1][1].transAxes, fontsize=11, color='black')


    # CDAM histogram
    #pixel_values = cdam.flatten() #.flatten()
    sns.histplot(cdam.flatten(), kde=False, bins=100, color="grey", edgecolor="black", ax=axs[1][2])
    axs[1][2].set_title("Raw CDAM histogram")
    axs[1][2].set_xlabel("CDAM Values")
    axs[1][2].set_ylabel("Frequency")
    # Calculate and display the mean and variance of the pixel values
    axs[1][2].text(0.5, -0.15, f"Mean: {cdam.flatten().mean():.3e} Var: {cdam.flatten().var():.3e}", ha='center', va='top', transform=axs[1][2].transAxes, fontsize=11, color='black')

    if save_name:
        if not os.path.exists("relevance_maps"):
            os.makedirs("relevance_maps")
        plt.savefig(f"relevance_maps/{save_name}_{key}.png", format="png", transparent=True, bbox_inches='tight')

    return None


def plot_attention_with_frame(
    original,
    maps,
    stats,
    ROI: np.ndarray | None = None,
    linewidths=1,
    loc_x=None,
    loc_y=None,
    n=None,
    titles=None,
    savename=None,
    figsize=(14, 7),
)->None:
    """Plot original image, attention maps, and stats using matplotlib."""

    with plt.rc_context({"font.size": 14}):
        num_plots = 1 + len(maps) + len(stats)
        fig, axs = plt.subplots(
            1, num_plots, figsize=figsize, layout="constrained"
        )
        axs = axs.ravel()

        # def maybe_add_rect(ax):
        #     if loc_x is not None and loc_y is not None and n is not None:
        #         ax.add_patch(
        #             patches.Rectangle(
        #                 (loc_x, loc_y),
        #                 n,
        #                 n,
        #                 linewidth=1,
        #                 edgecolor="r",
        #                 facecolor="none",
        #             )
        #         )

        def add_contour(ax):
            if ROI is not None:
                ax.contour(ROI, levels=[0.5], colors = 'red', linewidths=linewidths)
            if loc_x is not None and loc_y is not None and n is not None:
                ax.add_patch(
                    patches.Rectangle(
                        (loc_x, loc_y),
                        n,
                        n,
                        linewidth=linewidths,
                        edgecolor="r",
                        facecolor="none",
                    )
                )

        if titles is None:
            titles = [""] * (len(maps) + len(stats))

        # Original image
        axs[0].imshow(original)
        #maybe_add_rect(axs[0])
        add_contour(axs[0])
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        # Attention maps
        cmap_maps = get_attention_cmap(maps[0])
        for i, m in enumerate(maps, start=1):
            im = axs[i].imshow(m, cmap=cmap_maps)
            #maybe_add_rect(axs[i])
            add_contour(axs[i])
            axs[i].set_title(titles[i - 1])
            axs[i].axis("off")
        fig.colorbar(im, ax=axs[len(maps)], shrink=0.3)

        # Stats maps
        offset = 1 + len(maps)
        for i, m in enumerate(stats):
            ax = axs[offset + i]
            cmap = "plasma" if (m.min() >= 0 and m.max() <= 1) else get_attention_cmap(m)
            im = ax.imshow(m, cmap=cmap)
            ax.set_title(titles[len(maps) + i])
            ax.axis("off")
            fig.colorbar(im, ax=ax, shrink=0.3)

        if savename:
            plt.savefig(
                f"../results/plots/attention_{savename}.pdf",
                bbox_inches="tight",
                pad_inches=0,
            )

        plt.show()


def plot_attention_scores_before_and_after(
    am_before: np.ndarray,
    ams_reg: list[np.ndarray],
    titles: list[str] | None = None,
    savename: str | None = None,
    figsize=(16, 4),
    color='tab:red',
)->None:
    """
    Plot the attention scores after regularization vs. the scores before regularization for a set of regularized attention maps
    """

    nplots = len(ams_reg)

    if titles is None:
        titles = [""]*nplots

    assert len(titles) == nplots, "The length of the list of titles should be the same as the length of the list of attention maps."
    
    obs_min = am_before.min()
    obs_max = am_before.max()
    obs_line = np.linspace(obs_min,obs_max,100)

    fig, axes = plt.subplots(nrows=1, ncols=nplots, figsize=figsize)

    axes[0].set_ylabel("After reg.")

    for i, am_reg in enumerate(ams_reg):  

        axes[i].scatter(am_before,am_reg,color=color)
        axes[i].plot(obs_line,obs_line,'--')
        axes[i].set_xlabel("Before reg.")
        axes[i].set_xlim(obs_min,obs_max)
        axes[i].set_ylim(obs_min,obs_max)
        #axes[i].set_xticks([0.0, 0.001, 0.002, 0.003])
        axes[i].set_title(titles[i])
        axes[i].grid()

    for i in range(nplots):
        axes[i].set_rasterized(True)  

    if savename:
        plt.savefig(
            f"relevance_maps/attn_bef_aft_{savename}_p_reg.pdf", 
            format="pdf", 
            bbox_inches="tight", 
            dpi=300,
        )
        
    plt.show()


def plot_D_histogram(
    D_values, 
    D_errors, 
    ax=None, 
    show=True, 
    ylim=(0, 0.8),
    p_label="p reg.",
    l_label="lfdr reg.",
    pi0_label=r"$\pi_0$ reg.",
    xlabel=None,
    xtick_labels=None,  
):
    """
    D_values: list of (D_p, D_lfdr, D_pi0)
    D_errors: list of (err_p, err_lfdr, err_pi0)
    xtick_labels: optional list of labels for x ticks
    """

    if isinstance(D_values[0], float):
        D_values = [D_values]
        D_errors = [D_errors]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    colors = {
        "p": "tab:red",
        "lfdr": "gold",
        "pi0": "limegreen",
    }

    x = np.arange(1, len(D_values) + 1)
    width = 0.2

    D_p_vals = [d[0] for d in D_values]
    D_l_vals = [d[1] for d in D_values]
    D_pi_vals = [d[2] for d in D_values]

    err_p_vals = [e[0] for e in D_errors]
    err_l_vals = [e[1] for e in D_errors]
    err_pi_vals = [e[2] for e in D_errors]

    ax.bar(x - width, D_p_vals, width, yerr=err_p_vals,
           label=p_label, color=colors["p"])

    ax.bar(x, D_l_vals, width, yerr=err_l_vals,
           label=l_label, color=colors["lfdr"])

    ax.bar(x + width, D_pi_vals, width, yerr=err_pi_vals,
           label=pi0_label, color=colors["pi0"])

    ax.set_xticks(x)

    if xtick_labels is not None:
        if len(xtick_labels) != len(x):
            raise ValueError("xtick_labels must have the same length as D_values")
        ax.set_xticklabels(xtick_labels)
    else:
        ax.set_xticklabels([str(i) for i in x])

    ax.set_ylim(*ylim)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("D", fontsize=12)

    ax.grid()

    if show:
        plt.show()
        
    return ax


def plot_regularization_scatter(path, mode="mean_percentile", xlims=(0,100), ylims=(0, 100), savename: str = None):
    """
    Plot scatter comparison before vs after regularization.

    Parameters
    ----------
    path : str or Path
        Path to a CSV file OR a directory containing CSV files.
    mode : str
        "mean_percentile" or "nonzero_attention".
    y_limits : tuple
        Limits for the y-axis.
    """

    path = Path(path)

    configs = {
        "mean_percentile": {
            "x": "mean_perc_orig",
            "ys": {
                "p < 0.3": "mean_perc_p",
                "l < 0.3": "mean_perc_lfdr",
                r"$\pi_0$ reg.": "mean_perc_pi0",
            },
            "title": "Mean percentiles of scores in ROI to whole image",
        },
        "nonzero_attention": {
            "x": "nz_orig",
            "ys": {
                "p < 0.3": "nz_p",
                "l < 0.3": "nz_lfdr",
                r"$\pi_0$ reg.": "nz_pi0",
            },
            "title": "Percent of non-zero attention scores",
        },
    }

    if mode not in configs:
        raise ValueError("mode must be 'mean_percentile' or 'nonzero_attention'")

    cfg = configs[mode]
    required_cols = [cfg["x"]] + list(cfg["ys"].values())

    # Load data
    if path.is_file():
        df = pd.read_csv(path)

    elif path.is_dir():
        dfs = []
        for csv_file in path.glob("*.csv"):
            temp = pd.read_csv(csv_file)
            cols = [c for c in required_cols if c in temp.columns]
            dfs.append(temp[cols])

        if not dfs:
            raise ValueError("No CSV files found in directory.")

        df = pd.concat(dfs, ignore_index=True)

    else:
        raise ValueError("Provided path is neither a file nor a directory.")

    # Color scheme
    colors = {
        "p < 0.3": "tab:red",
        "l < 0.3": "gold",
        r"$\pi_0$ reg.": "limegreen"
    }

    alphas = {
        "p < 0.3": 0.7,
        "l < 0.3": 0.4,
        r"$\pi_0$ reg.": 0.3
    }

    plt.figure(figsize=(8, 6))
    #fig, ax = plt.subplots(figsize=(8,6))

    # Plot each regularization type
    for label, y_col in cfg["ys"].items():
        plt.scatter(
            df[cfg["x"]],
            df[y_col],
            color=colors[label],
            alpha=alphas[label],
            label=label
        )

    # Diagonal reference line
    plt.plot([0, 100], [0, 100], linestyle="--", color="blue")

    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.grid()

    plt.xlabel("Before regularization", fontsize=12)
    plt.ylabel("After regularization", fontsize=12)

    plt.legend(title="Regularization", loc="upper left")
    plt.title(cfg["title"])

    #plt.tight_layout()

    if savename:
        plt.savefig(
            f"../results/plots/{mode}_ROI_{savename}.pdf",
            format="pdf",
            bbox_inches="tight",
            dpi=300
        )
        
    plt.show()

