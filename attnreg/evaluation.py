import numpy as np
import pandas as pd
import os 
from scipy.stats import uniform
import re
import matplotlib.pyplot as plt

from attnreg.plotting import plot_D_histogram


# SIGNED RMSD
def rmsd(expected, observed):
    """Root Mean Square Deviation"""
    return np.sqrt(np.mean((expected - observed) ** 2))

def pval_srmsd(pvals, causal_indexes, detailed=False):
    """
    Signed RMSD measure of null p-value uniformity.
    
    Parameters
    ----------
    pvals : array-like
        Vector of association p-values to analyze.
        NA values allowed and removed. All remaining must be in [0,1].
    
    causal_indexes : array-like or None
        Indices of causal loci to exclude. If None, assumes all are null.
    
    detailed : bool, default False
        If True, return detailed info for plotting.
    
    Returns
    -------
    float or dict
        If detailed=False, returns the signed RMSD.
        If detailed=True, returns a dictionary with:
        - 'srmsd': signed RMSD
        - 'pvals_null': sorted null p-values (excluding NAs)
        - 'pvals_unif': expected order statistics under uniform distribution
    """
    if pvals is None:
        return {
            'srmsd': np.nan,
            'pvals_null': None,
            'pvals_unif': None
        } if detailed else np.nan

    pvals = np.asarray(pvals, dtype=np.float64)
    pvals = pvals[~np.isnan(pvals)]

    if causal_indexes is None:
        pvals_null = pvals
    else:
        causal_indexes = np.asarray(causal_indexes, dtype=int)
        if causal_indexes.size == 0:
            raise ValueError("non-NULL `causal_indexes` must have at least one index!")
        
        mask = np.ones(len(pvals), dtype=bool)
        mask[causal_indexes] = False
        pvals_null = pvals[mask]

        if len(pvals_null) == 0:
            raise ValueError("No loci were null (non-causal)!")

    pvals_null = np.sort(pvals_null)

    if np.any(pvals_null < 0) or np.any(pvals_null > 1):
        raise ValueError("Input p-values must be in [0, 1] range!")

    n = len(pvals_null)
    if n == 0:
        return {
            'srmsd': np.nan,
            'pvals_null': [],
            'pvals_unif': []
        } if detailed else np.nan

    expected_quantiles = uniform.ppf((np.arange(1, n + 1) - 0.5) / n)

    srmsd_value = rmsd(expected_quantiles, pvals_null)

    # Add sign: positive if median <= 0.5 (inflation), negative otherwise
    if np.median(pvals_null) > 0.5:
        srmsd_value = -srmsd_value

    if detailed:
        return {
            'srmsd': srmsd_value,
            'pvals_null': pvals_null,
            'pvals_unif': expected_quantiles
        }
    else:
        return srmsd_value


def compute_D_and_error(df, reg_column):
    """
    reg_column should be one of:
    'mean_perc_p', 'mean_perc_lfdr', 'mean_perc_pi0'
    """

    numerator = df[reg_column].sum()
    denominator = df["mean_perc_orig"].sum()

    D = numerator / denominator

    # Error propagation:
    # If D = A/B then:
    # Var(D) ≈ (1/B)^2 Var(A) + (A/B^2)^2 Var(B)

    var_A = df[reg_column].var(ddof=1) * len(df)
    var_B = df["mean_perc_orig"].var(ddof=1) * len(df)

    err = np.sqrt(
        (1 / denominator) ** 2 * var_A +
        (numerator / denominator ** 2) ** 2 * var_B
    )

    return D, err

def compute_all_D_and_plot(results_folder, base_filename=None, ylim=(0, 0.8), xlabel=None, xtick_labels=None, mapping_text=None, p_label="p < 0.3", l_label="l < 0.3", savename: str = None):
    
    D_vals = []
    D_errs = []
    #file_names = []
    
    def natural_key(s):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', s)]
    
    for file in sorted(os.listdir(results_folder), key=natural_key):
        if not file.endswith(".csv"):
            continue
    
        print(file)
    
        file_path = os.path.join(results_folder, file)
        df = pd.read_csv(file_path)
    
        D_p, err_p = compute_D_and_error(df, "mean_perc_p")
        D_lfdr, err_lfdr = compute_D_and_error(df, "mean_perc_lfdr")
        D_pi0, err_pi0 = compute_D_and_error(df, "mean_perc_pi0")
    
        D_vals.append((D_p, D_lfdr, D_pi0))
        D_errs.append((err_p, err_lfdr, err_pi0))
        #file_names.append(file)

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    ax = plot_D_histogram(D_vals, D_errs, ax=ax, show=False, ylim=ylim, xlabel=xlabel, xtick_labels=xtick_labels, p_label=p_label, l_label=l_label)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    if mapping_text is not None:
        ax.text(
            1.03,
            0.55,
            mapping_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
        )

    #plt.tight_layout()

    if savename:
        plt.savefig(
            f"../results/plots/D_histogram_{savename}.pdf",
            format="pdf",
            bbox_inches="tight",
            dpi=300
        )
        
    plt.show()

    return D_vals, D_errs
    

def compute_all_D_for_categories_and_plot(results_folder, base_filename, ylim=(0, 0.8), p_label="p < 0.3", l_label="l < 0.3", savename: str = None):
    """
    Imagenette validation dataset categories
    """
    
    order_of_categories = [
        "n03425413",
        "n02979186",
        "n03394916",
        "n01440764",
        "n03445777",
        "n02102040",
        "n03888257",
        "n03000684",
        "n03417042",
        "n03028079"
    ]

    D_vals = []
    D_errs = []

    # ---------------------------------------------------------
    # Process files in the given order
    # ---------------------------------------------------------
    for name in order_of_categories:

        file = f"{base_filename}_{name}.csv"
        file_path = os.path.join(results_folder, file)

        if not os.path.exists(file_path):
            print(f"Warning: {file} not found")
            continue

        df = pd.read_csv(file_path)

        D_p, err_p = compute_D_and_error(df, "mean_perc_p")
        D_lfdr, err_lfdr = compute_D_and_error(df, "mean_perc_lfdr")
        D_pi0, err_pi0 = compute_D_and_error(df, "mean_perc_pi0")

        D_vals.append((D_p, D_lfdr, D_pi0))
        D_errs.append((err_p, err_lfdr, err_pi0))

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    plot_D_histogram(D_vals, D_errs, ax=ax, show=False, ylim=ylim, p_label=p_label, l_label=l_label)

    # Legend outside
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    # Textbox mapping
    mapping_text = "\n".join(
        f"{i+1}: {name}" for i, name in enumerate(order_of_categories)
    )

    ax.text(
        1.03,
        0.55,
        mapping_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    #plt.tight_layout()

    if savename:
        plt.savefig(
            f"../results/plots/D_histogram_categories_{savename}.pdf",
            format="pdf",
            bbox_inches="tight",
            dpi=300
        )
        
    plt.show()

    return D_vals, D_errs
