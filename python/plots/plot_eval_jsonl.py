import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_metrics(csv_path: str, save_dir: str = None, show: bool = True,
                 name_boxplot: str = "metrics_boxplot.pdf",
                 name_histograms: str = "metrics_histograms.pdf"):
    
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    metric_cols = [
        "accuracy_global",
        "precision_0", "recall_0", "f1_0",
        "precision_1", "recall_1", "f1_1"
    ]
    support_cols = ["support_0", "support_1"]

    col_rename = {
        "accuracy_global": "Accuracy (global)",
        "precision_0": "Precision (class 0)",
        "recall_0": "Recall (class 0)",
        "f1_0": "F1-score (class 0)",
        "precision_1": "Precision (class 1)",
        "recall_1": "Recall (class 1)",
        "f1_1": "F1-score (class 1)",
        "support_0": "Support (class 0)",
        "support_1": "Support (class 1)"
    }

    for c in metric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            raise KeyError(f"Missing expected column: {c}")

    for c in support_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        else:
            raise KeyError(f"Missing expected column: {c}")

    # Create output directory if requested
    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = None

    # 1) Metrics histograms
    metrics_df = df[metric_cols].copy()
    n_metrics = len(metric_cols)
    ncols = 3
    nrows = int(np.ceil(n_metrics / ncols))

    fig1, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten()

    for i, col in enumerate(metric_cols):
        ax = axes[i]
        series = metrics_df[col].dropna()
        if series.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(col_rename.get(col, col))
            continue

        # Basic histogram
        ax.hist(series, bins=30)
        ax.set_title(col_rename.get(col, col))
        ax.set_xlabel(col_rename.get(col, col))
        ax.grid(axis="y", alpha=0.75)
        ax.set_ylabel("Number of samples")

        # Indicative lines: mean and median
        m = series.mean()
        med = series.median()
        ax.axvline(m, linestyle="--", linewidth=1)
        ax.axvline(med, linestyle=":", linewidth=1)
        ax.text(0.02, 0.95, f"mean = {m:.3f}\nmed = {med:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=9)

    # Remove unused axes if n_metrics < nrows*ncols
    for j in range(n_metrics, nrows*ncols):
        fig1.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if out:
        p = out / name_histograms
        fig1.savefig(p, dpi=150)
        print(f"Saved: {p}")

    # 2) Boxplot comparing metrics
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    box_df = metrics_df.dropna(how="any")
    ax2.boxplot([box_df[c] for c in metric_cols], labels=[col_rename[c] for c in metric_cols], vert=True)
    ax2.set_ylabel("Value")
    plt.xticks(rotation=25)
    ax2.yaxis.grid(True)
    plt.tight_layout()
    if out:
        p = out / name_boxplot
        fig2.savefig(p, dpi=150)
        print(f"Saved: {p}")

    # 4) Print descriptive statistics
    desc = metrics_df.describe().T 
    # ajouter mediane (describe ne donne pas mediane)
    medians = metrics_df.median()
    desc["median"] = medians
    print("\nDescriptive statistics (metrics):")
    print(desc[["count","mean","median","std","min","25%","50%","75%","max"]])

    if show:
        plt.show()

    # Return some useful objects if used as a function
    return {
        "df": df,
        "metrics_df": metrics_df,
        "figures": {
            "metrics_histograms": fig1,
            "metrics_boxplot": fig2
        },
        "summary": desc
    }

if __name__ == "__main__":
    # Train tiny
    csv_path = "../data/test_trainData_tiny.csv"
    outdir = "./plots"
    name_boxplot = "metrics_boxplot_train_tiny.pdf"
    name_histograms = "metrics_histograms_train_tiny.pdf"
    plot_metrics(csv_path, save_dir=outdir, show=False, name_boxplot=name_boxplot, name_histograms=name_histograms)

    # Test tiny
    csv_path = "../data/test_testData_tiny.csv"
    outdir = "./plots"
    name_boxplot = "metrics_boxplot_test_tiny.pdf"
    name_histograms = "metrics_histograms_test_tiny.pdf"
    plot_metrics(csv_path, save_dir=outdir, show=False, name_boxplot=name_boxplot, name_histograms=name_histograms)
