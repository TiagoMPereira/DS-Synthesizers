import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


def plot_histogram(original: pd.DataFrame, synthetic: pd.DataFrame,
                   save_path: str = None):
    columns = list(original.columns)
    assert list(synthetic.columns) == columns

    ncols = 2
    nrows = len(columns)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                           figsize=(ncols * 4, nrows * 4),
                           sharey='row', sharex='row')

    for i, feat in enumerate(columns):
        n_ax = ax[i, 0]
        original[feat].hist(ax=n_ax)
        n_ax.set_title(f"Original {feat}")

        n_ax = ax[i, 1]
        synthetic[feat].hist(ax=n_ax, color="orange")
        n_ax.set_title(f"Syntethic {feat}")

    if save_path:
        plt.savefig(save_path)


def plot_kde(original: pd.DataFrame, synthetic: pd.DataFrame,
             save_path: str = None):
    columns = list(original.columns)
    assert list(synthetic.columns) == columns

    ncols = 3
    nrows = int(math.ceil(len(columns)/ncols))

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                           figsize=(ncols * 4, nrows * 4))

    i, j = 0, 0
    for feat in columns:

        aux_df = pd.DataFrame()
        aux_df[f"original_{feat}"] = original[feat]
        aux_df[f"syntethic_{feat}"] = synthetic[feat]

        n_ax = ax[i, j]

        aux_df.plot.kde(ax=n_ax)
        n_ax.set_title(f"{feat}")

        j += 1
        if j == ncols:
            j = 0
            i += 1

    if save_path:
        plt.savefig(save_path)


def plot_pca(original: pd.DataFrame, synthetic: pd.DataFrame,
             save_path: str = None):
    pca = PCA(n_components=2)

    pca.fit(X=original)
    pca_original = pca.transform(original)
    pca_generated = pca.transform(synthetic)

    plt.figure(figsize=(8, 6))
    plt.title("PCA 2D - Data distribution")

    plt.scatter(pca_original[:, 0], pca_original[:, 1], label="original data")
    plt.scatter(pca_generated[:, 0], pca_generated[:, 1],
                label="synthetic data")
    plt.legend()

    if save_path:
        plt.savefig(save_path)


def plot_correlation(original: pd.DataFrame, synthetic: pd.DataFrame,
                     save_path: str = None):

    plt.figure(figsize=(10, 4))
    sns.heatmap(original.corr().round(3), annot=True, cmap='viridis')
    plt.title("Original data")
    if save_path:
        plt.savefig(save_path+"_original.jpg", bbox_inches="tight")

    plt.figure(figsize=(10, 4))
    sns.heatmap(synthetic.corr().round(3), annot=True, cmap='viridis')
    plt.title("Synthetic data")
    if save_path:
        plt.savefig(save_path+"_synthetic.jpg", bbox_inches="tight")

    plt.figure(figsize=(10, 4))
    sns.heatmap((original.corr() - synthetic.corr()).round(3), annot=True,
                cmap='viridis')
    plt.title("Original - Synthetic data")
    if save_path:
        plt.savefig(save_path+"_original-synthetic.jpg", bbox_inches="tight")
