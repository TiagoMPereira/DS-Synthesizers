import pandas as pd
from utils.stats import stats_ks, stats_jensenshannon, stats_wasserstein
from utils.plots import plot_histogram, plot_kde, plot_pca
import os


def generate_report(
    id_: str,
    X: pd.DataFrame,
    fake: pd.DataFrame
):
    base_path = f"./results/{id_}/"

    if not(os.path.exists(base_path)):
        os.makedirs(base_path)

    # Calculating metrics
    ks = stats_ks(X, fake)
    jensenshannon = stats_jensenshannon(X, fake)
    wasserstein = stats_wasserstein(X, fake)

    fake.to_csv(f'{base_path}output.csv', index=False)
    ks.to_csv(f'{base_path}ks.csv', index=False)
    jensenshannon.to_csv(f'{base_path}jensenshannon.csv', index=False)
    wasserstein.to_csv(f'{base_path}wasserstein.csv', index=False)

    plot_histogram(X, fake, f'{base_path}histogram.jpg')
    plot_kde(X, fake, f'{base_path}kde.jpg')
    plot_pca(X, fake, f'{base_path}pca.jpg')
