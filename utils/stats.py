import pandas as pd
import numpy as np
from scipy.stats import kstest, wasserstein_distance
from scipy.spatial.distance import jensenshannon


def _get_stats(values: np.array):
    stats = {
        "max": np.max(values),
        "min": np.min(values),
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values)
    }
    return stats


def stats_ks(original: pd.DataFrame, synthetic: pd.DataFrame):

    columns = list(original.columns)
    assert list(synthetic.columns) == columns

    stats = {
        col: kstest(original[col], synthetic[col]).statistic
        for col in columns
    }
    stats_values = np.array(list(stats.values()))
    _stats = _get_stats(stats_values)
    stats |= _stats

    result = pd.DataFrame(stats, index=[0])

    return result


def stats_jensenshannon(original: pd.DataFrame, synthetic: pd.DataFrame):
    columns = list(original.columns)
    assert list(synthetic.columns) == columns

    stats = {
        col: jensenshannon(original[col], synthetic[col]) for col in columns
    }
    stats_values = np.array(list(stats.values()))
    _stats = _get_stats(stats_values)
    stats |= _stats

    result = pd.DataFrame(stats, index=[0])
    return result


def stats_wasserstein(original: pd.DataFrame, synthetic: pd.DataFrame):
    columns = list(original.columns)
    assert list(synthetic.columns) == columns

    stats = {
        col: wasserstein_distance(original[col], synthetic[col])
        for col in columns
    }
    stats_values = np.array(list(stats.values()))
    _stats = _get_stats(stats_values)
    stats |= _stats

    result = pd.DataFrame(stats, index=[0])
    return result
