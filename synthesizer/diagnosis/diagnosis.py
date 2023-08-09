import pandas as pd


def duplicate_classes(occurences: dict):
    return occurences


def invert_classes(occurences: dict):
    sorted = pd.Series(occurences).sort_values().to_dict()
    _keys = list(sorted.keys())
    most_popular_class = _keys[-1]
    least_popular_class = _keys[0]

    occurences_per_class = occurences[most_popular_class] \
        + occurences[least_popular_class]
    
    return {
        _k: occurences_per_class - occurences[_k] for _k in occurences.keys()
    }


def fixed_number(occurences: dict, n_to_generate: int):
    return {
        _k: n_to_generate for _k in occurences.keys()
    }


def diagnostic(
    data: pd.DataFrame,
    target: str,
    proportion_strategy: str = "duplicate",
    n_samples: int = None
):
    if not target in data.columns:
        raise ValueError(f"Target '{target}' is not in the dataset")
    
    dataset_length = len(data)
    classes_proportion = {k: float(v) for k, v
                          in data[target].value_counts(1).items()}
    classes_occurences = {k: int(v) for k, v
                          in data[target].value_counts(0).items()}
    dataset_memo = data.memory_usage().sum() / 1000 / 1000

    # Calculates the number of rows to generate
    if proportion_strategy == "duplicate":
        rows_to_generate = duplicate_classes(classes_occurences)
    elif proportion_strategy == "inverse":
        rows_to_generate = invert_classes(classes_occurences)
    elif proportion_strategy == "fixed":
        if not n_samples:
            raise ValueError(f"To generate a fixed number of samples you must "
                             "provide 'n_samples' parameter")
        rows_to_generate = fixed_number(classes_occurences)
    else:
        raise ValueError(f"The strategy '{proportion_strategy}' is not "
                         "implemented")

    diagnosis = {
        "target": str(target),
        "dataset_length": int(dataset_length),
        "memory_usage": float(dataset_memo),
        "classes_occurences": classes_occurences,
        "classes_proportions": classes_proportion,
        "rows_to_generate": rows_to_generate
    }

    return diagnosis
