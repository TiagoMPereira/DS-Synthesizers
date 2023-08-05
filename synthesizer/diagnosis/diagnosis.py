import pandas as pd


def duplicate_classes(occurences: dict):
    return occurences


def diagnostic(
    data: pd.DataFrame,
    target: str,
    proportion_strategy: str = "duplicate"
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

