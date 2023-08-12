from sdv.sampling import Condition


def create_conditions(target: str, occurences: dict):

    conditions = []
    for class_, n_rows in occurences.items():
        if n_rows == 0:
            continue
        c = Condition(
            num_rows=n_rows,
            column_values={target: class_}
        )
        conditions.append(c)

    return conditions
