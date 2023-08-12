def minority_strategy(occurences: dict):
    min_occurences = min(occurences.values())
    max_occurences = max(occurences.values())
    occurences_gap = max_occurences - min_occurences
    new_occurences = {
        class_: occurences_gap
                if occurences[class_] == min_occurences else 0
                for class_ in occurences.keys()
    }
    return new_occurences


def not_minority_strategy(occurences: dict):
    min_occurences = min(occurences.values())
    max_occurences = max(occurences.values())
    new_occurences = {
        class_: max_occurences - occurences[class_]
                if occurences[class_] != min_occurences else 0
                for class_ in occurences.keys()
    }
    return new_occurences


def not_majority_strategy(occurences: dict):
    max_occurences = max(occurences.values())
    new_occurences = {
        class_: max_occurences - occurences[class_]
                for class_ in occurences.keys()
    }
    return new_occurences


def all_strategy(occurences: dict):
    max_occurences = max(occurences.values())
    new_occurences = {
        class_: max_occurences - occurences[class_]
                for class_ in occurences.keys()
    }
    return new_occurences