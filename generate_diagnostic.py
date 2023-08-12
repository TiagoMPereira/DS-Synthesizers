import pandas as pd
from synthesizer.diagnosis import diagnostic
import json
import os


if __name__ == "__main__":

    base_path = "./data/"
    base_save_path = "./storage/diagnostic/"
    datasets = [
        {"name": "preprocessed_body_performance", "target": "class"},
        {"name": "titanic", "target": "Survived"},
        {"name": "spaceship", "target": "Transported"}
    ]

    if not os.path.exists(base_save_path):
        os.mkdir(base_save_path)

    for dataset in datasets:
        name = dataset["name"]
        target = dataset["target"]
        data = pd.read_csv(f"{base_path}{name}.csv")

        diagnosis = diagnostic(data, target)

        with open(f"{base_save_path}{name}.json", "w") as fp:
            json.dump(diagnosis, fp)
