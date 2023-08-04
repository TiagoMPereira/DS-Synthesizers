import pandas as pd
from synthesizer.metadata import SDVMetadata
import os


if __name__ == "__main__":

    base_path = "./data/"
    base_save_path = "./storage/metadata/"
    datasets = [
        {"name": "preprocessed_body_performance"},
        {"name": "titanic"},
        {"name": "spaceship"}
    ]

    if not os.path.exists(base_save_path):
        os.mkdir(base_save_path)

    for dataset in datasets:
        name = dataset["name"]
        data = pd.read_csv(f"{base_path}{name}.csv")

        metadata = SDVMetadata()
        metadata.create_from_df(data)
        metadata.validate()
        if os.path.isfile(f"{base_save_path}{name}.json"):
            os.remove(f"{base_save_path}{name}.json")
        metadata.save(f"{base_save_path}{name}.json")
