from synthesizer.conditions import create_conditions
from synthesizer.diagnosis import diagnostic
from synthesizer.metadata import SDVMetadata
from synthesizer.synthesizers import (CTGAN, FASTML, TVAE, CopulaGAN,
                                      GaussianCopula)
import pandas as pd
from datetime import datetime


def _get_time():
    return f"[{datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}]"


def synthesize_data(
    name: str, data: pd.DataFrame, target_name: str,
    metadata: SDVMetadata = None, diagnosis: dict = {}, data_path: str = ""
):
    
    print(f"{_get_time()} - {name}")
    
    if not metadata:
        metadata = SDVMetadata()
        metadata.create_from_df(data)
    else:
        metadata.validate()

    if not diagnosis:
        diagnosis = diagnostic(data, target_name)
    
    synthesizers = {
        "ctgan": CTGAN(metadata.metadata),
        "fastml": FASTML(metadata.metadata),
        "tvae": TVAE(metadata.metadata),
        "copulagan": CopulaGAN(metadata.metadata),
        "gaussiancopula": GaussianCopula(metadata.metadata),
    }

    classes_to_generate = diagnosis["rows_to_generate"]

    for synt_name, synt in synthesizers.items():

        print(f"{_get_time()} - {synt_name} - {synt.name}")
        
        print(f"{_get_time()} - FIT")
        synt.fit(data)

        print(f"{_get_time()} - SAMPLING")
        conditions = create_conditions(target_name, classes_to_generate)
        generated_data = synt.sample_from_conditions(conditions)

        if data_path:
            save_path = f"{data_path}/synthetic_data/{name}_{synt_name}.csv"
            generated_data.to_csv(save_path)
            save_path = f"{data_path}/synthesizers/{name}_{synt_name}.synt"
            synt.save(save_path)


if __name__ == "__main__":

    base_path = "data/"
    save_path = "storage/"

    datasets = [
        {"name": "preprocessed_body_performance", "target": "class"},
        {"name": "titanic", "target": "Survived"},
        {"name": "spaceship", "target": "Transported"}
    ]

    for dataset in datasets:
        name = dataset["name"]
        target = dataset["target"]

        data = pd.read_csv(f"{base_path}{name}.csv")

        synthesize_data(name, data, target, data_path=save_path)

