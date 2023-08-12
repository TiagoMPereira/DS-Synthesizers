from synthesizer.metadata import SDVMetadata
from synthesizer.conditions import create_conditions
from synthesizer.synthesizers import FASTML, GaussianCopula, CTGAN, TVAE, CopulaGAN
import pandas as pd


if __name__ == "__main__":
    path = "./data/preprocessed_body_performance.csv"
    data = pd.read_csv(path)

    metadata = SDVMetadata()
    metadata.create_from_df(data)
    metadata.validate()

    s1 = FASTML(metadata.metadata)
    s2 = GaussianCopula(metadata.metadata)
    s3 = CTGAN(metadata.metadata)
    s4 = TVAE(metadata.metadata)
    s5 = CopulaGAN(metadata.metadata)

    for s in [s1,s2,s3,s4,s5]:
        print(s.name)
        s.fit(data)

        conditions = create_conditions("class", {2: 0, 3: 11, 1: 149, 4: 392})

        gen_data = s.sample_from_conditions(conditions)

        gen_data.to_csv(f"{s.name}_cond.csv")
