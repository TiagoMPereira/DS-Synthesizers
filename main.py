from synthesizer.metadata import SDVMetadata
import pandas as pd


if __name__ == "__main__":
    path = "./data/preprocessed_body_performance.csv"
    data = pd.read_csv(path)

    metadata = SDVMetadata()
    metadata.create_from_df(data)
    metadata.validate()

    metadata.save("meta.json")
    metajs = SDVMetadata.load_from_json("meta.json")
    metajs.validate()

    metadict = metadata.get_metadata_dict()
    meta_dict = SDVMetadata.load_from_dict(metadict)
    meta_dict.validate()
