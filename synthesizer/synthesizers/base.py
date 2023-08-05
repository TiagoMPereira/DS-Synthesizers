from sdv.metadata import SingleTableMetadata
import pandas as pd


class BaseSynthesizer(object):

    def __init__(self, metadata: SingleTableMetadata, name: str):
        self.metadata = metadata
        self.name = name
        self.locales = ['en-us']
        self.verbose = False
        self.cuda = False

    def _create_models(self):
        self.synthesizer = None

    def fit(self, data: pd.DataFrame) -> None:
        self.synthesizer.fit(data)

    def sample(self, n_rows: int) -> pd.DataFrame:
        generated_data = self.synthesizer.sample(num_rows=n_rows)
        return generated_data
