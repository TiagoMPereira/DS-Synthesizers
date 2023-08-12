from sdv.metadata import SingleTableMetadata
from time import time
import tracemalloc
import pandas as pd
import pickle as pkl


class BaseSynthesizer(object):

    def __init__(self, metadata: SingleTableMetadata, name: str):
        self.metadata = metadata
        self.name = name
        self.locales = ['en-us']
        self.verbose = False
        self.cuda = False
        self.fit_time = None
        self.sample_time = None
        self.fit_memo = None
        self.sample_memo = None

    def _create_models(self):
        self.synthesizer = None

    def _start_memory_measure(self):
        tracemalloc.start()
        memo = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        return memo

    def fit(self, data: pd.DataFrame) -> None:
        base_memo = self._start_memory_measure()
        tracemalloc.start()
        start_time = time()

        self.synthesizer.fit(data)

        self.fit_time = time() - start_time
        self.fit_memo = tracemalloc.get_traced_memory()[1] - base_memo
        tracemalloc.stop()

    def sample(self, n_rows: int) -> pd.DataFrame:
        generated_data = self.synthesizer.sample(num_rows=n_rows)
        return generated_data
    
    def sample_from_conditions(self, conditions: list) -> pd.DataFrame:
        base_memo = self._start_memory_measure()
        tracemalloc.start()
        start_time = time()

        generated_data = self.synthesizer.sample_from_conditions(conditions)

        self.sample_time = time() - start_time
        self.sample_memo = tracemalloc.get_traced_memory()[1] - base_memo
        tracemalloc.stop()

        return generated_data

    def save(self, name: str):
        with open(name+".snt", "wb") as fp:
            pkl.dump(self, fp)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as fp:
            syn = pkl.load(fp)

        return syn
