from sdv.metadata import SingleTableMetadata


class BaseSynthesizer(object):

    def __init__(self, metadata: SingleTableMetadata, name: str):
        self.metadata = metadata
        self.name = name
        self.locales = ['en-us']
        self.verbose = False
        self.cuda = False

    def _create_models(self):
        pass
