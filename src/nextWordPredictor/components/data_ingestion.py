import os

from src.nextWordPredictor.logging import logger
from src.nextWordPredictor.entity import DataIngestionConfig


class DataIngestion:
    """Two supported ways to get the corpus onto disk:

    1. Automatic (Kaggle API): set `kaggle_dataset: "owner/dataset-slug"` in
       config/config.yaml and provide KAGGLE_USERNAME / KAGGLE_KEY as
       environment variables. Requires the `kaggle` pip package.
    2. Manual: download the dataset yourself from Kaggle and place the raw
       file at the path given by `local_data_file` in config.yaml. This is
       the path of least friction and works with ANY Kaggle text dataset
       you choose for this task.
    """

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def _download_via_kaggle(self):
        try:
            import kaggle  # noqa: F401 -- import triggers auth check
        except ImportError:
            raise ImportError(
                "The 'kaggle' package is required to auto-download datasets. "
                "Install it with: pip install kaggle"
            )
        from kaggle.api.kaggle_api_extended import KaggleApi

        logger.info(f"Downloading Kaggle dataset: {self.config.kaggle_dataset}")
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            self.config.kaggle_dataset,
            path=self.config.root_dir,
            unzip=True,
        )
        logger.info(f"Kaggle dataset downloaded and unzipped to: {self.config.root_dir}")

    def fetch_data(self):
        if os.path.exists(self.config.local_data_file):
            logger.info(
                f"Found existing data file at '{self.config.local_data_file}'. "
                "Skipping download."
            )
            return

        if self.config.kaggle_dataset:
            self._download_via_kaggle()

        if not os.path.exists(self.config.local_data_file):
            raise FileNotFoundError(
                f"No data file found at '{self.config.local_data_file}' after "
                "ingestion. Either:\n"
                "  (a) manually download your chosen Kaggle dataset and place "
                "the file at that exact path, or\n"
                "  (b) set 'kaggle_dataset' in config/config.yaml to the "
                "dataset slug (owner/dataset-name) and ensure the downloaded "
                "file's name matches 'local_data_file', or\n"
                "  (c) update 'local_data_file' in config/config.yaml to "
                "match whatever filename Kaggle actually produced."
            )
