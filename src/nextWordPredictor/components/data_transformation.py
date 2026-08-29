import re
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from sklearn.model_selection import train_test_split

from src.nextWordPredictor.logging import logger
from src.nextWordPredictor.entity import DataTransformationConfig
from src.nextWordPredictor.config.configuration import ConfigurationManager
from src.nextWordPredictor.utils.common import save_pickle, save_json


class DataTransformation:
    """Builds the classic n-gram training set for next-word prediction:
    for every line in the corpus, generate every prefix of every length as
    a separate training example, with the next token as the label.

    e.g. "the cat sat down" becomes:
        ["the", "cat"]              -> "sat"
        ["the", "cat", "sat"]       -> "down"

    NOTE: sequences are pre-padded (padding="pre") so that every training
    example ends exactly at the position being predicted -- this matters
    for an LSTM the same way it mattered for the SimpleRNN sentiment model:
    the network's final hidden state is most influenced by whatever it
    processed last, so real content must be at the end of the padded
    sequence, not buried under trailing padding.
    """

    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def _load_corpus_text(self) -> str:
        ingestion_config = ConfigurationManager().get_data_ingestion_config()
        data_path = str(ingestion_config.local_data_file)

        if data_path.endswith(".csv"):
            if not ingestion_config.text_column:
                raise ValueError(
                    "local_data_file is a .csv but 'text_column' is not set "
                    "in config/config.yaml. Set it to the column name containing text."
                )
            df = pd.read_csv(data_path)
            lines = df[ingestion_config.text_column].dropna().astype(str).tolist()
        else:
            with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

        # Cap corpus size for tractability -- n-gram expansion grows
        # quadratically with average line length, so very large corpora
        # need this capped for reasonable training time on a laptop/CI runner.
        if self.config.max_lines and len(lines) > self.config.max_lines:
            logger.info(
                f"Corpus has {len(lines)} lines; capping to "
                f"max_lines={self.config.max_lines}"
            )
            lines = lines[: self.config.max_lines]

        return "\n".join(lines)

    @staticmethod
    def _clean_line(line: str) -> str:
        line = line.lower().strip()
        line = re.sub(r"[^a-z0-9'\s]", " ", line)
        line = re.sub(r"\s+", " ", line).strip()
        return line

    def transform(self):
        raw_text = self._load_corpus_text()
        lines = [self._clean_line(l) for l in raw_text.split("\n") if l.strip()]
        logger.info(f"Cleaned corpus down to {len(lines)} non-empty lines")

        tokenizer = Tokenizer(num_words=self.config.num_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(lines)
        vocab_size = min(
            self.config.num_words, len(tokenizer.word_index) + 1
        )
        logger.info(f"Fitted tokenizer. Vocabulary size: {vocab_size}")

        input_sequences = []
        for line in lines:
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                input_sequences.append(token_list[: i + 1])

        if not input_sequences:
            raise ValueError(
                "No training sequences were generated. Check that your "
                "corpus file/column actually contains usable text."
            )

        max_sequence_len = max(len(seq) for seq in input_sequences)
        logger.info(
            f"Generated {len(input_sequences)} n-gram sequences | "
            f"max_sequence_len={max_sequence_len}"
        )

        padded = pad_sequences(
            input_sequences, maxlen=max_sequence_len, padding="pre"
        )
        X, y = padded[:, :-1], padded[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_split,
            random_state=self.config.random_seed,
        )

        np.savez(self.config.train_sequences_file, X=X_train, y=y_train)
        np.savez(self.config.test_sequences_file, X=X_test, y=y_test)
        save_pickle(self.config.tokenizer_path, tokenizer)
        save_json(
            self.config.meta_file,
            {
                "vocab_size": vocab_size,
                "max_sequence_len": max_sequence_len,
            },
        )
        logger.info("Data transformation complete. Artifacts saved.")
