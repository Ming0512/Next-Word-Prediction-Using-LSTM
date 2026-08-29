import numpy as np
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model

from src.nextWordPredictor.config.configuration import ConfigurationManager
from src.nextWordPredictor.logging import logger
from src.nextWordPredictor.utils.common import load_pickle, load_json


class PredictionPipeline:
    """Loads the trained model + tokenizer once, then generates text by
    repeatedly predicting the next word and appending it to the seed text.

    Uses the SAME padding direction ("pre") used during training -- a
    mismatch here would silently degrade predictions the same way it did
    for the sentiment model.
    """

    def __init__(self):
        config = ConfigurationManager()
        prediction_config = config.get_prediction_config()

        logger.info(f"Loading trained model from: {prediction_config.trained_model_path}")
        self.model = load_model(prediction_config.trained_model_path)
        self.tokenizer = load_pickle(prediction_config.tokenizer_path)

        meta = load_json(prediction_config.meta_file)
        self.max_sequence_len = meta.max_sequence_len

        # Reverse lookup: token id -> word
        self.index_word = {index: word for word, index in self.tokenizer.word_index.items()}

    def _predict_next_token(self, seed_text: str, temperature: float = 0.0) -> int:
        token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
            [token_list], maxlen=self.max_sequence_len - 1, padding="pre"
        )
        probabilities = self.model.predict(token_list, verbose=0)[0]

        if temperature and temperature > 0:
            # Sample with temperature for more varied / "creative" output.
            probabilities = np.asarray(probabilities).astype("float64")
            probabilities = np.log(probabilities + 1e-9) / temperature
            probabilities = np.exp(probabilities)
            probabilities = probabilities / np.sum(probabilities)
            predicted_index = np.random.choice(len(probabilities), p=probabilities)
        else:
            # Greedy decoding (deterministic, most likely next word).
            predicted_index = int(np.argmax(probabilities))

        return predicted_index

    def generate(self, seed_text: str, num_words: int = 10, temperature: float = 0.0) -> str:
        result = seed_text.strip()
        for _ in range(num_words):
            predicted_index = self._predict_next_token(result, temperature=temperature)
            predicted_word = self.index_word.get(predicted_index, "")
            if not predicted_word:
                break
            result += " " + predicted_word
        return result
