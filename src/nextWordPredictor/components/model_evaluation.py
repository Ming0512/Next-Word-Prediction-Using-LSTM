import math

import numpy as np
from tensorflow.keras.models import load_model

from src.nextWordPredictor.logging import logger
from src.nextWordPredictor.entity import ModelEvaluationConfig
from src.nextWordPredictor.utils.common import save_json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate(self):
        test_data = np.load(self.config.test_sequences_file)
        X_test, y_test = test_data["X"], test_data["y"]

        logger.info(f"Loading trained model from: {self.config.trained_model_path}")
        model = load_model(self.config.trained_model_path)

        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
        perplexity = math.exp(test_loss) if test_loss < 20 else float("inf")

        logger.info(
            f"Test Loss: {test_loss} | Test Accuracy: {test_accuracy} | "
            f"Perplexity: {perplexity}"
        )

        save_json(
            path=self.config.metric_file_name,
            data={
                "test_loss": float(test_loss),
                "test_accuracy": float(test_accuracy),
                "perplexity": float(perplexity),
            },
        )
