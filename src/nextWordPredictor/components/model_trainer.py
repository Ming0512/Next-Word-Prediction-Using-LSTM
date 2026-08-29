import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.nextWordPredictor.logging import logger
from src.nextWordPredictor.entity import ModelTrainerConfig
from src.nextWordPredictor.utils.common import load_json


class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def _build_model(self, vocab_size: int, max_sequence_len: int) -> Sequential:
        input_length = max_sequence_len - 1
        units = self.config.lstm_units

        model = Sequential([
            Input(shape=(input_length,)),
            Embedding(input_dim=vocab_size, output_dim=self.config.embedding_dim),
            LSTM(units[0], return_sequences=True),
            Dropout(self.config.dropout),
            LSTM(units[1] if len(units) > 1 else units[0]),
            Dropout(self.config.dropout),
            Dense(vocab_size, activation="softmax"),
        ])

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train(self):
        meta = load_json(self.config.meta_file)
        vocab_size = meta.vocab_size
        max_sequence_len = meta.max_sequence_len

        train_data = np.load(self.config.train_sequences_file)
        X_train, y_train = train_data["X"], train_data["y"]

        gpus = tf.config.list_physical_devices("GPU")
        device = "/GPU:0" if gpus else "/CPU:0"
        logger.info(f"GPUs available: {gpus if gpus else 'none'} | using {device}")

        model = self._build_model(vocab_size, max_sequence_len)
        model.summary(print_fn=logger.info)

        mode = "max" if "accuracy" in self.config.monitor_metric else "min"
        checkpoint_callback = ModelCheckpoint(
            filepath=str(self.config.trained_model_path),
            monitor=self.config.monitor_metric,
            save_best_only=True,
            mode=mode,
            verbose=1,
        )
        early_stop_callback = EarlyStopping(
            monitor=self.config.monitor_metric,
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            mode=mode,
            verbose=1,
        )

        with tf.device(device):
            model.fit(
                X_train,
                y_train,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_split=self.config.validation_split,
                callbacks=[checkpoint_callback, early_stop_callback],
            )

        logger.info(f"Best model checkpoint saved to: {self.config.trained_model_path}")
