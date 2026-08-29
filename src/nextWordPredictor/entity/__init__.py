from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    local_data_file: Path
    kaggle_dataset: Optional[str]
    text_column: Optional[str]


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    tokenizer_path: Path
    train_sequences_file: Path
    test_sequences_file: Path
    meta_file: Path
    num_words: int
    max_lines: int
    test_split: float
    random_seed: int


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    trained_model_path: Path
    meta_file: Path
    train_sequences_file: Path
    embedding_dim: int
    lstm_units: List[int]
    dropout: float
    epochs: int
    batch_size: int
    validation_split: float
    early_stopping_patience: int
    monitor_metric: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    trained_model_path: Path
    test_sequences_file: Path
    metric_file_name: Path


@dataclass(frozen=True)
class PredictionConfig:
    trained_model_path: Path
    tokenizer_path: Path
    meta_file: Path
