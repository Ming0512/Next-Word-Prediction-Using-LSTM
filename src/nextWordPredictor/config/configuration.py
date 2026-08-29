from src.nextWordPredictor.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.nextWordPredictor.utils.common import read_yaml, create_directories
from src.nextWordPredictor.entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    PredictionConfig,
)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir=config.root_dir,
            local_data_file=config.local_data_file,
            kaggle_dataset=config.get("kaggle_dataset") or None,
            text_column=config.get("text_column") or None,
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        data_params = self.params.DataParams
        train_params = self.params.TrainingParams
        create_directories([config.root_dir])
        return DataTransformationConfig(
            root_dir=config.root_dir,
            tokenizer_path=config.tokenizer_path,
            train_sequences_file=config.train_sequences_file,
            test_sequences_file=config.test_sequences_file,
            meta_file=config.meta_file,
            num_words=data_params.num_words,
            max_lines=data_params.max_lines,
            test_split=data_params.test_split,
            random_seed=train_params.random_seed,
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        data_transformation_config = self.config.data_transformation
        model_params = self.params.ModelParams
        train_params = self.params.TrainingParams
        create_directories([config.root_dir])
        return ModelTrainerConfig(
            root_dir=config.root_dir,
            trained_model_path=config.trained_model_path,
            meta_file=data_transformation_config.meta_file,
            train_sequences_file=data_transformation_config.train_sequences_file,
            embedding_dim=model_params.embedding_dim,
            lstm_units=model_params.lstm_units,
            dropout=model_params.dropout,
            epochs=train_params.epochs,
            batch_size=train_params.batch_size,
            validation_split=train_params.validation_split,
            early_stopping_patience=train_params.early_stopping_patience,
            monitor_metric=train_params.monitor_metric,
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        model_trainer_config = self.config.model_trainer
        data_transformation_config = self.config.data_transformation
        create_directories([config.root_dir])
        return ModelEvaluationConfig(
            root_dir=config.root_dir,
            trained_model_path=model_trainer_config.trained_model_path,
            test_sequences_file=data_transformation_config.test_sequences_file,
            metric_file_name=config.metric_file_name,
        )

    def get_prediction_config(self) -> PredictionConfig:
        model_trainer_config = self.config.model_trainer
        data_transformation_config = self.config.data_transformation
        return PredictionConfig(
            trained_model_path=model_trainer_config.trained_model_path,
            tokenizer_path=data_transformation_config.tokenizer_path,
            meta_file=data_transformation_config.meta_file,
        )
