import pandas as pd

from helpers.functions import save_intermediary_dataset
from models.models import DatasetConfig
from helpers.logger import logger
from configurations.constants import LINE_BREAK
from services.preprocessing_services.missing_values_handler import handle_missing_values
from services.preprocessing_services.feature_encoding_service import encode_categorical_features
from services.preprocessing_services.dataset_spliting_service import split_training_testing_validation_datasets
from services.preprocessing_services.feature_scaling_service import scale_continuous_features


def preprocess_dataset(dataset: pd.DataFrame, config: DatasetConfig) -> None:
    logger.info(LINE_BREAK)
    logger.info("DATASET PREPROCESSING")
    __drop_unused_columns(dataset, config)
    handle_missing_values(dataset, config)
    encode_categorical_features(dataset, config)
    dataset_splits = split_training_testing_validation_datasets(dataset, config)
    scale_continuous_features(dataset_splits, config)


def __drop_unused_columns(dataset: pd.DataFrame, config: DatasetConfig) -> None:
    try:
        logger.info(LINE_BREAK)
        logger.info("Drop unused columns")
        logger.info(f"\tAll columns ({len(dataset.columns)}): {list(dataset.columns)}")
        columns_to_keep = [column.name for column in config.columns if not column.drop]
        columns_to_drop = [column for column in dataset.columns if column not in columns_to_keep]
        logger.info(f"\tColumns to keep ({len(columns_to_keep)}): {columns_to_keep}")
        logger.info(f"\tColumns to drop ({len(columns_to_drop)}): {columns_to_drop}")
        dataset.drop(columns=columns_to_drop, inplace=True)
        config.columns = [column for column in config.columns if not column.drop]
        logger.info("Unused columns dropped successfully")
        save_intermediary_dataset(dataset, config.working_directory_path, "dropped_unused_columns")
    except Exception as exception:
        logger.error(f"Error while dropping unused columns. Exception: {str(exception)}")
        raise
