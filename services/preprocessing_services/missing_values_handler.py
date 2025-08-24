from typing import Tuple, Dict

import pandas as pd

from configurations.constants import LINE_BREAK
from helpers.functions import calculate_statistic
from helpers.logger import logger
from models.models import DatasetConfig, ColumnConfig


def handle_missing_values(dataset: pd.DataFrame, config: DatasetConfig) -> None:
    try:
        logger.info(LINE_BREAK)
        logger.info("Handle missing values")
        missing_values, total_missing_values = __detect_missing_values(dataset)
        if not total_missing_values:
            logger.info("\tNo missing values in dataset")

        for column in config.columns:
            __handle_missing_value_for_column(dataset, missing_values, column)

        missing_values, total_missing_values = __detect_missing_values(dataset)
        if total_missing_values:
            logger.warning("\tMissing values still exist in the dataset.")
            return
        logger.info("Missing values handled successfully")
    except Exception as exception:
        logger.error(f"Error while handling missing values in dataset. Exception: {str(exception)}")
        raise


def __handle_missing_value_for_column(dataset: pd.DataFrame, missing_values: dict, column_config: ColumnConfig) -> None:
    if not missing_values[column_config.name]:
        logger.debug(f"\tNo missing values in column '{column_config.name}'")
        return
    if not column_config.missing:
        error_message = f"Missing values for column '{column_config.name}' cannot be handled because the handling method is not specified."
        logger.error(f"\t{error_message}")
        raise Exception(error_message)
    logger.info(f"\tHandling missing values for column '{column_config.name}' using '{column_config.missing}' method")
    imputation_value = calculate_statistic(dataset[column_config.name], column_config.missing)
    dataset[column_config.name] = dataset[column_config.name].fillna(imputation_value)
    logger.info(f"\tFilled missing values in column '{column_config.name}' with {column_config.missing} value: {imputation_value}")


def __detect_missing_values(dataset: pd.DataFrame) -> Tuple[Dict, int]:
    missing_values_dict = dict()
    total_missing_values = 0
    missing_values = dataset.isnull().sum()
    logger.info("\tMissing values:")
    for column, missing_value in zip(dataset.columns, missing_values):
        missing_values_dict[column] = missing_value
        total_missing_values += missing_value
        logger.info(f"\t\t{column}: {missing_value}")

    return missing_values_dict, total_missing_values
