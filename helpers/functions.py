import os
from pathlib import Path
from typing import Literal

from configurations.constants import INTERMEDIARY_DATASETS_FOLDER
from helpers.logger import logger

import pandas as pd

from models.enums import MissingValueImputationMethod


def calculate_statistic(data: pd.Series, statistic: MissingValueImputationMethod) -> float:
    if statistic == MissingValueImputationMethod.MEAN:
        return data.mean()
    if statistic == MissingValueImputationMethod.MEDIAN:
        return data.median()
    if statistic == MissingValueImputationMethod.MODE:
        return data.mode().iloc[0]


def save_intermediary_dataset(dataset: pd.DataFrame, working_directory_path: Path, dataset_type: Literal["dropped_columns", "missing_values_handled", "categorical_features_encoded", "continuous_features_scaled", "training", "testing", "validation"], sub_directory: str = None) -> None:
    dataset_name = f"dataset_{dataset_type}.csv"
    dataset_path = working_directory_path / INTERMEDIARY_DATASETS_FOLDER
    if sub_directory:
        dataset_path = dataset_path / sub_directory
    os.makedirs(dataset_path, exist_ok=True)
    dataset_path = dataset_path / dataset_name
    dataset.to_csv(dataset_path, index=False)
    logger.info(f"Dataset snapshot '{dataset_path}' saved.")
