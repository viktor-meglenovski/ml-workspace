import os
from pathlib import Path
from typing import Literal

from configurations.constants import INTERMEDIARY_DATASETS_FOLDER
from helpers.logger import logger

import pandas as pd

from models.enums import MissingValueImputationMethod
from models.models import DatasetSplits


def calculate_statistic(data: pd.Series, statistic: MissingValueImputationMethod) -> float:
    if statistic == MissingValueImputationMethod.MEAN:
        return data.mean()
    if statistic == MissingValueImputationMethod.MEDIAN:
        return data.median()
    if statistic == MissingValueImputationMethod.MODE:
        return data.mode().iloc[0]


def save_intermediary_dataset(dataset: pd.DataFrame, working_directory_path: Path, dataset_type: Literal["dropped_unused_columns", "missing_values_handled", "categorical_features_encoded", "continuous_features_scaled", "training", "testing", "validation"], sub_directory: str = None) -> None:
    dataset_name = f"dataset_{dataset_type}.csv"
    dataset_path = working_directory_path / INTERMEDIARY_DATASETS_FOLDER
    if sub_directory:
        dataset_path = dataset_path / sub_directory
    os.makedirs(dataset_path, exist_ok=True)
    dataset_path = dataset_path / dataset_name
    dataset.to_csv(dataset_path, index=False)
    logger.info(f"Dataset snapshot '{dataset_path}' saved.")


def save_dataset_splits(dataset_split: DatasetSplits, working_directory_path: Path, sub_directory: Literal["dataset_splits", "continuous_features_scaled"]) -> None:
    save_intermediary_dataset(dataset_split.training_dataset, working_directory_path, "training", sub_directory)
    save_intermediary_dataset(dataset_split.testing_dataset, working_directory_path, "testing", sub_directory)
    if dataset_split.validation_dataset is not None:
        save_intermediary_dataset(dataset_split.validation_dataset, working_directory_path, "validation", sub_directory)
