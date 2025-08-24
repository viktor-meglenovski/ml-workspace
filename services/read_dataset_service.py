import json
from typing import Dict, Tuple
import pandas as pd
import os

from configurations.constants import SAMPLE_DATASETS_PATH, LINE_BREAK, SAMPLE_DATASET_FILE_NAME, SAMPLE_CONFIG_FILE_NAME
from helpers.logger import logger
from models.models import DatasetConfig


def __list_sample_datasets() -> list:
    sample_datasets = os.listdir(SAMPLE_DATASETS_PATH)
    return sample_datasets


def pick_sample_dataset() -> str:
    logger.info(LINE_BREAK)
    logger.info("PICK SAMPLE DATASET")
    logger.info(LINE_BREAK)
    sample_datasets = __list_sample_datasets()
    logger.info("Choose one of the available datasets:")
    for i, dataset_name in enumerate(sample_datasets, 1):
        logger.info(f"\t[{i}] {dataset_name}")
    while True:
        try:
            dataset_number = int(input("> "))
            return sample_datasets[dataset_number - 1]
        except Exception:
            logger.warning("Invalid dataset number. Please specify a valid dataset number.")


def read_dataset(dataset_name: str) -> Tuple[pd.DataFrame, DatasetConfig]:
    try:
        dataset_folder_path = SAMPLE_DATASETS_PATH / dataset_name
        if not os.path.exists(dataset_folder_path):
            raise Exception(f"Folder for dataset: '{dataset_folder_path}' not found.")

        dataset_path = dataset_folder_path / SAMPLE_DATASET_FILE_NAME
        config_path = dataset_folder_path / SAMPLE_CONFIG_FILE_NAME

        if not os.path.exists(dataset_path):
            raise Exception(f"File '{SAMPLE_DATASET_FILE_NAME}' not found in dataset folder '{dataset_name}'")

        if not os.path.exists(config_path):
            raise Exception(f"File '{SAMPLE_CONFIG_FILE_NAME}' not found in dataset folder '{dataset_name}'")

        dataset = pd.read_csv(dataset_path)
        logger.info(f"Dataset file '{dataset_path}' read successfully.")

        with open(config_path, "r") as f:
            config_json = json.load(f)
        config = DatasetConfig(**config_json)
        logger.info(f"Dataset config file '{config_path}' read successfully.")

        return dataset, config
    except Exception as exception:
        logger.error(f"Error while reading files for dataset '{dataset_name}'. Exception: {str(exception)}")
        raise
