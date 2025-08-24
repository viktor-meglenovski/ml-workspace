from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from configurations.constants import LINE_BREAK
from helpers.functions import save_intermediary_dataset
from helpers.logger import logger
from models.models import DatasetConfig, DatasetSplits, DatasetSplitConfig


def split_training_testing_validation_datasets(dataset: pd.DataFrame, config: DatasetConfig) -> DatasetSplits:
    try:
        logger.info(LINE_BREAK)
        logger.info("Split dataset with following configuration:")
        dataset_split_config = config.dataset_split_config
        __print_dataset_split_configuration_info(dataset_split_config)

        training_dataset, testing_and_validation_dataset = train_test_split(
            dataset,
            train_size=dataset_split_config.training,
            random_state=dataset_split_config.random_seed
        )

        testing_dataset = testing_and_validation_dataset
        validation_dataset = None

        if dataset_split_config.validation:
            testing_dataset, validation_dataset = __split_testing_validation_dataset(dataset_split_config, testing_and_validation_dataset)

        logger.info("Original dataset split into datasets:")
        logger.info(f"\tTraining ({training_dataset.shape[0]} rows)")
        logger.info(f"\tTesting ({testing_dataset.shape[0]} rows)")
        if validation_dataset is not None:
            logger.info(f"\tValidation ({validation_dataset.shape[0]} rows)")

        dataset_splits = DatasetSplits(training_dataset=training_dataset,
                                       testing_dataset=testing_dataset,
                                       validation_dataset=validation_dataset)
        __save_dataset_splits(dataset_splits, config.working_directory_path)
        logger.info("Dataset splitting finished.")
        return dataset_splits
    except Exception as exception:
        logger.error(f"Error while splitting dataset in Training, Testing and Validation datasets. Exception: {str(exception)}")
        raise


def __print_dataset_split_configuration_info(dataset_split_config: DatasetSplitConfig) -> None:
    logger.info(f"\tTraining: {dataset_split_config.training * 100}%")
    logger.info(f"\tTesting: {dataset_split_config.testing * 100}%")
    logger.info(f"\tValidation: {dataset_split_config.validation * 100}%")
    logger.info(f"\tRandom Seed: {dataset_split_config.random_seed}")

    total = dataset_split_config.training + dataset_split_config.testing + dataset_split_config.validation
    if total != 1:
        raise Exception("The percentage split of the training, testing and validation datasets must add up to 100%")


def __split_testing_validation_dataset(dataset_split_config: DatasetSplitConfig, testing_and_validation_dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    validation_relative_fraction = dataset_split_config.validation / (dataset_split_config.testing + dataset_split_config.validation)
    testing_dataset, validation_dataset = train_test_split(
        testing_and_validation_dataset,
        test_size=validation_relative_fraction,
        random_state=dataset_split_config.random_seed
    )
    return testing_dataset, validation_dataset


def __save_dataset_splits(dataset_split: DatasetSplits, working_directory_path: Path) -> None:
    save_intermediary_dataset(dataset_split.training_dataset, working_directory_path, "training", "dataset_splits")
    save_intermediary_dataset(dataset_split.testing_dataset, working_directory_path, "testing", "dataset_splits")
    if dataset_split.validation_dataset is not None:
        save_intermediary_dataset(dataset_split.validation_dataset, working_directory_path, "validation", "dataset_splits")