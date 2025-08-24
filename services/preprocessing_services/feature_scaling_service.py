import json
from pathlib import Path
from typing import Tuple, List, Union

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from configurations.constants import LINE_BREAK, SCALERS_CONFIG_FILE, SCALERS_FOLDER
from helpers.functions import save_dataset_splits
from helpers.logger import logger
from models.enums import FeatureType, ContinuousFeatureScalingType
from models.models import DatasetConfig, DatasetSplits, ColumnConfig, ScalerConfig


def scale_continuous_features(dataset_splits: DatasetSplits, config: DatasetConfig) -> None:
    try:
        logger.info(LINE_BREAK)
        logger.info("Scale continuous features")

        all_continuous_features = [feature for feature in config.columns if feature.type == FeatureType.CONTINUOUS]

        all_continuous_features_names = [feature.name for feature in all_continuous_features]
        standard_scaler_features_names = [feature.name for feature in all_continuous_features if feature.scale == ContinuousFeatureScalingType.STANDARD]
        minmax_scaler_features_names = [feature.name for feature in all_continuous_features if feature.scale == ContinuousFeatureScalingType.MINMAX]

        __print_scaling_configurations(all_continuous_features_names, standard_scaler_features_names, minmax_scaler_features_names)

        scaler_configs = list()

        if standard_scaler_features_names:
            standard_scaler = __create_standard_scaler(dataset_splits, standard_scaler_features_names)
            __apply_scaler_to_datasets(dataset_splits, standard_scaler, standard_scaler_features_names)
            scaler_config = __save_scaler(config.working_directory_path, ContinuousFeatureScalingType.STANDARD, standard_scaler, standard_scaler_features_names)
            scaler_configs.append(scaler_config)

        if minmax_scaler_features_names:
            minmax_scaler = __create_minmax_scaler(dataset_splits, minmax_scaler_features_names)
            __apply_scaler_to_datasets(dataset_splits, minmax_scaler, minmax_scaler_features_names)
            scaler_config = __save_scaler(config.working_directory_path, ContinuousFeatureScalingType.MINMAX, minmax_scaler, minmax_scaler_features_names)
            scaler_configs.append(scaler_config)

        __save_scaler_configs(config.working_directory_path, scaler_configs)
        save_dataset_splits(dataset_splits, config.working_directory_path, "continuous_features_scaled")
    except Exception as exception:
        logger.error(f"Error while scaling continuous features. Exception: {str(exception)}")
        raise


def __print_scaling_configurations(all_continuous_features_names: list[str],
                                   standard_scaler_features_names: list[str],
                                   minmax_scaler_features_names: list[str]) -> None:
    logger.info(f"\tAll continuous features {len(all_continuous_features_names)}: {all_continuous_features_names}")
    logger.info(f"\tStandard scaler features {len(standard_scaler_features_names)}: {standard_scaler_features_names}")
    logger.info(f"\tMinMax scaler features {len(minmax_scaler_features_names)}: {minmax_scaler_features_names}")

    no_scaler_feature_names = [feature_name
                               for feature_name in all_continuous_features_names
                               if feature_name not in standard_scaler_features_names and feature_name not in minmax_scaler_features_names]
    logger.info(f"\tNo scaling features {len(no_scaler_feature_names)}: {no_scaler_feature_names}")


def __create_standard_scaler(dataset_splits: DatasetSplits, columns: List[str]) -> StandardScaler:
    logger.info(f"\tFitting Standard Scaler on training dataset on features: '{columns}'")
    scaler = StandardScaler()
    scaler.fit(dataset_splits.training_dataset[columns])
    logger.info(f"\tStandard Scaler fitted successfully on training dataset on features: '{columns}'")
    return scaler


def __create_minmax_scaler(dataset_splits: DatasetSplits, columns: List[str]) -> MinMaxScaler:
    logger.info(f"\tFitting MinMax Scaler on training dataset on features: '{columns}'")
    scaler = MinMaxScaler()
    scaler.fit(dataset_splits.training_dataset[columns])
    logger.info(f"\tMinMax Scaler fitted successfully on training dataset on features: '{columns}'")
    return scaler


def __apply_scaler_to_datasets(dataset_splits: DatasetSplits, scaler: Union[StandardScaler, MinMaxScaler], feature_names: List[str]) -> None:
    logger.info(f"\tApplying scaler to '{feature_names}'")
    dataset_splits.training_dataset[feature_names] = \
        scaler.transform(dataset_splits.training_dataset[feature_names])
    logger.info(f"\t\tApplied to training dataset")

    dataset_splits.testing_dataset[feature_names] = \
        scaler.transform(dataset_splits.testing_dataset[feature_names])
    logger.info(f"\t\tApplied to testing dataset")

    if dataset_splits.validation_dataset is not None:
        dataset_splits.validation_dataset[feature_names] = \
            scaler.transform(dataset_splits.validation_dataset[feature_names])
        logger.info(f"\t\tApplied to validation dataset")


def __save_scaler(working_directory_path: Path, scaler_type: ContinuousFeatureScalingType, scaler: Union[StandardScaler, MinMaxScaler], feature_names: List[str]) -> ScalerConfig:
    logger.info(f"\tSaving scaler type '{scaler_type.value}'")
    scaler_path = working_directory_path / SCALERS_FOLDER / f"{scaler_type.value}.pkl"
    joblib.dump(scaler, scaler_path)
    scaler_config = ScalerConfig(scaler_type=scaler_type, feature_names=feature_names, file_path=str(scaler_path))
    logger.info(f"\tScaler type '{scaler_type.value}' saved successfully at '{scaler_path}'.")
    return scaler_config


def __save_scaler_configs(working_directory_path: Path, scaler_configs: list) -> None:
    scalers_config_file_path = working_directory_path / SCALERS_CONFIG_FILE
    scalers_config_json = [json.loads(config.json()) for config in scaler_configs]
    with open(scalers_config_file_path, 'w') as f:
        json.dump(scalers_config_json, f, indent=4)
    logger.info(f"Scalers configurations saved at '{scalers_config_file_path}'")
