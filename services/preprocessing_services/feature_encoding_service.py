import json
from pathlib import Path
from typing import List, Union

import pandas as pd

from configurations.constants import LINE_BREAK, ENCODING_CONFIG_FILE
from helpers.functions import save_intermediary_dataset
from helpers.logger import logger
from models.models import DatasetConfig, ColumnConfig, BinaryOneHotEncodingConfig, MultiOneHotEncodingConfig, \
    OrdinalEncodingConfig, EncodingConfig
from models.enums import FeatureType, CategoricalEncodingType


def encode_categorical_features(dataset: pd.DataFrame, config: DatasetConfig) -> List[EncodingConfig]:
    try:
        logger.info(LINE_BREAK)
        logger.info("Encode categorical features")
        encoded_features = list()
        skipped_features = list()
        encoding_configs = list()
        for feature in config.columns:
            if feature.type != FeatureType.CATEGORICAL:
                continue
            encoding_config = __encode_feature(dataset, feature)
            if encoding_config:
                encoded_features.append(feature.name)
                encoding_configs.append(encoding_config)
            else:
                skipped_features.append(feature.name)
        if encoded_features:
            logger.info(f"Categorical features '{encoded_features}' encoded successfully")
        if skipped_features:
            logger.info(f"Skipped encodings for categorical features '{skipped_features}'")
        __save_encoding_configs(config.working_directory_path, encoding_configs)
        save_intermediary_dataset(dataset, config.working_directory_path, "categorical_features_encoded")
    except Exception as exception:
        logger.error(f"Error while encoding categorical features. Exception: {str(exception)}")
        raise


def __encode_feature(dataset: pd.DataFrame, feature: ColumnConfig) -> Union[EncodingConfig, None]:
    try:
        if feature.encode == CategoricalEncodingType.BINARY_ONEHOT:
            return __binary_onehot_encode_feature(dataset, feature.name)
        if feature.encode == CategoricalEncodingType.MULTI_ONEHOT:
            return __multi_onehot_encode_feature(dataset, feature.name)
        if feature.encode == CategoricalEncodingType.ORDINAL:
            category_mapping = feature.encoding_values
            return __ordinal_encode_feature(dataset, feature.name, category_mapping)
        logger.warning(f"\tSkipping encoding for categorical feature '{feature.name}' because no encoding type is specified.")
        return None
    except Exception as exception:
        logger.error(f"Error while encoding categorical feature '{feature.name}' using '{feature.encode.value}' method. Exception: {str(exception)}")
        raise


def __binary_onehot_encode_feature(dataset: pd.DataFrame, feature_name: str) -> BinaryOneHotEncodingConfig:
    logger.info(f"\tEncoding categorical feature '{feature_name}' using binary onehot encoding.")
    unique_vals = dataset[feature_name].dropna().unique().tolist()
    unique_vals = sorted(unique_vals, key=lambda x: str(x))
    mappings = {unique_vals[0]: 0, unique_vals[1]: 1}
    renamed_column = f"{feature_name}_{unique_vals[1]}"
    dataset[feature_name] = dataset[feature_name].map(mappings)
    dataset.rename(columns={feature_name: renamed_column}, inplace=True)
    logger.info(f"\tCategorical feature '{feature_name}' encoded successfully using binary onehot encoding and renamed to '{renamed_column}'.")
    encoding_config = BinaryOneHotEncodingConfig(original_feature_name=feature_name,
                                                 renamed_feature_name=renamed_column,
                                                 mappings=mappings)
    return encoding_config


def __multi_onehot_encode_feature(dataset: pd.DataFrame, feature_name: str) -> MultiOneHotEncodingConfig:
    logger.info(f"\tEncoding categorical feature '{feature_name}' using multi onehot encoding.")
    one_hot = pd.get_dummies(dataset[feature_name], prefix=feature_name)
    dataset.drop(columns=[feature_name], inplace=True)
    dataset[one_hot.columns] = one_hot
    logger.info(f"\tCategorical feature '{feature_name}' encoded successfully using multi onehot encoding.")
    resulting_columns = one_hot.columns.tolist()
    encoding_config = MultiOneHotEncodingConfig(original_feature_name=feature_name,
                                                resulting_columns=resulting_columns)
    return encoding_config


def __ordinal_encode_feature(dataset: pd.DataFrame, feature_name: str, category_mapping: dict) -> OrdinalEncodingConfig:
    logger.info(f"\tEncoding categorical feature '{feature_name}' using ordinal encoding with mapping: '{category_mapping}'.")
    unique_vals = dataset[feature_name].dropna().unique().tolist()
    missing_mappings = [value for value in unique_vals if value not in category_mapping.keys()]
    if missing_mappings:
        raise Exception(f"Missing mappings for ordinal encoding values: '{missing_mappings}'")
    dataset[feature_name] = dataset[feature_name].map(category_mapping)
    logger.info(f"\tCategorical feature '{feature_name}' encoded successfully using ordinal encoding with mapping: '{category_mapping}'.")
    encoding_config = OrdinalEncodingConfig(original_feature_name=feature_name,
                                            mappings=category_mapping)
    return encoding_config


def __save_encoding_configs(working_directory_path: Path, encoding_configs: list) -> None:
    encoding_config_file_path = working_directory_path / ENCODING_CONFIG_FILE
    encoding_config_json = [json.loads(config.json()) for config in encoding_configs]
    with open(encoding_config_file_path, 'w') as f:
        json.dump(encoding_config_json, f, indent=4)
    logger.info(f"Encoding configurations saved at '{encoding_config_file_path}'")
