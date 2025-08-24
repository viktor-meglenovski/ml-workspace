from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, List

import pandas as pd
from pydantic import BaseModel, Field
import os

from models.enums import FeatureType, CategoricalEncodingType, MissingValueImputationMethod, ContinuousFeatureScalingType
from configurations.constants import TEMP_FOLDER_PATH, SUB_FOLDERS
from helpers.logger import logger


class ColumnConfig(BaseModel):
    name: str
    type: FeatureType
    drop: bool
    missing: MissingValueImputationMethod = None
    scale: ContinuousFeatureScalingType = None
    encode: CategoricalEncodingType = None
    encoding_values: dict = None
    target: Optional[bool] = False


class DatasetSplitConfig(BaseModel):
    training: float
    testing: float
    validation: Optional[float] = 0
    random_seed: Optional[int] = None


class DatasetConfig(BaseModel):
    dataset_name: str
    task: Literal["classification", "regression"]
    columns: List[ColumnConfig]
    working_directory_path: Path = None
    dataset_split_config: DatasetSplitConfig

    def model_post_init(self, __context) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        working_directory_name = f"{self.dataset_name}_{timestamp}"
        self.working_directory_path = TEMP_FOLDER_PATH / working_directory_name
        os.makedirs(self.working_directory_path, exist_ok=True)
        logger.info(f"Working directory '{self.working_directory_path}' created")
        for sub_folder in SUB_FOLDERS:
            sub_folder_path = self.working_directory_path / sub_folder
            os.makedirs(sub_folder_path, exist_ok=True)
            logger.info(f"Sub directory '{sub_folder}' created")


class EncodingConfig(BaseModel, ABC):
    original_feature_name: str


class BinaryOneHotEncodingConfig(EncodingConfig):
    encoding_type: CategoricalEncodingType = CategoricalEncodingType.BINARY_ONEHOT
    renamed_feature_name: Optional[str] = None
    mappings: dict


class MultiOneHotEncodingConfig(EncodingConfig):
    encoding_type: CategoricalEncodingType = CategoricalEncodingType.MULTI_ONEHOT
    resulting_columns: list[str]


class OrdinalEncodingConfig(EncodingConfig):
    encoding_type: CategoricalEncodingType = CategoricalEncodingType.ORDINAL
    mappings: dict


class ScalerConfig(BaseModel):
    scaler_type: ContinuousFeatureScalingType
    feature_names: list[str]
    file_path: str


class DatasetSplits:
    def __init__(
        self,
        training_dataset: pd.DataFrame,
        testing_dataset: pd.DataFrame,
        validation_dataset: Optional[pd.DataFrame] = None
    ):
        self.training_dataset = training_dataset
        self.testing_dataset = testing_dataset
        self.validation_dataset = validation_dataset
