import pandas as pd

from configurations.constants import LINE_BREAK
from helpers.logger import logger
from models.models import DatasetConfig, DatasetSplits


def scale_continuous_features(dataset_splits: DatasetSplits, config: DatasetConfig) -> None:
    logger.info(LINE_BREAK)
    logger.info("Scale continuous features")
