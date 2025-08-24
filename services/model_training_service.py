import json
import os
from pathlib import Path
from typing import List, Tuple, Union, Any

import joblib
import pandas as pd

from configurations.constants import LINE_BREAK, MODELS_FOLDER, MODELS_PERFORMANCE_FILE
from helpers.logger import logger
from models.enums import ProblemType, ClassificationModels, RegressionModels
from models.models import DatasetSplits, DatasetConfig, DatasetsPredictorsAndTargets, ModelPerformance
from services.model_training_services.regression_models import get_regression_model, evaluate_regression_model
from services.model_training_services.classification_models import get_classification_model, evaluate_classification_model


def train_models(dataset_splits: DatasetSplits, dataset_config: DatasetConfig) -> None:
    try:
        logger.info(LINE_BREAK)
        logger.info("MODEL TRAINING")
        logger.info(f"\tProblem type: '{dataset_config.problem_type}'")
        if dataset_config.problem_type == ProblemType.UNSUPERVISED:
            raise NotImplementedError("Unsupervised training automation not implemented yet!")

        dataset_predictors_and_targets = __split_datasets_into_predictors_and_targets(dataset_splits, dataset_config)

        models_to_train = __get_models_to_train(dataset_config)
        logger.info(f"Training models ({len(models_to_train)}): {[model.value for model in models_to_train]}")

        model_evaluations = list()

        for i, model_name in enumerate(models_to_train, 1):
            logger.info(f"\t[{i}/{len(models_to_train)}] Training Model '{model_name.value}'")
            model_evaluation = __train_model(dataset_predictors_and_targets, model_name, dataset_config)
            model_evaluations.append(model_evaluation)

        logger.info("Model training finished")
        __save_model_evaluations(dataset_config.working_directory_path, model_evaluations)
    except Exception as exception:
        logger.error(f"Error while training models. Exception: {str(exception)}")


def __split_datasets_into_predictors_and_targets(dataset_splits: DatasetSplits, dataset_config: DatasetConfig) -> DatasetsPredictorsAndTargets:
    targets = [feature.name for feature in dataset_config.columns if feature.target]
    predictors = list(set(dataset_splits.training_dataset.columns) - set(targets))
    logger.info(f"\tPredictors ({len(predictors)}): '{predictors}'")
    logger.info(f"\tTargets ({len(targets)}): '{targets}'")

    training_x, training_y = __split_predictors_and_targets(dataset_splits.training_dataset, targets)
    testing_x, testing_y = __split_predictors_and_targets(dataset_splits.testing_dataset, targets)
    validation_x, validation_y = None, None
    if dataset_splits.validation_dataset is not None:
        validation_x, validation_y = __split_predictors_and_targets(dataset_splits.validation_dataset, targets)

    return DatasetsPredictorsAndTargets(training_x, training_y, testing_x, testing_y, validation_x, validation_y)


def __split_predictors_and_targets(dataset: pd.DataFrame, target_features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    predictors = dataset.drop(columns=target_features)
    targets = dataset[target_features].copy()
    return predictors, targets


def __get_models_to_train(dataset_config: DatasetConfig) -> List[Union[ClassificationModels, RegressionModels]]:
    if dataset_config.problem_type == ProblemType.REGRESSION:
        return [model for model in RegressionModels]
    elif dataset_config.problem_type == ProblemType.CLASSIFICATION:
        return [model for model in ClassificationModels]


def __train_model(dataset_predictors_and_targets: DatasetsPredictorsAndTargets, model_name: Union[ClassificationModels, RegressionModels], dataset_config: DatasetConfig):
    model = __get_model(dataset_config.problem_type, model_name)
    model.fit(dataset_predictors_and_targets.training_x, dataset_predictors_and_targets.training_y.squeeze())
    model_path = __save_model(model, model_name.value, dataset_config.working_directory_path, dataset_config.problem_type)
    testing_predictions = model.predict(dataset_predictors_and_targets.testing_x)
    model_evaluation = __evaluate_model(model_name.value, model_path, dataset_config.problem_type, dataset_predictors_and_targets.testing_y, testing_predictions)
    return model_evaluation


def __get_model(problem_type: ProblemType, model_name: Union[ClassificationModels, RegressionModels]) -> Any:
    if problem_type == ProblemType.REGRESSION:
        return get_regression_model(model_name)
    elif problem_type == ProblemType.CLASSIFICATION:
        return get_classification_model(model_name)


def __evaluate_model(model_name: str, model_path: str, problem_type: ProblemType, expected_values: pd.DataFrame, predicted_values: pd.DataFrame) -> ModelPerformance:
    if problem_type == ProblemType.REGRESSION:
        return evaluate_regression_model(model_name, model_path, expected_values, predicted_values)
    elif problem_type == ProblemType.CLASSIFICATION:
        return evaluate_classification_model(model_name, model_path, expected_values, predicted_values)


def __save_model(model: Any, model_name: str, working_directory_path: Path, problem_type: ProblemType) -> str:
    logger.info(f"\t\tSaving model '{model_name}'")
    model_path = working_directory_path / MODELS_FOLDER / problem_type.value
    os.makedirs(model_path, exist_ok=True)
    model_path = model_path / f"{model_name}.pkl"
    joblib.dump(model, model_path)
    logger.info(f"\t\tModel '{model_name}' saved successfully at '{model_path}'.")
    return str(model_path)


def __save_model_evaluations(working_directory_path: Path, model_evaluations: List[ModelPerformance]):
    models_performance_path = working_directory_path / MODELS_PERFORMANCE_FILE
    models_performance_json = [json.loads(model_performance.json()) for model_performance in model_evaluations]
    with open(models_performance_path, 'w') as f:
        json.dump(models_performance_json, f, indent=4)
    logger.info(f"Models performance evaluations saved at '{models_performance_path}'")
