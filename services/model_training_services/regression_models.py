import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Any

from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

from models.enums import RegressionModels
from models.models import RegressionModelPerformance


def get_regression_model(model_name: RegressionModels) -> Any:
    if model_name == RegressionModels.LINEAR_REGRESSION:
        model = LinearRegression()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def evaluate_regression_model(model_name: str, model_path: str, expected_values: pd.DataFrame, predicted_values: pd.DataFrame) -> RegressionModelPerformance:
    performance = RegressionModelPerformance(model=model_name,
                                      model_path=model_path,
                                      mean_absolute_error=mean_absolute_error(expected_values, predicted_values),
                                      mean_squared_error=mean_squared_error(expected_values, predicted_values),
                                      root_mean_squared_error=root_mean_squared_error(expected_values, predicted_values),
                                      r2_score=r2_score(expected_values, predicted_values))
    performance.pretty_print()
    return performance
