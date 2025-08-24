from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from models.enums import ClassificationModels
from models.models import ClassificationModelPerformance


def get_classification_model(model_name: ClassificationModels) -> Any:
    if model_name == ClassificationModels.LOGISTIC_REGRESSION:
        model = LogisticRegression(max_iter=1000)
    elif model_name == ClassificationModels.RANDOM_FOREST:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == ClassificationModels.GRADIENT_BOOSTING:
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_name == ClassificationModels.DECISION_TREE:
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == ClassificationModels.SUPPORT_VECTOR_MACHINE:
        model = SVC(probability=True, random_state=42)
    elif model_name == ClassificationModels.KNN:
        model = KNeighborsClassifier()
    elif model_name == ClassificationModels.NAIVE_BAYES:
        model = GaussianNB()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def evaluate_classification_model(model_name: str, model_path: str, expected_values: pd.DataFrame, predicted_values: pd.DataFrame) -> ClassificationModelPerformance:
    performance = ClassificationModelPerformance(model=model_name,
                                          model_path=model_path,
                                          accuracy=accuracy_score(expected_values, predicted_values),
                                          precision=precision_score(expected_values, predicted_values, average="weighted", zero_division=0),
                                          recall=recall_score(expected_values, predicted_values, average="weighted", zero_division=0),
                                          f1=f1_score(expected_values, predicted_values, average="weighted", zero_division=0))
    performance.pretty_print()
    return performance
