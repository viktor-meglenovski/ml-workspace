from enum import Enum


class MissingValueImputationMethod(Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"


class FeatureType(Enum):
    STR = "str"
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"


class CategoricalEncodingType(Enum):
    BINARY_ONEHOT = "binary_onehot"
    MULTI_ONEHOT = "multi_onehot"
    ORDINAL = "ordinal"


class ContinuousFeatureScalingType(Enum):
    STANDARD = "standard"
    MINMAX = "minmax"


class ProblemType(Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    UNSUPERVISED = "unsupervised"


class ClassificationModels(Enum):
    LOGISTIC_REGRESSION = "Logistic Regression"
    RANDOM_FOREST = "Random Forest"
    GRADIENT_BOOSTING = "Gradient Boosting"
    DECISION_TREE = "Decision Tree"
    SUPPORT_VECTOR_MACHINE = "Support Vector Machine"
    KNN = "K-Nearest Neighbors"
    NAIVE_BAYES = "Naive Bayes"


class RegressionModels(Enum):
    LINEAR_REGRESSION = "Linear Regression"
