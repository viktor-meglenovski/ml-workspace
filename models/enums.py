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
